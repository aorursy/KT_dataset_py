!pip install ../input/kaggle-efficientnet-repo/efficientnet-1.0.0-py3-none-any.whl
import numpy as np  # noqa

import pandas as pd

import argparse

import tensorflow as tf

from tqdm.auto import tqdm



from tensorflow.keras import layers as L

import efficientnet.tfkeras as efn





def normalize(image):

  # https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/main.py#L325-L326

  # https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_builder.py#L31-L32

  image -= tf.constant([0.485 * 255, 0.456 * 255, 0.406 * 255])  # RGB

  image /= tf.constant([0.229 * 255, 0.224 * 255, 0.225 * 255])  # RGB

  return image





def get_model_grapheme(input_size, backbone='efficientnet-b7', weights='imagenet', tta=False):

  print(f'Using backbone {backbone} and weights {weights}')

  x = L.Input(shape=input_size, name='imgs', dtype='float32')

  y = normalize(x)

  if backbone.startswith('efficientnet'):

    model_fn = getattr(efn, f'EfficientNetB{backbone[-1]}')



  y = model_fn(input_shape=input_size, weights=weights, include_top=False)(y)

  y = L.GlobalAveragePooling2D()(y)

  # 1292 of 1295 are present

  l1 = L.Dense(168, activation='softmax', name='l1')(y)

  model = tf.keras.Model(x, outputs=[l1])



  return model



def get_model_consonant(input_size, backbone='efficientnet-b3', weights='imagenet', tta=False):

  print(f'Using backbone {backbone} and weights {weights}')

  x = L.Input(shape=input_size, name='imgs', dtype='float32')

  y = normalize(x)

  if backbone.startswith('efficientnet'):

    model_fn = getattr(efn, f'EfficientNetB{backbone[-1]}')



  y = model_fn(input_shape=input_size, weights=weights, include_top=False)(y)

  y = L.GlobalAveragePooling2D()(y)

  #y = L.Dropout(0.2)(y)

  # 1292 of 1295 are present

  #y = L.Dense(168, activation='softmax')(y)

  #y= L.Dense(11, activation='softmax')(y)

  y= L.Dense(7, activation='softmax')(y) 

  #l1 = L.Dense(11,)

  #l2 =  

  model = tf.keras.Model(x, y)

    

  return model





def get_model_vowel(input_size, backbone='efficientnet-b3', weights='imagenet', tta=False):

  print(f'Using backbone {backbone} and weights {weights}')

  x = L.Input(shape=input_size, name='imgs', dtype='float32')

  y = normalize(x)

  if backbone.startswith('efficientnet'):

    model_fn = getattr(efn, f'EfficientNetB{backbone[-1]}')



  y = model_fn(input_shape=input_size, weights=weights, include_top=False)(y)

  y = L.GlobalAveragePooling2D()(y)

  #y = L.Dropout(0.2)(y)

  # 1292 of 1295 are present

  #y = L.Dense(168, activation='softmax')(y)

  y= L.Dense(11, activation='softmax')(y)

  #l2= L.Dense(7, activation='softmax')(y) 



  model = tf.keras.Model(x,y)

  

  return model





import cv2

import numpy as np

import os





def normalize_image(img, org_width, org_height, new_width, new_height):

  # Invert

  img = 255 - img

  # Normalize

  img = (img * (255.0 / img.max())).astype(np.uint8)

  # Reshape

  img = img.reshape(org_height, org_width)

  image_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

  return image_resized





def dump_images(args, org_width, org_height, new_width, new_height):

  labels = pd.read_csv(args.labels)

  iids = labels['image_id']

  root = labels['grapheme_root']

  vowel = labels['vowel_diacritic']

  consonant = labels['consonant_diacritic']

  labels = {a: (b, c, d) for a, b, c, d in zip(iids, root, vowel, consonant)}

  tuples = sorted(set(labels.values()))

  tuples_to_int = {v: k for k, v in enumerate(tuples)}

  print(f'Got {len(tuples)} unique combinations')

  for i in tqdm(range(0, 4)):

    df = pd.read_parquet(args.data_template % i)

    image_ids = df['image_id'].values

    df = df.drop(['image_id'], axis=1)

    for image_id, index in tqdm(zip(image_ids, range(df.shape[0])), total=df.shape[0]):

      normalized = normalize_image(df.loc[df.index[index]].values,

          org_width, org_height, new_width, new_height)

      r, v, c = labels[image_id]

      tuple_int = tuples_to_int[(r, v, c)]

      # e.g: 'Train_300_rt_29_vl_5_ct_0_ti_179.png'

      out_fn = os.path.join(args.image_dir, f'{image_id}_rt_{r}_vl_{v}_ct_{c}_ti_{tuple_int}.png')

      cv2.imwrite(out_fn, normalized)





def decode_predictions(y_pred, inv_tuple_map):

  # return predictions as tuple (root, vowel, consonant)

  y_argmax = np.argmax(y_pred, -1)

  decoded = []

  for yy in y_argmax:

    decoded.append(inv_tuple_map[int(yy)])

  return decoded





def decode_predictions_v2(y_pred1,y_pred2,y_pred3):

  # return predictions as tuple (root / 168, vowel / 11, consonant / 7) & ti 1292



  decoded = []

  for k in range(y_pred1.shape[0]):

    rr=y_pred1[k]

    vv=y_pred2[k]

    cc=y_pred3[k]

    

    rr = rr.argmax(-1)

    vv = vv.argmax(-1)

    cc = cc.argmax(-1)

    decoded.append((rr, vv, cc))



  return decoded





def process_batch(image_id_batch, img_batch, row_id, target, model1, model2, model3):

  img_batch = np.float32(img_batch)

  # deal with single image

  if img_batch.ndim != 4:

    img_batch = np.expand_dims(img_batch, 0)

  

  y_pred1 = model1.predict(img_batch)

  y_pred2 = model2.predict(img_batch)

  y_pred3 = model3.predict(img_batch)

  decoded = decode_predictions_v2(y_pred1,y_pred2,y_pred3)

  for iid, dd in zip(image_id_batch, decoded):

    row_id.append(iid + '_grapheme_root')

    target.append(dd[0])

    row_id.append(iid + '_vowel_diacritic')

    target.append(dd[1])

    row_id.append(iid + '_consonant_diacritic')

    target.append(dd[2])





def main():

  parser = argparse.ArgumentParser()

  parser.add_argument('--seed', type=int, default=123)

  parser.add_argument('--input_size', type=str, default='224,224')

  parser.add_argument('--batch_size', type=int, default=32)

  parser.add_argument('--backbone', type=str, default='efficientnet-b7')

  parser.add_argument('--weights', type=str, default='../input/b3out3/efficientb3_weights00000040.h5')

  args, _ = parser.parse_known_args()



  org_height = 137

  org_width = 236

  args.input_size = tuple(int(x) for x in args.input_size.split(','))

  np.random.seed(args.seed)

  tf.random.set_seed(args.seed)



  model1 = get_model_grapheme(input_size=args.input_size + (3, ),

      weights=None)

  

  model2 = get_model_vowel(input_size=args.input_size + (3, ),

      weights=None)

  

  model3 = get_model_consonant(input_size=args.input_size + (3, ),

      weights=None)



  print(f'Loading weights {args.weights}')

  model1.load_weights('../input/separatemodels/graphemeb7.h5')

  model2.load_weights('../input/vowelconsonant/weights00000096.h5')#vowel model

  model3.load_weights('../input/vowelconsonant/weights00000080.h5')  #consonant model

  #print(model.summary())

  row_id, target = [], []

  image_id_batch, img_batch = [], []

  for i in tqdm(range(4)):

    parquet_fn = f'../input/bengaliai-cv19/test_image_data_{i}.parquet'

    df = pd.read_parquet(parquet_fn)

    image_ids = df['image_id'].values

    df = df.drop(['image_id'], axis=1)

    for k in range(len(image_ids)):

      image_id = image_ids[k]

      img = df.iloc[k].values

      img = normalize_image(img, org_width, org_height, args.input_size[1], args.input_size[0])

      img_batch.append(np.dstack([img] * 3))

      image_id_batch.append(image_id)

      if len(img_batch) >= args.batch_size:

        process_batch(image_id_batch, img_batch, row_id, target, model1, model2, model3)

        image_id_batch, img_batch = [], []



  # process remaining batch

  if len(img_batch) > 0:

    process_batch(image_id_batch, img_batch, row_id, target, model1, model2, model3)

    image_id_batch, img_batch = [], []



  sub_fn = 'submission.csv'

  sub = pd.DataFrame({'row_id': row_id, 'target': target})

  sub.to_csv(sub_fn, index=False)

  print(f'Done wrote to {sub_fn}')



main()

!head submission.csv
!wc -l submission.csv