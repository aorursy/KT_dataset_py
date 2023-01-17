%pylab inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import pandas as pd
df_train = pd.read_csv('/kaggle/input/i2a2-bone-age-regression/train.csv')
df_train.head()
df_train_femele = df_train.query('patientSex == "F"')
df_train_femele.describe()
df_train_femele.hist(column = 'boneage', bins = 40, color = 'blue', alpha = 0.5, figsize=(20, 10))
plt.show()
first_female_group = df_train_femele.query('boneage < 50')
print('Less than 50 months: ', len(first_female_group))

second_female_group = df_train_femele.query('boneage >= 50 and boneage < 100')
print('Between 50 and 100 months: ', len(second_female_group))

third_female_group = df_train_femele.query('boneage >= 100 and boneage < 150')
print('Between 100 and 150 months: ', len(third_female_group))

fourth_female_group = df_train_femele.query('boneage >= 150 and boneage < 200')
print('Between 150 and 200 months: ', len(fourth_female_group))

fifth_female_group = df_train_femele.query('boneage >= 200')
print('More than 200 months: ', len(fifth_female_group))
df_train_male = df_train.query('patientSex == "M"')
df_train_male.hist(column = 'boneage', bins = 40, color = 'blue', alpha = 0.5, figsize=(20, 10))
plt.show()
first_male_group = df_train_male.query('boneage < 50')
print('Less than 50 months: ', len(first_male_group))

second_male_group = df_train_male.query('boneage >= 50 and boneage < 100')
print('Between 50 and 100 months: ', len(second_male_group))

third_male_group = df_train_male.query('boneage >= 100 and boneage < 150')
print('Between 100 and 150 months: ', len(third_male_group))

fourth_male_group = df_train_male.query('boneage >= 150 and boneage < 200')
print('Between 150 and 200 months: ', len(fourth_male_group))

fifth_male_group = df_train_male.query('boneage >= 200')
print('More than 200 months: ', len(fifth_male_group))
import cv2
import numpy as np
import os
def normalize_images(path):
    df = pd.read_csv(f'/kaggle/input/i2a2-bone-age-regression/{path}.csv')

    for filename in df['fileName']:
       _clean_image(path, filename)
def _clean_image(path, filename):
    image = cv2.imread(f"/kaggle/input/i2a2-bone-age-regression/images/{filename}")
    contours = _get_contours(image)
    
    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, -1, (0, 255, 0), 2)
    (x, y, _) = np.where(mask == 255)
 
    if len(x) > 0 and len(y) > 0:
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))

        image = image[topx:bottomx+1, topy:bottomy+1]
        print('contours', len(contours))

        height, width, _ = image.shape
        print('width', width)

        if width > 800:
            width_cutoff = width // 2
            image = image[:, width_cutoff:]

    if not os.path.exists(f'/kaggle/output/kaggle/working/clean-images-{path}'):
      os.makedirs(f'/kaggle/output/kaggle/working/clean-images-{path}')

    cv2.imwrite(f"/kaggle/output/kaggle/working/clean-images-{path}/{filename}", image)
def _get_contours(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_edges = cv2.Canny(image_gray, 40, 180)

    contours, hierarchy = cv2.findContours(image_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    return contours
# normalize_images('train')
df = pd.read_csv(f'/kaggle/input/i2a2-bone-age-regression/train.csv')
df = df.head(10)

for filename in df['fileName']:
  plt.title('Before')
  before = mpimg.imread(f"/kaggle/input/i2a2-bone-age-regression/images/{filename}")
  imgplot = plt.imshow(before)
  plt.show()

  plt.title('After')
  after = mpimg.imread(f"/kaggle/output/kaggle/working/clean-images-train/{filename}")
  imgplot = plt.imshow(after)
  plt.show()
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
image = load_img("/kaggle/output/kaggle/working/clean-images-train/1377.png")
data = img_to_array(image)

samples = expand_dims(data, 0)
datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='constant')

iterator = datagen.flow(samples, batch_size=1)
for i in range(9):
    plt.subplot(330 + 1 + i)
    batch = iterator.next()
    image_generated = batch[0].astype('uint8')
    plt.imshow(image_generated)
plt.show()
def image_generated(filename, number_samples):
  image = load_img(f"/kaggle/output/kaggle/working/clean-images-train/{filename}")
  data = img_to_array(image)

  samples = expand_dims(data, 0)
  datagen = ImageDataGenerator(
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='constant')

  iterator = datagen.flow(samples, batch_size=1)
  new_images = []
  for i in range(number_samples):
    batch = iterator.next()
    image_generated = batch[0].astype('uint8')
    new_images.append(image_generated)
  
  return new_images
def update_dataset(dataset, patientSex, number_samples):
  mean_bornage = dataset['boneage'].mean()

  for filename in dataset['fileName']:
    new_images = image_generated(filename, number_samples)
    for idx, image in enumerate(new_images):
      cv2.imwrite(f"/kaggle/output/kaggle/working/clean-images-train/{idx}-{filename}", image)

      new_data = pd.DataFrame({"fileName":[f"{idx}-{filename}"], "patientSex":[patientSex], "boneage": [mean_bornage]})
      dataset = pd.concat([dataset, new_data], ignore_index=True)
  
  return dataset
first_increased_female_group = update_dataset(first_female_group, 'F', 4).head(1000)
first_increased_female_group
second_female_group = second_female_group.sample(n=1000, random_state=1)
third_female_group = third_female_group.sample(n=1000, random_state=1)
fourth_female_group = fourth_female_group.sample(n=1000, random_state=1)

train_female_final = pd.concat([first_increased_female_group, second_female_group, third_female_group, fourth_female_group], ignore_index=True)
train_female_final.to_csv('/kaggle/output/kaggle/working/F-train.csv')
!ls -all -h /kaggle/output/kaggle/working
first_increased_male_group = update_dataset(first_male_group, 'M', 4).head(1000)
first_increased_male_group
fifth_increased_male_group = update_dataset(fifth_male_group, 'M', 4).head(1000)
fifth_increased_male_group
second_male_group = second_male_group.sample(n=1000, random_state=1)
third_male_group = third_male_group.sample(n=1000, random_state=1)
fourth_male_group = fourth_male_group.sample(n=1000, random_state=1)

train_male_final = pd.concat([first_increased_male_group, second_male_group, third_male_group, fourth_male_group, fifth_increased_male_group], ignore_index=True)
train_male_final.to_csv('/kaggle/output/kaggle/working/M-train.csv')
!ls -all -h /kaggle/output/kaggle/working
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf

from keras.applications.resnet import ResNet50, preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint   
from keras.layers import Conv2D, Dense, Dropout, Flatten, GlobalMaxPooling2D, MaxPooling2D
from keras.models import Model, Sequential
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
def init_model(patientSex):
    if not os.path.isfile(f'/kaggle/output/kaggle/working/model.hand.x-ray.weights.{patientSex}.best.hdf5'):
        images, outputs = _prepare_dataset(patientSex)
        model = _train(patientSex, images, outputs)
        print('New train!')
    else:
        model = _create_model()
        model.load_weights(f'/kaggle/output/kaggle/working/model.hand.x-ray.weights.{patientSex}.best.hdf5')
        print('Using network trained!')

    return model
def _preprocess_images(filename):
    image = load_img(filename, target_size=(128, 128))
    image = img_to_array(image)
    image = image.reshape((image.shape[0], image.shape[1], image.shape[2]))
    return preprocess_input(image) 

def _prepare_dataset(patientSex):
    df = pd.read_csv(f'/kaggle/output/kaggle/working/{patientSex}-train.csv')

    images = [_preprocess_images(f"/kaggle/output/kaggle/working/clean-images-train/{filename}") for filename in df['fileName']]
    images = np.array(images, dtype=np.float32)

    outputs = df['boneage']
    outputs = np.array(outputs, dtype=np.float32)

    return images, outputs
def _create_model():
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape= (128, 128, 3))
    
    number_of_frozen_layers = 0
    for i, layer in enumerate(base_model.layers):
      if i>=number_of_frozen_layers:
        break
      layer.trainable = False

    x = GlobalMaxPooling2D()(base_model.output)
    x = Flatten()(x)
    x = Dense(16, activation = 'relu')(x)
    x = Dense(1, activation = 'linear')(x)

    model = Model(base_model.input, x)
    model.summary()
    
    return model
def _train(patientSex, images, outputs):
    print('Number images:', len(images))
    print('Number outputs:', len(outputs))

    # divindo dataset de treinamento em treinamento, teste e validação
    seed = 42
    x_train, x_test, y_train, y_test = train_test_split(images, outputs, test_size = 0.20, random_state=seed)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = 0.20, random_state = seed)

    # normalização
    x_train = x_train.astype('float32')/255
    x_valid = x_valid.astype('float32')/255
    x_test = x_test.astype('float32')/255

    # mudando escala de idades para valores entre [0-1]
    max_bornage = outputs.max()
    y_train = y_train / max_bornage
    y_valid = y_valid / max_bornage
    y_test = y_test / max_bornage

    model = _create_model()
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])

    checkpointer = [ModelCheckpoint(filepath=f'/kaggle/output/kaggle/working/model.hand.x-ray.weights.{patientSex}.best.hdf5', save_best_only=True),
                    EarlyStopping(patience= 5)]
    history = model.fit(x_train, y_train,
           batch_size=32,
           epochs=100,
           verbose=1,
           validation_data=(x_valid, y_valid),
           callbacks=checkpointer)
    
    plt.plot(history.history['mse'])
    plt.show()
    
    # carregando os pesos que geraram a melhor precisão de validação
    model.load_weights(f'/kaggle/output/kaggle/working/model.hand.x-ray.weights.{patientSex}.best.hdf5')

    # avaliar e imprimir a precisão do teste
    loss, mse = model.evaluate(x_test, y_test, verbose=2)
    print("Testing set Mean Square Error: {:5.2f} MPG".format(mse))

    return model, x_test, y_test
images, outputs = _prepare_dataset('F')
model, x_test, y_test = _train('F', images, outputs)
max_bornage = outputs.max()
predict_boneages = max_bornage * model.predict(x_test, batch_size = 32, verbose = True)
real_boneages = max_bornage * y_test

ord_ind = np.argsort(y_test)
ord_ind = ord_ind[np.linspace(0, len(ord_ind)-1, 8).astype(int)]
fig, axs = plt.subplots(4, 2, figsize = (15, 30))
for (ind, ax) in zip(ord_ind, axs.flatten()):
    ax.imshow(x_test[ind, :,:,0], cmap = 'bone')
    ax.set_title('Age: %fY\nPredicted Age: %fY' % (real_boneages[ind], 
                                                   predict_boneages[ind]))
    ax.axis('off')
fig.savefig('image_predictions-F.png', dpi = 300)
fig, ax = plt.subplots(figsize = (7,7))
ax.plot(real_boneages, predict_boneages, 'r.', label = 'predictions')
ax.plot(real_boneages, real_boneages, 'b-', label = 'actual')
ax.legend(loc = 'upper right')
ax.set_xlabel('Actual Age (Months)')
ax.set_ylabel('Predicted Age (Months)')
images_M, outputs_M = _prepare_dataset('M')
model_M, x_test_M, y_test_M = _train('M', images_M, outputs_M)
max_bornage_M = outputs_M.max()
predict_boneages_M = max_bornage_M * model.predict(x_test_M, batch_size = 32, verbose = True)
real_boneages_M = max_bornage_M * y_test_M

ord_ind = np.argsort(y_test_M)
ord_ind = ord_ind[np.linspace(0, len(ord_ind)-1, 8).astype(int)]
fig, axs = plt.subplots(4, 2, figsize = (15, 30))
for (ind, ax) in zip(ord_ind, axs.flatten()):
    ax.imshow(x_test_M[ind, :,:,0], cmap = 'bone')
    ax.set_title('Age: %fY\nPredicted Age: %fY' % (real_boneages_M[ind], 
                                                   predict_boneages_M[ind]))
    ax.axis('off')
fig.savefig('image_predictions-M.png', dpi = 300)
fig, ax = plt.subplots(figsize = (7,7))
ax.plot(real_boneages_M, predict_boneages_M, 'r.', label = 'predictions')
ax.plot(real_boneages_M, real_boneages_M, 'b-', label = 'actual')
ax.legend(loc = 'upper right')
ax.set_xlabel('Actual Age (Months)')
ax.set_ylabel('Predicted Age (Months)')