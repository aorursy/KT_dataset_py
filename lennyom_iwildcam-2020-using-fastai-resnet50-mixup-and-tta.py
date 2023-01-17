import os



import pandas as pd

from matplotlib import pyplot as plt



from fastai import *

from fastai.vision import *



import json



%matplotlib inline
test_images = os.listdir("../input/iwildcam2020-256/256_images/test/images/")

train_images = os.listdir("../input/iwildcam2020-256/256_images/train/images/")
with open(r'/kaggle/input/iwildcam-2020-fgvc7/iwildcam2020_train_annotations.json') as json_file:

    train_data = json.load(json_file)
df_train = pd.DataFrame({'id': [item['id'] for item in train_data['annotations']],

                         'category_id': [item['category_id'] for item in train_data['annotations']],

                         'image_id': [item['image_id'] for item in train_data['annotations']],

                         'location': [item['location'] for item in train_data['images']],

                         'file_name': [item['file_name'] for item in train_data['images']]})



df_train.head()
df_train.shape
df_train = df_train[df_train['file_name'].isin(train_images)]
df_train.shape
cat_images = dict()

cat_count = dict()



annotations = train_data['annotations']

_images = train_data['images']

for i, annotation in enumerate(annotations):

    _img = annotation['image_id']

    cat = annotation['category_id']

    

    imgs = cat_images.get(cat, None)

    if imgs is None:

        cat_images[cat] = [{'image_id': _img, 'category': cat}]

    else:

        cat_images[cat].append({'image_id': _img, 'category': cat})

        

    count = cat_count.get(cat, 0)

    if count == 0:

        cat_count[cat] = 1

    else:

        cat_count[cat] += 1

        

n_train = dict()

n_val = dict()



for cat, count in cat_count.items():

    _train = math.floor(count * 0.70)

    if _train < 1:

        _train = 1

    _val = count - _train

    n_train[cat] = _train

    n_val[cat] = _val



train_images = []

val_images = []

for cat in cat_images.keys():

    random.shuffle(cat_images[cat])

    train_images += cat_images[cat][:n_train[cat]]

    val_images += cat_images[cat][n_train[cat]:]



val_img_dt = pd.DataFrame(val_images)



df_train['is_valid'] = np.where(df_train.image_id.isin(val_img_dt['image_id']), True, False)
loc_valid = df_train.loc[(df_train['is_valid'] == True)].location.unique()

loc_train = df_train.loc[(df_train['is_valid'] == False)].location.unique()



loc_valid.shape

df_train.category_id.unique().shape
df_train.groupby('is_valid').size()
df_train.drop(df_train.loc[df_train['file_name']=='87022118-21bc-11ea-a13a-137349068a90.jpg'].index, inplace=True)

df_train.drop(df_train.loc[df_train['file_name']=='8792549a-21bc-11ea-a13a-137349068a90.jpg'].index, inplace=True)
df_train.category_id.unique().shape
with open(r'/kaggle/input/iwildcam-2020-fgvc7/iwildcam2020_test_information.json') as f:

    test_data = json.load(f)
df_test = pd.DataFrame.from_records(test_data['images'])

df_test.head()
train, test = [ImageList.from_df(df, path='../input/iwildcam2020-256/256_images/', cols='file_name', folder=folder, suffix='') 

               for df, folder in zip([df_train, df_test], ['train/images', 'test/images'])]

trfm = get_transforms(max_rotate=20, max_zoom=1.3, max_lighting=0.4, max_warp=0.4,

                      p_affine=1., p_lighting=1.)

src = (train.use_partial_data(1)

        .split_from_df(col='is_valid')

        .label_from_df(cols='category_id')

        .add_test(test))

data = (src.transform(trfm, size = 128, padding_mode = 'reflection')

        .databunch(path=Path('.'), bs = 256).normalize(imagenet_stats))
print(data.classes)
org_classes = pd.DataFrame({"org_category": data.classes})

org_classes['Category'] = org_classes.index
def _plot(i,j,ax):

    x,y = data.train_ds[1]

    x.show(ax, y=y)



plot_multi(_plot, 3, 3, figsize=(8,8))
data.show_batch()
data.c
learn = cnn_learner(data, base_arch=models.resnet50, metrics=accuracy).mixup()
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.recorder.min_grad_lr
learn.fit_one_cycle(10, slice(0.01))
learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(10, slice(1e-5, 1e-4))
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
preds,y = learn.TTA(ds_type=DatasetType.Test)
pred_csv = pd.DataFrame(preds.numpy())

pred_csv['Id'] = learn.data.test_ds.items

pred_csv.to_csv("outout_preds.csv", index = False)
submission = pd.read_csv('../input/iwildcam-2020-fgvc7/sample_submission.csv')

id_list = list(submission.Id)

pred_list = list(np.argmax(preds.numpy(), axis=1))

pred_dict = dict((key, value.item()) for (key, value) in zip(learn.data.test_ds.items,pred_list))

pred_ordered = [pred_dict['../input/iwildcam2020-256/256_images/test/images/' + id + '.jpg'] for id in id_list]

submission_with_idx = pd.DataFrame({'Id':id_list,'Category':pred_ordered})

submission_fixed_labels = pd.merge(submission_with_idx, org_classes, on = 'Category', how='left')

submission_fixed_labels = submission_fixed_labels.drop(['Category'], axis = 1)

submission_fixed_labels.rename(columns={'org_category': 'Category'}, inplace=True)



submission_fixed_labels.to_csv("submission.csv".format(Category),index = False)

print("Done")