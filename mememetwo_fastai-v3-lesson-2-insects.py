from fastai import *

from fastai.vision import *
classes = ['ant','silverfish','spider']
folder = 'ant'

file = 'Ant_URLs.txt'
path = Path('data')

dest = path/folder

dest.mkdir(parents=True, exist_ok=True)
!cp -r ../input/* {path}/



#Path('../input').ls()

path.ls()
download_images(path/file, dest, max_pics=200)
dest.ls()
folder = 'silverfish'

file = 'Silverfish_URLs.txt'
dest = path/folder

dest.mkdir(parents=True, exist_ok=True)
download_images(path/file, dest, max_pics=200)
dest.ls()
folder = 'spider'

file = 'Spider_URLs.txt'
dest = path/folder

dest.mkdir(parents=True, exist_ok=True)
download_images(path/file, dest, max_pics=200)
dest.ls()
for c in classes:

    print(c)

    verify_images(path/c, delete=True, max_size=500)
np.random.seed(42)

data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,

        ds_tfms=get_transforms(), size=64, num_workers=0).normalize(imagenet_stats)
data.classes
data.show_batch(rows=3, figsize=(9,16))
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(2, max_lr=slice(1e-4,1e-3))
learn.save('stage-2')
learn.load('stage-2');
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
#from fastai.widgets import *
#ds, idxs = DatasetFormatter().from_toplosses(learn, n_imgs=100)
#ImageCleaner(ds, idxs, path)
#ds, idxs = DatasetFormatter().from_similars(learn)
# Alternative way to read the cleaned.csv file

#f = open(path/'cleaned.csv', 'r')

#textfilecontents = f.read()

#print(textfilecontents)
#df = pd.read_csv(path/'cleaned.csv')

df = pd.read_csv(path/'Data.csv')
for index, row in df.iterrows():

    file_path = path/row['name']

    if not file_path.is_file():

        print("Can't find file... dropping " + str(row['name']))

        df.drop(index, axis=0, inplace=True)
np.random.seed(42)

data = ImageDataBunch.from_df(df=df, path=path, valid_pct=0.2, size=224)

data
learn = cnn_learner(data, models.resnet34, metrics=error_rate)

learn.fit_one_cycle(4)
learn.save('clean1')

interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
interp.plot_top_losses(4, figsize=(11,11), heatmap=False)
interp.plot_top_losses(4, figsize=(11,11), heatmap=True)
import fastai

defaults.device = torch.device('cpu')
img = open_image(path/'silverfish'/'00000021.jpg')

img
# Use previously set classes

classes
# Cody to the rescue!  (It's ds_tfms, not tfms)

data2 = ImageDataBunch.single_from_classes(path, classes, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)

learn = cnn_learner(data2, models.resnet34).load('clean1')
pred_class,pred_idx,outputs = learn.predict(img)

pred_class
learn.export('insect_classifier_resnet34_224.pkl')
import shutil

shutil.rmtree(path/'ant')

shutil.rmtree(path/'silverfish')

shutil.rmtree(path/'spider')



path.ls()
#learn = create_cnn(data, models.resnet34, metrics=error_rate)
#learn.fit_one_cycle(5, max_lr=1e-5)
#learn.recorder.plot_losses()
#learn = create_cnn(data, models.resnet34, metrics=error_rate, pretrained=False)
#learn.fit_one_cycle(1)
# np.random.seed(42)

# data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.9, bs=32, 

#        ds_tfms=get_transforms(do_flip=False, max_rotate=0, max_zoom=1, max_lighting=0, max_warp=0

#                              ),size=224, num_workers=4).normalize(imagenet_stats)
# learn = create_cnn(data, models.resnet50, metrics=error_rate, ps=0, wd=0)

# learn.unfreeze()
# learn.fit_one_cycle(40, slice(1e-6,1e-4))