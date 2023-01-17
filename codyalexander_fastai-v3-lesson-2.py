from fastai import *

from fastai.vision import *
classes = ['cat','dog']
path = Path('data')

path.mkdir(parents=True, exist_ok=True)
!cp ../input/* {path}/
path.ls()
cat_folder = path/'cat'

cat_folder.mkdir(parents=True, exist_ok=True)



cat_file = '609_cat_links.txt'



dog_folder = path/'dog'

dog_folder.mkdir(parents=True, exist_ok=True)



dog_file = '650_dog_links.txt'
download_images(path/cat_file, cat_folder, max_pics=700)
download_images(path/dog_file, dog_folder, max_pics=700)
verify_images(cat_folder, delete=True, max_size=224)
verify_images(dog_folder, delete=True, max_size=224)
np.random.seed(42)

data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,

                                  ds_tfms=get_transforms(), size=224, num_workers=0).normalize(imagenet_stats)
data.classes

data
data.show_batch(rows=3, figsize=(7,7))
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
#from fastai.widgets import *
#ds, idxs = DatasetFormatter().from_toplosses(learn, n_imgs=100)
#ImageCleaner(ds, idxs, path)
df = pd.read_csv(path/'data.csv', index_col=0)
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
learn.save('clean-stage-1')
learn.fit_one_cycle(4)
learn.save('clean-stage-2')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(3, figsize=(15,11), heatmap=False)
interp.plot_top_losses(3, figsize=(15,11), heatmap=True)
import fastai

defaults.device = torch.device('cpu')
cat_img = open_image(cat_folder/'00000537.jpg')
cat_img.show()
classes = ['cat', 'dog']
data2 = ImageDataBunch.single_from_classes(path, classes, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
learn = cnn_learner(data2, models.resnet34).load('clean-stage-1')
pred_class,pred_idx,outputs = learn.predict(cat_img)

pred_class
learn.export('cat_dog_classifier_resnet34_224.pkl')
import shutil

shutil.rmtree(cat_folder)

shutil.rmtree(dog_folder)



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