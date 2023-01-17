from fastai.vision import *
!ls ../input
folder = 'house'

file = 'house.txt'
path = Path('data/total')

dest = path/folder

dest.mkdir(parents=True, exist_ok=True)
path.ls()
download_images(Path('../input')/file, dest, max_pics=200)
folder = 'nohouse'

file = 'nohouse.txt'
path = Path('data/total')

dest = path/folder

dest.mkdir(parents=True, exist_ok=True)

path.ls()
classes = ['house','nohouse']
download_images(Path('../input')/file, dest, max_pics=200)
for c in classes:

    print(c)

    verify_images(path/c, delete=True, max_size=500)
np.random.seed(42)

data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,

        ds_tfms=get_transforms(), size=224, num_workers=2).normalize(imagenet_stats)
data.show_batch()
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
learn = cnn_learner(data,models.resnet34,metrics=accuracy)
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5,1e-2)
learn.show_results()
learn.export()
path.ls()
# when you do prediction, you want to disable GPU, since not most server has GPU ready for prediction

defaults.device = torch.device('cpu')
!ls {path/'house'} | head -3
!ls {path/'nohouse'} | head -3
house_img = open_image(path/'house/00000002.jpg')

house_img
nohouse_img = open_image(path/'nohouse/00000001.jpg')

nohouse_img
learn = load_learner(path)
learn.predict(house_img)
learn.predict(nohouse_img)