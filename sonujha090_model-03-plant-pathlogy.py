!pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html -q
!pip install --upgrade kornia -q
!pip install allennlp==1.1.0.rc4 -q
!pip install --upgrade fastai -q
from fastai.vision.all import *
path = Path('../input/plant-pathology-2020-fgvc7')
Path.BASE_PATH = path
path.ls()
train = pd.read_csv(path/'train.csv')
print(train.shape)
train.head()
sumbission = pd.read_csv(path/'sample_submission.csv')
sumbission.head()
test = pd.read_csv(path/'test.csv')
print(test.shape)
test.head()
print('Total given Images = ', len(os.listdir(path/'images')))
print('Number of train images are {}, number of test images are {} therefore total images should be {}'.format(train.shape[0],test.shape[0], train.shape[0]+test.shape[0]))
(path/'images').ls()
train["label"] = (0*train.healthy + 1*train.multiple_diseases+2*train.rust + 3*train.scab)

train.drop(columns=["healthy","multiple_diseases","rust","scab"],inplace=True)
train.head()
get_x = lambda x: path/'images'/f'{x[0]}.jpg'
get_y = lambda x: x[1]
print(get_x(train.values[2]))
print(get_y(train.values[2]))
dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_x =get_x, get_y = get_y,
    splitter=RandomSplitter(),
    item_tfms=Resize(228))
dls = dblock.dataloaders(train)
dls.show_batch()
xb, yb = dls.one_batch()
xb.shape, yb.shape
learn = cnn_learner(dls, resnet34, metrics=accuracy)
learn.fine_tune(5)
test_images = get_image_files(path/'images')
test_images = L(x for x in test_images if x.name.startswith('Test'))
img1 = test_images[0]
print(img1)
learn.predict(img1)
test_dl = learn.dls.test_dl(test_images)
test_dl.show_batch()
preds = learn.get_preds(dl=test_dl)
sumbission.head()
# resultdf = pd.DataFrame(preds[0])
# resultdf.columns = sumbission.columns
# resultdf.head()
