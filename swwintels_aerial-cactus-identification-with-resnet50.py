%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *
PATH = "../input/"

sz=32

bs=512
tfms = get_transforms(flip_vert=True, max_rotate=90.)

data = ImageDataBunch.from_csv(PATH, ds_tfms=tfms,

        folder="train/train", csv_labels='train.csv', test="test/test",

        valid_pct=0.1, fn_col=0, label_col=1).normalize(imagenet_stats)
print(f'We have {len(data.classes)} different classes\n')

print(f'Classes: \n {data.classes}')
print (f'We have {len(data.train_ds)+len(data.valid_ds)+len(data.test_ds)} images in the total dataset')
data.show_batch(8, figsize=(20,15))
def get_ex(): return open_image('../input/train/train/000c8a36845c0208e833c79c1bffedd1.jpg')



def plots_f(rows, cols, width, height, **kwargs):

    [get_ex().apply_tfms(tfms[0], **kwargs).show(ax=ax) for i,ax in enumerate(plt.subplots(

        rows,cols,figsize=(width,height))[1].flatten())]
plots_f(4, 4, 8, 8, size=sz)
learn = create_cnn(data, models.resnet50, metrics=accuracy, path='../kaggle/working', model_dir='../kaggle/working/model',callback_fns=ShowGraph)
lrf=learn.lr_find()

learn.recorder.plot()
lr=5e-3
learn.fit_one_cycle(1,lr)
learn.save('cactus-stage-1')
learn.unfreeze()
lrf=learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(3, max_lr=slice(1e-6, 1e-4))
learn.save('cactus-stage-2')
preds_test,y_test=learn.get_preds(ds_type=DatasetType.Test)
preds_test_tta,y_test_tta=learn.TTA(ds_type=DatasetType.Test)
sub=pd.read_csv(f'{PATH}/sample_submission.csv').set_index('id')
clean_fname=np.vectorize(lambda fname: str(fname).split('/')[-1].split('.')[0]+".jpg")

fname_cleaned=clean_fname(data.test_ds.items)

fname_cleaned=fname_cleaned.astype(str)

fname_cleaned
sub.loc[fname_cleaned,'has_cactus']=to_np(preds_test[:,1])

sub.to_csv(f'submission.csv')

sub.loc[fname_cleaned,'has_cactus']=to_np(preds_test_tta[:,1])

sub.to_csv(f'submission_tta.csv')
classes = preds_test.argmax(1)

classes

sub.loc[fname_cleaned,'has_cactus']=to_np(classes)

sub.to_csv(f'submission_1_0.csv')