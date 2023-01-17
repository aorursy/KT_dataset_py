!pip install git+https://github.com/fastai/fastai.git
import fastai
from fastai import *
from fastai.vision import *
print(f'fastai version: {fastai.__version__}')
print(f'torch version: {torch.__version__}')
# import warnings
# warnings.filterwarnings('ignore')
coco = untar_data(URLs.COCO_TINY)
print(coco)
images, lbl_bbox = get_annotations(coco/'train.json')
!ls /tmp/.fastai/data/coco_tiny
print(images[0])
lbl_bbox[0]
def get_lrg(b):
    if not b: raise Exception()
    lrg_idx = np.array([bbox[2]*bbox[3] for bbox in b[0]]).argmax()  # largest by width * height
    return [[b[0][lrg_idx]], [b[1][lrg_idx]]]
lbl_bbox_lrg = [get_lrg(b) for b in lbl_bbox]
lbl_bbox_lrg[:3]
img2bbox = dict(zip(images, lbl_bbox_lrg))
get_y_func = lambda o:img2bbox[o.name]
data = (ObjectItemList.from_folder(coco)
        #Where are the images? -> in coco
        .random_split_by_pct()                          
        #How to split in train/valid? -> randomly with the default 20% in valid
        .label_from_func(get_y_func)
        .add_test_folder('train')
        #How to find the labels? -> use get_y_func
        .transform(get_transforms(), tfm_y=True, size=224)
        #Data augmentation? -> Standard transforms with tfm_y=True
        .databunch(bs=16, num_workers=0, collate_fn=bb_pad_collate))   
        #Finally we convert to a DataBunch and we use bb_pad_collate
idx = 32
fig, axes = plt.subplots(3,3, figsize=(9,9))
for i, ax in enumerate(axes.flat):
    img = data.train_ds[idx]
    # image is augmented each time it is retrived
    img[0].show(y=img[1], ax=ax)
data.show_batch(rows=2, ds_type=DatasetType.Valid, figsize=(6,6))
def loss_func(preds, targs, class_idx, **kwargs):
    return nn.L1Loss()(preds, targs.squeeze())
head_reg4 = nn.Sequential(
    Flatten(), 
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(25088,256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Dropout(0.5),
    nn.Linear(256, 4))
    # Maybe add nn.tanh since the values are [-1,1]
learn = create_cnn(data=data, arch=models.resnet18, pretrained=True, custom_head=head_reg4,
                   model_dir = '/tmp/models') ## For kaggle kernel
learn.loss_func = loss_func
print(learn.summary())
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(15, slice(0.03))
learn.recorder.plot_losses()
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(15, max_lr = slice(0.001, 0.001/5), pct_start=0.5)
learn.recorder.plot_losses()
# learn.show_results(rows=2)  # does not work
# returns ValueError: too many values to unpack (expected 2)
# line 362 of data.pay for def reconstruct(self, t, x):
#                             (bboxes, labels) = t
# here t is a tensor of tensor.Size([4])
# The reason is because my model only cares
preds, targs = learn.get_preds(ds_type=DatasetType.Valid)
targs = targs.squeeze()
print(preds.shape, targs.shape)
np.random.seed(24)
n = 10  # look at n samples, must be even
idxs = np.random.randint(0,len(data.valid_ds), size=n)
_, axes = plt.subplots(nrows=n//2, ncols=2, figsize = (n, n*2))
for i, ax in zip(idxs, axes.flat):
    img = data.valid_ds[i][0].data  # image resize after data is called else original image size
    img_name = Path(data.valid_ds.items[i]).name
    img_size = img.shape[1:]
    targ, pred = targs[i], preds[i]
    Image(img).show(ax=ax,
                    # target is white
                     y=ImageBBox.create(*img_size, 
                                        bboxes=targ.unsqueeze(0),
                                        scale=False),
                    title=img_name)
    # Prediction is red
    ImageBBox.create(*img_size, 
                     bboxes=pred.unsqueeze(0),
                     scale=False).show(ax=ax, color='red')
len(learn.data.test_ds)
learn.data.test_ds.tfm_y = False  # Will get "Exception: Not Implemented" since there is no y
learn.data.test_ds[0]
preds, targs = learn.get_preds(ds_type=DatasetType.Test)
learn.data.one_batch(DatasetType.Test)

