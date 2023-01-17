%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import error_rate
bs = 64
path = Path('/kaggle/input/emergency-vehicles-identification/Emergency_Vehicles')

path.ls()
path_anno = path/'train'

fn_paths = get_image_files(path_anno)
len(fn_paths)
train_df = pd.read_csv(path/'train.csv')

train_df.head()
def get_labels(file_path):

        for row in train_df.itertuples():

            if '/'+row.image_names in str(file_path):           

                return row.emergency_or_not
labels = list(map(get_labels, fn_paths))
len(labels)
tfms = get_transforms()

data = ImageDataBunch.from_lists(path, fn_paths, labels=labels, ds_tfms=tfms, size=224, bs=bs, valid_pct=0.25).normalize(imagenet_stats)
data.show_batch(rows=20, figsize=(20,20))
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
!mkdir -p /root/.cache/torch/checkpoints/

!cp /kaggle/input/resnet152/resnet152.pth /root/.cache/torch/checkpoints/resnet152.pth
learn = cnn_learner(data, models.resnet152, metrics=accuracy, model_dir=Path('/kaggle/input/resnet152'))
learn.model
learn.fit_one_cycle(10)
learn.model_dir = '/kaggle/output/resnet152/'
learn.save('/kaggle/output/resnet152/stage-1-152')
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(20, figsize=(20,20))
interp.plot_confusion_matrix(figsize=(20,20), dpi=100)
interp.most_confused(min_val=2)
learn.unfreeze()
learn.fit_one_cycle(4)
learn.load('/kaggle/output/resnet152/stage-1-152');
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(5, max_lr=slice(1e-5,1e-4))
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(50, figsize=(20,20))
interp.plot_confusion_matrix(figsize=(20,20), dpi=100)
interp.most_confused(min_val=1)
learn.save('/kaggle/output/resnet152/stage-2-152')
path = learn.path
learn.export('/kaggle/output/resnet152/emergency_vehicles.pkl')
defaults.device = torch.device('cpu')
lego_learn = load_learner('/kaggle/output/resnet152', 'emergency_vehicles.pkl')
pred_path = path/'test'

pred_fn_paths = get_image_files(pred_path)
for pred_fn_path in pred_fn_paths:

    img = open_image(pred_fn_path)

    pred_class,pred_idx,outputs = learn.predict(img)

    print(pred_fn_path, pred_class)
img = open_image('/kaggle/input/emergency-vehicles-identification/Emergency_Vehicles/test/841.jpg')

img
pred_class,pred_idx,outputs = lego_learn.predict(img)

print(str(pred_class))
img = open_image('/kaggle/input/emergency-vehicles-identification/Emergency_Vehicles/test/1287.jpg')

img
pred_class,pred_idx,outputs = lego_learn.predict(img)

print(str(pred_class))