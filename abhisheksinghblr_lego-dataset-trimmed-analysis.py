%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import error_rate
bs = 64
path = Path('/kaggle/input/lego-dataset-trimmed/Dataset')

path.ls()

path_anno = path/'train'

fn_paths = get_image_files(path_anno)
df = pd.read_csv(path/'train.csv')

df.head()
def get_labels(file_path):

        for row in df.itertuples():

            if '/'+row.name in str(file_path):           

                print (row.name, row.category)

                return row.category

    
labels = list(map(get_labels, fn_paths))
tfms = get_transforms()

data = ImageDataBunch.from_lists(path, fn_paths, labels=labels, ds_tfms=tfms, size=200, bs=bs, valid_pct=0.25

                                  ).normalize(imagenet_stats)
data.show_batch(rows=20, figsize=(20,20))
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
!mkdir -p /root/.cache/torch/checkpoints/

!cp /kaggle/input/fast-ai-models/resnet50-19c8e357.pth /root/.cache/torch/checkpoints/resnet50-19c8e357.pth
learn = cnn_learner(data, models.resnet50, metrics=accuracy, model_dir=Path('/kaggle/input/fast-ai-models'))
learn.model
learn.fit_one_cycle(10)
learn.model_dir = '/kaggle/output/fast-ai-models/'
learn.save('/kaggle/output/fast-ai-models/stage-1-50')
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(20, figsize=(20,20))
interp.plot_confusion_matrix(figsize=(20,20), dpi=100)
interp.most_confused(min_val=2)
learn.unfreeze()
learn.fit_one_cycle(2)
learn.load('/kaggle/output/fast-ai-models/stage-1-50');
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(5, max_lr=slice(1e-6,1e-4))
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(50, figsize=(20,20))
interp.plot_confusion_matrix(figsize=(20,20), dpi=100)
interp.most_confused(min_val=1)
learn.save('/kaggle/output/fast-ai-models/stage-2-50')
path = learn.path
learn.export('/kaggle/output/fast-ai-models/lego.pkl')
defaults.device = torch.device('cpu')
lego_learn = load_learner('/kaggle/output/fast-ai-models', 'lego.pkl')
pred_path = path/'predict'

pred_fn_paths = get_image_files(pred_path)
for pred_fn_path in pred_fn_paths:

    img = open_image(pred_fn_path)

    pred_class,pred_idx,outputs = lego_learn.predict(img)

    print(pred_fn_path, pred_class)
img = open_image('/kaggle/input/lego-dataset-trimmed/Dataset/predict/4001.png')

img
pred_class,pred_idx,outputs = lego_learn.predict(img)

print(str(pred_class))