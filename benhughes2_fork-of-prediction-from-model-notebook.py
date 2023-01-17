%reload_ext autoreload
%autoreload 2
%matplotlib inline
from fastai import *
from fastai.vision import *
import os, os.path
#fastai.defaults.device = torch.device('cpu')
path = Path('/kaggle/input/libbytest')
path.ls()
csv_train = Path('/kaggle/input/libbytestcsv/example_list_of_all_classes.csv')
csv_test = Path('/kaggle/input/libbytestcsv/libby_test.csv')
df_train = pd.read_csv(csv_train, header=None)
df_test = pd.read_csv(csv_test, header=None)

df_all = df_train.append([df_test])

df_test
imagelist_train = ImageList.from_csv(path,csv_train, header=None)
imagelist_test = ImageList.from_csv(path,csv_test, header=None)
path2 = Path('/kaggle/input/fullbirdpickle')
learn = load_learner(path2,'model_birds_full_BM_50eps.pkl')
tfms = get_transforms()
data_test = (ImageList.from_df(df_all, path)
        .split_by_list(imagelist_test, imagelist_test)
        .label_from_df(cols=1) 
        .transform(tfms, size=256)
        .databunch(bs=64, num_workers=0)
        .normalize(imagenet_stats))

data_test
data_test.classes = learn.data.classes
data_test.c2i = learn.data.c2i
learn.validate(data_test.valid_dl, metrics = [accuracy])
learn.data.valid_dl = data_test.valid_dl
preds,y,losses = learn.get_preds(with_loss = True)
classes = learn.data.classes
predictions = torch.argmax(preds, dim=1)
print(classes[predictions[500]])
print(classes[y[500]])
y
interp = ClassificationInterpretation(learn, preds, y, losses)
interp.plot_top_losses(9, figsize=(15,11),heatmap=True)
interp.most_confused(min_val=2)
#actual, predicted
losses,idxs = interp.top_losses()
len(imagelist_test)==len(losses)==len(idxs)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
testimagelist = ImageList.from_folder(path)
testimagelist
path2 = Path('/kaggle/input/fullbirdpickle')
learn = load_learner(path2,'model_birds_full_BM_50eps.pkl',test = testimagelist)
preds,y,losses = learn.get_preds(with_loss = True, ds_type=DatasetType.Test)