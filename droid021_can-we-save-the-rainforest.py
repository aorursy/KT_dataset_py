%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *
# ! pip install kaggle --upgrade
# ! mkdir -p ~/.kaggle/

# ! mv kaggle.json ~/.kaggle/



# For Windows, uncomment these two commands

# ! mkdir %userprofile%\.kaggle

# ! move kaggle.json %userprofile%\.kaggle
path = Path('../input')

path.ls()
# ! kaggle competitions download -c planet-understanding-the-amazon-from-space -f train-jpg.tar.7z -p {path}  

# ! kaggle competitions download -c planet-understanding-the-amazon-from-space -f train_v2.csv -p {path}  

# ! unzip -q -n {path}/train_v2.csv.zip -d {path}
# ! conda install -y -c haasad eidl7zip
# ! 7za -bd -y -so x {path}/train-jpg.tar.7z | tar xf - -C {path.as_posix()}
df = pd.read_csv(path/'train_v2.csv')

df.head()
tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
np.random.seed(42)

src = (ImageList.from_csv(path, 'train_v2.csv', folder='train-jpg', suffix='.jpg')

       .split_by_rand_pct(0.2)

       .label_from_df(label_delim=' '))
data = (src.transform(tfms, size=128)

        .databunch().normalize(imagenet_stats))
data.show_batch(rows=3, figsize=(12,9))
arch = models.resnet50
acc_02 = partial(accuracy_thresh, thresh=0.2)

f_score = partial(fbeta, thresh=0.2)

learn = cnn_learner(data, arch, metrics=[acc_02, f_score], model_dir='../working/')
learn.lr_find()
learn.recorder.plot()
lr = 0.01
learn.fit_one_cycle(5, slice(lr))
learn.save('stage-1-rn50')
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, slice(1e-5, lr/5))
learn.save('stage-2-rn50')
data = (src.transform(tfms, size=256)

        .databunch().normalize(imagenet_stats))



learn.data = data

data.train_ds[0][0].shape
learn.freeze()
learn.lr_find()

learn.recorder.plot()
lr=1e-2/2
learn.fit_one_cycle(5, slice(lr))
learn.save('stage-1-256-rn50')
learn.load('stage-1-256-rn50')
learn.unfreeze()
lr = 0.01

learn.fit_one_cycle(5, slice(1e-5, lr/5))
learn.recorder.plot_losses()
learn.save('stage-2-256-rn50')
learn.export('../working/export.pkl')
# ! kaggle competitions download -c planet-understanding-the-amazon-from-space -f test-jpg.tar.7z -p {path}  

# ! 7za -bd -y -so x {path}/test-jpg.tar.7z | tar xf - -C {path}

# ! kaggle competitions download -c planet-understanding-the-amazon-from-space -f test-jpg-additional.tar.7z -p {path}  

# # ! 7za -bd -y -so x {path}/test-jpg-additional.tar.7z | tar xf - -C {path}
test = ImageList.from_folder(path/'test-jpg').add(ImageList.from_folder(path/'test-jpg-additional'))

len(test)
learn = load_learner('../working/', test=test)
thresh = 0.2

labelled_preds = [' '.join([learn.data.classes[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]
labelled_preds[:5]
fnames = [f.name[:-4] for f in learn.data.test_ds.items]
df = pd.DataFrame({'image_name':fnames, 'tags':labelled_preds}, columns=['image_name', 'tags'])

df.head(10)
df.to_csv(path/'submission.csv', index=False)
# ! kaggle competitions submit planet-understanding-the-amazon-from-space -f {path/'submission.csv'} -m "My submission"