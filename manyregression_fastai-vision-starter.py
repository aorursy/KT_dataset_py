%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import error_rate
bs = 16
import os

from pathlib import Path

print(os.listdir("../input"))

work_p = Path("./")

p = Path("../input")
len(os.listdir(p/"test"))
# folders = ["train", "test"]

# for c in classes:

#     print(c)

#     verify_images(p/c, delete=False, max_size=500)
train_df = pd.read_csv(p/"train.csv")#.sample(frac=0.3, random_state=2)

print(train_df.shape); train_df.head()
labels_count = train_df.Id.value_counts()

train_names = train_df.index.values
for idx,row in train_df.iterrows():

    if labels_count[row['Id']] < 2:

        for i in row*math.ceil((2 - labels_count[row['Id']])/labels_count[row['Id']]):

            train_df = train_df.append(row,ignore_index=True)



print(train_df.shape)

# plt.hist(train_df.Id.value_counts()[1:],bins=100,range=[0,100]);

# plt.hist(train_df.Id.value_counts()[1:],bins=100,range=[0,100]);
name = f'res50-full-train'
np.random.seed(2)

data = (ImageDataBunch.from_df(work_p, train_df, folder=p/"train", test=p/"test", valid_pct=0.20, ds_tfms=get_transforms(), size=224, bs=bs)

        .normalize(imagenet_stats))
data.show_batch(rows=3, figsize=(7,6))
learn = create_cnn(data, models.resnet50, metrics=error_rate)

learn.fit_one_cycle(2, max_lr=slice(6.31e-07, 3e-07))
learn = create_cnn(data, models.resnet50, metrics=error_rate)
learn.fit_one_cycle(2, max_lr=3e-03)
learn = create_cnn(data, models.resnet50, metrics=error_rate)

learn.fit_one_cycle(2, max_lr=0.5e-02)
learn.recorder.plot_losses()
learn = create_cnn(data, models.resnet50, metrics=error_rate)

learn.fit_one_cycle(2, max_lr=slice(5e-02, 2.5e-02))
# learn.save("stage-1")
# learn.unfreeze()
# learn.fit_one_cycle(1)
# learn.load('stage-1');
learn.lr_find()
learn.recorder.plot()
# data_bigger = ImageDataBunch.from_df(work_p, train_df, folder=p/"train", valid_pct=0.20, ds_tfms=get_transforms(), size=448, bs=bs).normalize(imagenet_stats)

# learn_bigger = create_cnn(data_bigger, models.resnet34, metrics=error_rate)
# learn_bigger.fit_one_cycle(4)
# data_bigger.show_batch(rows=3, figsize=(7,6))
# interp = ClassificationInterpretation.from_learner(learn)

# losses,idxs = interp.top_losses()

# len(data.valid_ds)==len(losses)==len(idxs)
# interp.plot_top_losses(9, figsize=(15,11))
# learn.unfreeze()
# learn.fit_one_cycle(1)
# learn.load('stage-1');
# learn.lr_find()
# learn.recorder.plot()
# learn.unfreeze()

# learn.fit_one_cycle(2, max_lr=slice(1e-5,1e-4))
preds, _ = learn.get_preds(DatasetType.Test)
preds = torch.cat((preds, torch.ones_like(preds[:, :1])), 1)
def top_5_pred_labels(preds, classes):

    top_5 = np.argsort(preds.numpy())[:, ::-1][:, :5]

    labels = []

    for i in range(top_5.shape[0]):

        labels.append(' '.join([classes[idx] for idx in top_5[i]]))

    return labels



def create_submission(preds, data, name, classes=None):

    if not classes: classes = data.classes

    sub = pd.DataFrame({'Image': [path.name for path in data.test_ds.x.items]})

    sub['Id'] = top_5_pred_labels(preds, classes)

    sub.to_csv(f'{name}.csv', index=False)
create_submission(preds, learn.data, name, learn.data.classes)