from fastai.vision import *
folder = 'black'

file = 'black_bear.txt'
folder = 'grizzly'

file = 'grizzly_bear.txt'
folder = 'teddy'

file = 'teddy_bear.txt'
# 设置所有数据存放的根目录

path = Path('/kaggle/working/data/bears')

path.mkdir(parents=True, exist_ok=True)
# 在kaggle中，将导入的数据，从只读目录复制到工作区可写目录下

!cp /kaggle/input/* {path}/

# 安装tree命令

!apt get tree
!tree {path}
# 创建各个类别图像下载的目录

dest = path/folder

dest.mkdir(parents=True, exist_ok=True)

# 下载图像

download_images(path/file, dest, max_pics=200)
# 创建类别

classes = ['black', 'grizzly', 'teddy']

# 删除不能被打开的错误图像

for c in classes:

    print(c)

    verify_images(path/c, delete=True, max_size=500)
# 置固定的随机数种子，保证每次创建相同的验证集，以便调整超参数

np.random.seed(42)

# 默认训练集会在train目录下查找。用.设置为当前目录，并且划分验证集

data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,

        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
data.show_batch(rows=3, figsize=(12,8))
learn = cnn_learner(data, models.resnet34, metrics=error_rate)

learn.fit_one_cycle(4)

learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9, figsize=(12,8))

interp.plot_top_losses(9, figsize=(12,8), heatmap=False)

interp.plot_confusion_matrix()
learn.lr_find()

learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(1e-6,3e-4))

learn.save('stage-2')
from fastai.widgets import *
# 创建一个没有划分验证集的数据集。即包含所有数据的数据集

db = (ImageList.from_folder(path)

                   .split_none()

                   .label_from_folder()

                   .transform(get_transforms(), size=224)

                   .databunch()

     )
learn_cln = cnn_learner(db, models.resnet34, metrics=error_rate)

learn_cln.load('stage-2');

ds, idxs = DatasetFormatter().from_toplosses(learn_cln)

len(idxs)
doc(DatasetFormatter().from_toplosses)
# 取前25个进行clean

idxs = idxs[:25]
ImageCleaner(ds, idxs, path)
!tree
# 查看cleaned.csv 里面保存的是被清理之后的正确标签

!cat {path}/cleaned.csv -n
# 从cleaned.csv 创建训练集

db = (ImageList.from_csv(path, 'cleaned.csv', folder='.')

                    .split_none()

                    .label_from_df()

                    .transform(get_transforms(), size=224)

                    .databunch()

      )
learn_cln = cnn_learner(db, models.resnet34, metrics=error_rate)

learn_cln.load('stage-2');

ds, idxs = DatasetFormatter().from_similars(learn_cln)

len(idxs)
idxs = idxs[:10]
ImageCleaner(ds, idxs, path, duplicates=True)
np.random.seed(42)

data = ImageDataBunch.from_csv(path, folder=".", valid_pct=0.2, csv_labels='cleaned.csv',

        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
learn = cnn_learner(data, models.resnet34, metrics=error_rate)

learn.load('stage-2')

interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9, figsize=(12,8))

interp.plot_top_losses(9, figsize=(12,8),heatmap=False)

interp.plot_confusion_matrix()
learn = cnn_learner(data, models.resnet34, metrics=error_rate)

learn.fit_one_cycle(4)

interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9, figsize=(12,8))

interp.plot_top_losses(9, figsize=(12,8),heatmap=False)

interp.plot_confusion_matrix()
learn.lr_find()

learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(1e-6,3e-4))

interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9, figsize=(12,8))

interp.plot_top_losses(9, figsize=(12,8),heatmap=False)

interp.plot_confusion_matrix()