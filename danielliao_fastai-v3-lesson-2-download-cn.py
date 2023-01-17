from fastai.vision import *
# # 实验模式

# classes = ['teddys','grizzly','black'] # 构建图片类别
# # 实验模式

# folder = 'black' # 黑熊文件夹

# file = 'urls_black.txt' # 黑熊图片下载链接文档
# # 实验模式

# path = Path('/kaggle/working/data/bears') # 创建子目录path

# dest = path/folder # 再加一个子目录path

# # 但此刻，实际上没有真实文件夹被创建， /kaggle/working/下没新增任何文件或文件夹
# # 实验模式

# dest.mkdir(parents=True, exist_ok=True) # 真实创建这些子目录的文件夹

# # 现在增加了一个 data/文件夹，里面有一个 bears/文件夹，里面有一个 black/文件夹，里面空的
# # 实验模式

# !cp /kaggle/input/fastail2/* {path}/  

# # 将/kaggle/input/FastAI-L2下的数据复制到 /kaggle/working/data/bears/下面
# # 实验模式

# download_images(path/file, dest, max_pics=200) # 根据urls下载图片到指定文件夹

# # dest 实际地址是 "/kaggle/working/data/bears/black"

# # 如果下载报错，尝试增加`max_workers=0`， 如下

# # download_images(path/file, dest, max_pics=20, max_workers=0)
# # 实验模式

# # 重复上述操作到不同类别图片的文件夹

# folder = 'teddys'

# file = 'urls_teddys.txt'
# # 实验模式

# # 重复上述操作到不同类别图片的文件夹

# path = Path('/kaggle/working/data/bears')

# dest = path/folder

# dest.mkdir(parents=True, exist_ok=True)

# # 现在增加了一个 black/文件夹(里面空的)，在data/bears/下面
# # 实验模式

# download_images(path/file, dest, max_pics=200)
# # 实验模式

# # 重复上述操作到不同类别图片的文件夹

# folder = 'grizzly'

# file = 'urls_grizzly.txt'
# # 实验模式

# # 重复上述操作到不同类别图片的文件夹

# path = Path('/kaggle/working/data/bears')

# dest = path/folder

# dest.mkdir(parents=True, exist_ok=True)

# # 现在增加了一个 grizzly/文件夹(里面空的)，在data/bears/下面
# # 实验模式

# download_images(path/file, dest, max_pics=200)
# # 实验模式

# # 接下来移除所有无法打开的图片

# for c in classes:

#     print(c)

#     verify_images(path/c, delete=True, max_size=500)
# commit模式

path_mybears = Path('/kaggle/input/mybears/bears/bears')



np.random.seed(42)

data = ImageDataBunch.from_folder(path_mybears, train=".", valid_pct=0.2,

        ds_tfms=get_transforms(), size=64, num_workers=0).normalize(imagenet_stats)
# # 实验模式

# np.random.seed(42)

# data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,

#         ds_tfms=get_transforms(), size=64, num_workers=0).normalize(imagenet_stats)
data
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
data.show_batch(rows=3, figsize=(3,4))
learn = create_cnn(data, models.resnet34, metrics=error_rate, model_dir='/kaggle/working/')
learn.fit_one_cycle(5)
learn.save('/kaggle/working/stage-1') # 确保commit后能下载
learn.unfreeze() #解冻整个模型所有封冻的层
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(4, max_lr=slice(1e-5,1e-4))
# 如果表现不好，不必保存

learn.save('/kaggle/working/stage-2')
learn.load('/kaggle/working/stage-2');
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
# # 实验模式

# from fastai.widgets import *
# # 实验模式

# ds, idxs = DatasetFormatter().from_toplosses(learn, 

#                                              ds_type=DatasetType.Valid, 

#                                              n_imgs=10) # 设置 10张，为了速度结束
# # 实验模式

# ImageCleaner(ds, idxs, path)
# # 实验模式

# ds, idxs = DatasetFormatter().from_similars(learn, 

#                                             ds_type=DatasetType.Valid, 

#                                             n_imgs=10)
# # 实验模式

# ImageCleaner(ds, idxs, path, duplicates=True) # 确认在/kaggle/working/下运行
# # 实验模式

# np.random.seed(42)

# data = ImageDataBunch.from_csv("/kaggle/working/data/bears/", folder=".", csv_labels="cleaned.csv",

#         ds_tfms=get_transforms(), size=64, num_workers=0).normalize(imagenet_stats)

# data.classes
# commit模式

np.random.seed(42)

data = ImageDataBunch.from_csv(path_mybears, valid_pct=0.2, csv_labels='cleaned.csv',

        ds_tfms=get_transforms(), size=224, num_workers=0).normalize(imagenet_stats)
learn = create_cnn(data, models.resnet34, metrics=error_rate, model_dir="/kaggle/working/")

learn.load('/kaggle/working/stage-2');

learn.fit_one_cycle(4, max_lr=slice(1e-5,1e-4))

learn.save('/kaggle/working/stage-3')
# commit模式

learn.path = Path('/kaggle/working')
learn.export()
defaults.device = torch.device('cpu')
# commit模式

path = path_mybears
img = open_image(path/'black'/'00000021.jpg')

img
# commit模式

path = Path('/kaggle/working')
learn = load_learner(path)
pred_class,pred_idx,outputs = learn.predict(img)

pred_class