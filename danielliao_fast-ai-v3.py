# 闭眼，复制，粘贴，就好

%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import * # 加载所需的几乎所有工具

from fastai.metrics import error_rate # 补充工具
# bs = 64 # 一次处理64张图片，但如果内存不够（bus error)，则需要重启kernel, 将64缩小为16，降低内存需求

# bs = 16 # `0` + `0` = 重启kernel

bs = 8
# 下载解压数据包，并生成数据文件地址

path = untar_data(URLs.PETS); path
path.ls() # 查看数据文件内容
path_anno = path/'annotations'

path_img = path/'images' # 进一步找到直接存有数据文件夹
fnames = get_image_files(path_img) # 将所有数据图片，做成地址Path,放入一个list中

fnames[:5] # 查看list
np.random.seed(2) # 确保每次训练后，模型都被相同的验证数据验证

pat = r'/([^/]+)_\d+.jpg$' # 用re从文件名中提取图片label标注
# 生成数据集（包含训练集和验证集，测试集可选）

data = ImageDataBunch.from_name_re(path_img, # 图片数据文件夹path

                                   fnames, # 图片path的list

                                   pat, # regexpr 

                                   ds_tfms=get_transforms(), # 图片所需的处理

                                   size=224, # 图片裁剪大小, but having bus error,

#                                    size=56, # 缩小图片

                                   bs=bs # 一次处理图片数量

                                  ).normalize(imagenet_stats) # 图片处理的均值与方差
data.show_batch(rows=3, figsize=(7,6)) # 展示数据
print(data.classes) # 打印类别名称

len(data.classes),data.c # 类别总数
# 创建一个CNN模型，使用data作为数据，下载和调用resnet34作为模型框架和参数，同时打印错误率

learn = create_cnn(data, models.resnet34, metrics=error_rate) 
learn.model # 查看模型内在结构
learn.fit_one_cycle(1) # 训练模型，完整训练数据集一遍
learn.save('/kaggle/working/stage-1') # 将模型保存在工作目录下，commit后可下载
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
doc(interp.plot_top_losses)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.most_confused(min_val=2)
learn.unfreeze()
learn.fit_one_cycle(1)
learn.load('/kaggle/working/stage-1');
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(1, max_lr=slice(1e-6,1e-4))
learn.save('/kaggle/working/stage-2')
data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(),

                                   size=299, bs=bs//2).normalize(imagenet_stats)
learn = create_cnn(data, models.resnet50, metrics=error_rate)
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(1) # 4 次是正常数量
learn.save('/kaggle/working/stage-1-50')
learn.unfreeze()

learn.fit_one_cycle(1, max_lr=slice(1e-6,1e-4))
learn.load('/kaggle/working/stage-2-50');
interp = ClassificationInterpretation.from_learner(learn)
interp.most_confused(min_val=2)
path = untar_data(URLs.MNIST_SAMPLE); path
tfms = get_transforms(do_flip=False)

data = ImageDataBunch.from_folder(path, ds_tfms=tfms, size=26)
data.show_batch(rows=3, figsize=(5,5))
learn = create_cnn(data, models.resnet18, metrics=accuracy)

learn.fit(1)
df = pd.read_csv(path/'labels.csv')

df.head()
data = ImageDataBunch.from_csv(path, ds_tfms=tfms, size=28)
data.show_batch(rows=3, figsize=(5,5))

data.classes
data = ImageDataBunch.from_df(path, df, ds_tfms=tfms, size=24)

data.classes
fn_paths = [path/name for name in df['name']]; fn_paths[:2]
pat = r"/(\d)/\d+\.png$"

data = ImageDataBunch.from_name_re(path, fn_paths, pat=pat, ds_tfms=tfms, size=24)

data.classes
data = ImageDataBunch.from_name_func(path, fn_paths, ds_tfms=tfms, size=24,

        label_func = lambda x: '3' if '/3/' in str(x) else '7')

data.classes
labels = [('3' if '/3/' in str(x) else '7') for x in fn_paths]

labels[:5]
data = ImageDataBunch.from_lists(path, fn_paths, labels=labels, ds_tfms=tfms, size=24)

data.classes