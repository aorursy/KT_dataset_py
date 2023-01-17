%reload_ext autoreload

%autoreload 2

%matplotlib inline
import os

import numpy as np

import pandas as pd

import fastai.vision as fa # 载入工具

from fastai.metrics import error_rate  # 补充工具



print(os.listdir())
# batch_size，由于我们的内存不够，所以我们设置为8

bs=32
# 下载数据包，并生成数据文件地址

path=fa.untar_data(fa.URLs.PETS)

path
# 查看数据文件内容

path.ls()
# 进一步查找直接的数据文件

path_anno=path/'annotations'

path_img=path/'images'
# 将所有数据图片做成地址path，放在一个list中

fnames=fa.get_image_files(path_img)

fnames[:5]
# 设置随机数种子，保证每次的结果是一样的

np.random.seed(2)

# 使用正则表达式从文件中提取图片的label标注

pat=r'/([^/]+)_\d+.jpg$'
# 生成数据集，并且对图片做标准化处理

data=fa.ImageDataBunch.from_name_re(path=path_img,fnames=fnames,pat=pat,

                                ds_tfms=fa.get_transforms(),

                                 size=224,bs=bs).normalize(fa.imagenet_stats)
# 显示图片，显示的图片是随机的

data.show_batch(rows=3,figsize=(7,6))
# 打印类别名称

print(data.classes)

# 显示类别的个数，有两种方法可以显示

len(data.classes),data.c
# 床架一个cnn模型，使用data作为数据，下载和调用resnet34，

# 作为模型的参数和框架，并输出错误率

learn=fa.create_cnn(data=data,base_arch=fa.models.resnet34,metrics=error_rate)
# 查看模型的内部结构

learn.model
# 训练模型

learn.fit_one_cycle(1)
# 保存模型，将模型保存在工作目录下

learn.save("/kaggle/working/stage-1")
# 我们查看模型的训练结果

intrep=fa.ClassificationInterpretation.from_learner(learn)

losses,idxs=intrep.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
# 输出训练损失前9个，说明这些种类比较难分类

intrep.plot_top_losses(9,figsize=(15,11))
# 我们可以通过doc查看某个函数的源代码

fa.doc(intrep.plot_top_losses)
# 我们可以输出验证样本的混淆矩阵

intrep.plot_confusion_matrix(figsize=(12,12),dpi=60)

# 我们可以发现训练中的损失越高，在混淆矩阵中就越容易分类错误
# 我们输出最容易分类错误的几个类

# 分类错误的数量大于2

intrep.most_confused(min_val=2)
# 既然模型训练正常，我们就可以对模型进行解冻，使用更多层进行训练

learn.unfreeze()
# 使用更多层进行训练，我们可以看到效果反而变差了

learn.fit_one_cycle(1)
# 载入模型

learn.load("/kaggle/working/stage-1")
# 获得模型的学习率

learn.lr_find()
# 我们对模型的学习率进行画图

learn.recorder.plot()
# 我们选择学习率子1e-6和1e-4之间，重新训练模型

learn.unfreeze()

learn.fit_one_cycle(cyc_len=1,max_lr=slice(1e-6,1e-4))
# 保存模型

learn.save("/kaggle/working/stage-1")
learn50=fa.create_cnn(data=data,base_arch=fa.models.resnet50,metrics=error_rate)
# 查看模型结构

learn50.model
# 对于模型进行训练

learn50.fit_one_cycle(1)
# 保存模型

learn50.save("kaggle/working/stage-2")
# 查看混淆矩阵

interp=fa.ClassificationInterpretation.from_learner(learn50)
# 输出l损失最高的前9个样本

interp.plot_top_losses(9,figsize=(15,11))
# 画出样本的混淆矩阵

interp.plot_confusion_matrix(figsize=(12,12))
# 获得学习率曲线

learn50.lr_find()
learn50.recorder.plot()
# 解冻、使用学习率曲线

learn50.unfreeze()

learn50.fit_one_cycle(1,max_lr=slice(1e-6,1e-4)) # 这里的效果比之前的模型要好
# 保存模型

learn50.save("/kaggle/working/stage-2")