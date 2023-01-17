%reload_ext autoreload

%autoreload 2

%matplotlib inline
import os

import gc

import numpy as np

import pandas as pd

from fastai.text import *
path=untar_data(URLs.IMDB_SAMPLE)

path.ls()
df=pd.read_csv(path/"texts.csv")

df.head()
df.text[1]
# 构建文本的databunch

data_lm=TextDataBunch.from_csv(path,"texts.csv")
# 保存数据集

data_lm.save()
data=load_data(path)
data.show_batch()
data=TextClasDataBunch.from_csv(path,"texts.csv")

data.show_batch()
# 查看我们的词典前10个词

data.vocab.itos[:10]
# 查看我们的训练数据集

data.train_ds[0][0]
# 我们查看前10个单词的字典索引

data.train_ds[0][0].data[:10]
data=(TextList.from_csv(path,"texts.csv",cols="text")

     .split_from_df(2)  # 这里使用is_valid属性划分验证集和训练集

     .label_from_df(0)

     .databunch())
data.show_batch()
bs=128  # 因为语言模型会大量消耗我们的gpu，所以当我们gpu的内存不够时，我们可以减少这个值
# 获取数据

path=untar_data(URLs.IMDB)

path.ls()
# 查看一下训练集train文件夹下的文件

(path/"train").ls()
# 建立databunch

data_lm=(TextList.from_folder(path)   # 所有的文本文件都在path路径下

        .filter_by_folder(include=["train","test","unsup"])

        .split_by_rand_pct(0.1)   # 我们设置10%为验证集

        .label_for_lm()   # 为数据集建立文本标签

        .databunch(bs=bs))   # 设置batch_size

data_lm.save("/kaggle/working/data_lm.pkl")
# 建立模型(在视频上，作者使用了WT103,而在github上面的版本，使用了AWD-LSTM，这个模型是由维基百科语料训练的),同时这里为了防止过拟合，设置dropout=0.3

learn=language_model_learner(data_lm,AWD_LSTM,drop_mult=0.3)
learn.lr_find()
# 查看学习率曲线

learn.recorder.plot(skip_end=15)   # 修剪后面15%的曲线，使得曲线看起来不再那么光滑
learn.fit_one_cycle(1,max_lr=1e-2,moms=(0.8,0.7))   # 设置动量在0.7和0.8之间
# 保存模型

learn.save("/kaggle/working/fit_head")
# 对模型进行解冻训练

learn.unfreeze()

learn.fit_one_cycle(1,max_lr=1e-3,moms=(0.8,0.7))
learn.save("/kaggle/working/fine_tuned")
text="I like the movie because"

"\n".join([learn.predict(text,n_words=40,temperature=0.75) for i in range(2)])
learn.save_encoder("/kaggle/working/fine_tuned_enc")
path=untar_data(URLs.IMDB)
path.ls()
data_clas=(TextList.from_folder(path,vocab=data_lm.vocab)

          .split_by_folder(valid="test")

          .label_from_folder(classes=["neg","pos"])

          .databunch(bs=bs))

data_clas.save("/kaggle/working/data_clas.pkl")
data_clas.show_batch()
# 建立模型

learn=text_classifier_learner(data_clas,AWD_LSTM,drop_mult=0.5)

learn.load_encoder("/kaggle/working/fine_tuned_enc")    # 这里要用load_encoder
# 学习率曲线

learn.lr_find()
learn.recorder.plot(skip_start=15)
# 训练

learn.fit_one_cycle(1,max_lr=2e-2,moms=(0.8,0.7))
learn.save("/kaggle/working/first")
# 首先对最后两层模型进行训练

learn.freeze_to(-2)

learn.fit_one_cycle(1,slice(1e-2/(2.6**4),1e-2),moms=(0.8,0.7))  # 这里用到的学习率都是作者的经验所得
# 保存

learn.save("/kaggle/working/second")
# 对倒数第三层进行训练

learn.load("/kaggle/working/second")

learn.freeze_to(-3)

learn.fit_one_cycle(1,slice(5e-3/(2.6**4),5e-3),moms=(0.8,0.7))
learn.save("/kaggle/working/third")
learn.unfreeze()

learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))
learn.predict("I really loved that movie, it was awesome!")