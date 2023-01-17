# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# This ensures that any edits to libraries you make are reloaded here automatically,	

# and also that any charts or images displayed are shown in this notebook.	

%reload_ext autoreload

%autoreload 2

%matplotlib inline

#导入库

# Import libraries	

from fastai import *	

from fastai.vision import *	

from fastai.callbacks import CSVLogger, SaveModelCallback	



import warnings	

warnings.filterwarnings('ignore')
work_dir = Path('/kaggle/working/')

path = Path('../input')

train = 'train_images/train_images'

test =  path/'leaderboard_test_data/leaderboard_test_data'

holdout = path/'leaderboard_holdout_data/leaderboard_holdout_data'

sample_sub = path/'SampleSubmission.csv'

labels = path/'traininglabels.csv'

df = pd.read_csv(labels)

df_sample = pd.read_csv(sample_sub)
import seaborn as sns

sns.countplot(df.has_oilpalm)

(df.has_oilpalm==1).sum()
(df.has_oilpalm==0).sum()
df.describe()
df[df['score']<0.75]#There are some rows with low score, we will look into that later.
df.head()
#将排行榜预留数据（leaderboard holdout data ）与排行榜测试数据（leaderboard test data）组合起来

test_names = [f for f in test.iterdir()]

holdout_names = [f for f in holdout.iterdir()]

combined_test = test_names + holdout_names

len(combined_test)
#采用fast.ai的DataBlock API来构造数据，这是向模型输送数据集的一种简易的方法。

#创建一个ImageList来保存数据

src = (ImageList.from_df(df, path, folder=train)

      .random_split_by_pct(0.2, seed=14)

      .label_from_df('has_oilpalm')

      .add_test(combined_test))


data = (src.transform(get_transforms(flip_vert=True), size=164)	

           .databunch()	

           .normalize(imagenet_stats))
data.show_batch(3, figsize=(10,7))
learn = create_cnn(data, models.resnet50, 

                   metrics=[accuracy], #<---add aoc metric?

                   model_dir='/kaggle/working/models',

                  callback_fns=[ShowGraph, SaveModelCallback])
learn.lr_find()	

learn.recorder.plot()#选择一个接近坡度最陡之处的学习速率，在这个示例中是1e-2
#利用fit_one_cycle函数对模型训练5个周期 (对所有数据训练5个周期)。

learn.fit_one_cycle(5, slice(1e-2))
#注意到显示出来的结果，如training_loss 和valid_loss没有？后续，会用它们来监控模型的改进。

#在第四个循环，得到了最佳的模型。

#fast.ai在运行训练和验证数据集时，内部自动选取和保存最优的那个模型。

#竞赛组委会根据预测概率与观测目标has_oilpalm之间的工作特性曲线下的面积对参赛作品进行评价。

#默认情况下，Fast.ai没有提供这个评价标准的指标度量，所以我们将用到Scikit-Learning库。
from sklearn.metrics import roc_auc_score	

def auc_score(y_score,y_true):	

    return torch.tensor(roc_auc_score(y_true,y_score[:,1]))	

probs,val_labels = learn.get_preds(ds_type=DatasetType.Valid) 	

print('Accuracy',accuracy(probs,val_labels)),	

print('Error Rate', error_rate(probs, val_labels))	

print('AUC', auc_score(probs,val_labels))
#使用预训练模型和fast.ai的优点是，可以得到一个非常好的预测精度，在这个示例中，在没有多做其他工作的情况下，获得了99.48%精确度。

#将模型存盘，绘制出预测的混淆矩阵。

learn.save('resnet50-stg1')

interp = ClassificationInterpretation.from_learner(learn)	

interp.plot_confusion_matrix(dpi=120)

#混淆矩阵是一种图形化的方法，用来查看模型准确或不准确预测的图像数量

#第一阶段训练的混淆矩阵

#从这幅图中可以看出，模型准确地预测了2868幅没有油棕人工林的图像，对164幅油棕人工林的图像进行了正确的分类。

#将14幅含有油棕人工林的图像分类为无油棕人工林图像，并将2幅无油棕人工林图像分类为有油棕人工林图像。

#对于一个简单的模型来说这个结果还不错。
#接下来，找出这个训练迭代理想的学习率。

learn.lr_find()	

learn.recorder.plot()
#利用介于1e-6和1e-4之间的一个学习率最大值对模型进行拟合。

learn.fit_one_cycle(7, max_lr=slice(1e-6,1e-4))

#学习率在1e-6和1e-4的范围范围内，对模型进行7次循环训练

#在每个训练周期后，以图形的方式观察训练指标，从而监测模型的性能
#保存第二阶段的模型训练结果。

learn.save('resnet50-stg2')

#打印出模型的精度、错误率和曲线下面的面积。

probs,val_labels = learn.get_preds(ds_type=DatasetType.Valid) 	

print('Accuracy',accuracy(probs,val_labels)),	

print('Error Rate', error_rate(probs, val_labels))	

print('AUC', auc_score(probs,val_labels))
#你会注意到，此时，模型的准确度从99.44%提高到99.48%，

#错误率从0.0056降低到0.0052，AUC也有改善，从99.82%提高到99.87%。

interp = ClassificationInterpretation.from_learner(learn)	

interp.plot_confusion_matrix(dpi=120)

#通过与我们绘制的上一个混淆矩阵的比较，可以发现模型做出了更精准的预测。

#先前有油棕种植园的14张图片被错误分类,变为13张。



#你会注意到在训练过程中遵循了一个模式，在这个过程中调整了一些参数，这便是所谓的精调，绝大多数深度学习训练均遵循类似的迭代模式。
#我们将对数据执行更多的图像变换，通过这些变换对模型进行改进。

#关于每种变换的详细描述，可以在fast.ai的相关文档中找到：https://docs.fast.ai/vision.transform.html

tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)

data = (src.transform(tfms, size=200)

        .databunch().normalize(imagenet_stats))



#lhting：如果非空 None，则在概率p_lighting下使用由max_light控制的随机亮度和对比度变化。



#max_zoom:如果非1，或是一个比1更小的数，则在概率p_affine下使用max_zoom到1之间的随机缩放比



#max_warp：如果非空None，则使用概率p_affine应用-max_warp和max_warp之间的随机对称翘曲。



learn.data = data



data.train_ds[0][0].shape





#再次找出最优学习率：

learn.lr_find()	

learn.recorder.plot()

#选择学习率为1e-6,循环训练模型5次：

learn.fit_one_cycle(5, 1e-6)
learn.save('resnet50-stg3')
#打印出模型的精度、错误率和曲线下面的面积。

probs,val_labels = learn.get_preds(ds_type=DatasetType.Valid) 	

print('Accuracy',accuracy(probs,val_labels)),	

print('Error Rate', error_rate(probs, val_labels))	

print('AUC', auc_score(probs,val_labels))
tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)	

data = (src.transform(tfms, size=256)	

        .databunch().normalize(imagenet_stats))	

	

learn.data = data	

	

data.train_ds[0][0].shape
learn.lr_find()	

learn.recorder.plot()
learn.fit_one_cycle(5, slice(1e-4)) #学习率设置为1e-4，训练模型5次循环

 
learn.save('resnet50-stg4')
#打印出模型的精度、错误率和曲线下面的面积。

probs,val_labels = learn.get_preds(ds_type=DatasetType.Valid) 	

print('Accuracy',accuracy(probs,val_labels)),	

print('Error Rate', error_rate(probs, val_labels))	

print('AUC', auc_score(probs,val_labels))
p,t = learn.get_preds(ds_type=DatasetType.Test)
p = to_np(p); p.shape
ids = np.array([f.name for f in (test_names+holdout_names)]);ids.shape
#We only recover the probs of having palmoil (column 1)

sub = pd.DataFrame(np.stack([ids, p[:,1]], axis=1), columns=df_sample.columns)
sub.to_csv(work_dir/'sub.csv', index=False)
pd.read_csv(work_dir/'sub.csv')