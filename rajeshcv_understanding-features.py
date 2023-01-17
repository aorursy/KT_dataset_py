import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

print(os.listdir("../input"))

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
train.info()
likely_cat = {}

for c in train.columns:

    likely_cat[c] = 1.*train[c].nunique()/train[c].count() < 0.005

likely_cat= pd.Series(likely_cat)

likely_cat[likely_cat==True]
train.var_68.nunique()
trainvaluedist = pd.DataFrame(train.iloc[:,2:].max(axis=0),columns=["Max_value"])

trainvaluedist['Min_value'] = train.iloc[:,2:].min(axis=0)

trainvaluedist['Median_value'] = train.iloc[:,2:].median(axis=0)

trainvaluedist.head()
sns.set(rc={'figure.figsize':(24,12)})

line=sns.lineplot(data=trainvaluedist )

line= line.set(yticks=[-80,-60,-40,-30,-20,-10,0,10,20,30,40,60,80])

colzerototen= [c for c in train.iloc[:,2:].columns if (train.loc[:,c].min() >=0) & (train.loc[:,c].max()< 10) ]

print('Number of features with positive values and maximum value less than 10 :',len(colzerototen))

colzerototwenty= [c for c in train.iloc[:,2:].columns if (train.loc[:,c].min() >=0) & (train.loc[:,c].max() >= 10) & (train.loc[:,c].max() < 20)  ]

print('Number of features with positive values maximum value between 10 & 20 :',len(colzerototwenty))

colzeroandtwentyplus= [c for c in train.iloc[:,2:].columns if (train.loc[:,c].min() >=0) & (train.loc[:,c].max() >= 20)]

print('Number of features with positive values maximum value > 20 :',len(colzeroandtwentyplus))

colzerominus= [c for c in train.iloc[:,2:].columns if train.loc[:,c].max() <0 ]

print('Number of features with only negative values :',len(colzerominus))

colplustenminusten= [c for c in train.iloc[:,2:].columns if (train.loc[:,c].max() <= 10) & (train.loc[:,c].min() >=-10 )& (train.loc[:,c].min()< 0 )]

print('Number of features with negative values between 10 and -10 :',len(colplustenminusten))

colplustwentyminusten= [c for c in train.iloc[:,2:].columns if (train.loc[:,c].max() <= 20)& (train.loc[:,c].max() > 10) & (train.loc[:,c].min() >=-10 ) & (train.loc[:,c].min() < 0 )]

print('Number of features with max value between 10 and 20 and min value between  between 0 and -10  :',len(colplustwentyminusten))

colplustwentyminustwenty= [c for c in train.iloc[:,2:].columns if (train.loc[:,c].max() <= 20) &  (train.loc[:,c].min() < -10 ) & (train.loc[:,c].min() >= -20 )]

print('Number of features with max value less than 20 and min value between -10 and -20 :',len(colplustwentyminustwenty))

colplustwentyminustwentyless= [c for c in train.iloc[:,2:].columns if (train.loc[:,c].max() <= 20)& (train.loc[:,c].min() < -20 )]

print('Number of features with max value less than 20 and min value less than -20 :',len(colplustwentyminustwentyless))

colplustwentymoreminusten= [c for c in train.iloc[:,2:].columns if (train.loc[:,c].max() >20)& (train.loc[:,c].min()< 0 ) & (train.loc[:,c].min()>= -10 )]

print('Number of features with max value more than 20 and min value more than -10 :',len(colplustwentymoreminusten))

colplustwentymoreminustwenty= [c for c in train.iloc[:,2:].columns if (train.loc[:,c].max() >20)& (train.loc[:,c].min()< -10 ) & (train.loc[:,c].min()>= -20 )]

print('Number of features with max value more than 20 and min value between -10 and -20:',len(colplustwentymoreminustwenty))

colplustwentymoreminustwentymore= [c for c in train.iloc[:,2:].columns if (train.loc[:,c].max() >20)& (train.loc[:,c].min()< -20 )]

print('Number of features with max value more than 20 and min value less than -20:',len(colplustwentymoreminustwentymore))
sns.set(rc={'figure.figsize':(20,8)})

setpositive=train.loc[:,colzerototen].boxplot(rot=90)

setpositive=setpositive.set(yticks=[0,2.5,5,7.5,10],title="Features with  positive values and maximum value less than 10")
sns.set(rc={'figure.figsize':(20,16)})

plotlist =['hist'+ str(col) for col in colzerototen]



for k in range(len(colzerototen)):

     plt.subplot(4,4,k+1)

     plotlist[k] =plt.hist(train[colzerototen[k]])

     #plotlist[k].set(title=colzerototen[k])

    

sns.set(rc={'figure.figsize':(20,16)})

def sephist(col):

    yes = train[train['target'] == 1][col]

    no = train[train['target'] == 0][col]

    return yes, no



for num, alpha in enumerate(colzerototen):

    plt.subplot(4, 4, num+1)

    plt.hist(sephist(alpha)[0], alpha=0.75, label='yes', color='g')

    plt.hist(sephist(alpha)[1], alpha=0.25, label='no', color='r')

    plt.legend(loc='upper right')

    plt.title(alpha)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
sns.set(rc={'figure.figsize':(20,8)})

setpositive20=train.loc[:,colzerototwenty].boxplot(rot=90)

setpositive20=setpositive20.set(yticks=[0,5,10,15,20],title="Features with  positive values and maximum value between 10 & 20")
sns.set(rc={'figure.figsize':(16,24)})

for num, alpha in enumerate(colzerototwenty):

    plt.subplot(8, 4, num+1)

    plt.hist(sephist(alpha)[0], alpha=0.75, label='yes', color='g')

    plt.hist(sephist(alpha)[1], alpha=0.25, label='no', color='r')

    plt.legend(loc='upper right')

    plt.title(alpha)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
sns.set(rc={'figure.figsize':(20,8)})

setpositive20plus=train.loc[:,colzeroandtwentyplus].boxplot(rot=90)

setpositive20plus=setpositive20plus.set(yticks=[0,10,20,30,40],title="Features with  positive values and maximum value more than 20")
sns.set(rc={'figure.figsize':(16,20)})

for num, alpha in enumerate(colzeroandtwentyplus):

    plt.subplot(6, 4, num+1)

    plt.hist(sephist(alpha)[0], alpha=0.75, label='yes', color='g')

    plt.hist(sephist(alpha)[1], alpha=0.25, label='no', color='r')

    plt.legend(loc='upper right')

    plt.title(alpha)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
sns.set(rc={'figure.figsize':(16,8)})

setplustenminusten = train.loc[:,colplustenminusten].boxplot(rot=90)

setplustenminusten = setplustenminusten.set(yticks=[-10,-5,0,5,10],title="Features with  values between 10 and -10")
sns.set(rc={'figure.figsize':(16,16)})

for num, alpha in enumerate(colplustenminusten):

    plt.subplot(4, 4, num+1)

    plt.hist(sephist(alpha)[0], alpha=0.75, label='yes', color='g')

    plt.hist(sephist(alpha)[1], alpha=0.25, label='no', color='r')

    plt.legend(loc='upper right')

    plt.title(alpha)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
sns.set(rc={'figure.figsize':(16,8)})

setplustwentyminusten = train.loc[:,colplustwentyminusten].boxplot(rot=90)

setplustwentyminusten = setplustwentyminusten.set(yticks=[-10,-5,0,5,10,15,20],title="Features with  max value between 10 and 20 and min values between 0 and -10")
sns.set(rc={'figure.figsize':(16,16)})

for num, alpha in enumerate(colplustwentyminusten):

    plt.subplot(5, 4, num+1)

    plt.hist(sephist(alpha)[0], alpha=0.75, label='yes', color='g')

    plt.hist(sephist(alpha)[1], alpha=0.25, label='no', color='r')

    plt.legend(loc='upper right')

    plt.title(alpha)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
sns.set(rc={'figure.figsize':(20,8)})

setplustwentyminustwenty=train.loc[:,colplustwentyminustwenty].boxplot(rot=90)

setplustwentyminustwenty=setplustwentyminustwenty.set(yticks=[-20,-15,-10,-5,0,5,10,15,20],title="Features with  max value between 10 and 20 and min values between -10  and -20")
sns.set(rc={'figure.figsize':(16,16)})

for num, alpha in enumerate(colplustwentyminustwenty):

    plt.subplot(4, 4, num+1)

    plt.hist(sephist(alpha)[0], alpha=0.75, label='yes', color='g')

    plt.hist(sephist(alpha)[1], alpha=0.25, label='no', color='r')

    plt.legend(loc='upper right')

    plt.title(alpha)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
sns.set(rc={'figure.figsize':(20,8)})

setplustwentyminustwentyless=train.loc[:,colplustwentyminustwentyless].boxplot(rot=90)

setplustwentyminustwentyless=setplustwentyminustwentyless.set(yticks=[-40,-30,-20,-15,-10,-5,0,5,10,15,20],title="Features with  max value between 10 and 20 and min values less than -20")
sns.set(rc={'figure.figsize':(16,16)})

for num, alpha in enumerate(colplustwentyminustwentyless):

    plt.subplot(4, 4, num+1)

    plt.hist(sephist(alpha)[0], alpha=0.75, label='yes', color='g')

    plt.hist(sephist(alpha)[1], alpha=0.25, label='no', color='r')

    plt.legend(loc='upper right')

    plt.title(alpha)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
sns.set(rc={'figure.figsize':(20,8)})

setplustwentymoreminusten=train.loc[:,colplustwentymoreminusten].boxplot(rot=90)

setplustwentymoreminusten=setplustwentymoreminusten.set(yticks=[-10,-5,0,5,10,15,20,30,40,60],title="Features with  max value more than 20 and min values between 0 and -10")
sns.set(rc={'figure.figsize':(16,16)})

for num, alpha in enumerate(colplustwentymoreminusten):

    plt.subplot(7, 4, num+1)

    plt.hist(sephist(alpha)[0], alpha=0.75, label='yes', color='g')

    plt.hist(sephist(alpha)[1], alpha=0.25, label='no', color='r')

    plt.legend(loc='upper right')

    plt.title(alpha)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
sns.set(rc={'figure.figsize':(20,8)})

setplustwentymoreminustwenty=train.loc[:,colplustwentymoreminustwenty].boxplot(rot=90)

setplustwentymoreminustwenty=setplustwentymoreminustwenty.set(yticks=[-20,-15,-10,-5,0,5,10,15,20,30,40,60],title="Features with  max value more than 20 and min values between -10 and -20")
sns.set(rc={'figure.figsize':(16,16)})

for num, alpha in enumerate(colplustwentymoreminustwenty):

    plt.subplot(7, 4, num+1)

    plt.hist(sephist(alpha)[0], alpha=0.75, label='yes', color='g')

    plt.hist(sephist(alpha)[1], alpha=0.25, label='no', color='r')

    plt.legend(loc='upper right')

    plt.title(alpha)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
sns.set(rc={'figure.figsize':(20,8)})

setplustwentymoreminustwentymore=train.loc[:,colplustwentymoreminustwentymore].boxplot(rot=90)

setplustwentymoreminustwentymore=setplustwentymoreminustwentymore.set(yticks=[-40,-30,-20,-10,0,5,10,15,20,30,40,60],title="Features with  max value more than 20 and min values less than -20")
sns.set(rc={'figure.figsize':(16,16)})

for num, alpha in enumerate(colplustwentymoreminustwentymore):

    plt.subplot(6, 4, num+1)

    plt.hist(sephist(alpha)[0], alpha=0.75, label='yes', color='g')

    plt.hist(sephist(alpha)[1], alpha=0.25, label='no', color='r')

    plt.legend(loc='upper right')

    plt.title(alpha)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
traincorr = train.iloc[:,2:].corr()

traincorr.head()
sns.set(rc={'figure.figsize':(10,8)})

sns.heatmap(traincorr,xticklabels=traincorr.columns,yticklabels=traincorr.columns,cmap=sns.diverging_palette(240, 10, n=9))
from scipy.stats import ks_2samp

from tqdm import tqdm

ks_values =[]

p_values  = []

train_columns = train.iloc[:,2:].columns

for i in tqdm(train_columns):

    ks_values.append(ks_2samp(test[i] , train[i])[0])

    p_values.append(ks_2samp(test[i] , train[i])[1])

p_values_series = pd.Series(p_values, index = train_columns) 

dissimiliar_features= p_values_series[p_values_series <0.05].index
train['is_train'] = 1

test['is_train'] = 0

combined = pd.concat([train,test],sort=False)
sns.set(rc={'figure.figsize':(20,48)})

def  diffcheck(col):

    traindata = combined[combined['is_train'] == 1][col]

    testdata = combined[combined['is_train'] == 0][col]

    return traindata, testdata



for num, alpha in enumerate(dissimiliar_features):

    plt.subplot(12, 4, num+1)

    plt.hist(diffcheck(alpha)[0], alpha=0.75, label='train', color='g')

    plt.hist(diffcheck(alpha)[1], alpha=0.25, label='test', color='r')

    plt.legend(loc='upper right')

    plt.title(alpha)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
sns.set(rc={'figure.figsize':(20,48)})

fig, axes = plt.subplots(8,3)

for num, alpha in enumerate(list(dissimiliar_features[0:24])):

    a = sns.boxenplot(x='is_train',y=alpha,data=combined, ax=axes.flatten()[num])

# fig.delaxes(axes[11,2])

# fig.delaxes(axes[11,3])

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0) 

plt.title(alpha)

plt.show()

sns.set(rc={'figure.figsize':(20,48)})

fig, axes = plt.subplots(8,3)

for num, alpha in enumerate(list(dissimiliar_features[24:])):

    a = sns.boxenplot(x='is_train',y=alpha,data=combined, ax=axes.flatten()[num])

fig.delaxes(axes[7,1])

fig.delaxes(axes[7,2])

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0) 

plt.title(alpha)

plt.show()
# sns.set(rc={'figure.figsize':(16,48)})

# set1=train.iloc[:,47:92].hist(layout=(9,5),sharey=True)