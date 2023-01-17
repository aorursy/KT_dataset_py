# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
dataset=pd.read_csv("../input/adult.csv")

dataset.head()

dataset.columns
dataset["hours.per.week"].describe()

dataset["education.num"].median()
#得到所有的工作

dataset["workclass"].unique()
dataset["longhours"]=dataset["hours.per.week"]>40

dataset["longhours"]
#删除方差达不到最低标准的特征

import numpy as np

x=np.arange(30).reshape(10,3)

x
x[:,1]=1

x

from sklearn.feature_selection import VarianceThreshold

vt=VarianceThreshold()

xt=vt.fit_transform(x)

xt
dataset.head()
#卡方检验，互信息和信息熵

x1=dataset[['age',  'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']].values

y1=(dataset["income"]=='>50K').values

x1,y1
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

transfomer=SelectKBest(score_func=chi2,k=3)

xt_chi2=transfomer.fit_transform(x1,y1)

print(transfomer.scores_)

#输出数值越大表示相关性越好
#用皮尔逊相关系数计算相关性

#皮尔逊函数接收的x只能为一维数组

from scipy.stats import pearsonr

def mulity_pearsonr(x,y):

    scores,pvalues=[],[]

    for column in range(x.shape[1]):

        cur_score,cur_p=pearsonr(x[:,column],y)

        print(cur_score,cur_p)

        scores.append(abs(cur_score))

        pvalues.append(cur_p)

    print((np.array(scores),np.array(pvalues)))

    return (np.array(scores),np.array(pvalues))

#直接调用

#1，2，5列最好

transformer1=SelectKBest(score_func=mulity_pearsonr,k=3)

xt_pearsonr=transformer1.fit_transform(x1,y1)

print(transformer1.scores_)
#在分类器中看那个特征集合最好

#可见皮尔逊的特征集合较好

from sklearn.tree import DecisionTreeClassifier

from sklearn.cross_validation import cross_val_score

clf=DecisionTreeClassifier(random_state=14)

scores_chi2=cross_val_score(clf,xt_chi2,y1,scoring='accuracy')

scores_pearsonr=cross_val_score(clf,xt_pearsonr,y1,scoring='accuracy')

print(scores_chi2.mean())

print(scores_pearsonr.mean())