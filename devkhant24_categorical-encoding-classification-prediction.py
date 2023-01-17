# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.model_selection import cross_val_score

import xgboost

from scipy import stats

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')
df.drop('id',axis=1,inplace=True)
ob = df[['bin_3', 'bin_4', 'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5',

        'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_1', 'ord_2', 'ord_3', 'ord_4',

        'ord_5']]
dum1 = pd.get_dummies(df[['bin_3','bin_4','nom_0','nom_1','nom_2','nom_3','nom_4','ord_1','ord_2']],drop_first=True)
df.drop(['bin_3','bin_4','nom_0','nom_1','nom_2','nom_3','nom_4','ord_1','ord_2'],axis=1,inplace=True)
df = pd.concat([dum1,df],axis=1)
#for i in ob:

#    print(i,':',len(df[i].unique()))
def onehot(data,variable):

    t1 = [x for x in df[variable].value_counts().sort_values(ascending=False).head(10).index]

    for i in t1:

        df[variable+''+i] = np.where(df[variable]==i,1,0)
onehot(df,'nom_5')

onehot(df,'nom_6')

onehot(df,'nom_7')

onehot(df,'nom_8')

onehot(df,'nom_9')

onehot(df,'ord_3')

onehot(df,'ord_4')

onehot(df,'ord_5')
df.drop(['nom_5','nom_6','nom_7','nom_8','nom_9','ord_5','ord_3','ord_4'],axis=1,inplace=True)
dft = pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv')
dft.drop('id',axis=1,inplace=True)
dum1 = pd.get_dummies(dft[['bin_3','bin_4','nom_0','nom_1','nom_2','nom_3','nom_4','ord_1','ord_2']],drop_first=True)

dft.drop(['bin_3','bin_4','nom_0','nom_1','nom_2','nom_3','nom_4','ord_1','ord_2'],axis=1,inplace=True)

dft = pd.concat([dum1,dft],axis=1)
def onehot(data,variable):

    t1 = [x for x in dft[variable].value_counts().sort_values(ascending=False).head(10).index]

    for i in t1:

        dft[variable+''+i] = np.where(dft[variable]==i,1,0)
onehot(dft,'nom_5')

onehot(dft,'nom_6')

onehot(dft,'nom_7')

onehot(dft,'nom_8')

onehot(dft,'nom_9')

onehot(dft,'ord_3')

onehot(dft,'ord_4')

onehot(dft,'ord_5')

dft.drop(['nom_5','nom_6','nom_7','nom_8','nom_9','ord_5','ord_3','ord_4'],axis=1,inplace=True)
from imblearn.over_sampling import RandomOverSampler
x = df.drop('target',axis=1)

y = df['target']
os = RandomOverSampler(ratio=1)

x,y = os.fit_sample(x,y)
#z = np.abs(stats.zscore(df))

#df = df[(z<3).all(axis=1)]
bestfeatures = SelectKBest(score_func=chi2,k=10)

fit = bestfeatures.fit(x,y)
dfscore = pd.DataFrame(fit.scores_)

dfcolumn = pd.DataFrame(x.columns)

features = pd.concat([dfcolumn,dfscore],axis=1)

features.columns=['Specs','Score']

print(features.nlargest(20,'Score'))
#x = df[['nom_3_China','nom_3_Russia','ord_2_Cold','nom_4_Oboe','nom_0_Green','nom_1_Square','nom_2_Cat','nom_4_Theremin','nom_3_Finland',

#        'nom_2_Dog','ord_0','nom_0_Red','bin_1','ord_2_Lava Hot','ord_3l','ord_2_Freezing','ord_1_Novice','ord_3a','ord_1_Grandmaster','month']]
#test = dft[['nom_3_China','nom_3_Russia','ord_2_Cold','nom_4_Oboe','nom_0_Green','nom_1_Square','nom_2_Cat','nom_4_Theremin','nom_3_Finland',

 #       'nom_2_Dog','ord_0','nom_0_Red','bin_1','ord_2_Lava Hot','ord_3l','ord_2_Freezing','ord_1_Novice','ord_3a','ord_1_Grandmaster','month']]
lg = LogisticRegression(solver='lbfgs')

xg = xgboost.XGBClassifier()

dt = DecisionTreeClassifier(random_state=1)

rf = RandomForestClassifier(random_state=1)

svm = SVC(kernel='linear')

nb = GaussianNB()

knn = KNeighborsClassifier()

ss = StandardScaler()
x = ss.fit_transform(x)

test = ss.fit_transform(dft)
#import sklearn

#sorted(sklearn.metrics.SCORERS.keys())
#score = cross_val_score(lg,x,y,cv=5,scoring='accuracy')

#score.mean()
lg.fit(x,y)
yp = lg.predict(test)

pred = pd.DataFrame(yp)

pred.columns = ['target']
c = pd.read_csv('/kaggle/input/cat-in-the-dat/sample_submission.csv')
a = c['id']

submission = pd.concat([a,pred],axis=1)
submission.to_csv('sample_submission.csv',index=False)