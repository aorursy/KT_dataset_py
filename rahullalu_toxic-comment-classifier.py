# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.linear_model import RidgeClassifier

from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import make_moons, make_circles, make_classification

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import SGDClassifier

from sklearn import metrics

from sklearn.metrics import roc_curve, auc

from sklearn.linear_model.logistic import LogisticRegression

#from xgboost import XGBClassifier

from xgboost.sklearn import XGBClassifier

from lightgbm.sklearn import LGBMClassifier 







print('Reading data...')

train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')



df_train=train.copy()

for col in df_train.columns.values:

  df_train[col]=df_train[col].astype('str')



print('Transforming target field...')

df_train['target']=df_train.apply(lambda x: x.toxic+x.severe_toxic+x.obscene+x.threat+x.insult+x.identity_hate,axis=1)

df_train['target']=df_train.target.apply(lambda x: int(x,2))



df_train=df_train.drop(['toxic','severe_toxic','obscene','threat','insult','identity_hate'],axis=1)



print('Execution finished....')

# Any results you write to the current directory are saved as output.
df_target=pd.DataFrame(df_train.target.unique(),columns=['target'])

df_target['num']=df_target.target.apply(lambda x: len(df_train[df_train.target==x]))

df_target['percentage']=df_target.num.apply(lambda x: (x/len(df_train))*100)

df_target
def get_sparse(df,df1,df2):

  

  Tvect=TfidfVectorizer(ngram_range=(1,4))  



  vect=Tvect.fit(df)

  vect1=vect.transform(df)

  vect2=vect.transform(df1)

  vect3=vect.transform(df2)

  

  return(vect1,vect2,vect3)
def get_sparseH(df,df1,df2):

  

  Hvect=HashingVectorizer(ngram_range=(1,4))  



  vect=Hvect.fit(df)

  vect1=vect.transform(df)

  vect2=vect.transform(df1)

  vect3=vect.transform(df2)

  

  return(vect1,vect2,vect3)
df_y=df_train.target

df_X=df_train.comment_text

train_X,test_X,train_y,test_y=train_test_split(df_X,df_y)

print('train shape:',train_X.shape,train_y.shape)

print('test shape:',test_X.shape,test_y.shape)
df_pred_X=test.comment_text
%%time

sp_train,sp_test,sp_pred=get_sparseH(train_X,test_X,df_pred_X)

print(sp_train.shape,sp_test.shape,sp_pred.shape)
clf=LGBMClassifier(n_jobs=-1,objective='multi:softmax')

clf
%%time





clf.fit(sp_train,train_y)

pred_y=clf.predict(sp_test)

print('accuracy for classifier is:',accuracy_score(test_y,pred_y))





print(metrics.classification_report(test_y,pred_y))
%%time

df_final_sub=pd.DataFrame(test.id,columns=['id','target'])

df_final_sub['target']=clf.predict(sp_pred)

df_final=pd.DataFrame(df_final_sub.target.unique(),columns=['target'])

df_final['count']=df_final.target.apply(lambda x: len(df_final_sub[df_final_sub.target==x]))

df_final
df_target
df_final_sub1=df_final_sub.copy()

df_final_sub1['toxic']=df_final_sub1.target.apply(lambda x: '{:0>6}'.format(str(bin(x)).split('b')[1])[0])

df_final_sub1['severe_toxic']=df_final_sub1.target.apply(lambda x: '{:0>6}'.format(str(bin(x)).split('b')[1])[1])

df_final_sub1['obscene']=df_final_sub1.target.apply(lambda x: '{:0>6}'.format(str(bin(x)).split('b')[1])[2])

df_final_sub1['threat']=df_final_sub1.target.apply(lambda x: '{:0>6}'.format(str(bin(x)).split('b')[1])[3])

df_final_sub1['insult']=df_final_sub1.target.apply(lambda x: '{:0>6}'.format(str(bin(x)).split('b')[1])[4])

df_final_sub1['identity_hate']=df_final_sub1.target.apply(lambda x: '{:0>6}'.format(str(bin(x)).split('b')[1])[5])

df_final_sub1=df_final_sub1.drop(['target'],axis=1)

df_final_sub1.to_csv('final_sub.csv',index=False)