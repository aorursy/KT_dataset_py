import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings



from skimage.io import imshow, imread



warnings.filterwarnings('ignore')
df=pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')

df.head(3)
y=df['diagnosis'] #output labels

df.drop(columns=['Unnamed: 32','id'],inplace=True) #one is useless, the other is Nan
sns.heatmap(df.iloc[:,1:11].corr(),annot=True,fmt='.1g');
sns.heatmap(df.iloc[:,11:21].corr(),annot=True,fmt='.1g');
sns.heatmap(df.iloc[:,21:].corr(),annot=True,fmt='.1g');
df.drop(columns=['perimeter_mean','area_mean','compactness_mean','concave points_mean',

                      'perimeter_se','area_se',

                      'perimeter_worst','area_worst'],

                       inplace=True)

plt.figure(figsize=(14,10))

sns.heatmap(df.iloc[:,1:].corr(), annot=True, fmt='.1g');
df.drop(columns=['radius_worst','texture_worst','concavity_worst','concave points_worst',

                'texture_worst',], inplace=True)
#how many features does the new dataset have?

print ('The resulting dataset has',df.shape[1]-1, 'features')
#logistic regression for feature selection

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler



y=df['diagnosis']

enc=LabelEncoder()

y=enc.fit_transform(y.values)

x=df.drop(columns='diagnosis').values

x_tr,x_ts,y_tr,y_ts=train_test_split(x,y, random_state=7, test_size=0.2,stratify=y)

sc=StandardScaler()

x_tr_sc=sc.fit_transform(x_tr)

x_ts_sc=sc.transform(x_ts)



#after experiments we found this is best model

lr=LogisticRegression(C=1.0, random_state=7)

lr.fit(x_tr_sc,y_tr)

y_pred=lr.predict(x_ts_sc)

print('coefs', lr.coef_)

print('accuracy', accuracy_score(y_ts,y_pred))
#which are the most significant features, and how much they contribute

coefs=lr.coef_.reshape(18)

for ind in lr.coef_.argsort().reshape(18):

    print(df.columns[ind+1])

    print(coefs[ind])

    print('')
#find the ten most useful features

ab=np.abs(coefs)

cols=df.columns[ab.argsort()[:-11:-1]+1]

cols
#train a model with only the best features



x=df[cols].values

x_tr,x_ts,y_tr,y_ts=train_test_split(x,y, random_state=7, test_size=0.2,stratify=y)

sc=StandardScaler()

x_tr_sc=sc.fit_transform(x_tr)

x_ts_sc=sc.transform(x_ts)





lr=LogisticRegression(C=10.0, random_state=7)

lr.fit(x_tr_sc,y_tr)

y_pred=lr.predict(x_ts_sc)

print('coefs', lr.coef_)

print('accuracy', accuracy_score(y_ts,y_pred))



from sklearn.metrics import confusion_matrix



plt.figure(figsize=(3,3))

sns.heatmap(confusion_matrix(y_ts,y_pred), annot=True, fmt='d');
#make violin plots to visually evaluate features selected by lr

means=df.iloc[:,1:7]

ses=df.iloc[:,7:15]

worsts=df.iloc[:,15:]



means_sc=(means-means.mean())/(means.std())

ses_sc=(ses-ses.mean())/(ses.std())

worsts_sc=(worsts-worsts.mean())/(worsts.std())



means_sc=pd.concat([df['diagnosis'],means_sc],axis=1)

ses_sc=pd.concat([df['diagnosis'],ses_sc],axis=1)

worsts_sc=pd.concat([df['diagnosis'],worsts_sc],axis=1)



means_sc=pd.melt(means_sc, id_vars='diagnosis',

                 var_name='features',

                 value_name='value')

ses_sc=pd.melt(ses_sc, id_vars='diagnosis',

               var_name='features',

               value_name='value')

worsts_sc=pd.melt(worsts_sc, id_vars='diagnosis',

                  var_name='features',

                  value_name='value')
sns.violinplot(y='features',x='value', hue='diagnosis',

               data=means_sc, split=True);
sns.violinplot(y='features',x='value', hue='diagnosis',

               data=ses_sc, split=True);
sns.violinplot(y='features',x='value', hue='diagnosis', 

               data=worsts_sc, split=True);
from sklearn.tree import DecisionTreeClassifier



tree=DecisionTreeClassifier(max_depth=4)

tree.fit(x_tr,y_tr)

y_pred=tree.predict(x_ts)



print('accuracy:',accuracy_score(y_pred,y_ts))
plt.figure(figsize=(3,3))

sns.heatmap(confusion_matrix(y_ts,y_pred), annot=True, fmt='d');
enc.inverse_transform([0,1])
#create graph

from sklearn.tree import export_graphviz

import graphviz

from graphviz import Source



graph=Source(export_graphviz(tree,feature_names=df[cols].columns,

                   class_names=['B','M'],rounded=True,proportion = False, filled=True,precision=2))







display(graph)