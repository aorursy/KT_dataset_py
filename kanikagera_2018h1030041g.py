# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
Train = pd.read_csv("/kaggle/input/eval-lab-3-f464/train.csv")

Test = pd.read_csv("/kaggle/input/eval-lab-3-f464/test.csv")
Train.info()
Test.info()
TrainAnalysis= Train.select_dtypes(['object'])

for col in TrainAnalysis.columns: 

    print(col)

    print(TrainAnalysis[col].unique()) 
TrainAnalysis=TrainAnalysis.replace(r'^\s*$', np.nan, regex=True)
Train['TotalCharges'] = pd.to_numeric(Train['TotalCharges'],errors='coerce')

Test['TotalCharges'] = pd.to_numeric(Test['TotalCharges'],errors='coerce')
Train['TotalCharges'].fillna(0, inplace=True)

Test['TotalCharges'].fillna(0, inplace=True)
train= Train[Train.columns.drop(['custId','Satisfied'])]

test= Test[Test.columns.drop(['custId'])]
custId=Test['custId']
train_y=Train['Satisfied']
train = pd.get_dummies(train)

test = pd.get_dummies(test)
X= pd.concat([train, test])
newdataset= train

newdataset['Class']=train_y

import seaborn as sns

import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize=(20, 20))

corr = newdataset.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10,as_cmap=True),

            square=True, ax=ax)
from sklearn.preprocessing import StandardScaler

X_new = pd.DataFrame(StandardScaler().fit_transform(X),columns=X.columns)
X_new_2=X_new.drop(['Internet_No',                   

'Internet_Yes' ,               

'HighSpeed_No' ,'HighSpeed_Yes' ,               

'HighSpeed_No internet',

          'gender_Female','gender_Male'],axis=1)
# from sklearn.preprocessing import normalize

# X_normalized = normalize(X_new_2) 
from sklearn.cluster import SpectralClustering

from sklearn import metrics



    

model = SpectralClustering(n_clusters=2, affinity='poly',

                           assign_labels='kmeans',n_init=1000, random_state=42,n_neighbors=35 ,gamma=0.5,n_jobs=-1,eigen_solver='arpack')

y_pred=model.fit_predict(X_new_2)

fpr, tpr, thresholds = metrics.roc_curve(train_y, y_pred[:4930])

print(metrics.auc(fpr, tpr))
predict_y=y_pred[4930:]

sol_agg= pd.DataFrame({'custId':custId,'Satisfied':predict_y})

sol_agg.to_csv('km_spectral2.csv',index=False)
from sklearn.cluster import KMeans

from sklearn import metrics

kmeans_wo_outlier_concat= KMeans(n_clusters=2,max_iter=1000, n_init=20,random_state=42)

kmeans_wo_outlier_concat.fit(X_new_2)

y_pred=kmeans_wo_outlier_concat.fit_predict(X_new_2)

fpr, tpr, thresholds = metrics.roc_curve(train_y, y_pred[:4930])

print(metrics.auc(fpr, tpr))
# predict_y=y_pred[4930:]

# sol_agg= pd.DataFrame({'custId':custId,'Satisfied':predict_y})

# sol_agg.to_csv('kmeans.csv',index=False)