#from google.colab import drive



# This will prompt for authorization.

#drive.mount('/content/drive')
import numpy as np 

import pandas as pd 

import os
data = pd.read_csv('../input/train.csv')
pd.set_option('display.max_colwidth',-1)

#data.head()
columns=data.columns

for x in columns:

  data[x]=pd.Categorical(data[x]).codes
import matplotlib.pyplot as plt

import seaborn as sns

f, ax = plt.subplots(figsize=(150, 30))

corr = data.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax, annot = True);
drop_data=data

droplist=[]

for x in columns:

  if corr['Class'][x]<0.01:

    droplist.append(x)

    drop_data=drop_data.drop([x], axis = 1)
y=drop_data['Class']

X=drop_data.drop(['Class'],axis=1)

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)
from sklearn import preprocessing

#Performing Min_Max Normalization

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(X_train)

X_train = pd.DataFrame(np_scaled)

np_scaled_val = min_max_scaler.transform(X_val)

X_val = pd.DataFrame(np_scaled_val)

#X_train.head()
from sklearn.naive_bayes import GaussianNB as NB
nb = NB(priors=[0.25,0.75])

nb.fit(X_train,y_train)

nb.score(X_val,y_val)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score



y_pred_NB = nb.predict(X_val)

print(confusion_matrix(y_val, y_pred_NB))
	

from sklearn.metrics import roc_auc_score

print(roc_auc_score(y_val, y_pred_NB))
print(classification_report(y_val, y_pred_NB))
from sklearn.neighbors import KNeighborsClassifier

#KNeighborsClassifier?
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train,y_train)

knn.score(X_val,y_val)
y_pred_KNN = knn.predict(X_val)

cfm = confusion_matrix(y_val, y_pred_KNN, labels = [0,1])

print(cfm)
from sklearn.metrics import roc_auc_score

print(roc_auc_score(y_val, y_pred_KNN))
from sklearn.ensemble import AdaBoostClassifier
ad=AdaBoostClassifier()
ad.fit(X_train,y_train)

ad.score(X_val,y_val)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

y_pred_ada = ad.predict(X_val)

cfm = confusion_matrix(y_val, y_pred_ada, labels = [0,1])

print(cfm)
	

from sklearn.metrics import roc_auc_score

print(roc_auc_score(y_val, y_pred_ada))
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

rf2 = RandomForestClassifier(n_estimators = 8, max_depth = 10, min_samples_split = 9, class_weight={0:9000000,1:90000000})

rf2.fit(X_train, y_train)

rf2.score(X_val,y_val)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

y_pred_RF2 = rf2.predict(X_val)

confusion_matrix(y_val, y_pred_RF2)
from sklearn.metrics import roc_auc_score

print(roc_auc_score(y_val, y_pred_RF2))
test = pd.read_csv('../input/test.csv')

columns=test.columns

for x in columns:

  test[x]=pd.Categorical(test[x]).codes

IDarray=test['ID']



for x in droplist:

  test=test.drop([x],axis=1)


y_pred_RF = rf2.predict(test)

#print(y_pred_RF.size)
Submission = pd.DataFrame({'ID':IDarray,'Class':y_pred_RF})

#Submission.to_csv('submission_RF6.csv', index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(Submission)