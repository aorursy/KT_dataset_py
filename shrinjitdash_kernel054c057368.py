# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

traindigitrecognizer = pd.read_csv("../input/train.csv")
testdata = pd.read_csv("../input/test.csv")
# Any results you write to the current directory are saved as output.
print(traindigitrecognizer.shape)
print(testdata.shape)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
train1,test1=train_test_split(traindigitrecognizer,test_size=0.2,random_state=100)
print(train1.shape)

train_ydigit=train1['label']
test_ydigit=test1['label']
train_xdigit=train1.drop('label',axis=1)
test_xdigit=test1.drop('label',axis=1)
print(train_xdigit.shape)
from sklearn.ensemble import AdaBoostClassifier
model=AdaBoostClassifier(random_state=100)
model.fit(train_xdigit,train_ydigit)

prediction=model.predict(test_xdigit)

from sklearn.metrics import confusion_matrix
df_pred=pd.DataFrame({'actual':test_ydigit,'predicted':prediction})
df_pred.head()

df_pred['predstatus']=df_pred['actual']==df_pred['predicted']
df_pred[df_pred['predstatus']==True].shape[0]/df_pred.shape[0]*100

df_pred.head()
## creating the dataframe

adaboostdataframe=df_pred.iloc[:,0:2].head()
adaboostdataframe.reset_index(drop=True, inplace=True)
adaboostdataframe.columns=['actuallabel','predictedbyadaboost']
adaboostdataframe.head()
## true positive 
tp=df_pred[(df_pred['predicted']==1)&(df_pred['actual']==1)].shape[0]
tn=df_pred[(df_pred['predicted']==0)&(df_pred['actual']==0)].shape[0]
fp=tn=df_pred[(df_pred['predicted']==1)&(df_pred['actual']==0)].shape[0]
fn=df_pred[(df_pred['predicted']==0)&(df_pred['actual']==1)].shape[0]

print(tp,tn,fp,fn)
acc=(tp+tn)/(tp+tn+fp+fn)
acc
df_pred.head()
#from sklearn.neighbors import KNeighborsClassifier
#knn=KNeighborsClassifier(n_neighbors=5)
#knn.fit(train_xdigit,train_ydigit)
#predictingknn=knn.predict(test_xdigit)
#knn.score(test_xdigit,test_ydigit)
#knn.score(test_xdigit,test_ydigit)
from sklearn.tree import DecisionTreeClassifier
modeldecisiontree=DecisionTreeClassifier()
modeldecisiontree
modeldecisiontree.fit(train_xdigit,train_ydigit)
test_preddecisiontree=modeldecisiontree.predict(test_xdigit)
test_preddecisiontree
test_preddecisiontree=pd.DataFrame(test_preddecisiontree)
test_preddecisiontree.head()
test_ydigit.head()
newdataframe=pd.DataFrame(test_ydigit)
newdataframe.head()
newdataframe.reset_index(drop=True, inplace=True)

test_preddecisiontree.reset_index(drop=True,inplace=True)
decisiondataframe=pd.concat([newdataframe,test_preddecisiontree],axis=1)
decisiondataframe.head()
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(accuracy_score(test_ydigit,test_preddecisiontree ))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_ydigit,test_preddecisiontree)))  


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(accuracy_score(test_ydigit, randomforestprediction))  

from sklearn.ensemble import RandomForestClassifier
modelrandomforest=RandomForestClassifier(random_state=100,n_estimators=100)
modelrandomforest.fit(train_xdigit,train_ydigit)
randomforestprediction=modelrandomforest.predict(test_xdigit)
randomforestprediction
df_pred=pd.DataFrame({'actual':test_ydigit,'predictionbyrandomforest':randomforestprediction})
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(accuracy_score(test_ydigit, randomforestprediction))  

train_xdigit = train1.drop('label',axis =1)
train_ydigit = train1['label']
random_model = RandomForestClassifier()
random_model.fit(train_xdigit,train_ydigit)
random_predict = random_model.predict(testdata)

random_df = pd.DataFrame({'Label':random_predict})
random_df['ImageId'] = testdata.index+1

random_df[['ImageId','Label']].to_csv("Submission_File.csv",index = False)

random_df.head()
