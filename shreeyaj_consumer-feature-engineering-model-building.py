# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

test_df=pd.read_csv("/kaggle/input/consumer/Edureka_Consumer_Complaints_test.csv")
train_df=pd.read_csv("/kaggle/input/consumer/Edureka_Consumer_Complaints_train.csv")
def dataset_details(dataframe):
    #print(dataframe.info(),"\n---------------------")
    print("Shape:",dataframe.shape,"\n---------------------")
    print("Duplicate count is",dataframe.duplicated().sum(),"\n---------------------")
    print("Missing count is",dataframe.isna().sum().sum(),"\n---------------------")
    missing_val_col=pd.DataFrame(dataframe.isna().sum())
    missing_val_col.rename(columns={0:"missing_val_count"},inplace=True)
    print("Out of",dataframe.shape[1],"columns, below are the columns having missing values")
    print(missing_val_col[~(missing_val_col["missing_val_count"]==0)],"\n---------------------")
    nan_rows_df=pd.DataFrame(dataframe.apply(lambda x: sum(x.isna()),axis=1))
    nan_rows_count=nan_rows_df[~(nan_rows_df[0]==0)].count().tolist()[0]
    print("Out of",dataframe.shape[0],"rows,",nan_rows_count,"have missing values","\n---------------------")
    print(dataframe.dtypes)

print("-------------------------Analyzing Train dataset-------------------------")
dataset_details(train_df)
print(train_df["Company public response"].nunique())
print(train_df["Company public response"].value_counts())
train_df["Company public response"]=train_df["Company public response"].replace(to_replace=np.nan,value="Company chooses not to provide a public response")
test_df["Company public response"]=test_df["Company public response"].replace(to_replace=np.nan,value="Company chooses not to provide a public response")

print("------Train-------\n")
print("Missing values in Company public response",train_df["Company public response"].isnull().sum())
print("------Test-------\n")
print("Missing values in Company public response",test_df["Company public response"].isnull().sum())
###### New Column: Days difference on Train

train_df["Date received"]=pd.to_datetime(train_df["Date received"])
train_df["Date sent to company"]=pd.to_datetime(train_df["Date sent to company"])
train_df["Days difference"]=np.abs(train_df["Date sent to company"]-train_df["Date received"])
train_df["Days difference"]=train_df["Days difference"].apply(lambda x: str(x).strip()[0])

###### New Column: Days difference on Test

test_df["Date received"]=pd.to_datetime(test_df["Date received"])
test_df["Date sent to company"]=pd.to_datetime(test_df["Date sent to company"])
test_df["Days difference"]=np.abs(test_df["Date sent to company"]-test_df["Date received"])
test_df["Days difference"]=test_df["Days difference"].apply(lambda x: str(x).strip()[0])
train_df["Days difference"].value_counts()
train_df.describe(include="O")
train_new_df=train_df.drop(["Date received","Sub-product","Sub-issue","State","Consumer complaint narrative","Consumer consent provided?","Date sent to company","Complaint ID","Tags","ZIP code"],axis=1)
test_new_df=test_df.drop(["Date received","Sub-product","Sub-issue","State","Consumer complaint narrative","Consumer consent provided?","Date sent to company","Complaint ID","Tags","ZIP code"],axis=1)
col=train_new_df.drop(["Consumer disputed?","Company","Issue"],axis=1).columns.tolist()

from sklearn.preprocessing import LabelEncoder
lr=LabelEncoder()

##Encoding variable
train_new_df["Consumer disputed?"]=lr.fit_transform(train_new_df["Consumer disputed?"])
train_new_df["Company"]=lr.fit_transform(train_new_df["Company"])
test_new_df["Company"]=lr.fit_transform(test_new_df["Company"])
train_new_df["Issue"]=lr.fit_transform(train_new_df["Issue"])
test_new_df["Issue"]=lr.fit_transform(test_new_df["Issue"])


dummy_train=pd.get_dummies(train_new_df,columns=col)
dummy_test=pd.get_dummies(test_new_df,columns=col)
dummy_train.head()
    

dummy_test.head()
XX=dummy_train.drop("Consumer disputed?",axis=1)
YY=dummy_train["Consumer disputed?"]
from sklearn.model_selection import train_test_split
x_train1,x_test1,y_train1,y_test1=train_test_split(XX,YY,test_size=0.3,random_state=101)

x_train1.head()
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
#from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,f1_score

models=[LogisticRegression(max_iter=2000),RandomForestClassifier(),MultinomialNB(),XGBClassifier()]
for i in models:
    i.fit(x_train1,y_train1)
    pred=i.predict(x_test1)
    print(accuracy_score(y_test1,pred))
    result=f1_score(y_test1,pred, average = "macro")
    print("f1_score of",i,"is : ",result)
    print(confusion_matrix(y_test1,pred))
##KFOLD for accuracy
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kfold=KFold(n_splits=3,random_state=101)
models=[LogisticRegression(max_iter=2000),RandomForestClassifier(),MultinomialNB(),XGBClassifier()]
for i in models:
    print("Accuracy of",i,"is : ",np.round(np.mean(cross_val_score(i,XX,YY,cv=kfold,scoring='accuracy')),3))
from xgboost import XGBClassifier
XGBClassifier().get_params().keys() 
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
grid={"max_depth": range(2,10,1),"n_estimators": range(60,220,40),"learning_rate" : [0.1, 0.01,0.05]}
random=RandomizedSearchCV(estimator=XGBClassifier(), param_distributions=grid, n_jobs=-1, cv=10, scoring='accuracy')
random.fit(x_train1,y_train1)
print(random.best_score_,random.best_params_)
print(random.best_estimator_)
XXX=dummy_train.drop("Consumer disputed?",axis=1)
YYY=dummy_train["Consumer disputed?"]

from xgboost import XGBClassifier
xgb=XGBClassifier()
xgb.fit(XXX,YYY)
predictions=xgb.predict(dummy_test)
resul=pd.Series(predictions)
resul.to_csv("/kaggle/working/model_prediction.csv",index=False)
print("done")