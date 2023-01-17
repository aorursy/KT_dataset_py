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
#importing dataset
data=pd.read_csv(r'/kaggle/input/married-at-first-sight/mafs.csv')
#data exploration
data.info()
data.shape
data.describe()
data.head()
data.nunique()
#Total Missing Values
data.isna().sum()
#Class for output variable
data.Decision.value_counts(normalize=True)
#dividing into categorical and numerical data
numeric1=pd.DataFrame(data.describe())
#flags are incorrecly labelled as numeric
col_change=numeric1.columns
#need to convert these int into flags except for Age and Couple/Season
col_change
col_change=data[['DrPepperSchwartz', 'DrLoganLevkoff','DrJosephCilona', 'ChaplainGregEpstein', 'PastorCalvinRoberson','RachelDeAlto', 'DrJessicaGriffin', 'DrVivianaColes']]
for items in col_change.columns:
    col_change[items]=col_change[items].astype('category')
col_change.dtypes
#handling numerical data (finally)
from sklearn.preprocessing import StandardScaler
numeric=data[['Age']]
#scaling the numerical variables
scaler=StandardScaler()
scaler.fit(numeric)
numeric_scaled=pd.DataFrame(scaler.transform(numeric))

numeric_scaled.columns=["Age"]
#Handling categorical data, not taking in 'Status' as it will result in a data leak
categorical_data=data[['Gender','Occupation','Decision']]
categorical_data.head()
categorical_data['Decision']=categorical_data['Decision'].replace({'Yes':1,'No':0})
#setting the output variable
output=categorical_data['Decision']
categorical_data=categorical_data.drop(['Decision'],axis=1)
output.value_counts()
categorical_data['Occupation'].value_counts(normalize=True).sort_values(ascending=False)
#the Occupation wont affect the decision as it is so evently distributed. Just remove it. Lets only keep gender
categorical_data=categorical_data.drop(['Occupation'],axis=1)
categorical_data=pd.get_dummies(categorical_data,drop_first=True)
categorical_data.head()
#concatenate it all together
final=pd.concat([categorical_data,numeric_scaled,col_change],axis=1)
final.head()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,f1_score,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
X_train,X_test,y_train,y_test=train_test_split(final,output,test_size=0.2,random_state=1)
#model fittin- Logistic Reg
Log=LogisticRegression()
Log.fit(X_train,y_train)
preds_log=Log.predict(X_test)
#confusion matrix
confusion_matrix(y_test,preds_log)

print(classification_report(y_train,Log.predict(X_train)))
#LOL
print(classification_report(y_test,preds_log))
#LOL
roc_auc_score(y_test,preds_log)
#Lets see if the Random Forest works any better
random=RandomForestClassifier()
random.fit(X_train,y_train)
preds_rand=random.predict(X_test)
accuracy_score(y_test,preds_rand)
print(classification_report(y_test,preds_rand))
print(confusion_matrix(y_test,preds_rand))
#Dataset is not good :(