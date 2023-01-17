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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd

CTF_Data = pd.read_csv("../input/Change_the_Future_data.csv")

CTF_Data.head()
CTF_Data.isnull().sum()
CTF_Data.info()
CTF_Data.count()
CTF_Data.columns
((CTF_Data.groupby('Are_you_interested_in_future_to').Are_you_interested_in_future_to.count())/(CTF_Data.Are_you_interested_in_future_to.count()))*100
sns.countplot(x = 'Age',hue = 'Are_you_interested_in_future_to',data = CTF_Data)
sns.countplot(x = 'Gender',hue = 'Are_you_interested_in_future_to',data = CTF_Data)
sns.countplot(x = 'Area_Locality',hue = 'Are_you_interested_in_future_to',data = CTF_Data)
sns.countplot(x = 'Education',hue = 'Are_you_interested_in_future_to',data = CTF_Data)
sns.countplot(x = 'Employment_Status',hue = 'Are_you_interested_in_future_to',data = CTF_Data)
sns.countplot(x = 'Type_of_House',hue = 'Are_you_interested_in_future_to',data = CTF_Data)
sns.countplot(x = 'Approximate_Annual_Family_Incom',hue = 'Are_you_interested_in_future_to',data = CTF_Data)
CTF_Data_A   = (CTF_Data.groupby(['Are_you_interested_in_future_to','Age']).Are_you_interested_in_future_to.count())/(CTF_Data.groupby('Are_you_interested_in_future_to').Are_you_interested_in_future_to.count())*100

CTF_Data_G   = (CTF_Data.groupby(['Are_you_interested_in_future_to','Gender']).Are_you_interested_in_future_to.count())/(CTF_Data.groupby('Are_you_interested_in_future_to').Are_you_interested_in_future_to.count())*100

CTF_Data_AL  = (CTF_Data.groupby(['Are_you_interested_in_future_to','Area_Locality']).Are_you_interested_in_future_to.count())/(CTF_Data.groupby('Are_you_interested_in_future_to').Are_you_interested_in_future_to.count())*100

CTF_Data_E   = (CTF_Data.groupby(['Are_you_interested_in_future_to','Education']).Are_you_interested_in_future_to.count())/(CTF_Data.groupby('Are_you_interested_in_future_to').Are_you_interested_in_future_to.count())*100

CTF_Data_ES  = (CTF_Data.groupby(['Are_you_interested_in_future_to','Employment_Status']).Are_you_interested_in_future_to.count())/(CTF_Data.groupby('Are_you_interested_in_future_to').Are_you_interested_in_future_to.count())*100

CTF_Data_AI  = (CTF_Data.groupby(['Are_you_interested_in_future_to','Approximate_Annual_Family_Incom']).Are_you_interested_in_future_to.count())/(CTF_Data.groupby('Are_you_interested_in_future_to').Are_you_interested_in_future_to.count())*100

CTF_Data_TOH = (CTF_Data.groupby(['Are_you_interested_in_future_to','Type_of_House']).Are_you_interested_in_future_to.count())/(CTF_Data.groupby('Are_you_interested_in_future_to').Are_you_interested_in_future_to.count())*100

CTF_Data_AG = (CTF_Data.groupby(['Are_you_interested_in_future_to','Age','Gender']).Are_you_interested_in_future_to.count())/(CTF_Data.groupby('Are_you_interested_in_future_to').Are_you_interested_in_future_to.count())*100

CTF_Data_ALA = (CTF_Data.groupby(['Are_you_interested_in_future_to','Area_Locality','Age']).Are_you_interested_in_future_to.count())/(CTF_Data.groupby('Are_you_interested_in_future_to').Are_you_interested_in_future_to.count())*100

CTF_Data_ALE = (CTF_Data.groupby(['Are_you_interested_in_future_to','Area_Locality','Education']).Are_you_interested_in_future_to.count())/(CTF_Data.groupby('Are_you_interested_in_future_to').Are_you_interested_in_future_to.count())*100

CTF_Data_EG = (CTF_Data.groupby(['Are_you_interested_in_future_to','Education','Gender']).Are_you_interested_in_future_to.count())/(CTF_Data.groupby('Are_you_interested_in_future_to').Are_you_interested_in_future_to.count())*100

CTF_Data_ESAL = (CTF_Data.groupby(['Are_you_interested_in_future_to','Employment_Status','Area_Locality']).Are_you_interested_in_future_to.count())/(CTF_Data.groupby('Are_you_interested_in_future_to').Are_you_interested_in_future_to.count())*100

CTF_Data_AITOH = (CTF_Data.groupby(['Are_you_interested_in_future_to','Approximate_Annual_Family_Incom','Type_of_House']).Are_you_interested_in_future_to.count())/(CTF_Data.groupby('Are_you_interested_in_future_to').Are_you_interested_in_future_to.count())*100

CTF_Data_ESAI = (CTF_Data.groupby(['Are_you_interested_in_future_to','Employment_Status','Approximate_Annual_Family_Incom']).Are_you_interested_in_future_to.count())/(CTF_Data.groupby('Are_you_interested_in_future_to').Are_you_interested_in_future_to.count())*100
CTF_Data_A
CTF_Data_G
CTF_Data_AL
CTF_Data_E
CTF_Data_ES
CTF_Data_AI
CTF_Data_TOH
CTF_Data_AG
CTF_Data_ALA
CTF_Data_ALE
CTF_Data_EG
CTF_Data_v1 = CTF_Data
CTF_Data_v1.columns
Age_dummy = pd.get_dummies(CTF_Data_v1['Age'])

Gender_dummy = pd.get_dummies(CTF_Data_v1['Gender'])

AL_dummy = pd.get_dummies(CTF_Data_v1['Area_Locality'])

Education_dummy = pd.get_dummies(CTF_Data_v1['Education'])

ES_dummy = pd.get_dummies(CTF_Data_v1['Employment_Status'])

AI_dummy = pd.get_dummies(CTF_Data_v1['Approximate_Annual_Family_Incom'])

TOF_dummy = pd.get_dummies(CTF_Data_v1['Type_of_House'])

CTF_Data_v1 = pd.concat([CTF_Data_v1,Age_dummy],axis = 1)

CTF_Data_v1 = pd.concat([CTF_Data_v1,Gender_dummy],axis = 1)

CTF_Data_v1 = pd.concat([CTF_Data_v1,AL_dummy],axis = 1)

CTF_Data_v1 = pd.concat([CTF_Data_v1,Education_dummy],axis = 1)

CTF_Data_v1 = pd.concat([CTF_Data_v1,ES_dummy],axis = 1)

CTF_Data_v1 = pd.concat([CTF_Data_v1,AI_dummy],axis = 1)

CTF_Data_v1 = pd.concat([CTF_Data_v1,TOF_dummy],axis = 1)

CTF_Data_v1['y'] =  np.where(CTF_Data_v1.Are_you_interested_in_future_to == 'Yes',1,0)

CTF_Data_v1 = CTF_Data_v1.drop(columns = ['Are_you_interested_in_future_to','Age','Gender','Area_Locality','Education','Employment_Status','Approximate_Annual_Family_Incom','Type_of_House'],axis = 1)
CTF_Data_v1.head()
X_Input = CTF_Data_v1

X_Input = X_Input.drop(columns = 'y',axis = 1)

X_Input.columns
corr = X_Input.corr(method = 'pearson')

sns.heatmap(corr,annot = True)

plt.show
rows,cols = X_Input.shape

flds = list(X_Input.columns)

corr_x = X_Input.corr().values



for i in range(cols):

    for j in range(i+1, cols):

        if(corr_x[i,j] > 0.9):

            print(flds[i],flds[j],corr_x[i,j])

        else:

            print('No high correlation has seen in the variables')

            
corr_x = CTF_Data_v1.corr()

sns.heatmap(corr_x,annot = True)
corr_target = abs(corr_x['y'])

corr_target
y = CTF_Data_v1['y']
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X_Input, y, test_size=0.3, random_state=0)

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

import statsmodels.api as sm

logit_model=sm.Logit(y_train,X_train)

result=logit_model.fit()

print(result.summary2())
X_Input_v1 = X_Input[['Number_of_family_members','Mumbai_South','Mumbai_Suburban_District','Othert','Thane_District','Other','Owned','Rental']]

X_Input_v1.head()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X_Input_v1, y, test_size=0.3, random_state=0)

from sklearn.metrics import roc_auc_score,average_precision_score,auc,roc_curve,precision_recall_curve

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

import statsmodels.api as sm

logit_model=sm.Logit(y_train,X_train)

result=logit_model.fit()

print(result.summary2())
y_pred_test_v1 = logreg.predict_proba(X_test)[:,1]

y_pred_train_v1 = logreg.predict_proba(X_train)[:,1]
fpr , tpr ,thresold = roc_curve(y_test,y_pred_test_v1)

roc_auc = auc(fpr,tpr)

plt.plot(fpr,tpr,label = 'ROC curve (area = %0.2f)'% roc_auc)

plt.xlabel("False Positve rate")

plt.ylabel("True Positive rate")

plt.legend(loc = 'lower right')

print("ROC Curve")
y_pred_test_v1 = np.where(y_pred_test_v1 > 0.7,1,0)

y_pred_train_v1 = np.where(y_pred_train_v1 > 0.7,1,0)
matrix = confusion_matrix(y_test,y_pred_test_v1)

sns.heatmap(matrix ,annot = True,cbar = True)
matrix = confusion_matrix(y_train,y_pred_train_v1)

sns.heatmap(matrix ,annot = True,cbar = True)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_test_v1))