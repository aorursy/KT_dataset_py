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

%matplotlib inline

from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier

from matplotlib import pyplot

import seaborn as sn

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier

from sklearn import metrics

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, auc, roc_curve, recall_score, classification_report 

import sklearn

from sklearn import preprocessing

path = '../input/health-insurance-cross-sell-prediction/'

train_df = pd.read_csv(path + "train.csv")

test_df = pd.read_csv(path + "test.csv")
print("The Shape of Train Data Set :",train_df.shape)

print("The Shape of Test Data Set :", test_df.shape)
print("Columns of Train Data Set: \n",train_df.columns)

print("-------------------------")

print("-------------------------")

print("Columns of Test Data Set: \n",test_df.columns)
train_df.head(4)
train_df.info()
sn.countplot(x="Gender", data = train_df)
sn.countplot(x="Driving_License", data = train_df)
sn.countplot(x="Previously_Insured", data = train_df)
sn.countplot(x="Vehicle_Age", data = train_df)
sn.countplot(x="Vehicle_Damage", data = train_df)
sn.countplot(x="Response", data = train_df)
sn.distplot(train_df.Age)
sn.distplot(train_df.Annual_Premium)
Numerical_Features = ['Age', 'Driving_License', 'Vehicle_Age','Annual_Premium','Vintage']

train_df[Numerical_Features].describe()
fig = plt.figure(figsize =(10, 7)) 

plt.boxplot(train_df.Age) 

plt.show() 
fig = plt.figure(figsize =(8, 7)) 

plt.boxplot(train_df.Annual_Premium) 

plt.show() 
fig = plt.figure(figsize =(8, 7)) 

plt.boxplot(train_df.Vintage) 

plt.show() 
train_df[Numerical_Features].corr()
X= train_df[['Gender', 'Age', 'Driving_License', 'Region_Code','Previously_Insured', 'Vehicle_Age','Vehicle_Damage', 'Annual_Premium','Policy_Sales_Channel', 'Vintage']]

y= train_df['Response']
#there is some categorical features that need to be encoded

X_features = list(X.columns)

encoded_Data_df= pd.get_dummies(X[X_features],drop_first=True)

X=encoded_Data_df
radm_clf = RandomForestClassifier( max_depth=15,n_estimators=20,max_features = 'auto')

## Fitting the model with the training set

radm_clf.fit(X,y )
feature_rank = pd.DataFrame( { 'feature': X.columns,'importance': radm_clf.feature_importances_ } )

## Sorting the features based on their importances with mosti important feature at top.

feature_rank = feature_rank.sort_values('importance', ascending =False)

plt.figure(figsize=(8, 6))

sn.barplot( y = 'feature', x = 'importance', data = feature_rank )
sn.countplot("Response", data=train_df)
from imblearn.over_sampling import SMOTE

oversample = SMOTE()

X, y = oversample.fit_resample(X, y)
print("Shape of X after over-Sampling:", X.shape)

print("Shape of y after over-Sampling:", y.shape)
sn.countplot(y)
#Dropping feature "Driving_License"

X= X.drop(['Driving_License'],axis=1)
## Initializing the StandardScaler

X_scaler = StandardScaler()

## Standardize all the feature columns

X_scaled = X_scaler.fit_transform(X)

X=X_scaled
#Splitting the dataset into the traing set and test set

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=43)
gboost_clf = GradientBoostingClassifier()
gboost_clf.fit(X_train,y_train)
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve
def plot_ROC(fpr, tpr, m_name):

    roc_auc = sklearn.metrics.auc(fpr, tpr)

    plt.figure(figsize=(10,8))

    lw = 2

    plt.plot(fpr, tpr, color='blue',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc, alpha=0.5)    

    plt.plot([0, 1], [0, 1], color='red', lw=lw, linestyle='--', alpha=0.5)    

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xticks(fontsize=16)

    plt.yticks(fontsize=16)

    plt.grid(True)

    plt.xlabel('False Positive Rate', fontsize=16)

    plt.ylabel('True Positive Rate', fontsize=16)

    plt.title('Receiver operating characteristic for %s'%m_name, fontsize=20)

    plt.legend(loc="lower right", fontsize=16)

    plt.show()
Gboost_preds = gboost_clf.predict_proba(X_test)

Gboost_class = gboost_clf.predict(X_test)

Gboost_score = roc_auc_score(y_test, Gboost_preds[:,1], average = 'weighted')

(fpr, tpr, thresholds) = roc_curve(y_test, Gboost_preds[:,1])

plot_ROC(fpr, tpr, 'Gboost')
#there is some categorical features that need to be encoded

Features = list(test_df.columns)

encoded_Data_df= pd.get_dummies(test_df[Features],drop_first=True)

test_df=encoded_Data_df
Test = test_df.drop(["id","Driving_License"],axis=1)

Test.head(4)
pred = gboost_clf.predict(Test)
submit = pd.DataFrame(index=test_df.index)

submit["id"] = test_df.id

submit["Response"] = pred

submit.set_index('id').reset_index(inplace=True)

submit.head()
submit.to_csv("Submission.csv")