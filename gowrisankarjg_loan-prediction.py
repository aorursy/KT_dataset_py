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

import numpy as np                     # For mathematical calculations 

import seaborn as sns                  # For data visualization 

import matplotlib.pyplot as plt        # For plotting graphs 

%matplotlib inline 



from sklearn.preprocessing import LabelEncoder

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.feature_selection import RFE, RFECV

from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import NearestNeighbors

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, ExtraTreesClassifier,GradientBoostingClassifier

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

from sklearn.model_selection import cross_val_score, KFold

from imblearn.over_sampling import SMOTE

from sklearn.utils import resample

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV



import warnings                        # To ignore any warnings warnings.filterwarnings("ignore")

warnings.filterwarnings("ignore")
data_train = pd.read_csv('/kaggle/input/train.csv')

train=data_train.copy()

data_test = pd.read_csv('/kaggle/input/test.csv')
data_train.head()
print(data_train.isna().sum())

print("Total missing values:",data_train.isna().sum().sum())
pd.DataFrame({"Skewness": data_train.skew(), "Kurtosis": data_train.kurt()})
print(data_train['Gender'].value_counts())

print(data_test['Gender'].value_counts())
def get_combined_data(train,test):

    targets = train.Loan_Status

    train.drop('Loan_Status', 1, inplace=True)

    combined = train.append(test)

    combined.reset_index(inplace=True)

    combined.drop(['index', 'Loan_ID'], inplace=True, axis=1)

    return combined

df=get_combined_data(data_train,data_test)
df.head()
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)

df['Self_Employed'].fillna('No', inplace=True)

df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(),inplace=True)

df['Credit_History'].fillna(1,inplace=True)

df['Married'].fillna('Yes',inplace=True)

df['Gender'].fillna('Male',inplace=True)

df['Dependents'].fillna('0',inplace=True)

df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']

df['TotalIncome_log'] = np.log(df['TotalIncome'])
df.isna().sum().sum()
numerical_features = data_train.select_dtypes(include=np.number)

categorical_features = data_train.select_dtypes(include=np.object)

print("numeric_features: ", numerical_features.columns)

print("categorical_features: ", categorical_features.columns)
var_mod=categorical_features.columns.values[1:]

le=LabelEncoder()

for i in var_mod:

        df[i]=le.fit_transform(df[i])
def recover_train_test_target():

    global df, train

    targets = train['Loan_Status'].map({'Y':1,'N':0})

    train = df.head(614)

    test = df.iloc[614:]

    return train, test, targets



train, test, targets = recover_train_test_target()
parameters = {'bootstrap': False,

              'min_samples_leaf': 3,

              'n_estimators': 100,

              'min_samples_split': 10,

              'max_features': 'sqrt',

              'max_depth': 6}



model = RandomForestClassifier(**parameters)

model.fit(train, targets)
def compute_score(clf, X, y, scoring='accuracy'):

    xval = cross_val_score(clf, X, y, cv = 10, scoring=scoring)

    return np.mean(xval)



compute_score(model, train, targets)
X_train, X_test, y_train, y_test = train_test_split(train, targets, random_state=42, test_size=0.2)



model = LogisticRegression()



grid = {'C':[0.001,0.01,0.1,1,5,10],

       'penalty':['l1','l2'],

        'class_weight':['balanced']}

# cv = KFold(n_splits = 10, shuffle=True, random_state = 7)

clf = GridSearchCV(model, grid, n_jobs=8, cv=None,scoring='f1_macro')

clf.fit(X_train, y_train)
clf.best_score_, clf.best_params_
model = LogisticRegression(C= 5, class_weight= 'balanced', penalty= 'l2')

model.fit(X_train, y_train)



prediction = model.predict_proba(X_test) # predicting on the validation set

prediction_int = prediction[:,1] >= 0.56 # if prediction is greater than or equal to 0.3 than 1 else 0

y_pred = prediction_int.astype(np.int)



print("f1_score:",f1_score(y_test, y_pred)) # calculating f1 score

print("Accuracy on train data:",model.score(X_train,y_train))

print("Accuracy on test data:",model.score(X_test,y_test))
pd.crosstab(y_pred,y_test)
print(classification_report(y_test,y_pred))