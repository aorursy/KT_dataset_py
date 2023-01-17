# data visualization libraries:
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# to ignore warnings:
import warnings
warnings.filterwarnings('ignore')

# to display all columns:
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score,roc_curve
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier 
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import numpy as np, pandas as pd, os, gc
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
import lightgbm as lgb
import gc
from time import time
import datetime
from tqdm import tqdm_notebook
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import roc_auc_score
warnings.simplefilter('ignore')
sns.set()
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np,gc # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns', 500)
import numpy as np
import pandas as pd 
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score,roc_curve
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from warnings import filterwarnings
filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
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
data = pd.read_csv("/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
data.head(10)
data.info()
data.nunique()
data.select_dtypes(include='object').nunique()
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data.select_dtypes(include='object').nunique()
fig, ax = plt.subplots(7, 3, figsize=(16,16))
sns.countplot('gender', data=data,hue='Churn', ax=ax[0][0])
sns.countplot('Partner', data=data,hue='Churn', ax=ax[0][1])
sns.countplot('Dependents', data=data,hue='Churn', ax=ax[0][2])
sns.countplot('PhoneService', data=data,hue='Churn', ax=ax[1][0])
sns.countplot('MultipleLines', data=data,hue='Churn', ax=ax[1][1])
sns.countplot('InternetService', data=data,hue='Churn', ax=ax[1][2])
sns.countplot('OnlineSecurity', data=data,hue='Churn', ax=ax[2][0])
sns.countplot('OnlineBackup', data=data,hue='Churn', ax=ax[2][1])
sns.countplot('DeviceProtection', data=data,hue='Churn', ax=ax[2][2])
sns.countplot('TechSupport', data=data,hue='Churn', ax=ax[3][0])
sns.countplot('StreamingTV', data=data,hue='Churn', ax=ax[3][1])
sns.countplot('StreamingMovies', data=data,hue='Churn', ax=ax[3][2])
sns.countplot('Contract', data=data,hue='Churn', ax=ax[4][0])
sns.countplot('PaperlessBilling', data=data,hue='Churn', ax=ax[4][1])
sns.countplot('PaymentMethod', data=data,hue='Churn', ax=ax[4][2])
sns.countplot('SeniorCitizen', data=data,hue='Churn', ax=ax[5][0])
sns.distplot(data['tenure'], kde=True, ax=ax[5][1])
sns.distplot(data['MonthlyCharges'], kde=True, ax=ax[5][2])
sns.distplot(data['TotalCharges'], kde=True, ax=ax[6][0])
plt.tight_layout()
for i in data.select_dtypes(include='object'):
    data[i] = LabelEncoder().fit_transform(data[i])
data.head(10)
data['tenure'] = (data['tenure']-min(data['tenure']))/(max(data['tenure'])-min(data['tenure']))
data['MonthlyCharges'] = (data['MonthlyCharges']-min(data['MonthlyCharges']))/(max(data['MonthlyCharges'])-min(data['MonthlyCharges']))
data['TotalCharges'] = (data['TotalCharges']-min(data['TotalCharges']))/(max(data['TotalCharges'])-min(data['TotalCharges']))
data.describe()
f, ax = plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(), annot=True, linewidth=.5, fmt='.1f', ax=ax)
def getCorrelatedFeature(corrdata, threshold):
    feature = []
    value = []
    
    for i, index in enumerate(corrdata.index):
        if abs(corrdata[index]) > threshold:
            feature.append(index)
            value.append(corrdata[index])
            
    df = pd.DataFrame(data=value, index=feature, columns=['corr value'])
    
    return df
data = data.dropna()
corrmat = data.corr()
threshold = 0.15
corr_df = getCorrelatedFeature(corrmat['Churn'], threshold)
corr_df.index.values
train, test = train_test_split(data[['SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'Contract', 'PaperlessBilling', 'MonthlyCharges',
       'TotalCharges', 'Churn']], test_size=0.20, random_state=4)

y_test=test["Churn"]
X_test=test.drop(["Churn"], axis=1)

y=train['Churn']
X=train.drop(['Churn'], axis=1).astype('float64')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=4)
model=GradientBoostingClassifier().fit(X_train, y_train)
y_pred=model.predict(X_test)
accuracy_score(y_test, y_pred)
model
GBM_params = {"loss":[ 'deviance', 'exponential'],
             "min_samples_split":[2,3,5, 10],
             "n_estimators":[100,200,500,1000],
             "min_samples_leaf":[1,2,5, 10],
             }
GBM_model = GridSearchCV(model, GBM_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)
GBM_model.best_params_
mdl_tuned = GradientBoostingClassifier(min_samples_split= 2,
                                        min_samples_leaf= 2,
                                       learning_rate= 0.1,
                                       max_depth= 3,
                                       n_estimators= 100,
                                       subsample= 1).fit(X_train, y_train)
y_pred=mdl_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
Importance = pd.DataFrame({'Importance':mdl_tuned.feature_importances_*100},
                         index = X_train.columns)

Importance.sort_values(by = 'Importance',axis = 0,ascending = True).plot(kind = 'barh',color = '#d62728',figsize=(10,6), edgecolor='white')
plt.xlabel('Variable Importance')
plt.gca().legend_ = None
preds = mdl_tuned.predict(X_test)
real_y_test=pd.DataFrame(y_test)
real_y_test["Predictions"]=preds
real_y_test.head()
accuracy_score(real_y_test.loc[:,"Churn"],real_y_test.loc[:,"Predictions"] )
