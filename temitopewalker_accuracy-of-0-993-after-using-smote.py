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
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold,cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
data.shape
data.head()
# I will drop the unnamed Feature since it looks empty and also drop the Id also since we dont need it for out prediction
sns.heatmap(data.isnull())
data1=data.drop(['id','Unnamed: 32'], axis=1)
sns.pairplot(data1)
ax=sns.countplot(data1.diagnosis)
import matplotlib.pyplot as plt
plt.figure(figsize=(20,20))
sns.heatmap(data1.corr(),annot=True,cmap='coolwarm')
num_cols = data1.select_dtypes(exclude=['object'])

fig = plt.figure(figsize=(10,15))

for col in range(len(num_cols.columns)):
    fig.add_subplot(5,6,col+1)
    sns.boxplot(x=data1.diagnosis,y=num_cols.iloc[:,col])
    plt.xlabel(num_cols.columns[col])

plt.tight_layout()
encording=LabelEncoder()
data1['diagnosis'] = encording.fit_transform(data1['diagnosis'])
y= data1['diagnosis']
X=data1.drop(['diagnosis'],axis=1)
scaler = StandardScaler()
X_scaled=scaler.fit_transform(X)
X_scaled=pd.DataFrame(X_scaled)
X.head()
import pickle
def fit_model(X,y,algo_name,algorithm,gridSeachParams,cv):
    np.random.seed(10)
    X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2)
    
    grid = GridSearchCV(estimator=algorithm,
                        param_grid=gridSeachParams,
                       cv=cv,scoring='accuracy',
                        verbose=1,n_jobs=1)
    grid_result = grid.fit(X_train, y_train)
    best_params = grid_result.best_params_
    pred=grid_result.predict(X_test)
    cm = confusion_matrix(y_test,pred)
    ##pickle.dump(grid_result,open(algo_name, 'wb'))
    print('Best Params :', best_params)
    print('Classification Report :' , classification_report(y_test,pred))
    print('Accuracy Score :' + str(accuracy_score(y_test,pred)))
    plt.figure()
    plt.title(algo_name)
    sns.heatmap(cm,annot=True,label=algo_name,fmt='d')
params = {
    'C': [0.1,1,100,1000],
    'gamma':[0.0001, 0.001, 0.01, 1, 3, 5]
}
fit_model(X,y,'SVC',SVC(),params,cv=10)
params = {
            'n_estimators' : (100,500,1000,2000)
}
fit_model(X,y,'Random Forest',RandomForestClassifier(),params,cv=5)
params = {
            'n_estimators' : (100,500,1000,2000)
}
fit_model(X,y,'XGB Classifier',XGBClassifier(),params,cv=5)
#Balancing the model by over smapling it 
from imblearn.over_sampling import SMOTE
sm=SMOTE(random_state=2)
x_bal , y_bal =sm.fit_resample(X_scaled,y)
sns.countplot(y_bal)
y_bal.shape
def fit_model(X,y,algo_name,algorithm,gridSeachParams,cv):
    np.random.seed(10)
    X_train, X_test, y_train, y_test = train_test_split(
    x_bal, y_bal, test_size=0.2)
    
    grid = GridSearchCV(estimator=algorithm,
                        param_grid=gridSeachParams,
                       cv=cv,scoring='accuracy',
                        verbose=1,n_jobs=1)
    grid_result = grid.fit(X_train, y_train)
    best_params = grid_result.best_params_
    pred=grid_result.predict(X_test)
    cm = confusion_matrix(y_test,pred)
    ##pickle.dump(grid_result,open(algo_name, 'wb'))
    print('Best Params :', best_params)
    print('Classification Report :' , classification_report(y_test,pred))
    print('Accuracy Score :' + str(accuracy_score(y_test,pred)))
    plt.figure()
    plt.title(algo_name)
    sns.heatmap(cm,annot=True,label=algo_name,fmt='d')
params = {
            'n_estimators' : (100,500,1000,2000)
}
fit_model(x_bal,y_bal,'Random Forest',RandomForestClassifier(),params,cv=5)
params = {
            'n_estimators' : (100,500,1000,2000)
}
fit_model(x_bal,y_bal,'XGB Classifier',XGBClassifier(),params,cv=5)
