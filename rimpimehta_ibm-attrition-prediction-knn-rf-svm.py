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
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

%matplotlib inline 

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import  metrics 
from sklearn.metrics import confusion_matrix, accuracy_score, make_scorer


#loading data

data=pd.read_csv("../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")
data.head()
data.info()
data.describe()
import pandas_profiling as pp
pp.ProfileReport(data)
# taking categorical features only
df_cat=data.loc[:,data.dtypes==np.object]
#Droping 'Over18'
#Attrition is target variable so will drop it too
df_cat=df_cat.drop(['Attrition','Over18'], axis=1)
df_cat
#taking numerical fetures only
df_num=data.loc[:,data.dtypes==np.int64]
df_num
#droping numeric features which are constant
df_num=df_num.drop(['EmployeeCount','StandardHours'], axis=1)
df_num
corr = df_num.corr()
plt.figure(figsize=(10, 10))
ax = sns.heatmap(
    corr, 
    vmin=-0, vmax=1,
    cmap=sns.diverging_palette(40, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
sns.countplot(x="BusinessTravel",data=df_cat)
sns.countplot(x="Department",data=df_cat)

sns.countplot(x="EducationField",data=df_cat)
sns.countplot(x="Gender",data=df_cat)
sns.countplot(x="JobRole",data=df_cat)
sns.countplot(x="MaritalStatus",data=df_cat)
sns.countplot(x="OverTime",data=df_cat)
#conveting categorical variable into numeric 
df_cat = pd.get_dummies(df_cat)
df_cat.head()
#transforming target/Attrition variable
target_map = {'Yes':1, 'No':0}
# Use the pandas apply method to numerically encode our attrition target variable
Y= data["Attrition"].apply(lambda x: target_map[x])
Y
#merging categorical and numerical data
new_df= pd.concat([df_cat, df_num], axis=1)
new_df
corr =new_df.corr()
plt.figure(figsize=(10, 10))
ax = sns.heatmap(
    corr, 
    vmin=-0, vmax=1,
    cmap=sns.diverging_palette(40, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
#analysing target variable
sns.countplot(x="Attrition",data=data)
data.Attrition.value_counts()
from sklearn.preprocessing import RobustScaler

X=new_df
scaler=RobustScaler()
scaled_df=scaler.fit_transform(X)
X=scaled_df

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.3,random_state=42)
from sklearn.neighbors import KNeighborsClassifier
model= KNeighborsClassifier(n_neighbors=3,n_jobs=-1)
model.fit(X_train,Y_train)
Y_predict=model.predict(X_test)
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,Y_predict))
print(confusion_matrix(Y_test,Y_predict))


probs=model.predict_proba(X_test)
preds = probs[:,1]

auc = roc_auc_score(Y_test, preds)
print('AUC: %.2f' % auc)
fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
plot_roc_curve(fpr, tpr)
from imblearn.over_sampling import SMOTE
smote=SMOTE()
from imblearn.over_sampling import SMOTE
smote=SMOTE()
X_train_smote,Y_train_smote=smote.fit_sample(X_train,Y_train)

from collections import Counter
print("Before smote",Counter(Y_train))
print("after smote",Counter(Y_train_smote))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

model = KNeighborsClassifier(n_jobs=-1)
params = {'n_neighbors':[3,5,7,9],
          'leaf_size':[1,2,3,5],
          'weights':['uniform', 'distance'],
          'algorithm':['auto', 'ball_tree','kd_tree','brute'],
          'n_jobs':[-1]}
#Making models with hyper parameters sets
model1 = GridSearchCV(model, param_grid=params, n_jobs=1)
#Learning
model1.fit(X_train_smote,Y_train_smote)
#The best hyper parameters set
print("Best Hyper Parameters:\n",model1.best_params_)
#Prediction
prediction=model1.predict(X_test)
#importing the metrics module
from sklearn import metrics
#evaluation(Accuracy)
print("Accuracy:",metrics.accuracy_score(prediction,Y_test))
#evaluation(Confusion Metrix)
print("Confusion Metrix:\n",metrics.confusion_matrix(prediction,Y_test))
#evaluation(ROC_AUC)
probs=model1.predict_proba(X_test)
preds = probs[:,1]
auc = roc_auc_score(Y_test, preds)
print('ROC_AUC: %.2f' % auc)

fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
plot_roc_curve(fpr, tpr)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer

# Choose the type of classifier. 
clf = RandomForestClassifier()

# Choose some parameter combinations to try
parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train_smote,Y_train_smote)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
clf.fit(X_train_smote,Y_train_smote)

predictions = clf.predict(X_test)
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test,predictions))
#evaluation(ROC_AUC)
probs=model1.predict_proba(X_test)
preds = probs[:,1]
auc = roc_auc_score(Y_test, preds)
print('ROC_AUC: %.2f' % auc)

fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
plot_roc_curve(fpr, tpr)
from sklearn.model_selection import GridSearchCV
from sklearn import svm
#making the instance
model=svm.SVC(probability=True)
#Hyper Parameters Set
params = {'C': [6,7,8,9,10,11,12], 
          'kernel': ['linear','rbf']}
#Making models with hyper parameters sets
model1 = GridSearchCV(model, param_grid=params, n_jobs=-1)
#Learning
model1.fit(X_train_smote,Y_train_smote)
#The best hyper parameters set
print("Best Hyper Parameters:\n",model1.best_params_)
#Prediction
prediction=model1.predict(X_test)
#importing the metrics module
from sklearn import metrics
#evaluation(Accuracy)
print("Accuracy:",metrics.accuracy_score(prediction,Y_test))
#evaluation(Confusion Metrix)
print("Confusion Metrix:\n",metrics.confusion_matrix(prediction,Y_test))
#evaluation(ROC_AUC)
probs=model1.predict_proba(X_test)
preds = probs[:,1]
auc = roc_auc_score(Y_test, preds)
print('ROC_AUC: %.2f' % auc)

fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
plot_roc_curve(fpr, tpr)
