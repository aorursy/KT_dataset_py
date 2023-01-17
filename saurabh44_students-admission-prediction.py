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
EnrolData= pd.read_csv('../input/EnrolData.csv')
EnrolData.head()
# removing column which are not relevant these data are just for referencing purpose hence should be removed
noncol = ['Academic Period','Unique ID','Application Date','Enrolled']
EnrolData = EnrolData.drop(noncol,axis=1)
EnrolData.hist()
plt.gcf().set_size_inches(15, 15)
sns.set(color_codes=True)
len(np.unique(EnrolData['Common Application- Paper']))==1
len(np.unique(EnrolData['Pre-Dental']))==1
len(np.unique(EnrolData['Pre-Law']))==1
len(np.unique(EnrolData['Pre-Veterinarian']))==1
# Only Pre-Veterinarian has single unique value 0 so dropping the value
EnrolData = EnrolData.drop('Pre-Veterinarian',axis=1)
EnrolData.isna().sum()
plt.figure(figsize=(21,7), dpi=300)
sns.countplot(x='State Province',data=EnrolData)
plt.savefig('State Province.png')
plt.show()
# Since the data is categorical and it is just 1.2 % missing data we can impute it with Mode
EnrolData['State Province'].fillna(EnrolData['State Province'].mode()[0],inplace=True)
sns.countplot(x='Admissions Population Description', data = EnrolData)
plt.show()
# Since the data is categorical and it is just 3.2 % missing data we can impute it with Mode
EnrolData['Admissions Population Description'].fillna(EnrolData['Admissions Population Description'].mode()[0],inplace=True)
# For institutional aid offered we are assuing zero aid offer for missing value
EnrolData['Institutional Aid Offered'].fillna(0,inplace=True)
EnrolData= EnrolData.dropna(axis=0)
EnrolData['Sat Verbal'].fillna(EnrolData['Sat Verbal'].mean(),inplace=True)
EnrolData['Sat Total Score'].fillna(EnrolData['Sat Total Score'].mean(),inplace=True)
EnrolData['Sat Mathematics'].fillna(EnrolData['Sat Mathematics'].mean(),inplace=True)
# looking at unique values of categorical variables
print(EnrolData['State Province'].unique())
print(EnrolData['Student Population'].unique())
print(EnrolData['Admissions Population Description'].unique())
print(EnrolData['Residency Description'].unique())
print(EnrolData['Admissions Athlete'].unique())
# Applying labelEncoder in categorical data
from sklearn.preprocessing import LabelEncoder

column_names_for_onehot = ["State Province","Student Population","Admissions Population Description","Residency Description","College Description","Major Description","Gender","Admitted"]
encoded_EnrolData = pd.get_dummies(EnrolData, columns=column_names_for_onehot, drop_first=True)

# Applying Decision Tree and Random forest method
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, accuracy_score
# Ridge Regression alpha parameter
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error 

X_value = encoded_EnrolData.iloc[:,:-1].values
Y_Value = encoded_EnrolData.iloc [:,-1].values

x_train, x_test, y_train, y_test = train_test_split(X_value, Y_Value, test_size=0.2, random_state=0)



sc = StandardScaler()  
X_train_scaled = sc.fit_transform(x_train)
X_test_scaled = sc.transform(x_test)

# number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 20, stop = 300, num = 10)]
# number of features at every split
max_features = ['auto', 'sqrt']
criterion=['entropy','gini']
bootstrap = [True,False]
# max depth
max_depth = [int(x) for x in np.linspace(2, 50, num = 11)]
max_depth.append(None)
# create random grid
random_grid = {
 'n_estimators': n_estimators,
 'max_features': max_features,
 'max_depth': max_depth,
 'criterion': criterion,
 'bootstrap': bootstrap 
 }
# Random search of parameters
rfc_random = RandomizedSearchCV(RandomForestClassifier(), param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the model
rfc_random.fit(X_train_scaled, y_train)
# print results
print(rfc_random.best_params_)
ridge=RidgeClassifier(max_iter=10e5)
parameters={'alpha': [1e-15, 1e-10, 1e-7,1e-3, 1e-2,1e-1,1,5,10,20,30,40,50,70,100,200]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error', cv=5)
ridge_regressor.fit(X_train_scaled, y_train)
print(ridge_regressor.best_params_)

names_of_classifier = ["Random Forest","Decision Tree","Linear SVM","K-Nearest Neighbors",  "Ridge"]

classifier = [
    RandomForestClassifier(n_estimators= 268, criterion="entropy", bootstrap = False, max_depth=50, max_features = 'auto',class_weight="balanced"),
    DecisionTreeClassifier(max_depth=9),
    SVC(kernel="linear", C=0.03),
    KNeighborsClassifier(6, n_jobs=-1),
    RidgeClassifier(alpha=20)]

for name, classifier in zip(names_of_classifier, classifier):
    classifier.fit(X_train_scaled,y_train)
    
    y_predict=classifier.predict(X_test_scaled)
#     y_Train_predict=classifier.predict(X_train_scaled)
    print("Classifier: ",name)
    print("\nAccuracy for Test Set: ",accuracy_score(y_test, y_predict))
    print( "Mean Squared Error for Test Set: ",round(mean_squared_error(y_test,y_predict), 3))
    print("Confusion matrix for Test Set \n",confusion_matrix(y_test,y_predict))
    print(classification_report(y_test,y_predict))
    fpr, tpr, thresholds= metrics.roc_curve(y_test,y_predict)
    auc = metrics.roc_auc_score(y_test,y_predict, average='macro', sample_weight=None)
    print("ROC Curve for for Test Set \n")
    sns.set_style('darkgrid')
    sns.lineplot(fpr,tpr,color ='blue')
    plt.show()
    
    
#     print("\nAccuracy for Train Set: ",accuracy_score(y_train, y_Train_predict))
#     print( "Mean Squared Error for Train Set: ",round(mean_squared_error(y_train,y_Train_predict), 3))
#     print("Confusion matrix for Train Set \n",confusion_matrix(y_train,y_Train_predict))
#     print(classification_report(y_train,y_Train_predict))
#     fpr_train, tpr_train, thresholds_train= metrics.roc_curve(y_train,y_Train_predict)
#     auc_train = metrics.roc_auc_score(y_train,y_Train_predict, average='macro', sample_weight=None)
#     print("ROC Curve for for Train Set \n")
#     sns.set_style('darkgrid')
#     sns.lineplot(fpr_train,tpr_train,color ='red')
#     plt.show()
    
    print("--------------------------------------xxx--------------------------------------\n\n")