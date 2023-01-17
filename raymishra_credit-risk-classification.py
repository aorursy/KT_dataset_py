# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as  plt
import sklearn
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data= pd.read_csv("../input/german-credit-data-with-risk/german_credit_data.csv")
data= data.drop(['Unnamed: 0'], axis=1)
data
#LabelBinarizer converts the string categorical variable to binary 
from sklearn.preprocessing import LabelBinarizer
lb= LabelBinarizer()
data["Risk"]= lb.fit_transform(data["Risk"])
#using seaborn library, we have used a frequency plot
sns.countplot('Risk', data=data)
plt.title('Risk Distribution', fontsize=14)
plt.show()
Saving_accounts= data["Saving accounts"]
print("No. of null values in savings:")
print(Saving_accounts.isnull().values.sum())
Checking_accounts= data["Checking account"]
print("No. of null values in Checking:")
print(Checking_accounts.isnull().values.sum())

print(data["Saving accounts"].value_counts())
print(data["Checking account"].value_counts())
ax1 = sns.scatterplot(x="Age", y="Duration", hue="Risk", data=data)

ax2 = sns.scatterplot(x="Credit amount", y="Age", hue="Risk", data=data)
data["Saving accounts"].fillna('NoSavingAcc', inplace= True)
data["Checking account"].fillna('NoCheckAcc', inplace= True)

sns.countplot('Checking account', data=data)
plt.title('Checking account distribution', fontsize=14)
plt.show()
sns.countplot('Saving accounts', data=data)
plt.title('Saving accounts distribution', fontsize=14)
plt.show()
interval = (0, 12, 24, 36, 48, 60, 72)
cats =['year1', 'year2', 'year3', 'year4', 'year5', 'year6']
data["Duration"] = pd.cut(data.Duration, interval, labels=cats)

interval = (18, 25, 35, 60, 120)

cats = ['Student', 'Youth', 'Adult', 'Senior']
data["Age"] = pd.cut(data.Age, interval, labels=cats)

from sklearn.preprocessing import LabelEncoder
lb= LabelEncoder()
data["Saving accounts"]= lb.fit_transform(data["Saving accounts"])
data["Checking account"]= lb.fit_transform(data["Checking account"])
data["Age"]=lb.fit_transform(data["Age"])
data["Sex"]= lb.fit_transform(data["Sex"])
data["Housing"]=lb.fit_transform(data["Housing"])
data["Duration"]= lb.fit_transform(data["Duration"])
data = data.merge(pd.get_dummies(data.Purpose, drop_first=True, prefix='Purpose'), left_index=True, right_index=True)
del data["Purpose"]
data
X= data.drop("Risk", axis= 1)
y= data["Risk"]

from sklearn.preprocessing import StandardScaler
SC= StandardScaler()
X= SC.fit_transform(X)
X=pd.DataFrame(X)
X
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.30, random_state= 100)
y_df=pd.DataFrame(y_train)

from imblearn.over_sampling import SMOTE
def sampling_func(X, y):
    smote= SMOTE( ratio= 'minority')
    x_sm, y_sm= smote.fit_sample(X, y)
    return x_sm, y_sm


def plot_2d_space(X, y, label='Classes'):   
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()
#plot the re-sampled data on 2D space
X_sampled, y_sampled = sampling_func(X_train, y_train)
plot_2d_space(X_sampled, y_sampled, 'SMOTE')
X_sampled= pd.DataFrame(X_sampled)
y_sampled= pd.DataFrame(y_sampled)
y_sampled.columns= y_df.columns
y_sampled["Risk"].value_counts()
df= pd.concat([X_sampled, y_sampled], axis= 1)
df

#converting to numpy array for use in model
y_sampled=y_sampled.values
#visaulization of the plot with equal risk distribution 
colors = ["#0101DF", "#DF0101"]
sns.countplot('Risk', data=df, palette=colors)
plt.title('Equally Distributed Classes', fontsize=14)
plt.show()
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report #To evaluate our model
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
# Algorithmns models to be compared
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
#use the logistic regression classifier on the dataset
classifier = LogisticRegression()
#setting the Gridsearch parameters to fit the model using best estimates
parameters = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_search = GridSearchCV(estimator= classifier,param_grid= parameters, cv=5,  n_jobs= -1)
#fitting the model and predictingon the test set 
grid_search.fit(X_sampled, y_sampled.ravel())
y_pred = grid_search.predict(X_test)

#The confusion matrix plots the predicted positives and negatives
cm= confusion_matrix(y_test, y_pred)
labels = ['Bad', 'Good']
print(classification_report(y_test, y_pred, target_names=labels))
print('accuracy is ',accuracy_score(y_pred,y_test))
print(cm)
#using SVC classifier
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = SVC()
grid_search = GridSearchCV(estimator= svc, param_grid= parameters, cv=5, n_jobs= -1)
grid_search.fit(X_sampled, y_sampled.ravel())
y_pred = grid_search.predict(X_test)
cm= confusion_matrix(y_test, y_pred)
labels = ['Bad', 'Good']
print(classification_report(y_test, y_pred, target_names=labels))
print('accuracy is ',accuracy_score(y_pred,y_test))
print(cm)
#RandomForestClassifier
parameters = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
 'criterion' :['gini', 'entropy']
}
classifier= RandomForestClassifier()
grid_search= GridSearchCV(estimator=classifier, param_grid=parameters, cv= 5, n_jobs= -1)
grid_search.fit(X_sampled, y_sampled.ravel())
y_pred = grid_search.predict(X_test)
cm= confusion_matrix(y_test, y_pred)
labels = ['Bad', 'Good']
print(classification_report(y_test, y_pred, target_names=labels))
print('accuracy is ',accuracy_score(y_pred,y_test))
print(cm)
#Parameters for XGBoost Classifier
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }
classifier = XGBClassifier(learning_rate=0.01, n_estimators=600, objective='binary:logistic',
                    silent=True, nthread=1)
random_search = RandomizedSearchCV(classifier, param_distributions=params,  n_jobs=4, cv=10, verbose=2, random_state=0 )

X_test= pd.DataFrame(X_test)
X_sampled.columns= X_train.columns
random_search.fit(X_sampled, y_sampled.ravel())
y_pred= random_search.predict(X_test)
cm= confusion_matrix(y_test, y_pred)
labels = ['Bad', 'Good']
print(classification_report(y_test, y_pred, target_names=labels))
print('accuracy is ',accuracy_score(y_pred,y_test))
print(cm)
GNB = GaussianNB()
# Fitting with train data
GNB.fit(X_sampled, y_sampled.ravel())
y_pred= GNB.predict(X_test)
cm= confusion_matrix(y_test, y_pred)
labels = ['Bad', 'Good']
print(classification_report(y_test, y_pred, target_names=labels))
print('accuracy is ',accuracy_score(y_pred,y_test))
print(cm)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
Model=LinearDiscriminantAnalysis()
Model.fit(X_sampled,y_sampled.ravel())
y_pred=Model.predict(X_test)
labels = ['Bad', 'Good']
print(classification_report(y_test,y_pred, target_names= labels))
print(confusion_matrix(y_pred,y_test))
#Accuracy Score
print('accuracy is ',accuracy_score(y_pred,y_test))
from sklearn.neighbors import KNeighborsClassifier

Model = KNeighborsClassifier(n_neighbors=8)
Model.fit(X_train, y_train)

y_pred = Model.predict(X_test)

# Summary of the predictions made by the classifier
labels = ['Bad', 'Good']
print(classification_report(y_test, y_pred, target_names= labels))
print(confusion_matrix(y_test, y_pred))
# Accuracy score

print('accuracy is',accuracy_score(y_pred,y_test))