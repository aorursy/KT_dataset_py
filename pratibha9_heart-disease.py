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
df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
df.head()
#CHCEKING VALUT TYPE AND NOT-NULL VALUES
df.info()
# CHECKING MEAN STANDARD DEVIATION AND OTHER INFORMATIONS FOR COLUMNS
df.describe()
#CHECK COVARIANCE RELATION BETWEEN COLUMNS 
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize = (15,10))
#df.corr()
sns.heatmap(df.corr(),annot = True,linecolor = "RED")
plt.show()
plt.scatter(df['target'],df['age'])
plt.xlabel("Target")
plt.ylabel("Age")
plt.show()
# Filtering patients
patients = df[df['target'] == 1]
sns.distplot(patients['age'],color = "RED")
plt.figure(figsize = (10,5))
sns.violinplot( df["sex"] ,df["target"], color="RED")
plt.title("Rleation between gender and target")
plt.show()
#sns.distplot( df["target"] , color="red")
#sns.plt.legend()
#plt.scatter(df['sex'],df['target'])

print(df['sex'].value_counts())
female = df[df['sex'] == 0]
female_patients = female[female['target'] == 1]
male = df[df['sex'] == 1]
male_patients = male[male['target'] == 1]
print("Probability for women having heart disease" + str(len(female_patients)/len(female)))
print("Probability for men having heart disease" + str(len(male_patients)/len(male)))
plt.figure(figsize = (10,5))
sns.swarmplot(df['target'],df['cp'], size = 8)
plt.xlabel("Target")
plt.ylabel("Chest Pain")
plt.title("Relation between chest pain and target")
plt.show()
plt.figure(figsize = (10,5))
sns.swarmplot(df['target'],df['trestbps'], size = 8)
plt.xlabel("Target")
plt.ylabel("Blood Pressure on Rest Time")
plt.title("Relation between Rest Time Blood Pressure and target")
plt.show()

plt.figure(figsize = (10,5))
sns.swarmplot(df['target'],df['chol'], size = 8)
plt.title("Relation between cholestral and target")
plt.show()
pd.crosstab(df['fbs'],df['target']).plot(kind="bar",figsize=(10,7),color=['RED','BLUE' ])
#plt.xlabel('Target')
#plt.ylabel('Frequency of Disease or Not')
plt.show()
pd.crosstab(df['fbs'],df['target'])
pd.crosstab(df['restecg'],df['target']).plot(kind="bar",figsize=(10,7),color=['yellow','purple' ])
#plt.xlabel("")

plt.figure(figsize = (10,5))
sns.swarmplot(df['target'],df['thalach'], size = 8)
plt.title("Relation between maximum heart rate achieved and target")
plt.show()

X = df[['sex','cp','fbs','restecg','exang','slope','ca','thal']].astype(object)
dummies = pd.get_dummies(X)
dummies.info()
df.drop(['sex','cp','fbs','restecg','exang','slope','ca','thal'], inplace = True,axis = 1)
df = pd.concat([df,dummies] ,axis = 1)
df
#CHANGING NUMERICAL VALUES TO THEIR STANDARD VALUE
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
nm = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
df[nm] = scale.fit_transform(df[nm])
df
df.shape

from sklearn.model_selection import train_test_split
y = df.target
df.drop('target',inplace = True,axis = 1)
print(y.shape)
print(df.shape)
x_train, x_test, y_train, y_test = train_test_split(df,y,test_size = 0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
lr = LogisticRegression()
lr.fit(x_train,y_train)
pred = lr.predict(x_test)
train_acc = accuracy_score(y_train,lr.predict(x_train))
acc = accuracy_score(y_test,pred)
print("Training_Accuracy" + str(train_acc))
print("Test_Accuracy" + str(acc))
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)
pred1 = nb.predict(x_test)
acc1 = accuracy_score(y_test,pred1)
train_acc1 = accuracy_score(y_train,nb.predict(x_train))
print("Training_Accuracy" + str(train_acc1))
print("Test_Accuracy" + str(acc1))
from sklearn.svm import SVC
svm = SVC()
svm.fit(x_train, y_train)
pred2 = svm.predict(x_test)
acc2 = accuracy_score(y_test,pred2)
acc2
from sklearn.model_selection import GridSearchCV
params = {"C": np.logspace(-4, 4, 20),
          "solver": ["liblinear"]}


grid_search_cv = GridSearchCV(lr, params, scoring="accuracy", n_jobs=-1, verbose=1, cv=5, iid=True)
grid_search_cv.fit(x_train, y_train)
grid_search_cv.best_estimator_
log_reg = LogisticRegression(C=0.23357214690901212, 
                             solver='liblinear')

log_reg.fit(x_train, y_train)

pred5 = log_reg.predict(x_test)
acc5 = accuracy_score(y_test,pred5)
acc5
from sklearn.metrics import classification_report
print(classification_report(y_test,pred5))
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,pred5)
cm
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=1000,random_state = 42)
rf.fit(x_train,y_train)
rf_pred = rf.predict(x_test)
print(accuracy_score(y_test,rf_pred))
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
param_dist = {"max_depth": [3, None],"max_features": sp_randint(1, x_train.shape[1]),"min_samples_split": sp_randint(2, 11),"bootstrap": [True, False],"n_estimators": sp_randint(100, 500)}
random_search = RandomizedSearchCV(rf, param_distributions=param_dist,
                                   n_iter=10, cv=5, iid=False, random_state=42)
random_search.fit(x_train, y_train)
print(random_search.best_params_)
rf1 = RandomForestClassifier(bootstrap = False, max_depth = None, max_features = 3, min_samples_split = 7,n_estimators = 408)
rf1.fit(x_train,y_train)
rf1_pred = rf1.predict(x_test)
print(accuracy_score(y_test,rf1_pred))
