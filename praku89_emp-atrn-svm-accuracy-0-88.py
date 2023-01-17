# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
emp_atr=pd.read_csv("../input/HR-Employee-Attrition.csv")

emp_atr.shape
emp_atr.info()
emp_atr.describe().transpose()
emp_atr.Attrition.value_counts()
emp_atr.head(10)
emp_atr.tail(10)
emp_atr.isna().sum()
emp_atr[emp_atr.duplicated()]
emp_atr.columns
cat_col = emp_atr.select_dtypes(exclude=np.number).columns

num_col = emp_atr.select_dtypes(include=np.number).columns

print(cat_col)

print(num_col)
for i in cat_col:

    print(emp_atr[i].value_counts())
# Get discrete numerical value

num_col_disc=[]

num_col_medium=[]

num_col_cont=[]

print("Attributes with their distinct count")

for i in num_col:

    if emp_atr[i].nunique() <=10:

        print(i,"==",emp_atr[i].nunique(),"== disc")

        num_col_disc.append(i)

    elif (emp_atr[i].nunique() >10 and emp_atr[i].nunique() <100):

        num_col_medium.append(i)    

        print(i,"==",emp_atr[i].nunique(),"== medium")

    else:

        num_col_cont.append(i)

        print(i,"==",emp_atr[i].nunique(),"== cont")

#print(num_col_disc)

#print(num_col_medium)

#print(num_col_cont)
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
fig, ax = plt.subplots(3, 3, figsize=(30, 40))

for variable, subplot in zip(cat_col, ax.flatten()):

    cp=sns.countplot(emp_atr[variable], ax=subplot,order = emp_atr[variable].value_counts().index,hue=emp_atr['Attrition'])

    cp.set_title(variable,fontsize=40)

    cp.legend(fontsize=30)

    for label in subplot.get_xticklabels():

        label.set_rotation(90)

        label.set_fontsize(36)                

    for label in subplot.get_yticklabels():

        label.set_fontsize(36)        

        cp.set_ylabel('Count',fontsize=40)    

plt.tight_layout()

fig, ax = plt.subplots(3, 3, figsize=(30, 40))

for variable, subplot in zip(num_col_disc, ax.flatten()):

    cp=sns.countplot(emp_atr[variable], ax=subplot,order = emp_atr[variable].value_counts().index,hue=emp_atr['Attrition'])

    cp.set_title(variable,fontsize=40)

    cp.legend(fontsize=30)

    for label in subplot.get_xticklabels():

        #label.set_rotation(90)

        label.set_fontsize(36)                

    for label in subplot.get_yticklabels():

        label.set_fontsize(36)        

        cp.set_ylabel('Count',fontsize=40)

plt.tight_layout()
plt.figure(figsize=(20, 10))

sns.countplot(emp_atr["WorkLifeBalance"],hue=emp_atr["Attrition"])

#emp_atr["WorkLifeBalance"]
plt.figure(figsize=(20, 10))

sns.boxplot(data=emp_atr[num_col_cont],orient="h")
plt.figure(figsize=(20, 10))

sns.barplot(data=emp_atr[num_col_cont],orient="h")
#fill_num_attrition=lambda x: 1 if x=="Yes" else 0

#type(fill_num_attrition)

emp_atr["num_attrition"]=emp_atr["Attrition"].apply(lambda x: 1 if x=="Yes" else 0)

emp_atr["num_attrition"].value_counts()
emp_atr_cov=emp_atr.cov()

emp_atr_cov
plt.figure(figsize=(40,20))

sns.heatmap(emp_atr_cov,vmin=-1,vmax=1,center=0,annot=True)
# Importing necessary package for creating model

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics
# one hot encoding num_attrition

cat_col_rm_tgt=cat_col[1:]

num_col=emp_atr.select_dtypes(include=np.number).columns

one_hot=pd.get_dummies(emp_atr[cat_col_rm_tgt])

emp_atr_df=pd.concat([emp_atr[num_col],one_hot],axis=1)

emp_atr_df.head(10)

X=emp_atr_df.drop(columns=['num_attrition'])

y=emp_atr_df[['num_attrition']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

logreg = LogisticRegression()

logreg.fit(X_train, y_train)
train_Pred = logreg.predict(X_train)
metrics.confusion_matrix(y_train,train_Pred)
metrics.accuracy_score(y_train,train_Pred)
test_Pred = logreg.predict(X_test)
metrics.confusion_matrix(y_test,test_Pred)
metrics.accuracy_score(y_test,test_Pred)
from sklearn.metrics import classification_report

print(classification_report(y_test, test_Pred))
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from math import sqrt
X_train, X_test, y_train, y_test = train_test_split(

X, y, test_size = 0.3, random_state = 100)

y_train=np.ravel(y_train)

y_test=np.ravel(y_test)

#y_train = y_train.ravel()

#y_test = y_test.ravel()
accuracy_train_dict={}

accuracy_test_dict={}

df_len=round(sqrt(len(emp_atr_df)))

for k in range(3,df_len):

    K_value = k+1

    neigh = KNeighborsClassifier(n_neighbors = K_value, weights='uniform', algorithm='auto')

    neigh.fit(X_train, y_train) 

    y_pred_train = neigh.predict(X_train)

    y_pred_test = neigh.predict(X_test)    

    train_accuracy=accuracy_score(y_train,y_pred_train)*100

    test_accuracy=accuracy_score(y_test,y_pred_test)*100

    accuracy_train_dict.update(({k:train_accuracy}))

    accuracy_test_dict.update(({k:test_accuracy}))

    print ("Accuracy for train :",train_accuracy ," and test :",test_accuracy,"% for K-Value:",K_value)
elbow_curve_train = pd.Series(accuracy_train_dict,index=accuracy_train_dict.keys())

elbow_curve_test = pd.Series(accuracy_test_dict,index=accuracy_test_dict.keys())

elbow_curve_train.head(10)
ax=elbow_curve_train.plot(title="Accuracy of train VS Value of K ")

ax.set_xlabel("K")

ax.set_ylabel("Accuracy of train")

ax=elbow_curve_test.plot(title="Accuracy of test VS Value of K ")

ax.set_xlabel("K")

ax.set_ylabel("Accuracy of test")
from sklearn.naive_bayes import GaussianNB
NB=GaussianNB()

NB.fit(X_train, y_train)
GaussianNB(priors=None,var_smoothing=1e-09)
train_pred=NB.predict(X_train)

accuracy_score(train_pred,y_train)
test_pred=NB.predict(X_test)

accuracy_score(test_pred,y_test)
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import accuracy_score

from sklearn import tree

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PolynomialFeatures
clf_gini = DecisionTreeClassifier(criterion = "gini",

                               max_depth=3, min_samples_leaf=5)

clf_gini.fit(X_train, y_train)
clf_entropy = DecisionTreeClassifier(criterion = "entropy",

 max_depth=3, min_samples_leaf=5)

clf_entropy.fit(X_train, y_train)
y_pred = clf_gini.predict(X_test)

y_pred
y_pred_en = clf_entropy.predict(X_test)

y_pred_en
print ("Accuracy is ", accuracy_score(y_test,y_pred)*100)
print ("Accuracy is ", accuracy_score(y_test,y_pred_en)*100)
def fit_predict(train, test, y_train, y_test, scaler, max_depth, 

                criterion = 'entropy', max_features = 1, min_samples_split = 4):

    train_scaled = scaler.fit_transform(train)

    test_scaled = scaler.transform(test)        

    dt = DecisionTreeClassifier(criterion = criterion, max_depth=max_depth, 

                                 max_features=max_features,

                               min_samples_split=min_samples_split)

    dt.fit(train_scaled, y_train)

    y_pred = dt.predict(test_scaled)

    print(accuracy_score(y_test, y_pred))
dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

print(accuracy_score(y_test, y_pred))
for i in range(1, 20):

    print('Accuracy score using max_depth =', i, end = ': ')

    fit_predict(X_train, X_test, y_train, y_test, StandardScaler(), i)
for i in np.arange(0.1, 1.0, 0.1):

    print('Accuracy score using max features =', i, end = ': ')

    fit_predict(X_train, X_test, y_train, y_test, StandardScaler(), max_depth = 1, max_features=i)
for i in range(2, 10):

    print('Accuracy score using min samples split =', i, end = ': ')

    fit_predict(X_train, X_test, y_train, y_test, StandardScaler(), 1, max_features=0.1, min_samples_split=i)
for i in ['gini', 'entropy']:

    print('Accuracy score using criterion =', i, end = ': ')

    fit_predict(X_train, X_test, y_train, y_test, StandardScaler(), 1, 

                max_features=0.1, min_samples_split=2, criterion = i)
def create_poly(train,test,degree):

    poly = PolynomialFeatures(degree=degree)

    train_poly = poly.fit_transform(train)

    test_poly = poly.fit_transform(test)

    return train_poly,test_poly
for degree in [1,2,3,4]:

    train_poly, test_poly = create_poly(X_train, X_test, degree)

    print('Polynomial degree',degree)

    fit_predict(train_poly, test_poly, y_train, y_test, StandardScaler(), 1, 

                max_features=0.1, min_samples_split=2, criterion = 'gini')

    print(10*'-')

    

train_poly, test_poly = create_poly(X_train, X_test, 2)

original_score = .8231292517006803

best_score = 0.8412698412698413

improvement = np.abs(np.round(100*(original_score - best_score)/original_score,2))

print('overall improvement is {} %'.format(improvement))
from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV



# Choose the type of classifier. 

clf = DecisionTreeClassifier()





# Choose some parameter combinations to try

parameters = {'max_features': ['log2', 'sqrt','auto'], 

              'criterion': ['entropy', 'gini'],

              'max_depth': [i for i in range(1,20)], 

              'min_samples_split': [i for i in range(2,10)],

              'min_samples_leaf': [i for i in range(2,10)]

             }



# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)



# Run the grid search

grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)

grid_obj = grid_obj.fit(X_train, y_train)



# Set the clf to the best combination of parameters

clf = grid_obj.best_estimator_



# Fit the best algorithm to the data. 

print(clf.fit(X_train, y_train))



#Predict target value and find accuracy score

y_pred = clf.predict(X_test)

print("Accuracy score is ",accuracy_score(y_test, y_pred))
# using Naive Bayes

from sklearn.svm import SVC

import warnings

warnings.filterwarnings("ignore")

def fit_predict(train, test, y_train, y_test, scaler, kernel = 'linear', C = 1.0, degree = 3):

    train_scaled = scaler.fit_transform(train)

    test_scaled = scaler.transform(test)        

    lr = SVC(kernel = kernel, degree = degree, C = C)

    lr.fit(train_scaled, y_train)

    y_pred = lr.predict(test_scaled)

    print(accuracy_score(y_test, y_pred))
y_train=np.ravel(y_train)

y_test=np.ravel(y_test)

for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:

    print('Accuracy score using {0} kernel:'.format(kernel), end = ' ')

    fit_predict(X_train, X_test, y_train, y_test, StandardScaler(), kernel)
for с in np.logspace(-1,3 ,base = 2, num = 6):

    print('Accuracy score using penalty = {0} with rbf kernel:'.format(с), end = ' ')

    fit_predict(X_train, X_test, y_train, y_test, StandardScaler(), 'linear', с)
for degree in range(2, 6):

    print('Accuracy score using degree = {0} with poly kernel:'.format(degree), end = ' ')

    fit_predict(X_train, X_test, y_train, y_test, StandardScaler(), 'linear', 0.5, degree = degree)