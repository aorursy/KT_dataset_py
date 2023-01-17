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
emp_attr = pd.read_csv('../input/HR-Employee-Attrition.csv')
emp_attr.shape
emp_attr.info()
emp_attr.head()
emp_attr.nunique()
emp_attr.corr()
import seaborn as sns
ax = sns.barplot(x ='Attrition', y= 'PercentSalaryHike', data=emp_attr)
# for col in emp_attr.columns:

#     pd.crosstab(emp_attr[col],emp_attr.Attrition).plot(kind='bar',color = ('blue','red'),figsize=(5,5))
emp_attr[['EmployeeCount','StandardHours']].nunique()
emp_attr.drop(columns=['EmployeeCount','StandardHours'],inplace=True)
emp_attr.info()
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

emp_attr.iloc[:, 1] = labelencoder.fit_transform(emp_attr.iloc[:, 1])

emp_attr.iloc[:, 2] = labelencoder.fit_transform(emp_attr.iloc[:, 2])

emp_attr.iloc[:, 4] = labelencoder.fit_transform(emp_attr.iloc[:, 4])

emp_attr.iloc[:, 7] = labelencoder.fit_transform(emp_attr.iloc[:, 7])

emp_attr.iloc[:, 10] = labelencoder.fit_transform(emp_attr.iloc[:, 10])

emp_attr.iloc[:, 14] = labelencoder.fit_transform(emp_attr.iloc[:, 14])

emp_attr.iloc[:, 16] = labelencoder.fit_transform(emp_attr.iloc[:, 16])

emp_attr.iloc[:, 20] = labelencoder.fit_transform(emp_attr.iloc[:, 20])

emp_attr.iloc[:, 21] = labelencoder.fit_transform(emp_attr.iloc[:, 21])
emp_attr.info()
emp_attr.head()
lower_bnd = lambda x: (x.quantile(0.25) - (1.5 * ( x.quantile(0.75) - x.quantile(0.25) )))

upper_bnd = lambda x: (x.quantile(0.75) + (1.5 * ( x.quantile(0.75) - x.quantile(0.25) )))
emp_attr.shape
emp_attr.plot(kind='box', figsize=(70,30))
# for col in emp_attr.columns:

#     emp_attr = emp_attr.loc[(emp_attr[col] >= lower_bnd(emp_attr[col])) & (emp_attr[col] <= upper_bnd(emp_attr[col]))]
# emp_attr = emp_attr.loc[(emp_attr['MonthlyIncome'] >= lower_bnd(emp_attr['MonthlyIncome'])) & (emp_attr['MonthlyIncome'] <= upper_bnd(emp_attr['MonthlyIncome']))]
emp_attr.plot(kind='box', figsize=(70,30))
emp_attr.shape
# for col in emp_attr.columns:

#     pd.crosstab(emp_attr[col],emp_attr.Attrition).plot(kind='bar',color = ('blue','red'),figsize=(5,5))
from math import ceil

def int_repl(df,col,n):

    a = df[col].unique()

    a.sort()

    cutoff = ceil(len(a)/n)

    x = 0

    y = 0

    res = '{'

    for i in a:

        if x == cutoff:

            y = y + 1

            x = 0

        res = res + '{0}:{1},'.format(i,y)

        x = x + 1

    res = res[0:len(res)-1] + '}'

    df[col].replace(eval(res),inplace=True)
emp_attr.columns
lst1 = ['Age', 'DistanceFromHome',

        'TotalWorkingYears',

       'YearsAtCompany', 'HourlyRate','PercentSalaryHike','YearsInCurrentRole',

       'YearsSinceLastPromotion','YearsWithCurrManager','NumCompaniesWorked','TrainingTimesLastYear'

       ,'JobRole','EducationField']
lst2 = ['DailyRate','MonthlyIncome', 'MonthlyRate']
emp_attr.nunique()
for i in lst1:

    int_repl(emp_attr,i,5)
for i in lst2:

    int_repl(emp_attr,i,5)
emp_attr.drop(columns='EmployeeNumber',inplace=True)
emp_attr.nunique()
for col in emp_attr.columns:

    pd.crosstab(emp_attr[col],emp_attr.Attrition).plot(kind='bar',color = ('blue','red'),figsize=(5,5))
from sklearn.model_selection import train_test_split

from sklearn import preprocessing

X = emp_attr.drop(columns=['Attrition','Over18'])

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

y = emp_attr['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

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
import matplotlib.pyplot as plt 

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
from sklearn.preprocessing import Imputer

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

accuracy_dict = {}

accuracy_list = []

for K in range(25):

    K_value = K + 1

    neigh = KNeighborsClassifier(n_neighbors=K_value,weights='uniform',algorithm='auto')

    neigh.fit(X_train, y_train)

    y_pred=neigh.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)

    accuracy_dict.update({K_value:accuracy})

    accuracy_list.append(accuracy)

    print("Accuracy is",accuracy_score(y_test,y_pred)*100,"% for K-Value",K_value)
key_max = max(accuracy_dict.keys(), key=(lambda k: accuracy_dict[k]))



print( "The Accuracy value is ",accuracy_dict[key_max], "with k= ", key_max)
elbow_curve = pd.DataFrame(accuracy_list,columns = ['accuracy'])

elbow_curve.plot()



print(classification_report(y_test, test_Pred))
from sklearn.naive_bayes import GaussianNB

bayes = GaussianNB()

bayes.fit(X_train, y_train)

y_pred=bayes.predict(X_train)
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import PolynomialFeatures

dt = DecisionTreeClassifier(criterion = 'entropy', max_depth=18, 

                                 max_features=1,

                               min_samples_split=4)

# X1 = emp_attr.drop(columns=['Attrition','Over18'])

# X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.3, random_state=0)

# scaler = preprocessing.StandardScaler()

# X_train = scaler.fit_transform(X_train)

# y_train = scaler.transform(y_train) 

dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

print(accuracy_score(y_test, y_pred))
for i in range(1, 20):

    print('Accuracy score using max_depth =', i, end = ': ')

    dt = DecisionTreeClassifier(criterion = 'entropy', max_depth=i, 

                                 max_features=1,

                               min_samples_split=4)

    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)

    print(accuracy_score(y_test, y_pred))
for i in np.arange(0.1, 1.0, 0.1):

    print('Accuracy score using max_features =', i, end = ': ')

    dt = DecisionTreeClassifier(criterion = 'entropy', max_depth=1, 

                                 max_features=i,

                               min_samples_split=4)

    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)

    print(accuracy_score(y_test, y_pred))
for i in ['entropy','gini']:

    print('Accuracy score using criterion =', i, end = ': ')

    dt = DecisionTreeClassifier(criterion = i, max_depth=1, 

                                 max_features=0.1,

                               min_samples_split=4)

    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)

    print(accuracy_score(y_test, y_pred))
for i in range(2, 10):

    print('Accuracy score using min_samples_split =', i, end = ': ')

    dt = DecisionTreeClassifier(criterion = 'gini', max_depth=1, 

                                 max_features=0.1,

                               min_samples_split=i)

    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)

    print(accuracy_score(y_test, y_pred))
from sklearn import svm

steps = [('scaler', preprocessing.StandardScaler()), ('SVM', svm.SVC())]

from sklearn.pipeline import Pipeline

pipeline = Pipeline(steps) # define the pipeline object.
from sklearn.model_selection import cross_val_score

cvscores = cross_val_score(pipeline, X_train, y_train, n_jobs=-1)



print ("The pipeline CV score is:")

print (cvscores.mean().round(3), "+/-", cvscores.std().round(3))
pipeline.fit(X_train,y_train)

y_pred = pipeline.predict(X_test)

print(accuracy_score(y_test, y_pred))