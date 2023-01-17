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
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn import preprocessing

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
bank = pd.read_csv("../input/UnivBank.csv")
bank.shape
bank.head()
bank.describe()
bank.info()
bank.isna().sum()
bank.duplicated().sum()
import seaborn as sns

from matplotlib import pyplot as plot

sns.boxplot(data=bank,orient='h')

bank.plot(kind = 'box',figsize = (15,10))
bank.CreditCard.value_counts()
# sns.pairplot(data = bank, hue = 'CreditCard')
bank.corr()
plot.figure(figsize=(16, 10))

sns.heatmap(bank.corr(),cmap="BuPu",annot=True)
bank.drop(columns= ['ID','Age','Income'],inplace=True)
bank.columns[1]
fig,axes = plot.subplots(7,2,figsize=(12,9))

credit_yes = bank[bank.CreditCard == 1]

credit_no = bank[bank.CreditCard == 0]

ax = axes.ravel()

for i in range(11):

    _,bins=np.histogram(bank.iloc[:,i],bins=40)

    ax[i].hist(credit_yes.iloc[:,i],bins=bins,color='r',alpha=.5)

    ax[i].hist(credit_no.iloc[:,i],bins=bins,color='g',alpha=.3)

    ax[i].set_title(bank.columns[i],fontsize=9)

    ax[i].axes.get_xaxis().set_visible(False)

    ax[i].set_yticks(())

ax[0].legend(['yes','no'],loc='best',fontsize=8)

plot.tight_layout()

plot.show()
# # Initialize a decision tree model

# from sklearn.tree import DecisionTreeClassifier

# estimator = DecisionTreeClassifier(max_depth=5, random_state=0)



# # other ideas for classifiers:

# # from sklearn.linear_model import LogisticRegression

# # from sklearn.neighbors import KNeighborsClassifier

# # from sklearn.svm import SVC

# # from sklearn.naive_bayes import GaussianNB

# # from sklearn.tree import DecisionTreeClassifier

# # from sklearn.ensemble import RandomForestClassifier





# # preprocess the data using a standard scaler

# from sklearn.preprocessing import StandardScaler

# transformer = StandardScaler()





# # combine the steps in a pipeline:

# # step1: transformer

# # step2: model

# # a pipeline can contain N transformer steps and an optional

# # final estimator step

# from sklearn.pipeline import make_pipeline

# pipeline = make_pipeline(transformer,

#                          estimator)





# # Do a cross validation:

# from sklearn.model_selection import cross_val_score

# cvscores = cross_val_score(pipeline, X, y, n_jobs=-1)



# print ("The pipeline CV score is:")

# print (cvscores.mean().round(3), "+/-", cvscores.std().round(3))





# from sklearn.model_selection import GridSearchCV

# from sklearn.svm import SVC

# model = SVC()

# params = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}

# gsmodel = GridSearchCV(model, params, n_jobs=-1)

# gsmodel.fit(X, y)

# print ("Best Model:")

# print (gsmodel.best_estimator_)

# print

# print ("Best Parameters:")

# print (gsmodel.best_params_)

# print

# print ("Best Score:")

# print (gsmodel.best_score_)
X_train, X_test, y_train, y_test = train_test_split(bank.drop(columns=['CreditCard']), bank['CreditCard'], stratify=bank['CreditCard'],test_size=0.25, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

train_Pred = logreg.predict(X_train)

test_Pred = logreg.predict(X_test)

confusion_matrix(y_test,test_Pred)

accuracy_score(y_test,test_Pred)
X_test
from sklearn.model_selection import cross_val_score

scores = cross_val_score(logreg,bank.drop(columns=['CreditCard']), bank['CreditCard'], cv = 6)

print(scores)
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import PolynomialFeatures

dt = DecisionTreeClassifier(max_depth=5, random_state=0)

dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

print(accuracy_score(y_test, y_pred))

confusion_matrix(y_test,y_pred)
from sklearn.model_selection import cross_val_score

scores = cross_val_score(dt,bank.drop(columns=['CreditCard']), bank['CreditCard'], cv = 6)

print(scores)
from sklearn.svm import SVC

kernel = 'linear'

C = 1.0

degree = 3

lr = SVC(kernel = kernel, degree = degree, C = C)

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print(accuracy_score(y_test, y_pred))
confusion_matrix(y_test,y_pred)
def fit_predict(train, test, y_train, y_test, scaler, kernel = 'linear', C = 1.0, degree = 3):

    train_scaled = scaler.fit_transform(train)

    test_scaled = scaler.transform(test)        

    lr = SVC(kernel = kernel, degree = degree, C = C)

    lr.fit(train_scaled, y_train)

    y_pred = lr.predict(test_scaled)

    print(accuracy_score(y_test, y_pred))
for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:

    print('Accuracy score using {0} kernel:'.format(kernel), end = ' ')

    fit_predict(X_train, X_test, y_train, y_test, StandardScaler(), kernel)
for с in np.logspace(3-1, -5, base = 2, num = 6):

    print('Accuracy score using penalty = {0} with poly kernel:'.format(с), end = ' ')

    fit_predict(X_train, X_test, y_train, y_test, StandardScaler(), 'rbf', с)
for degree in range(2, 6):

    print('Accuracy score using degree = {0} with poly kernel:'.format(degree), end = ' ')

    fit_predict(X_train, X_test, y_train, y_test, StandardScaler(), 'poly', 0.5743, degree = degree)
fit_predict(X_train, X_test, y_train, y_test, StandardScaler(), 'poly', 0.5743, degree = degree)