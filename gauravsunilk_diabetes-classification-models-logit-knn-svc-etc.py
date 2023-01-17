import pandas as pd

import numpy as np

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train=pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
train.head()
train.describe()
train.info()


train.isnull().sum()
train['SkinThickness'].hist()
train['Insulin'].hist()
train['BMI'].hist()
#Imputing the null values with respective column median values since the data is skewed.

train['SkinThickness'].replace(np.NaN,train['SkinThickness'].median(),inplace=True)

train['Insulin'].replace(np.NaN,train['Insulin'].median(),inplace=True)

train['BMI'].replace(np.NaN,train['BMI'].median(),inplace=True)
train.isnull().sum()
train.boxplot()
#Number of pregnancies imputing values greater than (mean+3*std. dev) or (Q3+1.5*IQR) to the one which is greater

train['Pregnancies'] = np.where(train['Pregnancies'] > 14, 14,train['Pregnancies'])

#Concentration of plasma glucose imputing values lesser than (mean-3*std. dev) or (Q1-1.5*IQR) to the one which is greater

train['Glucose'] = np.where(train['Glucose'] < 33, 33,train['Glucose'])

# Normal diastolic blood pressure levels

train['BloodPressure'] = np.where(train['BloodPressure'] < 60, 60,train['BloodPressure'])

#Tricep skinfold thickness imputing values greater than (mean+3*std. dev) or (Q3+1.5*IQR) to the one which is greater

train['SkinThickness'] = np.where(train['SkinThickness'] > 65, 65,train['SkinThickness'])

# Insulin concentration in the serum after 2 hours should be in the range < 177) (which is considered normal). Since, we have too many outliers we scale them to Q3+1.5*IQR

train['Insulin'] = np.where(train['Insulin'] > 390, 390,train['Insulin'])

# BMI will be scaled to a minimum of Q1-1.5*IQR

train['BMI'] = np.where(train['BMI'] < 15, 15,train['BMI'])

# Probability cannot be greater than 1

train['DiabetesPedigreeFunction'] = np.where(train['DiabetesPedigreeFunction'] >1, 1,train['DiabetesPedigreeFunction'])



train.describe()


#Dividing the dataset into X and Y variables

x=train.drop('Outcome',axis=1)

y=train['Outcome']
from pandas.plotting import scatter_matrix

attributes=train.columns.values

scatter_matrix(train[attributes], figsize = (25,25), c=y, alpha = 0.6, marker = 'O')
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=0)
sc=StandardScaler()

x_sc=sc.fit_transform(x_train)

xt_sc=sc.fit_transform(x_test)
#Hyperparameter Tuning theb Logistic Regression model

from sklearn.linear_model import LogisticRegression



c_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

train_score_l1 = []

train_score_l2 = []

test_score_l1 = []

test_score_l2 = []

for c in c_range:

    log_l1 = LogisticRegression(penalty = 'l1', C = c, solver='liblinear')

    log_l2 = LogisticRegression(penalty = 'l2', C = c, solver='liblinear')

    log_l1.fit(x_sc, y_train)

    log_l2.fit(x_sc, y_train)

    train_score_l1.append(log_l1.score(x_sc, y_train))

    train_score_l2.append(log_l2.score(x_sc, y_train))

    test_score_l1.append(log_l1.score(xt_sc, y_test))

    test_score_l2.append(log_l2.score(xt_sc, y_test))
print('Train_score_l1:',train_score_l1)

print('Test_score_l1:',test_score_l1)

print('Train_score_l2:',train_score_l2)

print('Test_score_l2:',test_score_l2)
from sklearn.model_selection import GridSearchCV

logit = LogisticRegression()

param = { 'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty': ['l1','l2']}

logistic = GridSearchCV(logit,param,scoring='accuracy',cv=5)

logistic.fit(x_sc,y_train)
logistic.best_params_
logit=LogisticRegression(penalty='l2',C=0.1)

logit.fit(x_sc,y_train)

print([logit.score(x_sc,y_train),logit.score(xt_sc,y_test)])
#Feature importance

import matplotlib.pyplot as plt

%matplotlib inline

coef = list(logit.coef_[0])

labels = list(x.columns)

features = pd.DataFrame()

features['Features'] = labels

features['importance'] = coef



features.sort_values(by=['importance'], ascending=True, inplace=True)

features['positive'] = features['importance'] > 0

features.set_index('Features', inplace=True)

features.importance.plot(kind='barh', figsize=(11, 6),color = features.positive.map({True: 'blue', False: 'red'}))

plt.xlabel('Importance')
logit.coef_
#Plotting train and test scores

plt.plot(c_range, train_score_l1, label = 'Train score, penalty = l1')

plt.plot(c_range, test_score_l1, label = 'Test score, penalty = l1')

plt.plot(c_range, train_score_l2, label = 'Train score, penalty = l2')

plt.plot(c_range, test_score_l2, label = 'Test score, penalty = l2')

plt.legend()

plt.xlabel('Regularization parameter: C')

plt.ylabel('Accuracy')

plt.xscale('log')
#Decision boundary

from mlxtend.plotting import plot_decision_regions



X_b = x_sc[10:50, [1,5]]

y_b = np.array(y_train[10:50])



lreg = LogisticRegression()

lreg.fit(X_b, y_b) 



plot_decision_regions(X_b, y_b, clf = lreg)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier



train_score_array = []

test_score_array = []



for k in range(1,20):

    knn = KNeighborsClassifier(k)

    knn.fit(x_sc, y_train)

    train_score_array.append(knn.score(x_sc, y_train))

    test_score_array.append(knn.score(xt_sc, y_test))


print(train_score_array)

print(test_score_array)
knn = KNeighborsClassifier()

param = { 'n_neighbors':[7,8,9]}

knnc = GridSearchCV(knn,param,scoring='accuracy',cv=5)

knnc.fit(x_sc,y_train)


knnc.best_params_
x_axis = range(1,20)

%matplotlib inline

plt.plot(x_axis, train_score_array, label = 'Train Score', c = 'g')

plt.plot(x_axis, test_score_array, label = 'Test Score', c='b')

plt.xlabel('k')

plt.ylabel('Accuracy')

plt.legend()
#Decision boundary

from mlxtend.plotting import plot_decision_regions



X_b = x_sc[100:150,[1,5]]

y_b = np.array(y_train[100:150])



knn = KNeighborsClassifier(9)

knn.fit(X_b, y_b) 



plot_decision_regions(X_b, y_b, clf = knn)
from sklearn.svm import LinearSVC, SVC
#Decision boundary

X_b = x_sc[100:150,[1,5]]

y_b = np.array(y_train[100:150])



svc = LinearSVC()

svc.fit(X_b, y_b) 



plot_decision_regions(X_b, y_b, clf = svc)
#LinearSVC 

svc=LinearSVC(C=1)

svc.fit(x_sc,y_train)

print(svc.score(x_sc,y_train))

print(svc.score(xt_sc,y_test))
from matplotlib import gridspec

import itertools

C = 10

clf1 = LinearSVC(C=C)

clf2 = SVC(kernel='linear', C=C)

clf3 = SVC(kernel='rbf', gamma=0.75, C=C)

clf4 = SVC(kernel='poly', degree=3, C=C)



gs = gridspec.GridSpec(2, 2)

fig = plt.figure(figsize=(10, 8))



labels = ['LinearSVC',

          'SVC with linear kernel',

          'SVC with RBF kernel',

          'SVC with polynomial (degree 3) kernel']



for clf, lab, grd in zip([clf1, clf2, clf3, clf4],

                         labels,

                         itertools.product([0, 1],

                         repeat=2)):

    clf.fit(X_b, y_b)

    ax = plt.subplot(gs[grd[0], grd[1]])

    fig = plot_decision_regions(X=X_b, y=y_b,

                                clf=clf, legend=2)

    plt.title(lab)
from matplotlib import gridspec

import itertools

C = 10

clf1 = SVC(kernel='rbf', gamma=0.01, C=C)

clf2 = SVC(kernel='rbf', gamma=0.1, C=C)

clf3 = SVC(kernel='rbf', gamma=1, C=C)

clf4 = SVC(kernel='rbf', gamma=10, C=C)



X_b = x_sc[100:150,[1,5]]

y_b = np.array(y_train[100:150])



models = (SVC(kernel='rbf', gamma=0.01, C=10),

         SVC(kernel = 'rbf', gamma = 0.1, C = 10),

         SVC(kernel = 'rbf', gamma = 1, C = 10),

         SVC(kernel = 'rbf', gamma = 10, C = 10))





gs = gridspec.GridSpec(2, 2)

fig = plt.figure(figsize=(10, 8))



labels = ['gamma = 0.01',

          'gamma = 0.1',

          'gamma = 1',

          'gamma = 10']



for clf, lab, grd in zip([clf1, clf2, clf3, clf4],

                         labels,

                         itertools.product([0, 1],

                         repeat=2)):

    clf.fit(X_b, y_b)

    ax = plt.subplot(gs[grd[0], grd[1]])

    fig = plot_decision_regions(X=X_b, y=y_b,

                                clf=clf, legend=2)

    plt.title(lab)
train_score_array = []

test_score_array = []

g=[0.01,0.1,1,10,100]

for i in g:

  svc = SVC(kernel='rbf', gamma=i, C=1)

  svc.fit(x_sc, y_train)

  train_score_array.append(svc.score(x_sc, y_train))

  test_score_array.append(svc.score(xt_sc, y_test))
print(train_score_array)

print(test_score_array)
svc=SVC(C=1, kernel='rbf', gamma=0.1)

svc.fit(x_sc,y_train)

print(svc.score(x_sc,y_train))

print(svc.score(xt_sc,y_test))
#After trying various values for max_depth the maximum accuracy obtained was at max_depth=5

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(random_state=0, max_depth=5)



dtree.fit(x_sc, y_train)

print("Accuracy on training set: {:.3f}".format(dtree.score(x_sc, y_train)))

print("Accuracy on test set: {:.3f}".format(dtree.score(xt_sc, y_test)))
features=x.columns.values
def plot_feature_importances_cancer(model):

    n_features = x.shape[1]

    plt.barh(range(n_features), model.feature_importances_, align='center')

    plt.yticks(np.arange(n_features), features)

    plt.xlabel("Feature importance")

    plt.ylabel("Feature")

    plt.ylim(-1, n_features)



plot_feature_importances_cancer(dtree)