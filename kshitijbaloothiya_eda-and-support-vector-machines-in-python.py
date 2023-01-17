

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

data = pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")
print(data.head())


    
y = data['status'].tolist()
ctr1 = 0
ctr2 = 0
Y = []
for i in range(0,len(y)):
    if y[i]=='Placed':
        ctr1 = ctr1 + 1
        Y.append(1)
    else:
        ctr2 = ctr2 + 1
        Y.append(0)
ctr_1 = [ctr1, ctr2]
labels1 = ['placed', 'not placed']
ypos1 = np.arange(len(labels1))
plt.xticks(ypos1, labels1)
plt.ylabel('Number')
plt.bar(ypos1, ctr_1)
plt.show()


ctr3 = 0
ctr4 = 0
gen = data['gender'].tolist()
for i in range(0,len(gen)):
    if gen[i] == 'M':
            if Y[i] == 1:
                ctr3 = ctr3 + 1
    elif gen[i] == 'F':
        if Y[i] == 1:
               ctr4 = ctr4 + 1
ctr_2 = [ctr3 ,ctr4]
labels2 = ['Male', 'Female']
ypos2 = np.arange(len(labels2))
plt.xticks(ypos2, labels2)
plt.ylabel('Placed number')
plt.bar(ypos2, ctr_2, color = ['red', 'green'])

plt.show()
ctr5 = 0
ctr6 = 0
ctr7 = 0
st = data['hsc_s'].tolist()
for i in range(0,len(st)):
    if (st[i].lower()) == 'commerce':
        if Y[i] == 1:
            ctr5 = ctr5 + 1
    elif (st[i].lower()) == 'science':
        if Y[i] == 1:
            ctr6 = ctr6 + 1
    elif (st[i].lower()) == 'arts':
        if Y[i] == 1:
            ctr7 = ctr7 + 1
ctr_3 = [ctr5, ctr6, ctr7]
labels3 = ['Commerce', 'Science', 'Arts']
ypos3 = np.arange(len(labels3))
plt.xticks(ypos3, labels3)
plt.ylabel('Placed number')
plt.bar(ypos3, ctr_3, color = ['red', 'green', 'blue'])
deg = data['degree_t'].tolist()
ctr7 = 0
ctr8 = 0
for i in range(0,len(deg)):
    if (deg[i].lower()) == 'sci&tech':
        if Y[i] == 1:
            ctr7 = ctr7 + 1
    elif (deg[i].lower()) == 'comm&mgmt':
        if Y[i] == 1:
            ctr8 = ctr8 + 1
ctr_4 = [ctr7, ctr8]
labels4 = ['Sci&tech', 'Comm&mgmt']
ypos4 = np.arange(0,len(labels4))
plt.xticks(ypos4, labels4)
plt.ylabel("Placed number")
plt.bar(ypos4, ctr_4, color = ['red', 'green'])
#Numberical data

print(data.head())
X1 = data[['gender', 'ssc_b','hsc_b','hsc_s','degree_t', 'workex', 'specialisation']]
X1 = pd.get_dummies(X1)
print(X1.shape)

X2 = data[['ssc_p','hsc_p', 'degree_p','etest_p', 'mba_p']]

#concat the data

X = pd.concat([X1, X2], axis=1, sort=False)

y = pd.DataFrame(Y) # 1 represents placed and 0 represents not placed


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, preprocessing
from sklearn.feature_selection import SelectFromModel

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2)  # Spliting the data
X_train = preprocessing.StandardScaler().fit(X_train).transform(X_train)   # Preprocessing the data
X_test = preprocessing.StandardScaler().fit(X_test).transform(X_test)
LR = LogisticRegression()
LR.fit(X_train, y_train)            # Fiting the logistic regression
yhat = LR.predict(X_test)
print("Logistic regression accuracy:", metrics.accuracy_score(y_test, yhat)) #Finding out the accuracy
print(X_train)
#Feature selection using L1 regularization

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
sel = SelectFromModel(LogisticRegression())
sel.fit(X_train, y_train)
selected_feat = X_train.columns[(sel.get_support())]
print("Optimum number of features from L1 regularisation:", len(selected_feat))
X_train_lasso = sel.fit_transform(X_train, y_train)
X_test_lasso = sel.transform(X_test)
mdl_lasso = LogisticRegression()
mdl_lasso.fit(X_train_lasso, y_train)
score_lasso = mdl_lasso.score(X_test_lasso, y_test)
print("Score with L1 regularisation:",score_lasso)


from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2)  # Spliting the data
X_train = preprocessing.StandardScaler().fit(X_train).transform(X_train)   # Preprocessing the data
X_test = preprocessing.StandardScaler().fit(X_test).transform(X_test)
mdl = SVC(gamma='auto')
mdl.fit(X_train, y_train)
yhat_svm = mdl.predict(X_test)
print("Support vector machine accuracy:", metrics.accuracy_score(yhat_svm, y_test))
svc_accuracy = metrics.accuracy_score(yhat_svm, y_test)