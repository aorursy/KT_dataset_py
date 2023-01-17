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
pd.set_option('display.max_rows', 100)
#read the training data
df_train = pd.read_csv("../input/titanic/train.csv")
df_train.head()
y = np.array(df_train['Survived'])
y = y.reshape((891,1))

df_train.drop("Survived", axis =1, inplace =True)
df_train.head()
print("No. of training exapmles = {} and No. of features = {}".format(df_train.shape[0],df_train.shape[1]))
#read the test data
df_test = pd.read_csv("../input/titanic/test.csv")
df_test.head()
print("No. of exapmles in test data = {} and No. of features = {}".format(df_test.shape[0],df_test.shape[1]))
# combine both train and test data
df = pd.concat([df_train, df_test], axis = 0)
df.head()
# Let's check the dimension of the dataset 
df.shape
#Let's check for null values in the dataset
df.info()
plt.bar(df["Cabin"].isnull().value_counts().index,df["Cabin"].isnull().value_counts() ,
        tick_label = ["Null Values", "Non-Null Values"], color='cornflowerblue')
# Cabin has more than 50% null values, therefore let's discard it.
df.drop("Cabin", axis =1, inplace = True)
df.head()
#Plot Histogram of Age
plt.figure(figsize=(12,8)) # figure ration 16:9
sns.set()
sns.distplot(df["Age"],hist = False, label="Age distribution of Passengers")

# Let's take median age of the passengers to fill the null values in the age column.
df.update(df["Age"].fillna(df["Age"].median()))
df.head()
#Plot Histogram of Fare
plt.figure(figsize=(12,8)) # figure ration 16:9
sns.set()
sns.distplot(df["Fare"],hist = False, label="Fare distribution of Passengers")
# Let's take median Fare of the passengers to fill the null value in the age column.
df.update(df["Fare"].fillna(df["Fare"].median()))
df.head()
plt.bar(df["Embarked"].value_counts().index,df["Embarked"].value_counts() , color='cornflowerblue')


df.update(df["Embarked"].fillna(df["Embarked"].mode()[0]))
df.head()
# Verifying if there are any null values
df.info()
# Correlation between features (included non-object-datatype features only )
sns.heatmap(df.corr().round(2), cmap="YlGnBu", annot=True)
df['fam_size'] = df["SibSp"]+df["Parch"]+1
df.drop(["SibSp","Parch"], axis = 1, inplace = True)
df.head()
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder() 
df['Name']= label_encoder.fit_transform(df['Name'])
df['Ticket']= label_encoder.fit_transform(df['Ticket'])
df['Embarked']= label_encoder.fit_transform(df['Embarked'])
df['Sex']= label_encoder.fit_transform(df['Sex'])


df.head()
# column index is stored in col
col = df.columns
df.head()
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler().fit(df)
df= pd.DataFrame(scaler.transform(df), columns =col)
df.head()
df_train_final = df[0:891]
df_train_final 
df_test_final = df[891:1309]
df_test_final 
X= df_train_final[col]
X.head()
from sklearn.model_selection import train_test_split

X_train, X_test,Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=94)

X_train.shape, X_test.shape, Y_train.shape, Y_test.shape
def score_evaluator(clf, X_train, Y_train, X_test, Y_test, y_pred):
    
    from sklearn import metrics
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_val_score, KFold
    
    scores = cross_val_score(clf, X_train, Y_train, cv=10)

    kfold = KFold(n_splits=10, shuffle=True)
    kf_cv_scores = cross_val_score(clf, X_train, Y_train, cv=kfold )
    print("Mean cross-validation score: {0:.2f}    and     K-fold CV average score: {0:.2f} "
                                                          .format(scores.mean(),kf_cv_scores.mean()))
    
    print()
    
    print("Accuracy : {0:.4f} ".format(accuracy_score(Y_test, y_pred)))
    print()

import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier as RFC

# Lets check for various loss function

loss =['gini', 'entropy']

print("-----------------------------------------Random Forest Classifier---------------------------------\n")
for l in loss: 
    print("-------------Loss function = {}---------------------------------".format(l))
    clf = RFC(random_state=0,criterion = l)
    clf = clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    
    score_evaluator(clf, X_train, Y_train, X_test, Y_test, y_pred)
print()
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn import tree

loss =['gini', 'entropy']

print("-----------------------------------------Decision Tree Classifier---------------------------------\n")
for l in loss:
    print("-----------Loss function = {}------------------\n".format(l))
    clf = DTC(random_state=0, criterion = l)
    clf = clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)

    score_evaluator(clf, X_train, Y_train, X_test, Y_test, y_pred)
    
print()

from sklearn.svm import SVC

kernel =['linear', 'poly', 'rbf', 'sigmoid']
C = [0.1, 1, 10] # Regularization constant or penalty (L2 norm is used)

print("---------Support Vector Classifier-------------------")
for k in kernel:
    for c in C:
        print("------------penalty = {}--------kernel = {}--------------".format(c,k))
        clf = SVC(random_state=0, kernel = k, C = c)
        clf = clf.fit(X_train, Y_train)
        y_pred = clf.predict(X_test)
        
        score_evaluator(clf, X_train, Y_train, X_test, Y_test, y_pred)

       

from sklearn.neighbors import KNeighborsClassifier as KNC

algo = ['auto', 'ball_tree', 'kd_tree', 'brute']
neighbor = np.arange(7,13)  # check for different neighborhood size

print("---------K-Neighbors Classifier-------------------------------------")

for n in neighbor:
    for a in algo:
        print("------------No. of neighbours = {}--------algorithm used = {}--------------".format(n,a))
        clf = KNC(n_neighbors= n, algorithm = a)
        clf = clf.fit(X_train, Y_train)
        y_pred = clf.predict(X_test)
        
        
        score_evaluator(clf, X_train, Y_train, X_test, Y_test, y_pred)

from sklearn.linear_model import LogisticRegression as LR

solver= ['lbfgs','newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
C = [ 0.1, 1, 10] # Check for different penalty values


print("---------Logistic  Regression-------------------------------------")
for s in solver:
    for c in C:
        
        print("------------Solver = {}, Penalty = {}--------------".format(s,c))
        clf = LR(solver =  s, C = c)
        clf = clf.fit(X_train, Y_train)
        y_pred = clf.predict(X_test)
        
        score_evaluator(clf, X_train, Y_train, X_test, Y_test, y_pred)

    
        
from xgboost import XGBClassifier as XGB
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score

xgbc = XGB()
xgbc.fit(X_train, Y_train)


scores = cross_val_score(xgbc, X_train, Y_train, cv=5)
 
kfold = KFold(n_splits=10, shuffle=True)
kf_cv_scores = cross_val_score(xgbc, X_train, Y_train, cv=kfold )

print("Mean cross-validation score: {0:.2f}    and     K-fold CV average score: {0:.2f} "
                                                          .format(scores.mean(),kf_cv_scores.mean()))


ypred = xgbc.predict(X_test)


print()
print("Accuracy : {0:.4f} ".format(accuracy_score(Y_test, y_pred)))
print()


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

n_features = X_train.shape[1]

model = Sequential()
model.add(Dense(12, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(4, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='sigmoid'))

# compile the model
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
# fit the model
model.fit(X_train, Y_train, epochs=150, batch_size=32, verbose=1)

# evaluate the model
loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print()
print('Test Accuracy: %.3f' % acc)

X_submit = df_test_final[col]

X_submit.head()
clf = RFC(random_state=0,criterion = "gini")
clf = clf.fit(X_train, Y_train)
y_pred = clf.predict(X_submit)



y_pred_submit = model.predict(X_submit)
y_pred_submit = np.where(y_pred_submit > 0.5, 1, 0)
y_pred_submit
#read the submission data file
df_submit = pd.read_csv("../input/titanic/gender_submission.csv")
df_submit["Survived"] = y_pred_submit

df_submit.to_csv("gender_submission.csv", index=False)

