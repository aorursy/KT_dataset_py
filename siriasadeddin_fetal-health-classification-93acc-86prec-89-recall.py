import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import f1_score

from sklearn import preprocessing

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from sklearn import metrics
df=pd.read_csv('../input/fetal-health-classification/fetal_health.csv')
df.T
df.shape
columns=df.columns

columns_new=[]

for i in columns:

    columns_new.append(any(df[i].isnull()|df[i].isnull()))

df=df.drop(columns[columns_new],axis=1)
df.shape
ax = sns.countplot(df.fetal_health,label="Count")       # M = 212, B = 357

df.fetal_health.value_counts()
ax = sns.boxplot( palette="Set2", orient="h",data=df[df.fetal_health==1])
ax = sns.boxplot( palette="Set2", orient="h",data=df[df.fetal_health==2])
ax = sns.boxplot( palette="Set2", orient="h",data=df[df.fetal_health==3])
X_train, X_test, y_train, y_test=train_test_split(

    df.drop(['fetal_health'], axis=1),

    df[['fetal_health']],

    stratify=df[['fetal_health']],

    shuffle=True,

    test_size=0.3,

    random_state=41)
{'train':X_train.shape,'test':X_test.shape}
corrMatrix = X_train.corr()

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(corrMatrix, annot=True,ax=ax)

plt.show()
correlated_features = set()

for i in range(len(corrMatrix .columns)):

    for j in range(i):

        if abs(corrMatrix.iloc[i, j]) > 0.7:

            colname = corrMatrix.columns[i]

            correlated_features.add(colname)

print(correlated_features)
X_train.drop(labels=correlated_features, axis=1, inplace=True)

X_test.drop(labels=correlated_features, axis=1, inplace=True)
corrMatrix = X_train.corr()

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(corrMatrix, annot=True,ax=ax)

plt.show()
{'train':X_train.shape,'test':X_test.shape}
constant_filter = VarianceThreshold(threshold=0.0)

constant_filter.fit(X_train)

X_train = constant_filter.transform(X_train)

X_test = constant_filter.transform(X_test)



{'train':X_train.shape,'test':X_test.shape}
mm_scaler = preprocessing.StandardScaler()

X_train = pd.DataFrame(mm_scaler.fit_transform(X_train))



X_test = pd.DataFrame(mm_scaler.transform(X_test))
def conf_matrix(matrix,pred):

    class_names= [0,1]# name  of classes

    fig, ax = plt.subplots()

    tick_marks = np.arange(len(class_names))

    plt.xticks(tick_marks, class_names)

    plt.yticks(tick_marks, class_names)

    # create heatmap

    sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')

    ax.xaxis.set_label_position("top")

    plt.tight_layout()

    plt.title('Confusion matrix', y=1.1)

    plt.ylabel('Actual label')

    plt.xlabel('Predicted label')

    plt.show()
# Random Forest Classification

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(class_weight="balanced",n_estimators=200,random_state = 1)

rf.fit(X_train, y_train.values.ravel())

y_pred=rf.predict(X_test)

acc = metrics.accuracy_score(y_pred,y_test.values.ravel())*100

print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(acc))
# make class predictions with the model

y_pred = rf.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_pred,y_test,normalize='true')

conf_matrix(cnf_matrix,y_test)

# calculate prediction

report = classification_report(y_pred,y_test)

print(report)
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train, y_train.values.ravel())



y_pred=nb.predict(X_test)

acc = metrics.accuracy_score(y_pred,y_test.values.ravel())*100



print("Accuracy of Naive Bayes: {:.2f}%".format(acc))
# make class predictions with the model

y_pred = nb.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_pred,y_test,normalize='true')

conf_matrix(cnf_matrix,y_test)

# calculate prediction

report = classification_report(y_pred,y_test)

print(report)
from sklearn.svm import SVC

svm = SVC(random_state = 1)

svm.fit(X_train, y_train.values.ravel())



y_pred=svm.predict(X_test)

acc = metrics.accuracy_score(y_pred,y_test.values.ravel())*100



print("Test Accuracy of SVM Algorithm: {:.2f}%".format(acc))
# make class predictions with the model

y_pred = svm.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_pred,y_test,normalize='true')

conf_matrix(cnf_matrix,y_test)

# calculate prediction

report = classification_report(y_pred,y_test)

print(report)
# KNN Model

from sklearn.neighbors import KNeighborsClassifier



# try ro find best k value

score = []



for i in range(1,20):

    knn = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k

    knn.fit(X_train, y_train.values.ravel())

    score.append(knn.score(X_test, y_test.values.ravel()))

    

plt.plot(range(1,20), score)

plt.xticks(np.arange(1,20,1))

plt.xlabel("K neighbors")

plt.ylabel("Score")

plt.show()



acc = max(score)*100

print("Maximum KNN Score is {:.2f}%".format(acc))
knn = KNeighborsClassifier(n_neighbors =1)  # n_neighbors means k

knn.fit(X_train, y_train.values.ravel())  
# make class predictions with the model

y_pred = knn.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_pred,y_test,normalize='true')

conf_matrix(cnf_matrix,y_test)

# calculate prediction

report = classification_report(y_pred,y_test)

print(report)
import xgboost as xgb

xgbo=xgb.XGBClassifier(random_state=42,learning_rate=0.01)

xgbo.fit(X_train, y_train.values.ravel())
# make class predictions with the model

y_pred = xgbo.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_pred,y_test.values.ravel(),normalize='true')

conf_matrix(cnf_matrix,y_test)

# calculate prediction

report = classification_report(y_pred,y_test.values.ravel())

print(report)
from sklearn.ensemble import VotingClassifier



eclf1 = VotingClassifier(estimators=[('knn', knn),('rf', rf),('nb', nb)],

                         voting='hard')



eclf1 = eclf1.fit(X_train, y_train)

print(eclf1.predict(X_test))
# make class predictions with the model

y_pred = eclf1.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_pred,y_test,normalize='true')

conf_matrix(cnf_matrix,y_test)

# calculate prediction

report = classification_report(y_pred,y_test)

print(report)