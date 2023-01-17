

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn import preprocessing

from sklearn.tree import DecisionTreeClassifier

from sklearn import tree

from sklearn import metrics
df=pd.read_csv('../input/heart-disease-uci/heart.csv')
df.T
columns=df.columns

columns_new=[]

for i in columns:

    columns_new.append(any(df[i].isnull()|df[i].isnull()))

df=df.drop(columns[columns_new],axis=1)
df.shape
sns.countplot(x="target", data=df)

plt.show()
pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(15,6),color=['blue','orange' ],alpha=0.7)

plt.title('Heart Disease Frequency for Sex')

plt.xlabel('Sex (0 = Female, 1 = Male)')

plt.xticks(rotation=0)

plt.legend(["Haven't Disease", "Have Disease"])

plt.ylabel('Frequency')

plt.show()
pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(20,6))

plt.title('Heart Disease Frequency for Ages')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.savefig('heartDiseaseAndAges.png')

plt.show()
ax = sns.boxplot( palette="Set2", orient="h",data=df[df.target==1])
ax = sns.boxplot( palette="Set2", orient="h",data=df[df.target==0])
from keras.utils import to_categorical

# one hot encode

encoded1 = pd.DataFrame(to_categorical(df.restecg),columns=['restecg0','restecg1','restecg2'])

encoded2 = pd.DataFrame(to_categorical(df.cp),columns=['cp0','cp1','cp2','cp3'])

encoded3 = pd.DataFrame(to_categorical(df.thal),columns=['thal0','thal1','thal2','thal3'])

encoded4 = pd.DataFrame(to_categorical(df.ca),columns=['ca0','ca1','ca2','ca3','ca4'])

encoded5 = pd.DataFrame(to_categorical(df.slope),columns=['slope0','slope1','slope2'])



df = pd.concat([df,encoded1,encoded2,encoded3,encoded4,encoded5],axis=1).drop(['restecg','cp','thal','ca','slope'],axis=1)
df.T
X_train, X_test, y_train, y_test=train_test_split(

    df.drop(['target'], axis=1),

    df[['target']],

    test_size=0.3,

    random_state=41)
print(X_train.shape)

print(X_test.shape)
for column in X_train.columns:

    

    df_train1 = X_train[(y_train.target==0) & (X_train[column]<np.mean(X_train.loc[y_train.target==0,column])+3*np.std(X_train.loc[y_train.target==0,column]))]

    df_test1 = X_test[(y_test.target==0) & (X_test[column]<np.mean(X_train.loc[y_train.target==0,column])+3*np.std(X_train.loc[y_train.target==0,column]))]

    

    label_train1 = y_train[(y_train.target==0) & (X_train[column]<np.mean(X_train.loc[y_train.target==0,column])+3*np.std(X_train.loc[y_train.target==0,column]))]

    label_test1 = y_test[(y_test.target==0) & (X_test[column]<np.mean(X_train.loc[y_train.target==0,column])+3*np.std(X_train.loc[y_train.target==0,column]))]

    

    df_train2 = X_train[(y_train.target==1) & (X_train[column]<np.mean(X_train.loc[y_train.target==1,column])+3*np.std(X_train.loc[y_train.target==1,column]))]

    df_test2 = X_test[(y_test.target==1) & (X_test[column]<np.mean(X_train.loc[y_train.target==1,column])+3*np.std(X_train.loc[y_train.target==1,column]))]

    

    label_train2 = y_train[(y_train.target==1) & (X_train[column]<np.mean(X_train.loc[y_train.target==1,column])+3*np.std(X_train.loc[y_train.target==1,column]))]

    label_test2 = y_test[(y_test.target==1) & (X_test[column]<np.mean(X_train.loc[y_train.target==1,column])+3*np.std(X_train.loc[y_train.target==1,column]))]
X_train=pd.concat([df_train1,df_train2])

y_train=pd.concat([label_train1,label_train2])



X_test=pd.concat([df_test1,df_test2])

y_test=pd.concat([label_test1,label_test2])
print(X_train.shape)

print(X_test.shape)
corrMatrix = X_train.corr()

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(corrMatrix, annot=True,ax=ax)

plt.show()
correlated_features = set()

for i in range(len(corrMatrix .columns)):

    for j in range(i):

        if abs(corrMatrix.iloc[i, j]) > 0.85:

            colname = corrMatrix.columns[i]

            correlated_features.add(colname)

print(correlated_features)
X_train.drop(labels=correlated_features, axis=1, inplace=True)

X_test.drop(labels=correlated_features, axis=1, inplace=True)
print(X_train.shape)

print(X_test.shape)
constant_filter = VarianceThreshold(threshold=0.0)

constant_filter.fit(X_train)

X_train = constant_filter.transform(X_train)

X_test = constant_filter.transform(X_test)



X_train.shape, X_test.shape
mm_scaler = preprocessing.StandardScaler()

X_train = pd.DataFrame(mm_scaler.fit_transform(X_train))

X_test=pd.DataFrame(mm_scaler.transform(X_test))
# Random Forest Classification

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(class_weight="balanced",n_estimators=200,random_state = 1)

rf.fit(X_train, y_train.values.ravel())

y_pred=rf.predict(X_test)

acc = metrics.accuracy_score(y_pred,y_test.values.ravel())*100

print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(acc))
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train, y_train.values.ravel())



y_pred=nb.predict(X_test)

acc = metrics.accuracy_score(y_pred,y_test.values.ravel())*100



print("Accuracy of Naive Bayes: {:.2f}%".format(acc))
from sklearn.svm import SVC

svm = SVC(random_state = 1)

svm.fit(X_train, y_train.values.ravel())



y_pred=svm.predict(X_test)

acc = metrics.accuracy_score(y_pred,y_test.values.ravel())*100



print("Test Accuracy of SVM Algorithm: {:.2f}%".format(acc))
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
knn = KNeighborsClassifier(n_neighbors = 7)  # n_neighbors means k

knn.fit(X_train, y_train.values.ravel())    
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression(max_iter=50)

logreg.fit(X_train, y_train.values.ravel())

y_pred=logreg.predict(X_test)

acc = metrics.accuracy_score(y_pred,y_test.values.ravel())*100

print("Test Accuracy of Logistic Regression Algorithm: {:.2f}%".format(acc))
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout



# define the keras model

model = Sequential()

model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(8, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

# compile the keras model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset

model.fit(X_train, y_train, epochs=100, batch_size=32)

# evaluate the keras model

_, accuracy = model.evaluate(X_test, y_test)
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
# make class predictions with the model

y_pred = rf.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_pred,y_test)

conf_matrix(cnf_matrix,y_test)

# calculate prediction

report = classification_report(y_pred,y_test)

print(report)
# make class predictions with the model

y_pred = nb.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_pred,y_test)

conf_matrix(cnf_matrix,y_test)

# calculate prediction

report = classification_report(y_pred,y_test)

print(report)
# make class predictions with the model

y_pred = svm.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_pred,y_test)

conf_matrix(cnf_matrix,y_test)

# calculate prediction

report = classification_report(y_pred,y_test)

print(report)
# make class predictions with the model

y_pred = knn.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_pred,y_test)

conf_matrix(cnf_matrix,y_test)

# calculate prediction

report = classification_report(y_pred,y_test)

print(report)
# make class predictions with the model

y_pred = model.predict_classes(X_test)

cnf_matrix = metrics.confusion_matrix(y_pred,y_test)

conf_matrix(cnf_matrix,y_test)

report = classification_report(y_pred,y_test)

print(report)