import pandas as pd

df = pd.read_csv('../input/xAPI-Edu-Data/xAPI-Edu-Data.csv')

df
df.describe()
df.shape
df.isnull().sum()
df.dtypes
df.select_dtypes(exclude=['object'])
df.select_dtypes(include=['object']).columns
from sklearn.preprocessing import StandardScaler,LabelEncoder

label=LabelEncoder()

def encode_labels(df,labels_to_encode):

    for column in labels_to_encode:

        df[column] = label.fit_transform(df[column])

    return df

df_labelled = encode_labels(df,['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID',

       'SectionID', 'Topic', 'Semester', 'Relation', 'ParentAnsweringSurvey',

       'ParentschoolSatisfaction', 'StudentAbsenceDays'])
df_labelled
df_labelled.hist(figsize = (20,20))
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
#considering X as independent variable and y as target variable

X = pd.DataFrame(sc_X.fit_transform(df_labelled.drop(["Class"],axis = 1),),columns=['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID',

       'SectionID', 'Topic', 'Semester', 'Relation','raisedhands','VisITedResources','AnnouncementsView','Discussion', 'ParentAnsweringSurvey',

       'ParentschoolSatisfaction', 'StudentAbsenceDays'])
X
X.head()
y=df_labelled.Class
#train_test_split is used to split data into training and testing

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=42, stratify=y)
#model_KNN

from sklearn.neighbors import KNeighborsClassifier

test_scores = []

train_scores = []

for i in range(1,15):

    knn = KNeighborsClassifier(i)

    knn.fit(X_train,y_train)

    train_scores.append(knn.score(X_train,y_train))

    test_scores.append(knn.score(X_test,y_test))
max_train_score = max(train_scores)

train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]

print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))
max_test_score = max(test_scores)

test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]

print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))
import numpy as np

error_rate = []

for i in range(1,40):    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
#visulalization for error rate vs K value so that it makes easier to finalize on value of k

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

plt.figure(figsize=(12,5))

p = sns.lineplot(range(1,15),train_scores,marker='*',label='Train Score')

p = sns.lineplot(range(1,15),test_scores,marker='o',label='Test Score')
knn = KNeighborsClassifier(7)

knn.fit(X_train,y_train)

knn.score(X_test,y_test)
from sklearn.metrics import confusion_matrix

#let us get the predictions using the classifier we had fit above

y_pred = knn.predict(X_test)

confusion_matrix(y_test,y_pred)

pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
y_pred = knn.predict(X_test)

from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

a=knn.fit(X_train,y_train)

y_train_pred = cross_val_predict(a, X_train, y_train, cv=3)

confusion_matrix(y_train, y_train_pred)
from sklearn.metrics import accuracy_score

knn_pred = a.predict(X_test)

accuracy_score(y_test, knn_pred)