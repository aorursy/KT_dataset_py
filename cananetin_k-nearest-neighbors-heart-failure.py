#import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
#read file

heart_fail = pd.read_csv("../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")

heart_fail.head()
#check out the data set

heart_fail.info()
heart_fail.columns
print("Proportion of death occurence: ",np.round(heart_fail[heart_fail["DEATH_EVENT"]==1].shape[0]/heart_fail.shape[0],decimals=2)*100,"%")
X = heart_fail[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',

       'ejection_fraction', 'high_blood_pressure', 'platelets',

       'serum_creatinine', 'serum_sodium', 'sex',"time", 'smoking']]



y = heart_fail[["DEATH_EVENT"]]
from sklearn import preprocessing



X = preprocessing.StandardScaler().fit(X).transform(X)

X[:3]
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
from sklearn.neighbors import KNeighborsClassifier



k = 4

neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train,np.ravel(y_train))

yhat = neigh.predict(X_test)
from sklearn import metrics

print("Train set's accuracy: ", metrics.accuracy_score(y_train,neigh.predict(X_train)))

print("Test set's accuracy: ", metrics.accuracy_score(y_test,yhat))
# What about K=6?



k6 = 6

neigh6 = KNeighborsClassifier(n_neighbors=k6).fit(X_train,np.ravel(y_train))

y_hat6 = neigh6.predict(X_test)



print("Train set's accuracy: ", metrics.accuracy_score(y_train,neigh.predict(X_train)))

print("Test set's accuracy: ", metrics.accuracy_score(y_test,y_hat6))
test_scores = []

train_scores = []



for i in range(1,15):



    knn = KNeighborsClassifier(i)

    knn.fit(X_train,np.ravel(y_train))

    

    train_scores.append(knn.score(X_train,np.ravel(y_train)))

    test_scores.append(knn.score(X_test,y_test))

    

max_train_score = max(train_scores)

train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]

print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))
max_test_score = max(test_scores)

test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]

print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))
k2 = 2

neigh2 = KNeighborsClassifier(n_neighbors=k2).fit(X_train,np.ravel(y_train))

y_hat2 = neigh2.predict(X_test)



print("Train set's accuracy: ", metrics.accuracy_score(y_train,neigh.predict(X_train)))

print("Test set's accuracy: ", metrics.accuracy_score(y_test,y_hat2))
import seaborn as sns

plt.figure(figsize=(12,8))

p = sns.lineplot(range(1,15),train_scores,marker='*',label='Train Score')

p = sns.lineplot(range(1,15),test_scores,marker='o',label='Test Score')
from sklearn.metrics import confusion_matrix

con_mat = confusion_matrix(y_test, y_hat2)

con_mat
import math

total_accuracy = (con_mat[0, 0] + con_mat[1, 1]) / float(np.sum(con_mat))

class1_accuracy = (con_mat[0, 0] / float(np.sum(con_mat[0, :])))

class2_accuracy = (con_mat[1, 1] / float(np.sum(con_mat[1, :])))

print(con_mat)

print('Total accuracy: %.5f' % total_accuracy)

print('Class1 accuracy: %.5f' % class1_accuracy)

print('Class2 accuracy: %.5f' % class2_accuracy)

print('Geometric mean accuracy: %.5f' % math.sqrt((class1_accuracy * class2_accuracy)))
from sklearn.metrics import classification_report

print(classification_report(y_test, y_hat2))
import seaborn as sns 



fig = plt.figure(figsize=(10,8))

ax= plt.subplot()

sns.heatmap(con_mat, annot=True, ax = ax)



# labels, title and ticks

ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')

ax.set_title('Confusion Matrix')

ax.xaxis.set_ticklabels(['not-dead', 'dead']); ax.yaxis.set_ticklabels(['not-dead', 'dead'])

plt.show()