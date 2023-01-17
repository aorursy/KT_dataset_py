import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('../input/heartdiseasecsv/Heartdisease.csv')
data.head()
data.info()
#checking whether the data is balanced or imbalanced

pd.value_counts(data['target'])


import seaborn as sns
sns.countplot(x='target',data=data)
correlation=data.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, annot = True)
data.isnull().sum()
#converting categorical variables into dummy variables
F_data = pd.get_dummies(data, columns=['sex','cp','fbs','restecg','exang','slope','thal'])
F_data.head()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

Columnstoscale = ['age','trestbps','chol','thalach']

F_data[Columnstoscale] = scaler.fit_transform(F_data[Columnstoscale])

F_data.head()
from sklearn.model_selection import train_test_split

X = F_data.drop(['target'], axis=1)
Y = F_data['target']

X_train,X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.30)
from sklearn.neighbors import KNeighborsClassifier

Classifier = KNeighborsClassifier(n_neighbors=5,metric='euclidean')
Classifier.fit(X_train,Y_train)
Y_pred = Classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score

print("Accuracy score:{}". format(accuracy_score(Y_pred,Y_test)))
print("CONFUSION MATRIX")
print(confusion_matrix(Y_pred,Y_test))
#finding out the best K value
from sklearn.model_selection import cross_val_score

knn_scores = []

for k in range(1,10):
    Knn_classifier = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    score = cross_val_score(Knn_classifier,X_train,Y_train,cv=10) 
    knn_scores.append(score.mean())
    
    
plt.plot([k for k in range(1,10)], knn_scores, color ='blue')

for i in range(1,10):
    plt.text(i, knn_scores[i-1], (i,knn_scores[i-1]))
    plt.xticks([i for i in range(1, 10)])
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Scores')
    plt.title('K Neighbors Classifier scores for different K values')
Best_Classifier = KNeighborsClassifier(n_neighbors=6,metric='euclidean')
Best_Classifier.fit(X_train,Y_train)
Y_pred = Best_Classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

print("Accuracy score:{}". format(accuracy_score(Y_pred,Y_test)))
print("CONFUSION MATRIX")
print(confusion_matrix(Y_pred,Y_test))
print('CLASSIFICATION REPORT')
print(classification_report(Y_pred,Y_test))
