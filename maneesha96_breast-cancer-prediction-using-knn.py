import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
bc = pd.read_csv(r'../input/data.csv')
bc.head(10)
bc.isnull().sum()
bc.dtypes
drop_cols = ['Unnamed: 32','id']
bc = bc.drop(drop_cols, axis = 1)
bc.shape
bc['diagnosis'] = bc['diagnosis'].map({'M': 1, 'B': 0})
bc.head()
bc['diagnosis'].value_counts()
# plotting the labels with the frequency 
Labels = ['Benign','Malignant']
classes = pd.value_counts(bc['diagnosis'], sort = True)
classes.plot(kind = 'bar', rot=0)
plt.title("Transaction class distribution")
plt.xticks(range(2), Labels)
plt.xlabel("Class")
plt.ylabel("Frequency")
# Plotting the features with each other.
groups = bc.groupby('diagnosis')
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.perimeter_mean, group.texture_mean, marker='o', ms=3.5, linestyle='', 
            label = 'Malignant' if name == 1 else 'Benign')
ax.legend()
plt.xlabel("perimeter_mean")
plt.ylabel("texture_mean")
plt.show()    

groups = bc.groupby('diagnosis')
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.radius_mean, group.texture_mean, marker='o', ms=3.5, linestyle='', 
            label = 'Malignant' if name == 1 else 'Benign')
ax.legend()
plt.ylabel("texture_mean")
plt.xlabel("radius_mean")
plt.show()    
groups = bc.groupby('diagnosis')
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.area_mean, group.texture_mean, marker='o', ms=3.5, linestyle='', 
            label = 'Malignant' if name == 1 else 'Benign')
ax.legend()
plt.ylabel("texture_mean")
plt.xlabel("area_mean")
plt.show()    
groups = bc.groupby('diagnosis')
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.smoothness_mean, group.compactness_mean, marker='o', ms=3.5, linestyle='', 
            label = 'Malignant' if name == 1 else 'Benign')
ax.legend()
plt.ylabel("compactness_mean")
plt.xlabel("smoothness_mean")
plt.show()   
import seaborn as sns
plt.figure(figsize=(12,12)) 
sns.heatmap(bc.corr(),annot=True,cmap='cubehelix_r') 
plt.show()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
bc_labels = pd.DataFrame(bc['diagnosis'])
bc_features = bc.drop(['diagnosis'], axis = 1)
X_train,X_test,y_train,y_test = train_test_split(bc_features,bc_labels,test_size=0.20)
#Performing cross validation
neighbors = []
cv_scores = []
from sklearn.model_selection import cross_val_score
#perform 10 fold cross validation
for k in range(1,10):
    neighbors.append(k)
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn,X_train,y_train,cv=10, scoring = 'accuracy')
    cv_scores.append(scores.mean())
    
#Misclassification error versus k
MSE = [1-x for x in cv_scores]

#determining the best k
optimal_k = neighbors[MSE.index(min(MSE))]
print('The optimal number of neighbors is %d ' %optimal_k)

#plot misclassification error versus k

plt.figure(figsize = (10,6))
plt.plot(neighbors, MSE)
plt.xlabel('Number of neighbors')
plt.ylabel('Misclassification Error')
plt.show()
knn=KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train,y_train)

from sklearn.metrics import classification_report
y_pred = knn.predict(X_test)
# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))

# Accuracy score
print('accuracy is',accuracy_score(y_pred,y_test))
#calculating confusion matrix for knn
tn,fp,fn,tp=confusion_matrix(y_test,y_pred).ravel()
print("K-Nearest Neighbours")
print("Confusion Matrix")
print("tn =",tn,"fp =",fp)
print("fn =",fn,"tp =",tp)
Labels = ['Benign','Malignant']
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=Labels, yticklabels=Labels, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()