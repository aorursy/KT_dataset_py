import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import collections 
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve


# load data from file into pandas dataframe
data=pd.read_table('/kaggle/input/fruits-with-colors-dataset/fruit_data_with_colors.txt')

# show top 5
print(data.shape)
data.head()


y = data["fruit_label"]

X = data.drop(['fruit_name', 'fruit_subtype', "fruit_label"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=81)
best_means = collections.deque(5*[0], 5)
best_k = []
k_range = range(1, 30)
k_scores = []

start = time.time()

for i in k_range:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    mean=knn.score(X_test,y_test)
    k_scores.append(mean)
    #mean=cross_val_score(knn, X_train, y_train, cv=10).mean()
    if mean > min(best_means): 
        best_means.append(mean)
        best_k.append(i)
        
end = time.time()

print(best_means)
print(best_k)
print((end - start)/60)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Score')
plt.show()
knn = KNeighborsClassifier(n_neighbors=23)
knn.fit(X_train_reduced.T, y_train)
XX, YY = np.meshgrid(np.linspace(-6, 5, 50), np.linspace(-3, 4, 50))
pos = np.vstack([XX.ravel(), YY.ravel()])
preds = knn.predict(pos.T)
plt.scatter(pos[0], pos[1],c=[colors[x-1] for x in preds])
plt.show()


lda = LinearDiscriminantAnalysis(n_components=2).fit(X_train, y_train)
X_train_reduced = lda.transform(X_train).T
X_test_reduced = lda.transform(X_test).T

best_means = collections.deque(5*[0], 5)
best_k = []
k_range = range(1, 30)
k_scores = []

start = time.time()

for i in range(1,30):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_reduced.T, y_train)
    mean=knn.score(X_test_reduced.T,y_test)
    k_scores.append(mean)
    #mean=cross_val_score(knn, X_train, y_train, cv=10).mean()
    if mean > min(best_means): 
        best_means.append(mean)
        best_k.append(i)
        
end = time.time()

print(best_means)
print(best_k)
print((end - start)/60)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train_reduced.T, y_train)
preds = knn.predict(X_test_reduced.T)
#print(preds)
#print("X_test_reduced")
#print(X_test_reduced.T)
#print("y_test")
#np.sort(preds)   
#print("preds")
#print(y_test)
plt.scatter(X_train_reduced[0],X_train_reduced[1],
            c=[colors[x-1] for x in y_train],
            marker='o')
plt.scatter(X_test_reduced[0], 
         X_test_reduced[1],
        c=[colors[x-1] for x in preds],
         marker='*', s=200)

plt.show()
XX, YY = np.meshgrid(np.linspace(-6, 5, 50), np.linspace(-3, 4, 50))
pos = np.vstack([XX.ravel(), YY.ravel()])
preds = knn.predict(pos.T)
plt.scatter(pos[0], pos[1],c=[colors[x-1] for x in preds])
plt.show()
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train_reduced.T, y_train)
mean=knn.score(X_test_reduced.T,y_test)

y_scores = knn.predict_proba(X_test_reduced.T)
print(y_scores)
fpr, tpr, threshold = roc_curve(y_test, y_scores[:, 1])
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of kNN')
plt.show()