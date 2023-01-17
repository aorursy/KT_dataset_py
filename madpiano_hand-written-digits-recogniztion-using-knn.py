import numpy as np # needed for array
import pandas as pd # needed for reading CSV
import matplotlib.pyplot as plt # for common graph plotting
data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')

print(data_train.shape)
print(data_test.shape)


X = data_train.values[0:5000, 1:] # us 5000 to test the code
y = data_train.values[0:5000, 0]
X_data_test = data_test.values

print(X.shape)
print(y.shape)
print(X_data_test.shape)


print(y[20])

plt.imshow(X[20].reshape((28, 28)))
plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

#neigh = KNeighborsClassifier(n_neighbors=3)
#neigh.fit(X, y) 
range_k = range(1, 7)
scores = []
best_k = 0

for k in range_k:
    
    start = time.time()
    print('k = ', str(k), 'begin')
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    scores.append(accuracy)
    end = time.time()
    
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print('total time used is ', str(end - start), 'secs.')
    
#best_k = max(scores)
print('best k is ' )
    


plt.plot(range_k, scores)
plt.xlabel('range of k')
plt.ylabel('accuracy scores')
plt.show()

#start = time.time()
#print('k = ', str(k), 'begin')
knn = KNeighborsClassifier(n_neighbors = 4)
knn.fit(X_train, y_train)

y_pred2 = knn.predict(X_data_test)

#accuracy = accuracy_score(y_test, y_pred)
#scores.append(accuracy)
#end = time.time()

#print(classification_report(y_test, y_pred))
#print(confusion_matrix(y_test, y_pred))
#print('total time used is ', str(end - start), 'secs.')
#does it predict correctly for a certain number?
print(y_pred2[100])

plt.imshow(X_data_test[100].reshape((28,28)))
plt.show()

print(len(y_pred))

# save submission to csv
pd.DataFrame({"ImageId": list(range(1,len(y_pred)+1)),"Label": y_pred}).to_csv('Digit_Recogniser_Result.csv', index=False,header=True)

