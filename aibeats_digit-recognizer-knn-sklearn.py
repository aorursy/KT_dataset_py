import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
load_data_rows = 42000
def load_data(train_row):
    data_dir = '../input/'
    train = pd.read_csv(data_dir + 'train.csv')
    print(train.shape)
    X_train = train.values[0:train_row,1:] 
    y_train = train.values[0:train_row,0] 
    
    
    Pred_test = pd.read_csv(data_dir + 'test.csv').values  
    print(Pred_test.shape)
    return X_train, y_train, Pred_test

Origin_X_train, Origin_y_train, Origin_X_test = load_data(load_data_rows)
import matplotlib
import matplotlib.pyplot as plt

def plot_digit(X,y,row):
    print (y[row])
    plt.imshow(X[row].reshape((28, 28)))
    plt.show()

plot_digit(Origin_X_train,Origin_y_train, 205)
from sklearn.model_selection import train_test_split

X_train,X_vali, y_train, y_vali = train_test_split(Origin_X_train,
                                                   Origin_y_train,
                                                   test_size = 0.2,
                                                   random_state = 0)

print(X_train.shape, X_vali.shape, y_train.shape, y_vali.shape)
import time
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

ans_k = 0
ans_accuracy = 0
k_range = range(1, 8)
scores = []

for k in k_range:
    print("k = " + str(k) + " begin ")
    start = time.time()
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_vali)
    accuracy = accuracy_score(y_vali,y_pred)
    scores.append(accuracy)
    if accuracy > ans_accuracy:
        ans_accuracy = accuracy
        ans_k = k
    end = time.time()
    print(classification_report(y_vali, y_pred))  
    print(confusion_matrix(y_vali, y_pred))  
    
    print("Complete time: " + str(end-start) + " Secs.")
    
print('Finisged KNN training and the best k is' + str(ans_k))
print (scores)
plt.plot(k_range,scores)
plt.xlabel('Value of K')
plt.ylabel('Testing accuracy')
plt.show()
k = ans_k

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(Origin_X_train,Origin_y_train)
y_pred = knn.predict(Origin_X_test)
# change this line to y_pred = knn.predict(Origin_X_test) for full test
row = 1050
plot_digit(Origin_X_test, y_pred, row)
print(len(y_pred))

# save submission to csv
pd.DataFrame({"ImageId": list(range(1,len(y_pred)+1)),"Label": y_pred}).to_csv('Digit_Recognizer_Result.csv', index=False,header=True)
      
result = pd.read_csv('Digit_Recognizer_Result.csv')
    
result
result.head(20)