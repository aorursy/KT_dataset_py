import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
class KNN():
    def fit(self, X, y):
        self.dataset = X
        self.labels = y


    ########core-start###########   
    def predict(self, X, k=3):
        dists = np.sqrt(np.square(self.dataset-X).sum(axis=1))
        dist_idxes = dists.argsort()
        
        label_cnt = {}
        for i in range(k):
            label = self.labels[dist_idxes[i]]
            label_cnt[label] = label_cnt.get(label,0) + 1
        
        ans = max(label_cnt, key=label_cnt.get)
        return(ans)
     ########core-end###########  
def test():
    knn = KNN()
    X = np.array(range(12)).reshape(3,4)
    y = np.array(range(4))
    knn.fit(X,y)
    _X = np.array(range(4))
    _y = knn.predict(_X)
    print(_y == 0)
    
    _X = np.array(range(4,8))
    _y = knn.predict(_X)
    print(_y == 1)

test()
load_data_rows = 300
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

ans_k = 0
ans_accuracy = 0
k_range = range(1, 8)


knn = KNN()
knn.fit(X_train,y_train)

        
for k in k_range:
    print("k = " + str(k) + " begin ")
    
    start = time.time()
    y_pred = []
    for i in range(X_vali.shape[0]):
        pred = knn.predict(X_vali[i], k)
        y_pred.append(pred)
        
    accuracy = accuracy_score(y_vali,y_pred)
    print ('k = '+ str(k) , ' accuracy =' + str(accuracy))        
    if accuracy > ans_accuracy:
        ans_accuracy = accuracy
        ans_k = k
    end = time.time()
    print(classification_report(y_vali, y_pred))  
    print(confusion_matrix(y_vali, y_pred))  

    print("Complete time: " + str(end-start) + " Secs.")
    
print('Finisged KNN training and the best k is ' + str(ans_k))
k = ans_k

knn = KNN()
knn.fit(Origin_X_train,Origin_y_train)
y_pred = knn.predict(Origin_X_test,k)
row = 1050
plot_digit(Origin_X_test, y_pred, row)
print(len(y_pred))

# save submission to csv
pd.DataFrame({"ImageId": list(range(1,len(y_pred)+1)),"Label": y_pred}).to_csv('Digit_Recognizer_Result.csv', index=False,header=True)
      
result = pd.read_csv('Digit_Recognizer_Result.csv')
    
result
result.head(20)