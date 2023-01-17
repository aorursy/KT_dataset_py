import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
import matplotlib.pyplot as plt # 画图常用库
data_dir = '../input/'
def data_input(data_dir,train_row):
    Org_Train_data = pd.read_csv(data_dir+'train.csv')
    Pred_Test = pd.read_csv(data_dir+'test.csv').values
    train_X = Org_Train_data.values[0:train_row,1:]
    train_Y = Org_Train_data.values[0:train_row,0]
    return train_X,train_Y,Pred_Test
train_row = 400 
Origin_X_train, Origin_y_train, Origin_X_test = data_input(data_dir, train_row)

    
plt.imshow(Origin_X_train[4,:].reshape(28,28))
plt.show()
rows = 3
classes = ['1','2','3','4','5','6','7','8','9']
for y,clss in enumerate(classes):
    idx = np.nonzero([i==y for i in  Origin_y_train])
    idx = np.random.choice(idx[0],rows)
    for i,idx in enumerate(idx):
        plot_index = y+1+len(classes)*i
        plt.subplot(rows, len(classes), plot_index)
        plt.imshow(Origin_X_train[idx].reshape(28,28))
        plt.axis("off")
plt.show()
        
    
    
from sklearn.model_selection import train_test_split
X_train,X_vali,Y_train,Y_vali = train_test_split(Origin_X_train,Origin_y_train,test_size = 0.2, random_state = 0)
import time
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

k_range = range(1,8)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train,Y_train)
    Y_pred = knn.predict(X_vali)
    accuracy = accuracy_score(Y_vali,Y_pred)
    scores.append(accuracy)
    print(classification_report(Y_vali,Y_pred))
    print(confusion_matrix(Y_vali,Y_pred)) 

    k=scores.index(max(scores))+1
    print (k)
    
  
plt.plot(k_range,scores)
plt.xlabel('k value')
plt.ylabel('accuracy')
plt.show()
knn = KNeighborsClassifier(n_neighbors = k)
knn.fit(Origin_X_train,Origin_y_train)
y_pred = knn.predict(Origin_X_test[:500])
print (y_pred[200])
plt.imshow(Origin_X_test[200].reshape(28,28))
plt.show()
print(len(y_pred))
pd.DataFrame({"ImageId": list(range(1,len(y_pred)+1)),"Label": y_pred}).to_csv('Digit_Recogniser_Result.csv', index=False,header=True)
