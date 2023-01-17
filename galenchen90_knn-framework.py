import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
import matplotlib.pyplot as plt # 画图常用库

def load_data(data_dir, train_row):

    train = pd.read_csv(data_dir + "train.csv")
    print(train.shape)
    pred_test = pd.read_csv(data_dir + "test.csv").values
    

    x_train = train.values[0:train_row,1:]
    y_train = train.values[0:train_row,0]

  
    return x_train , y_train, pred_test

data_dire = "../input/"
train_rows=5000
origin_x_train,origin_y_train,origin_x_test = load_data(data_dire,train_rows)



# print(origin_x_train,origin_x_test,origin_y_train)
# row =224
# print(origin_y_train[row])
# plt.imshow(origin_x_train[row].reshape((28,28)))
# plt.show()

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
rows=4
print(classes)
for y, cls in enumerate(classes):
    idxs = np.nonzero([i==y for i in origin_y_train])
    idxs = np.random.choice(idxs[0],rows)
    for i , idx in enumerate(idxs):
        plt_idx =i*len(classes)+y+1
        plt.subplot(rows,len(classes),plt_idx)
        plt.imshow(origin_x_train[idx].reshape((28,28)))
        plt.axis("off")
        if i ==0 :
                   plt.title(cls)
plt.show


from sklearn.model_selection import train_test_split

x_train,x_vali,y_train,y_vali = train_test_split(origin_x_train,origin_y_train,test_size=0.2,
                                                random_state =0)
print(x_train.shape,x_vali.shape,y_train.shape,y_vali.shape)
import time
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

ans_k =0
k_range = range(1,8)
scores=[]

for k in k_range:
    print("k = " + str(k)+ " begin")
    start=time.time()
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    y_pred = knn.predict(x_vali)
    accuracy = accuracy_score (y_vali, y_pred)
    scores.append(accuracy)
    end = time.time()
    print(classification_report(y_vali,y_pred))
    print(confusion_matrix(y_vali,y_pred))
    print("Complete time: "+ str(end-start)+ " Secs")



print(scores)
plt.plot(k_range,scores)
plt.xlabel('value of k')
plt.ylabel('testing accuracy')
plt.show()
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(origin_x_train,origin_y_train)
y_pred = knn.predict(origin_x_test[:400])


test_row= 1
print (y_pred[test_row])
print(origin_x_test.shape)
plt.imshow(origin_x_test[test_row].reshape((28,28)))
plt.show()

print(len(y_pred))
pd.DataFrame({"ImageId": list(range(1,len(y_pred)+1)),"Label": y_pred}).to_csv(
    'Digit_Recogniser_Result.csv', index=False,header=True)

