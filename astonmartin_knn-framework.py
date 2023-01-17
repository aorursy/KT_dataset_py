import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data_dir="../input/"

def load_data(data_dir, train_row):
    train=pd.read_csv(data_dir + "train.csv")
    print(train.shape)
    X_train=train.values[0:train_row, 1:]
    y_train=train.values[0:train_row, 0]
    
    test=pd.read_csv(data_dir + "test.csv").values
    
    return X_train, y_train, test

train_row = 5000
O_X_train, O_y_train, O_test=load_data(data_dir, train_row)
    
row = 15
print(O_y_train[row])
plt.imshow(O_X_train[row].reshape((28, 28)))
plt.show()
classes=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
rows=4
print(classes)

for y, cls in enumerate(classes):
    #print(type(y))
    print(cls, end=' ')
    idxs=np.nonzero([i==y for i in O_y_train])
    #print(idxs.shape)
    idxs=np.random.choice(idxs[0], rows)
    #print(idxs.shape)
    for i, idx in enumerate(idxs):
        plt_idx=i*len(classes) + y + 1
        plt.subplot(rows, len(classes), plt_idx)
        plt.imshow(O_X_train[idx].reshape((28, 28)))
        plt.axis("off")
        if i == 0:
            plt.title(cls)
plt.show()
        
from sklearn.model_selection import train_test_split
X_train, X_validation, y_train, y_validation=train_test_split(O_X_train, O_y_train, test_size=0.2, random_state=0)

print(X_train.shape, X_validation.shape, y_train.shape, y_validation.shape)
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

ans_k=0

k_range=range(1, 8)
scores=[]

for k in k_range:
    print("k=" + str(k) + " begin ")
    start=time.time()
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred=knn.predict(X_validation)
    accuracy=accuracy_score(y_validation, y_pred)
    scores.append(accuracy)
    end=time.time()
    print(classification_report(y_validation, y_pred))
    print(confusion_matrix(y_validation, y_pred))
    
    print("Complete time: "+str(end-start) + " Sec.")
print(scores)
plt.plot(k_range, scores)
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.show()
k=3
knn=KNeighborsClassifier(n_neighbors=k)
knn.fit(O_X_train, O_y_train)
y_pred=knn.predict(O_test)
print(len(y_pred))

pd.DataFrame({"ImageID": list(range(1, len(y_pred)+1)), "Label": y_pred}).to_csv("Digit_Recogniser_Result.csv",index=False, header=True)
