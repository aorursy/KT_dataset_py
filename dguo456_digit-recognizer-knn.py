import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data_dir = "../input/"

# load csv files to numpy arrays
def load_data(data_dir, train_row):
    train = pd.read_csv(data_dir + "train.csv")
    print(train.shape)
#     print(train.head())
#     print(train[0:])
    X_train = train.values[0:train_row,1:] # feature columns
    y_train = train.values[0:train_row,0] # label column
    
    Pred_test = pd.read_csv(data_dir + "test.csv").values  # test data (ndarray)
#     print(Pred_test.shape)
#     print(pd.read_csv(data_dir + "test.csv").head())
    return X_train, y_train, Pred_test

train_row = 42000 # select 5000 samples from training dataset
Origin_X_train, Origin_y_train, Origin_y_test = load_data(data_dir, train_row)
print(Origin_X_train.shape, Origin_y_train.shape, Origin_y_test.shape)
print(Origin_X_train)
import matplotlib
import matplotlib.pyplot as plt
row = 3

print (Origin_y_train[row]) # show the row's digit

plt.imshow(Origin_X_train[row].reshape((28, 28)))
plt.show()
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
rows = 4

print(classes)
for y, cls in enumerate(classes):
    idxs = np.nonzero([i == y for i in Origin_y_train])
    idxs = np.random.choice(idxs[0], rows)
    for i , idx in enumerate(idxs):
        plt_idx = i * len(classes) + y + 1
        plt.subplot(rows, len(classes), plt_idx)
        plt.imshow(Origin_X_train[idx].reshape((28, 28)))
        plt.axis("off")
        if i == 0:
            plt.title(cls)
        
plt.show()
from sklearn.model_selection import train_test_split

X_train,X_test, y_train, y_test = train_test_split(Origin_X_train,
                                                   Origin_y_train,
                                                   test_size = 0.2,
                                                   random_state = 0)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
import time
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

ans_k = 0

k_range = range(1,8)
scores = []

# finding the best k value by enumerating from 1-7
for k in k_range:
    print("k = " + str(k) + " begin ")
    start = time.time()
    knn = KNeighborsClassifier(n_neighbors=k) # select KNN as our training model
    knn.fit(X_train,y_train) # train the model
    y_pred = knn.predict(X_test) # validate the model on validation dataset
    accuracy = accuracy_score(y_test,y_pred)
    scores.append(accuracy)
    end = time.time()
    print(classification_report(y_test, y_pred))  
    print(confusion_matrix(y_test, y_pred))  
    
    print("Complete time: " + str(end-start) + " Secs.")

print (scores)
plt.plot(k_range,scores)
plt.xlabel('Value of K')
plt.ylabel('Testing accuracy')
plt.show()
k = 3

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(Origin_X_train,Origin_y_train)
y_pred = knn.predict(Origin_y_test)
print (y_pred[200])
plt.imshow(Origin_y_test[200].reshape((28, 28)))
plt.show()
print(len(y_pred))

# save submission to csv
pd.DataFrame({"ImageId": list(range(1,len(y_pred)+1)),"Label": y_pred}).to_csv('Digit_Recogniser_Result.csv', index=False,header=True)