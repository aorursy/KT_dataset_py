# import libraries

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt
data_dir = ""

train_data = "../input/train.csv"



# load csv files to numpy arrays

def load_data(data_dir, input_file):

    data = pd.read_csv(data_dir + input_file)

    

#    print(train.shape)

    return(data)

   

train_data = load_data(data_dir, train_data)

X_train = train_data.values[:,1:] 

y_train = train_data.values[:,0] 
#Check the data

print(train_data.shape)

print(train_data[0:5])
print(X_train.shape, y_train.shape)
# Show the image of a single data

import matplotlib

import matplotlib.pyplot as plt

row = 4



print (y_train[row])



plt.imshow(X_train[row].reshape((28, 28)))

plt.show()
# Show the images of multiple rows data

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

rows = 5



print(classes)

for y, cls in enumerate(classes):

    idxs = np.nonzero([i == y for i in y_train])

    idxs = np.random.choice(idxs[0], rows)

    for i , idx in enumerate(idxs):

        plt_idx = i * len(classes) + y + 1

        plt.subplot(rows, len(classes), plt_idx)

        plt.imshow(X_train[idx].reshape((28, 28)))

        plt.axis("off")

        if i == 0:

            plt.title(cls)

        



plt.show()
# Split Training Dataset into two sets: train and validation

from sklearn.model_selection import train_test_split



X_train,X_vali, y_train, y_vali = train_test_split(X_train,

                                                   y_train,

                                                   test_size = 0.2,

                                                   random_state = 0)

# If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.



print(X_train.shape, X_vali.shape, y_train.shape, y_vali.shape)
#Build KNN model and run the train set and predict Y for the validation set

import time

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

from sklearn.neighbors import KNeighborsClassifier



ans_k = 0



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

    end = time.time()

    print(classification_report(y_vali, y_pred))  

    print(confusion_matrix(y_vali, y_pred))  

    

    print("Complete time: " + str(end-start) + " Secs.")
print (scores)

plt.plot(k_range,scores)

plt.xlabel('Value of K')

plt.ylabel('Testing accuracy')

plt.show()


for k, score in zip(k_range, scores):

    if score == min(scores):

        k_min=k

        break
#Input Test Dataset

test_data = pd.read_csv("../input/test.csv").values
import time

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

from sklearn.neighbors import KNeighborsClassifier



k = k_min



start = time.time()

knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(X_train,y_train)

y_pred = knn.predict(test_data)

end = time.time()



print("Complete time: " + str(end-start) + " Secs.")
# Test the prediction results

print (y_pred[200])



plt.imshow(test_data[200].reshape((28, 28)))

plt.show()
# Output csv



pd.DataFrame({"ImageId": list(range(1,len(y_pred)+1)),"Label": y_pred}).to_csv('Digit_Recogniser_Result.csv', index=False,header=True)