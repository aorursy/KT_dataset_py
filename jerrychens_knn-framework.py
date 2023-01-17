# Import Library
import numpy as np  # arrays
import pandas as pd  # read csv
import matplotlib.pyplot as plt  # plot
import time  # system time

from sklearn.model_selection import train_test_split  # split the train-test set of training data
from sklearn.neighbors import KNeighborsClassifier # KNN library
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # training performace
# directory of the *.csv
data_dir = '../input/'

# a function to load csv files into numpy arrays
def load_data(data_dir, train_row):
    # read train data
    train = pd.read_csv(data_dir + "train.csv") # load file as a DataFrame
    
    # print("Training data")
    # print(train.shape) # print the size of training data
    # print(train.head()) # print the structure of the data - first 5 rows
    
    # Convert to np array
    img_train = train.values[0:train_row, 1:] # pd.values returns a Numpy representation of the DataFrame.
    ans_train = train.values[0:train_row, 0]

    # read test data
    test = pd.read_csv(data_dir + 'test.csv')
    
    # Convert to np array
    img_test = test.values
    
    return img_train, ans_train, img_test

train_row = 5000 # Train part of the data, total is 42,000
data_img_train, data_ans_train, data_img_test = load_data(data_dir, train_row) # call load_data function to load data
# Test how many data in the training set is 5
np.sum(data_ans_train == 5)
# Shape of the data
print(data_img_train.shape, data_ans_train.shape, data_img_test.shape)
# check a certain "row" in "train_row"
row = 6 
print(data_ans_train[row]) # print the answer (label)
plt.imshow(data_img_train[row].reshape(28,28)) # show the image
plt.show()
# display what the handwritings look like
classes = ["0","1","2","3","4","5","6","7","8","9"] # list of classes 

display_rows = 4

for index, value in enumerate(classes):
    matched_index = np.nonzero([i == index for i in data_ans_train]) # array of matched index. index of non-zero is at first element: True = 1, False = 0
    random_matched_index = np.random.choice(matched_index[0], display_rows) # generate a uniform random sample from "matched_index[0]" of size "rows"
    for index2, value2 in enumerate(random_matched_index):
        plt_index = index2 * len(classes) + index + 1 
        plt.subplot(display_rows, len(classes), plt_index)
        plt.imshow(data_img_train[value2].reshape(28,28))
        plt.axis("off")
        if index2 == 0:
            plt.title(value)
plt.show()
data_img_train, data_img_vali, data_ans_train, data_ans_vali = train_test_split(data_img_train, data_ans_train, test_size=0.2, random_state = 0)
print(data_img_train.shape, data_img_vali.shape, data_ans_train.shape, data_ans_vali.shape)

k_range = range(1, 8) # try k-th nearest neighbor
scores = [] # placeholder for scores

for k in k_range:
    print('k = ' + str(k))
    
    start = time.time()
    
    # Using 80% training data to train a model by KNN
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(data_img_train, data_ans_train)
    
    end = time.time()
    
    # Using 20% training data to validate the results
    data_ans_pred = knn.predict(data_img_vali)
    
    accuracy = accuracy_score(data_ans_vali, data_ans_pred)
    scores.append(accuracy)
    
    print(classification_report(data_ans_vali, data_ans_pred))
    print(confusion_matrix(data_ans_vali, data_ans_pred))
    
    print('Training time ' + str(end - start) + ' secs.')
print(scores)
print (scores)
plt.plot(k_range,scores)
plt.xlabel('K')
plt.ylabel('Testing accuracy')
plt.show()
k_best = 3

# Using 80% training data to train a model by KNN
knn_best = KNeighborsClassifier(n_neighbors = k_best)
knn_best.fit(data_img_train, data_ans_train)

# Using 20% training data to validate the results
test_sample_size = 300
knn_best_pred = knn_best.predict(data_img_test[:test_sample_size])

# Validate
x = np.random.choice(test_sample_size, 1)
random_x = int(x[0]) # convert numpy array (choice) to integer
print ("Prediction: " + str(knn_best_pred[random_x]))
plt.imshow(data_img_test[random_x].reshape((28, 28))) # actual image
plt.show()
pd.DataFrame({"ImageId": list(range(1,len(knn_best_pred)+1)),
              "Label": knn_best_pred}).to_csv('Digit_Recogniser_Result.csv',index=False,header=True)

