import numpy as np

from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns



digits = load_digits()

print(digits.data.shape)



def show_nums(data=digits.data, target=digits.target, prediction=None, start=0, count=1, cols=5):

    plt.figure(figsize=(30,6))

    rows = (count/cols)

    if count%cols > 0:

        rows+=1

    for index, (image, label) in enumerate(zip(data[start:start+count], target[start:start+count])):

        plt.subplot(rows,cols,index+1)

        plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)

        title = "Number: {}".format(target[start+index])

        if prediction is not None:

            title+=", Prediction: {}".format(prediction[start+index])

        plt.title(title)



# explore data - show some sample digit images

show_nums(data=digits.data, target=digits.target, start=12, count=6, cols=6)
# split the data into training and test data and then build the LogisticRegression regression model.

# Using LogisticRegression as we have to classify the given digit's dataset into 10 classes (0 to 9 digits)

from sklearn.linear_model import LogisticRegression

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.33, random_state=42)

print("Shape of training dataset: {}".format(x_train.shape))

print("Shape of training labels: {}".format(y_train.shape))

print("Shape of test dataset: {}".format(x_test.shape))

print("Shape of test labels: {}".format(y_test.shape))



model = LogisticRegression()

model.fit(x_train, y_train)

predict = model.predict(x_test)

# calculate the accuracy score of model

print("Accuracy score of model: {} %".format(model.score(x_test, y_test)*100))
# print confusion matrix to know how many predictions were correct and incorrect, kind a detailed view for accuracy of model.

from sklearn import metrics



confusion_metrics = metrics.confusion_matrix(y_test, predict)

print(confusion_metrics)
#print numbers where prediction was wrong

mis_match = []

count=0

for indx, (target, prediction) in enumerate(zip(y_test, predict)):

    if target != prediction:

        count+=1

        mis_match.append(x_test[indx])

        mis_match.append(target)

        mis_match.append(prediction)

mis_match = np.reshape(mis_match, (count, 3))

show_nums(data= mis_match[:, 0], target=mis_match[:, 1], prediction=mis_match[:, 2], start=0, count=len(mis_match))