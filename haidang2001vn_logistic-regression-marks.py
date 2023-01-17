import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
# load the data from the file



# Path: Duong dan chua file dataset

!ls ..

path ='..//input//marks.csv'   # thay doi duong dan thich hop

data = pd.read_csv(path, header=None)

print(data)
 # X = feature values, all the columns except the last column

X = data.iloc[:, :-1]

print(X.shape)



# y = target values, last column of the data frame

y = data.iloc[:, -1]

print(y.shape)



# filter out the applicants that got admitted

admitted = data.loc[y == 1]



# filter out the applicants that din't get admission

not_admitted = data.loc[y == 0]

# plots

plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')

plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10,label='Not Admitted')

plt.legend(['Admitted','Not Admitted'])

# preparing the data for building the model

X_new = np.c_[np.ones((X.shape[0], 1)), X] # them mot dau tien tat ca gia tri la 1

print(X_new.shape)

y_new = y[:, np.newaxis]

print(y.shape)

# Using scikit-learn

from sklearn import svm

model = svm.SVC(kernel='rbf')

# model = LogisticRegression(fit_intercept=False, solver='newton-cg', multi_class='multinomial')  # False for calculating the bias

model.fit(X_new, y_new)

# parameters = model.coef_

predicted_classes = model.predict(X_new)

accuracy = accuracy_score(y_new.flatten(),predicted_classes)

print('The accuracy score using scikit-learn is {}'.format(accuracy))

print("The model parameters using scikit learn")

print(parameters)

print("confusion_matrix")

print(confusion_matrix(predicted_classes,y_new))
# plotting the decision boundary

# As there are two features

# wo + w1x1 + w2x2 = 0

# x2 = - (wo + w1x1)/(w2)



x_values = [np.min(X_new[:, 1] - 2), np.max(X_new[:, 2] + 2)]

y_values = -(parameters[0][0] + np.dot(parameters[0][1], x_values)) / parameters[0][2]



# plots

plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')

plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10,label='Not Admitted')

plt.legend(['Admitted','Not Admitted'])

plt.plot(x_values, y_values, label='Decision Boundary')

plt.xlabel('Marks in 1st Exam')

plt.ylabel('Marks in 2nd Exam')

plt.legend()

plt.show()
# slipt train data and test data

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_new, y_new, test_size = 0.2)

print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)

print(Y_test.shape)

Train_admit = X_train[(Y_train[:]==1).flatten()]

print(Train_admit.shape)

Train_nonadmit = X_train[(Y_train[:]==0).flatten()]

print(Train_nonadmit.shape)

plt.scatter(Train_admit[:, 1], Train_admit[:, 2], s=10,label='Admitted')

plt.scatter(Train_nonadmit[:, 1], Train_nonadmit[:, 2], s=10,label='Not Admitted')

plt.legend()

plt.title("Trainning data")

# plt.legend(['Admitted','Not Admitted'])
model2 = LogisticRegression(fit_intercept=False, multi_class='multinomial', solver='newton-cg')

# model2 = svm.SVC(kernel='poly')

model2.fit(X_train, Y_train)

parameters = model2.coef_

predicted_classes = model2.predict(X_test)

accuracy = accuracy_score(predicted_classes,Y_test)

print('The accuracy score using scikit-learn is {}'.format(accuracy))

print("The model parameters using scikit learn")

print(parameters)

print("confusion_matrix")

print(confusion_matrix(predicted_classes,Y_test))
Test_admit = X_test[(Y_test[:]==1).flatten()]

print(Test_admit.shape)

Test_nonadmit = X_test[(Y_test[:]==0).flatten()]

print(Test_nonadmit.shape)

plt.scatter(Test_admit[:, 1], Test_admit[:, 2], s=10,label='Admitted')

plt.scatter(Test_nonadmit[:, 1], Test_nonadmit[:, 2], s=10,label='Not Admitted')

plt.legend()

plt.title("test data")
plt.scatter(Test_admit[:, 1], Test_admit[:, 2], s=10,label='Admitted')

plt.scatter(Test_nonadmit[:, 1], Test_nonadmit[:, 2], s=10,label='Not Admitted')

plt.legend()

x_values = [np.min(X_test[:, 1] - 20), np.max(X_test[:, 2] + 20)]

y_values = -(parameters[0][0] + np.dot(parameters[0][1], x_values)) / parameters[0][2]

# plots

plt.plot(x_values, y_values, label='Decision Boundary')

plt.xlabel('Marks in 1st Exam')

plt.ylabel('Marks in 2nd Exam')

plt.legend()

plt.show()
