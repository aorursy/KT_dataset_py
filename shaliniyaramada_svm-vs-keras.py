#In this notebook, we will compare Ml model SVM with Neural networks using MNIST dataset
# Suppressing War

# Importing Pandas and NumPy

import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import numpy as np
# libraries

import pandas as pd

import numpy as np

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import scale



import matplotlib.pyplot as plt

import seaborn as sns

from PIL import Image

from pathlib import Path
from keras.models import Sequential

from keras.layers import Dense, Dropout, BatchNormalization, Activation

from keras import regularizers

# Importing all datasets



df = pd.read_csv('../input/digit-recognizer/train.csv')

#test = pd.read_csv('../input/test.csv')

# test = pd.read_csv("test.csv")

df.info()
#Selecting 20% of data as train and 80% as test 

train = df.iloc[0:8000,:]

test = df.iloc[8000:,:]

#Describe data

train.describe()

#checking nulls in train data

train.loc[:, train.isnull().sum() > 0].isnull().sum().sort_values()

#checking nulls in test data



test.loc[:, test.isnull().sum() > 0].isnull().sum().sort_values()

#shape of train and test

print(train.shape)

print(test.shape)
#seperating train and test sets before scaling, as we dont want to let test data learn from train



X_train = train.drop('label',axis = 1)

y_train = train["label"]

X_test = test.drop('label',axis= 1)

y_test = test["label"]

y_train
#reshape

visual_train =  X_train.values.reshape(-1,28,28,1)

#visualizng the data

for i in range(9):

    plt.subplot(330 + 1 + i)

    plt.imshow(visual_train[i][:,:,0], cmap=plt.get_cmap('gray'))

plt.show()
#Scaling

#using scale fron scikit-learn



X_train = scale(X_train)

X_test = scale(X_test)
#free memory of train and test dataframes

del train

del test
# linear model



model_linear = SVC(kernel='linear')

model_linear.fit(X_train, y_train)



# predict

y_pred = model_linear.predict(X_test)

# confusion matrix and accuracy



# accuracy

print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")



# cm

print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))
#for class-wise accuracy



from sklearn.metrics import classification_report

import numpy as np



# Y_test = np.argmax(y_test, axis=1) # Convert one-hot to index

report = classification_report(y_test, y_pred)

print(report)
##True positives

cm = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred)

TP = np.diag(cm)

TP



#max number for wrong prediction 

max(cm[cm != TP])

#just to see what numbers are incorrectly predicted

l = sorted(cm[cm != TP],reverse = True)[0:4]

print("Number of 3's predicted as 5:",l[0] )

print("Number of 4's predicted as 9:",l[1])

print("NUmber of 7's predicted as 9:",l[2])

print("Number of 9's predicted as 7:",l[3])
# non-linear model

# using rbf kernel, C=1, default value of gamma



# model

non_linear_model = SVC(kernel='rbf')



# fit

non_linear_model.fit(X_train, y_train)



# predict

y_pred = non_linear_model.predict(X_test)
# confusion matrix and accuracy



# accuracy

print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")



# cm

print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))
#for class accuracy



from sklearn.metrics import classification_report

import numpy as np



# Y_test = np.argmax(y_test, axis=1) # Convert one-hot to index

report = classification_report(y_test, y_pred)

print(report)
# creating a KFold object with 5 splits 

folds = KFold(n_splits = 5, shuffle = True, random_state = 101)



# Set the parameters by cross-validation

hyper_params = [ {'gamma': [1e-2,1e-3,1e-4],

                     'C': [1,10,100]}]



# specify model

model = SVC(kernel="rbf")



# set up GridSearchCV()

model_cv = GridSearchCV(estimator = model, 

                        param_grid = hyper_params, 

                        scoring= 'accuracy', 

                        cv = folds, 

                        verbose = 1,

                        return_train_score=True)      



# fit the model

model_cv.fit(X_train, y_train)                  

# cv results

cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results
# converting C to numeric type for plotting on x-axis

cv_results['param_C'] = cv_results['param_C'].astype('int')



# # plotting

plt.figure(figsize=(16,6))



# subplot 1/3

plt.subplot(131)

gamma_01 = cv_results[cv_results['param_gamma']==0.01]



plt.plot(gamma_01["param_C"], gamma_01["mean_test_score"])

plt.plot(gamma_01["param_C"], gamma_01["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.01")

plt.ylim([0.60, 1])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')



# subplot 2/3

plt.subplot(132)

gamma_001 = cv_results[cv_results['param_gamma']==0.001]



plt.plot(gamma_001["param_C"], gamma_001["mean_test_score"])

plt.plot(gamma_001["param_C"], gamma_001["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.001")

plt.ylim([0.60, 1])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')





# subplot 3/3

plt.subplot(133)

gamma_0001 = cv_results[cv_results['param_gamma']==0.0001]



plt.plot(gamma_0001["param_C"], gamma_0001["mean_test_score"])

plt.plot(gamma_0001["param_C"], gamma_0001["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.0001")

plt.ylim([0.60, 1])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')

# model with optimal hyperparameters



# model

model = SVC(C=10, gamma=0.001, kernel="rbf")



model.fit(X_train, y_train)

y_pred = model.predict(X_test)



# metrics

print("accuracy", metrics.accuracy_score(y_test, y_pred), "\n")

print(metrics.confusion_matrix(y_test, y_pred), "\n")



from sklearn.metrics import classification_report

import numpy as np



# Y_test = np.argmax(y_test, axis=1) # Convert one-hot to index

report = classification_report(y_test, y_pred)

print(report)
print("accuracy", metrics.accuracy_score(y_test, y_pred), "\n")

#keras
# create model

nn_model = Sequential()

nn_model.add(Dense(35, input_dim=784, activation='relu'))

nn_model.add(Dropout(0.3))

nn_model.add(Dense(21, activation = 'relu'))

nn_model.add(Dense(10, activation='softmax'))
#For NN y_train conversion
from keras.utils import np_utils

from keras.utils.np_utils import to_categorical





y_train = to_categorical(y_train, 10)

# y_val_10 = to_categorical(y_val, 10)
y_train
nn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
nn_model.fit(X_train,y_train, epochs=10, batch_size=10)
scores_train = nn_model.evaluate(X_train, y_train)

print("\n%s: %.2f%%" % (nn_model.metrics_names[1], scores_train[1]*100))
predictions = nn_model.predict(X_test)

predictions = np.argmax(predictions, axis = 1)

predictions
y_test = to_categorical(y_test, 10)
scores_test = nn_model.evaluate(X_test,y_test)

print("\n%s: %.2f%%" % (nn_model.metrics_names[1], scores_test[1]*100))
y_test