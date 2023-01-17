import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from keras.models import Sequential

from keras.layers import Dense, Dropout, BatchNormalization

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, auc

import matplotlib.pyplot as plt

from keras import optimizers

from google.colab import drive
drive.mount('/content/drive/')
project_path = '/content/drive/My Drive/AIML Notes/Intro_to_NN/'
dataset_file = project_path + 'creditcard.csv'
data = pd.read_csv(dataset_file)
data.head()
data = data.drop("Time", axis = 1)
X_data = data.iloc[:, :-1]
X_data.shape
X_data.head()
y_data = data.iloc[:, -1]
y_data.shape
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2, random_state = 7)
X_train = preprocessing.normalize(X_train)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
model = Sequential()

model.add(Dense(64, input_shape = (29,), activation = 'relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(1, activation = 'sigmoid'))
sgd = optimizers.Adam(lr = 0.001)
model.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, batch_size = 700, epochs = 10, verbose = 1)
X_test = preprocessing.normalize(X_test)
results = model.evaluate(X_test, y_test)

print(model.metrics_names)

print(results)    
Y_pred_cls = model.predict_classes(X_test, batch_size=200, verbose=0)

print('Accuracy Model1 (Dropout): '+ str(model.evaluate(X_test,y_test)[1]))

print('Recall_score: ' + str(recall_score(y_test,Y_pred_cls)))

print('Precision_score: ' + str(precision_score(y_test, Y_pred_cls)))

print('F-score: ' + str(f1_score(y_test,Y_pred_cls)))

confusion_matrix(y_test, Y_pred_cls)
Y_pred_prob = model.predict_proba(X_test).ravel()



precision, recall, thresholds_pr = precision_recall_curve(y_test, Y_pred_prob)
AUC_PRcurve= auc(recall, precision)


plt.figure(1)

#plot PR curve

plt.plot(precision, recall, label = "AUC = {:0.2f}".format(AUC_PRcurve))

plt.xlabel('Precision', fontsize = 14)

plt.ylabel('Recall', fontsize = 14)

plt.title('Precision-Recall Curve', fontsize = 18)

plt.legend(loc='best')

plt.show()