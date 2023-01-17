import numpy as np
np.random.seed(10)

import pandas as pd 

train = pd.DataFrame(pd.read_csv("../input/train.csv", index_col=[0], header=0))
test  = pd.DataFrame(pd.read_csv("../input/test.csv", index_col=[0], header=0))
display(train.head())
train.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)
test.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)
display(train.head())
train['Age'].fillna(train['Age'].mean(), inplace=True)
train['Fare'].fillna(train['Fare'].mean(), inplace=True)
train['SibSp'].fillna(-1, inplace=True)
train['Parch'].fillna(-1, inplace=True)

test['Age'].fillna(train['Age'].mean(), inplace=True)
test['Fare'].fillna(train['Fare'].mean(), inplace=True)
test['SibSp'].fillna(-1, inplace=True)
test['Parch'].fillna(-1, inplace=True)
train = pd.get_dummies(train, dummy_na=True, columns=['Pclass', 'Sex', 'Embarked']).astype(float)
test = pd.get_dummies(test, dummy_na=True, columns=['Pclass', 'Sex', 'Embarked']).astype(float)

display(train.head())
display(test.head())
X_train = train.drop(columns=["Survived"])[:-120]
y_train = train["Survived"][:-120]

X_val = train.drop(columns=["Survived"])[-120:]
y_val = train["Survived"][-120:]

X_test = test

print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_val: ",   X_val.shape)
print("y_val: ",   y_val.shape)
print("X_test: ",   X_test.shape)
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
model.summary()
import time

epochs = 1750
start_time = time.time()

history = model.fit(X_train.as_matrix(), y_train.as_matrix(), epochs=epochs, batch_size=32, 
                    validation_data=(X_val.as_matrix(), y_val.as_matrix()), verbose=0, shuffle=True)

print("Tempo gasto: %d segundos" % (time.time() - start_time), "\r\nÉpocas: %d" % (epochs))
import matplotlib.pyplot as plt
plt.plot(history.history['acc'], color="r")
plt.plot(history.history['val_acc'], color="g")
plt.title('Curva de Treinamento')
plt.ylabel('Acurácia')
plt.xlabel('Época')
plt.legend(['Treinamento', 'Validação'], loc='lower right')
plt.show()

plt.plot(history.history['loss'], color="r")
plt.plot(history.history['val_loss'], color="g")
plt.title('Curva de Treinamento')
plt.ylabel('Erro')
plt.xlabel('Época')
plt.legend(['Treinamento', 'Validação'], loc='upper left')
plt.show()
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns
import numpy as np

print("Acurácia no Dataset de Treinamento:", accuracy_score(y_train.as_matrix(), np.round(model.predict(X_train.as_matrix()))), "\r\n")

confusionMatrixDF = pd.DataFrame( confusion_matrix(y_train.as_matrix(), np.round(model.predict(X_train.as_matrix()))),
                                 index=('Sobrevivente', 'Vítima'), columns=('Sobrevivente', 'Vítima'))

heatmap = sns.heatmap(confusionMatrixDF, annot=True, fmt="d", cmap="Blues",  vmin=0)
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print("Acurácia no Dataset de Validação:", accuracy_score(y_val.as_matrix(), np.round(model.predict(X_val.as_matrix()))), "\r\n")

confusionMatrixDF = pd.DataFrame( confusion_matrix(y_val.as_matrix(), np.round(model.predict(X_val.as_matrix()))),
                                 index=('Sobrevivente', 'Vítima'), columns=('Sobrevivente', 'Vítima'))

heatmap = sns.heatmap(confusionMatrixDF, annot=True, fmt="d", cmap="Blues",  vmin=0)
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
y_test_pred = model.predict(X_test.as_matrix())

X_test_submission = X_test.copy()
X_test_submission['Survived'] = np.round(y_test_pred).astype(int)
X_test_submission['Survived'].to_csv('submission.csv', header=True)