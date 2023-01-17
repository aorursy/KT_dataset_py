import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
dataset = pd.read_csv('../input/churn-predictions-personal/Churn_Predictions.csv')

X = dataset.iloc[:, 3:13]

y = dataset.iloc[:, 13]
geography=pd.get_dummies(X["Geography"],drop_first=True)

gender=pd.get_dummies(X['Gender'],drop_first=True)
## Concatenate the Data Frames

X=pd.concat([X,geography,gender],axis=1)

## Drop Unnecessary columns

X=X.drop(['Geography','Gender'],axis=1)
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
X.head()
import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LeakyReLU,PReLU,ELU

from keras.layers import Dropout

from keras.optimizers import Adam
classifier = Sequential()

classifier.add(Dense(units = 6,activation='relu',input_dim = 11))

classifier.add(Dense(units = 6,activation='relu'))

classifier.add(Dense(units = 1,  activation = 'sigmoid'))

classifier.compile(optimizer = Adam(learning_rate=0.01), loss = 'binary_crossentropy', metrics = ['accuracy'])

model_history=classifier.fit(X_train, y_train,validation_split=0.33, batch_size = 10, epochs = 20)
print(model_history.history.keys())

# summarize history for accuracy

plt.plot(model_history.history['accuracy'])

plt.plot(model_history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# summarize history for loss

plt.plot(model_history.history['loss'])

plt.plot(model_history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)



# Calculate the Accuracy

from sklearn.metrics import accuracy_score

score=accuracy_score(y_pred,y_test)
print(cm)
def plot_confusion_matrix(cm, classes):

    title='Confusion matrix',

    cmap=plt.cm.Blues

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)
cm_plot_labels = ['exited','not_exited']

plot_confusion_matrix(cm=cm, classes=cm_plot_labels)
!pip install jovian
import jovian
jovian.commit(project='keras-on-dataframe-02')