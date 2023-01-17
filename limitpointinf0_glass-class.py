import numpy as np

import pandas as pd

from sklearn import preprocessing,neighbors,svm

from sklearn.model_selection import cross_val_score, train_test_split

import seaborn as sns

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.utils import to_categorical

from keras import optimizers



df = pd.read_csv("../input/glass.csv")

print(df.head())
corr = df.drop('Type',1).corr()

plt.figure(figsize=(20,10))

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True)

plt.title("Glass Correlation Heatmap")



plt.show()
for i in range(len(corr.index)):

    for j in range(len(corr.columns)):

        if (np.abs(corr.iloc[i,j]) >= 0.5) & (np.abs(corr.iloc[i,j]) !=1):

            print(corr.index.values[i],corr.columns[j],corr.iloc[i,j])

            plt.figure(i)

            x = list(df[corr.index.values[i]])

            y = list(df[corr.columns[j]])

            plt.scatter(x,y)

            plt.show()

            
X = np.array(df.drop(['Type'],1).astype(float))

X = preprocessing.scale(X)

y = np.array(df['Type'])



for i in range(1,5):

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    clf = neighbors.KNeighborsClassifier(n_neighbors=i, p=1)

    clf.fit(X_train,y_train)

    accuracy = clf.score(X_test,y_test)

    print('accuracy',accuracy)
X = np.array(df.drop(['Type'],1).astype(float))

X = preprocessing.scale(X)

y = np.array(df['Type'])



for i in range(1,5):

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    clf = svm.SVC(C=1e3)

    clf.fit(X_train,y_train)

    accuracy = clf.score(X_test,y_test)

    print(accuracy)
from keras.optimizers import SGD

from keras.callbacks import EarlyStopping



X = np.array(df.drop(['Type'],1).astype(float))

X = preprocessing.scale(X)

y = np.array(df['Type']-1)

y = to_categorical(y)



print(len(y[0]))

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)



in_shape = (X.shape[1],)



model = Sequential()

model.add(Dense(50, activation='relu', input_shape=in_shape))

model.add(Dropout(0.2))

model.add(Dense(50, activation='relu'))

model.add(Dropout(0.2))

# model.add(Dense(25, activation='relu'))

# model.add(Dropout(0.2))

# model.add(Dense(1000, activation='relu'))

# model.add(Dense(1000, activation='relu'))

model.add(Dense(7, activation='softmax'))



early_stopping = EarlyStopping(monitor='val_loss', patience=5)

model.compile(loss='categorical_crossentropy',optimizer=SGD(),metrics=['accuracy'])

hist = model.fit(X_train, y_train,epochs=1000, batch_size=5, callbacks=[early_stopping], verbose=1, validation_data=(X_train,y_train))
plt.plot(hist.history['loss'])

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.show()
score = model.evaluate(X_test,y_test,batch_size=1)



print(score)