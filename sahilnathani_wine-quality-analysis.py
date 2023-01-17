import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns  
df = pd.read_csv("../input/winequality-red.csv", encoding='ISO-8859-1')
df.head()
df.columns
df.corr()
df['quality'].unique()
df = df.drop(['residual sugar', 'density'], 1)



z = []

for each in df['quality']:

    if each>=3 and each<7:

        z.append(0)

    else:

        z.append(1)



df['quality'] = z        



y = np.array(df['quality'])

y = y.reshape(-1, 1)

x = np.array(df.drop(['quality'], 1))



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=47)



from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB, MultinomialNB

from sklearn.neural_network import MLPClassifier



names = ['knn', 'scv', 'rfc', 'abc', 'dtc', 'gnb', 'mnb', 'mlp']



classifiers = [KNeighborsClassifier(3), SVC(C=0.009, gamma=3), RandomForestClassifier(n_estimators=50), AdaBoostClassifier(), 

               DecisionTreeClassifier(), GaussianNB(), MultinomialNB(), MLPClassifier()]



for n, c in zip(names, classifiers):

    c.fit(x_train, y_train)

    score = c.score(x_test, y_test)

    print('The accuracy achieved by classifier', n, 'is', score*100)

    

       
#Obtained after grid Serach

rfc = RandomForestClassifier(max_depth= 6,

 min_samples_leaf =3,

 min_samples_split= 4,

 n_estimators= 50)

rfc.fit(x_train, y_train)

accuracy = rfc.score(x_test, y_test)

print(accuracy*100)
for each in df.columns:

    plt.figure(figsize=(6, 6))

    plt.xlabel(each)

    df[each].hist()
from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras import Sequential



model = Sequential()

model.add(Dense(12, input_dim=9, kernel_initializer='normal', activation='relu'))



model.add(Dense(8, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.summary()



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



history = model.fit(x_train, y_train, epochs=100, batch_size=1000, shuffle=True, verbose=1,

                    validation_split=0.3)

print(history.history.keys())

# "Loss"

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['loss', 'val_loss'], loc='upper left')

plt.show()
y_pred