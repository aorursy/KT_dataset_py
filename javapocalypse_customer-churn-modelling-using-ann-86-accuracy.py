import numpy as np

import pandas as pd





from numpy.random import seed

seed(1)

from tensorflow import set_random_seed

set_random_seed(1)





dataset = pd.read_csv('../input/Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values

y = dataset.iloc[:, 13].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder



labelencoder_X_1 = LabelEncoder()

X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()

X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])



onehotencoder = OneHotEncoder(categorical_features=[1])

X = onehotencoder.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from keras.models import Sequential

from keras.layers import Dense





def deep_model():

    classifier = Sequential()

    classifier.add(Dense(units=12, kernel_initializer='uniform', activation='relu', input_dim=12))

    classifier.add(Dense(units=12, kernel_initializer='uniform', activation='relu'))

    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

    classifier.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    return classifier
from keras.models import load_model



# classifier = load_model('Churn_Modelling.h5')
# Uncomment to train new model otherwise it will used trained model

classifier = deep_model()

classifier.fit(X_train, y_train, batch_size=4, epochs=128)





y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)



from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_test, y_pred)

print(cm)



accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])

print("Accuracy: "+ str(accuracy*100)+"%")
classifier.save('Churn_Modelling.h5')