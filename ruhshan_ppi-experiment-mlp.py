import numpy as np

import pandas as pd

from sklearn import svm



from sklearn.model_selection import cross_val_score

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



df = pd.read_csv("../input/data_5000.csv", delimiter=";", header=None)

#shuffling the whole rows of whole set to mix positive and negatives

df = df.sample(frac=1,random_state=1)



#slicing X and y

##with Amino acid statistics

#X = df.iloc[:,0:9432]

##without amino acid statistics

X = df.iloc[:, 0:9413]

#normalizing X

#scaler = StandardScaler()

#scaler.fit(X)

#X = scaler.transform(X)

y = df.iloc[:,9432]

#subsetting training set and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.80,test_size=0.20)
#declaring classifier

clfn = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(1000,500), 

                     random_state=1, learning_rate='invscaling', 

                     verbose=True ,activation='relu')

#fitting model

clfn.fit(X_train, y_train)

#generating prediction for test set

pred = clfn.predict(X_test)

#calculating accuracy

score = accuracy_score(y_test, pred)

print("Achieved accuracy: {}%".format(score*100))