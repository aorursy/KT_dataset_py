import pandas as pd 

import numpy as np 

from sklearn.linear_model import LogisticRegression



df = pd.read_csv("../input/creditcard.csv")

df_orig = df.copy(deep = True)        # keep a copy for cross-validation
lrn = LogisticRegression(penalty = 'l2', C= 1, class_weight='balanced' )
def create_sets():

    df = df_orig.copy(deep = True)

    df = df.sample(frac=1).reset_index(drop=True)        #shuffle

    

    y = df.Class.tolist()

    df = df.drop('Class', 1)

    X = df.as_matrix()

    

    # create test and training set

    p = 0.2                      #fraction of test sample

    X_test = X[:int(p*len(y))]

    y_test = y[:int(p*len(y))]

    X_train = X[int(p*len(y)):]

    y_train = y[int(p*len(y)):]

    return X_test, y_test, X_train, y_train
def train_test():

    X_test, y_test, X_train, y_train = create_sets()

    

    lrn.fit(X_train, y_train)

    

    y_predict = lrn.predict(X_test)



    # count the errors:

    c_0 = 0

    c_1 = 0

    for i in range(len(y_test)):

        if (y_test[i] == 0) and (y_predict[i] == 1):

            c_0 += 1

        if (y_test[i] == 1) and (y_predict[i] == 0):

            c_1 += 1



    n_fraud = np.sum(y_test)

    return (100*c_0)/(len(y_test)-n_fraud), (100*c_1)/(n_fraud)
N = 10        #number of iterations

f_1 = 0      #counts the errors of the first kind (already in percent)

f_2 = 0      #counts the errors of the second kind

for n in range(N):

    a, b = train_test()

    f_1 += a

    f_2 += b



print("Error of first kind  = {}%".format(((10*f_1)//N)/10))

print("Error of second kind = {}%".format(((10*f_2)//N)/10))