import numpy as np

import pandas as pd



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.svm import SVR



import math
#import the initial dataset and shuffle

df = pd.read_csv("../input/zhao_data.csv")

# df = df.sample(frac=1).reset_index(drop=True)

df.head()
#split into X and y

X = df[['P', 'T', 'MW', 'Tb', 'Vc', 'Zc']]

print(X.head(), "\n")



y = df[['Viscosity']]

y = pd.DataFrame(np.asarray([math.log(i) for i in df[['Viscosity']].values]))

print(y.head(), "\n")





#split into train and test

X_train = X.iloc[:int(0.8*y.shape[0]), :]

scalerX = StandardScaler()

scalerX.fit(X)

X_train = pd.DataFrame(scalerX.transform(X_train))



X_test = X.iloc[int(0.8*y.shape[0]):, :]

X_test = pd.DataFrame(scalerX.transform(X_test))



y_train = y.iloc[:int(0.8*y.shape[0]), :]

scalerY = StandardScaler()

scalerY.fit(y)

y_train = pd.DataFrame(scalerY.transform(y_train))



y_test = y.iloc[int(0.8*y.shape[0]):, :]

y_test = pd.DataFrame(scalerY.transform(y_test))



print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
C_space = np.linspace(80, 150, num=10)

e_space = np.linspace(0.001, 0.01, num=10)



# C_space = [99.6]

# e_space = [0.0086]



best = {}

best['score'] = 0

best['C'] = 0

best['e'] = 0



for c in C_space:

    print(c)

    for e in e_space:

        rgs = SVR(verbose=0, C=c, epsilon=e, gamma='scale')

        rgs.fit(X_train.values,y_train.values.ravel())

        score = rgs.score(X_test.values, y_test.values.ravel())

        

        if score > best['score']:

            best['score'] = score

            best['C'] = c

            best['e'] = e    
best