# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data_train = pd.read_csv('../input/train.csv')
(N_train, N_vars) = data_train.shape

X = np.ones((N_train, 7), dtype=float)

Y = np.zeros((N_train, 1))


for index, row in data_train.iterrows():

    X[index,0] = row['Pclass']

    X[index,1] = (1.0 if row['Sex']=='female' else 0.0)

    if np.isnan(row['Age']):

        X[index,2] = 0

    else:

        X[index,2] = row['Age']

    X[index,3] = row['SibSp']

    X[index,4] = row['Parch']

    X[index,5] = row['Fare']

    

    Y[index] = row['Survived']



print(X)

print(X.shape)
print("X = ", X.shape)

print("Y = ", Y.shape)
### Calculate theta parameters for LS fit

theta = np.dot( np.dot( np.linalg.pinv( np.dot(X.T, X) ), X.T ), Y )
print(theta)

print(theta.shape)
### Test model on training data

Y_train_pred = np.dot(X, theta)

Y_train_pred[(Y_train_pred > 0.5)] = 1

Y_train_pred[(Y_train_pred < 0.5)] = 0



err = Y_train_pred - Y

N_err = np.dot(err.T,err).item()
print("Wrong predicted = ", N_err, "out of ", N_train)

print("Accuracy = ", 1 - N_err / N_train)
### Load test data

data_test = pd.read_csv('../input/test.csv')
### Arrange test data into numpy array for prediction

(N_test, N_vars) = data_test.shape

X_t = np.ones((N_test, 7), dtype=float)

Y_t = np.zeros((N_test, 1))



rows_to_remove = []

for index, row in data_test.iterrows():

    X_t[index,0] = row['Pclass']

    X_t[index,1] = (1.0 if row['Sex']=='female' else 0.0)

    if np.isnan(row['Age']):

        X_t[index,2] = 0

    else:

        X_t[index,2] = row['Age']

    X_t[index,3] = row['SibSp']

    X_t[index,4] = row['Parch']

    if np.isnan(row['Fare']):

        X_t[index,5] = 0

    else:

        X_t[index,5] = row['Fare']



print(X_t)

print(X_t.shape)
### Predict test data

Y_test_pred = np.dot(X_t, theta)

Y_test_pred[(Y_test_pred > 0.5)] = 1

Y_test_pred[(Y_test_pred < 0.5)] = 0



print("Predited survived: ", Y_test_pred.sum(), " out of ", N_test)
### Save prediction results to file

with open('titanic_predictions.csv', 'w') as save_file:

    save_file.write('PassengerId,Survived\n')

    for index, row in data_test.iterrows():

        save_file.write(str(row['PassengerId']))

        save_file.write(',')

        save_file.write(str(int(Y_test_pred[index].item())))

        save_file.write('\n')



print("File witten successfully!")
### Read written file

with open('titanic_predictions.csv', 'r') as save_file:

    print(save_file.read())