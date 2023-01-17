# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import RobustScaler

from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
class AdalineGD(object):

    """ADAptive LInear NEuron Classifier.

    Parameters

    ----------

    eta: float

        learning rate (between 0.0 and 1.0)

    n_iter : int

        passes over the training dataset

    

    Attributes:

    -----------

    w_ : 1d-array

        Weights after fitting

    errors_

        number of misclassifications after each epoch

    """

    

    def __init__(self, eta=0.01, n_iter = 5000):

        self.eta = eta

        self.n_iter = n_iter

    

    def fit(self, X, y):

        """Fit Training Data

        Parameters:

        -----------

        X: {Array like} shape = [n_samples, n_features]

            training vectors,

            where n_samples is the number of samples and n_features is the number of features

            

        y: array-like, shape = [n_samples]

            target values

        

        returns: self : object

        """

        self.w_ = np.zeros(1 + X.shape[1])

        self.cost_ = []

        

        self.errors_ = 0

        

        for i in range(self.n_iter):

            output = self.net_input(X)

            self.errors_ = (y - output)

            

            self.w_[1:] += self.eta * X.T.dot(self.errors_)

            self.w_[0] = self.eta * self.errors_.sum()

            cost = (self.errors_**2).sum() / 2.0

            self.cost_.append(cost)

        return self

    

    def net_input(self, X):

        """Calculate net input"""

        #return np.dot(X, self.w_[1:]) + self.w_[0]

        output = X.dot(self.w_[1:]) + self.w_[0]

        return output

    

    def activation(self, X):

        """Calculate the linear activation weights

        later to be used in the redshift determination"""

        return self.net_input(X)

    

    def predict(self, X):

        """Return the class label after unite step"""

        activation_weights = self.activation(X)

        return [np.where(activation_weights > 0.0, 1, -1),activation_weights]
def mrs_married(x):

    if "Mrs" in x:

        return True

    else:

        return False



def mr_married(x):

    if "Mr" in x:

        return True

    else:

        return False

def miss(x):

    if "Miss" in x:

        return True

    if "Ms" in x:

        return True

    else:

        return False
df_titanic = pd.read_csv("../input/train.csv")
df_titanic = df_titanic.drop("Cabin",axis=1)

df_titanic = df_titanic.drop("Embarked",axis=1)

df_titanic = df_titanic.drop("Ticket",axis=1)



df_titanic['miss'] = df_titanic['Name'].map(miss)

df_titanic['mrmar'] = df_titanic['Name'].map(mr_married)

df_titanic['mrsmar'] = df_titanic['Name'].map(mrs_married)

df_titanic['sibs'] = df_titanic.SibSp + df_titanic.Parch

df_titanic = df_titanic.drop(['SibSp','Parch'],axis=1)

df_titanic = df_titanic.drop('Name',axis=1)



df_titanic = pd.get_dummies(df_titanic.drop("PassengerId",axis=1))
df_titanic.Fare = df_titanic.Fare.fillna(df_titanic.Fare.mean())

df_titanic.Age = df_titanic.Age.fillna(df_titanic.Age.mean())

rs = RobustScaler()

df_titanic.Fare = rs.fit_transform(df_titanic.Fare.reshape(-1,1))

df_titanic.Age = rs.fit_transform(df_titanic.Age.reshape(-1,1))
df_titanic = df_titanic.sort("Survived")



# Compute the indexes for where the sorted survival classes begin and end

did_not_survive_ix = df_titanic[df_titanic.Survived == 0].count()[0]

did_survive_ix = df_titanic[df_titanic.Survived == 1].count()[0]



# Set the tuples for the reference frame

did_survive_ref = (did_not_survive_ix, did_not_survive_ix + 50)

did_not_survive_ref = (0,50)



# Set the tuples for the train frame

did_survive_train = (did_not_survive_ix + 50, did_not_survive_ix + did_survive_ix)

did_not_survive_train = (50,did_not_survive_ix)



# Set the tuples for the test frame

did_survive_test = (did_not_survive_ix, did_not_survive_ix + 50)

did_not_survive_test = (did_not_survive_ix-50, did_not_survive_ix)



ref_df = df_titanic.iloc[did_not_survive_ref[0]:did_not_survive_ref[1]]

ref_df = ref_df.append(df_titanic.iloc[did_survive_ref[0]:did_survive_ref[1]])



train_df = df_titanic.iloc[did_not_survive_train[0]:did_not_survive_train[1]]

train_df = train_df.append(df_titanic.iloc[did_survive_train[0]:did_survive_train[1]])



test_df = df_titanic.iloc[did_not_survive_test[0]:did_not_survive_test[1]]

test_df = test_df.append(df_titanic.iloc[did_survive_test[0]:did_survive_test[1]])
ada = AdalineGD(eta=0.00001,n_iter=10000)
ada.fit(train_df.drop('Survived',axis=1).values.astype(float),train_df.Survived.values.astype(float))
ref_pred = ada.predict(ref_df.drop('Survived',axis=1).values.astype(float))
predictions_array = []

def score(test_df,ref_pred):

    for i in range(test_df.shape[0]):

        predicted_value = test_df.iloc[i].copy(deep=True)

        predicted_value.Survived = 0

        pred_df = train_df.append(predicted_value)

        ada.fit(pred_df.drop('Survived',axis=1).values.astype(float),pred_df.Survived.values.astype(float))

        predictions_array.append(ada.predict(ref_df.drop('Survived',axis=1).values.astype(float))[1])

        

    np_pred = ref_pred[1]

    predictions = []

    for new_pred in predictions_array:

        predictions.append(np.where((new_pred[0:50] - np_pred[0:50]).sum() - (new_pred[50:100] - np_pred[50:100]).sum()>0,1,0))

    

    print(accuracy_score(test_df.Survived,predictions))



score(test_df,ref_pred)
mlp = MLPClassifier(random_state=17)

mlp.fit(train_df.drop("Survived",axis=1),train_df.Survived)

y_pred = mlp.predict(test_df.drop("Survived",axis=1))

y_true = test_df.Survived

print(accuracy_score(y_true,y_pred))