import numpy as np

import pandas as pd

from sklearn.metrics import accuracy_score,mean_squared_error

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm_notebook

import random
data_train=pd.read_csv('../input/train.csv')

data_test=pd.read_csv('../input/test.csv')
print(data_train.shape,data_test.shape)
data_test.head(4)
Cols1=(data_test['PassengerId'])
data_train.head(5)
sns.countplot(x='Survived',data=data_train)
sns.countplot(x='Survived',hue='Embarked',data=data_train)
sns.countplot(x='Survived',hue='Pclass',data=data_train)
data_test.isnull().sum()
data_train.isnull().sum()
sns.heatmap(data_test.isnull())

plt.show()
sns.heatmap(data_train.isnull())

plt.show()
full_data=[data_train,data_test]
for data in full_data:

    data['Family_size']=data['SibSp']+data['Parch']+1

    data['IsAlone']=0

    data.loc[data['Family_size']==1,'IsAlone']=1
data_train.head(5)
for data in full_data:

    data['Embarked']=data['Embarked'].fillna('S')
for data in full_data:

    data['Fare']=data['Fare'].fillna(data['Fare'].median())
random.seed(0)

for data in  full_data:

    avg_age=data['Age'].mean()

    std_age=data['Age'].std()

    data_na_size=data['Age'].isnull().sum()

    data_random_list=np.random.randint(avg_age-std_age,avg_age+std_age,size=data_na_size)

    data['Age'][np.isnan(data['Age'])]=data_random_list

    data['Age']=data['Age'].astype(int)
data_train['Age'].isnull().sum()
data_train['Embarked'].unique()
for data in full_data:

    data['Sex']=data['Sex'].map({'female':0,'male':1}).astype(int)

    data['Embarked']=data['Embarked'].map({'S':0,'Q':1,'C':2}).astype(int)
for data in full_data:

    data.loc[ data['Fare'] <= 7.91, 'Fare']= 0

    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare']= 1

    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare']= 2

    data.loc[ data['Fare'] > 31, 'Fare'] = 3

    data['Fare'] = data['Fare'].astype(int)

    

    data.loc[ data['Age'] <= 16, 'Age']= 0

    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1

    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2

    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3

    data.loc[ data['Age'] > 64, 'Age'] = 4
drope=['PassengerId','Name','Family_size', 'SibSp', 'Parch','Cabin','Ticket']

data_train=data_train.drop(drope,axis=1)

data_test=data_test.drop(drope,axis=1)
data_train.head(5)
data_test.head(5)
sns.heatmap(data_train.isnull())

plt.show()
X_train=np.array(data_train.drop('Survived',axis=1))

Y_train=np.array(data_train['Survived'])

X_val=np.array(data_test)
class FFSNNetwork:

  

  def __init__(self, n_inputs, hidden_sizes=[2]):

    self.nx = n_inputs

    self.ny = 1

    self.nh = len(hidden_sizes)

    self.sizes = [self.nx] + hidden_sizes + [self.ny]

    

    self.W = {}

    self.B = {}

    random.seed(0)

    for i in range(self.nh+1):

      self.W[i+1] = np.random.randn(self.sizes[i], self.sizes[i+1])

      self.B[i+1] = np.zeros((1, self.sizes[i+1]))

  

  def sigmoid(self, x):

    return 1.0/(1.0 + np.exp(-x))

  

  def forward_pass(self, x):

    self.A = {}

    self.H = {}

    self.H[0] = x.reshape(1, -1)

    for i in range(self.nh+1):

      self.A[i+1] = np.matmul(self.H[i], self.W[i+1]) + self.B[i+1]

      self.H[i+1] = self.sigmoid(self.A[i+1])

    return self.H[self.nh+1]

  

  def grad_sigmoid(self, x):

    return x*(1-x) 

    

  def grad(self, x, y):

    self.forward_pass(x)

    self.dW = {}

    self.dB = {}

    self.dH = {}

    self.dA = {}

    L = self.nh + 1

    self.dA[L] = (self.H[L] - y)

    for k in range(L, 0, -1):

      self.dW[k] = np.matmul(self.H[k-1].T, self.dA[k])

      self.dB[k] = self.dA[k]

      self.dH[k-1] = np.matmul(self.dA[k], self.W[k].T)

      self.dA[k-1] = np.multiply(self.dH[k-1], self.grad_sigmoid(self.H[k-1]))

    

  def fit(self, X, Y, epochs=1, learning_rate=1, initialise=True, display_loss=False):

    

    # initialise w, b

    if initialise:

        random.seed(0)

    for i in range(self.nh+1):

        self.W[i+1] = np.random.randn(self.sizes[i], self.sizes[i+1])

        self.B[i+1] = np.zeros((1, self.sizes[i+1]))

      

    if display_loss:

      loss = {}

    

    for e in tqdm_notebook(range(epochs), total=epochs, unit="epoch"):

      dW = {}

      dB = {}

      for i in range(self.nh+1):

        dW[i+1] = np.zeros((self.sizes[i], self.sizes[i+1]))

        dB[i+1] = np.zeros((1, self.sizes[i+1]))

      for x, y in zip(X, Y):

        self.grad(x, y)

        for i in range(self.nh+1):

          dW[i+1] += self.dW[i+1]

          dB[i+1] += self.dB[i+1]

        

      m = X.shape[1]

      for i in range(self.nh+1):

        self.W[i+1] -= learning_rate * dW[i+1] / m

        self.B[i+1] -= learning_rate * dB[i+1] / m

      

      if display_loss:

        Y_pred = self.predict(X)

        loss[e] = mean_squared_error(Y_pred, Y)

    

    if display_loss:

      plt.plot(loss.values())

      plt.xlabel('Epochs')

      plt.ylabel('Mean Squared Error')

      plt.show()

      

  def predict(self, X):

    Y_pred = []

    for x in X:

      y_pred = self.forward_pass(x)

      Y_pred.append(y_pred)

    return np.array(Y_pred).squeeze()
ffsnn = FFSNNetwork(6, [6,5,2])

ffsnn.fit(X_train, Y_train, epochs=1000, learning_rate=.01, display_loss=True)
Y_pred_train = ffsnn.predict(X_train)

Y_pred_binarised_train = (Y_pred_train >= 0.5).astype("int").ravel()

Y_pred_val = ffsnn.predict(X_val)

Y_pred_binarised_val = (Y_pred_val >= 0.5).astype("int").ravel()

accuracy_train = accuracy_score(Y_pred_binarised_train, Y_train)



print("Training accuracy", round(accuracy_train, 2))
submission = pd.DataFrame({"PassengerId": Cols1,"Survived":(Y_pred_binarised_val)})
submission.head(5)
filename = 'Titanic Predictions 1.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)