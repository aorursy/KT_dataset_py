import pandas as pd

import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



from sklearn.model_selection import train_test_split
data = pd.read_csv('../input/heart.csv')
data.head()
data = data.sample(frac=1)
data.head()
data.columns.values
data.info()
data.corr()
data.describe()
data.head()
data[['sex','target']].groupby(['sex'], as_index=False).mean().sort_values(by='target', ascending=False)
data[['cp','target']].groupby(['cp'], as_index=False).mean().sort_values(by='target', ascending=False)
data[['fbs','target']].groupby(['fbs'], as_index=False).mean().sort_values(by='target', ascending=False)
data[['restecg','target']].groupby(['restecg'], as_index=False).mean().sort_values(by='target', ascending=False)
data[['exang','target']].groupby(['exang'], as_index=False).mean().sort_values(by='target', ascending=False)
data[['slope','target']].groupby(['slope'], as_index=False).mean().sort_values(by='target', ascending=False)
data[['ca','target']].groupby(['ca'], as_index=False).mean().sort_values(by='target', ascending=False)
data[['thal','target']].groupby(['thal'], as_index=False).mean().sort_values(by='target', ascending=False)
data.head()
g = sns.FacetGrid(data, col='target')

g.map(plt.hist, 'age', bins=20)
g = sns.FacetGrid(data, row='sex', col='target')

g.map(plt.hist, 'trestbps', bins=20)
g = sns.FacetGrid(data, col='target')

g.map(plt.hist, 'chol', bins=20)
g = sns.FacetGrid(data, col='target')

g.map(plt.hist, 'thalach', bins=20)
g = sns.FacetGrid(data, col='target')

g.map(plt.hist, 'oldpeak', bins=20)
data["ageband"] = pd.qcut(data['age'],5)
data.head()
data[['ageband','target']].groupby(['ageband'], as_index=False).mean()
combine = [data]
for dataset in combine:

    dataset.loc[dataset['age']<=45, 'age'] = 0

    dataset.loc[(dataset['age']>45) & (dataset['age']<=53), 'age'] = 1

    dataset.loc[(dataset['age']>53) & (dataset['age']<=58), 'age'] = 2

    dataset.loc[(dataset['age']>58) & (dataset['age']<=62), 'age'] = 3

    dataset.loc[(dataset['age']>62) & (dataset['age']<=77), 'age'] = 4
data.head()
combine = [data]
data["tresband"] = pd.qcut(data['trestbps'],5)
data[['tresband','target']].groupby(['tresband'], as_index=False).mean()
data.head()
for dataset in combine:

    dataset.loc[dataset['trestbps']<=120, 'trestbps'] = 0

    dataset.loc[(dataset['trestbps']>120) & (dataset['trestbps']<=126), 'trestbps'] = 1

    dataset.loc[(dataset['trestbps']>126) & (dataset['trestbps']<=134), 'trestbps'] = 2

    dataset.loc[(dataset['trestbps']>134) & (dataset['trestbps']<=144), 'trestbps'] = 3

    dataset.loc[(dataset['trestbps']>144) & (dataset['trestbps']<=200), 'trestbps'] = 4
data.head()
combine =[data]

data["cholband"] = pd.qcut(data['chol'],5)

data[['cholband','target']].groupby(['cholband'], as_index=False).mean()
for dataset in combine:

    dataset.loc[dataset['chol']<=204, 'chol'] = 0

    dataset.loc[(dataset['chol']>204) & (dataset['chol']<=230), 'chol'] = 1

    dataset.loc[(dataset['chol']>230) & (dataset['chol']<=254), 'chol'] = 2

    dataset.loc[(dataset['chol']>254) & (dataset['chol']<=285.2), 'chol'] = 3

    dataset.loc[(dataset['chol']>285.2) & (dataset['chol']<=564), 'chol'] = 4
data.head()
data = data.drop(['ageband','tresband','cholband'], axis=1)
data.head()
combine =[data]

data["thalband"] = pd.qcut(data['thalach'],5)

data[['thalband','target']].groupby(['thalband'], as_index=False).mean()
for dataset in combine:

    dataset.loc[dataset['thalach']<=130, 'thalach'] = 0

    dataset.loc[(dataset['thalach']>130) & (dataset['thalach']<=146), 'thalach'] = 1

    dataset.loc[(dataset['thalach']>146) & (dataset['thalach']<=159), 'thalach'] = 2

    dataset.loc[(dataset['thalach']>159) & (dataset['thalach']<=170), 'thalach'] = 3

    dataset.loc[(dataset['thalach']>170) & (dataset['thalach']<=202), 'thalach'] = 4
data.head()
combine =[data]
data["olband"] = pd.cut(data['oldpeak'],5)

data[['olband','target']].groupby(['olband'], as_index=False).mean()
for dataset in combine:

    dataset.loc[dataset['oldpeak']<=1.24, 'oldpeak'] = 0

    dataset.loc[(dataset['oldpeak']>1.24) & (dataset['oldpeak']<=2.48), 'oldpeak'] = 1

    dataset.loc[(dataset['oldpeak']>2.48) & (dataset['oldpeak']<=3.72), 'oldpeak'] = 2

    dataset.loc[(dataset['oldpeak']>3.72) & (dataset['oldpeak']<=4.96), 'oldpeak'] = 3

    dataset.loc[(dataset['oldpeak']>4.96) & (dataset['oldpeak']<=6.2), 'oldpeak'] = 4
data.oldpeak = data.oldpeak.astype(int)
data = data.drop(['thalband','olband'], axis=1)
data.head(20)
data.shape
X = data[data.columns[:-1]]

y = data[data.columns[-1]]
X.head()
y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
logreg = LogisticRegression()

logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_test)

logreg.score(X_test,y_test)*100
svc = SVC()

svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

svc.score(X_test,y_test)*100
knn = KNeighborsClassifier(n_neighbors=2)

knn.fit(X_train,y_train)

Y_pred = knn.predict(X_test)

knn.score(X_test,y_test)*100
gaussian = GaussianNB()

gaussian.fit(X_train,y_train)

Y_pred = gaussian.predict(X_test)

gaussian.score(X_test,y_test)*100
perceptron = Perceptron()

perceptron.fit(X_train,y_train)

Y_pred = perceptron.predict(X_test)

perceptron.score(X_test, y_test)*100
linear_svc = LinearSVC()

linear_svc.fit(X_train,y_train)

Y_pred = linear_svc.predict(X_test)

linear_svc.score(X_test,y_test)*100
sgd = SGDClassifier()

sgd.fit(X_train, y_train)

Y_pred = sgd.predict(X_test)

sgd.score(X_test,y_test)*100
dec = DecisionTreeClassifier()

dec.fit(X_train,y_train)

Y_pred = dec.predict(X_test)

dec.score(X_test,y_test)*100
ran = RandomForestClassifier()

ran.fit(X_train, y_train)

Y_pred = ran.predict(X_test)

ran.score(X_test, y_test)*100
import scipy.optimize as opt
#sigmoid function



def sigmoid(Z):

  return 1/(1+np.exp(-Z))
#initialize function



def initialize(X):

  

  m = X.shape[0]

  temp = np.ones((m,1))

  X.insert(loc=0, column='ones', value = np.ones((m,1))) #adding the column of ones to the X matrix

  

  return X
#defining the cost function



# input ---->

# theta.shape = (n,)

# X.shape = (m,n)

# y.shape = (m,1)



#output ---->

# grad.shape = (n,1)





def computeCost(theta,X,y):

  

  m,n = X.shape

  theta = theta.reshape(n,1)    #changing shape of theta from (n,) to (n,1)

  

  a = sigmoid(np.dot(X,theta))

  f = np.multiply(y,np.log(a))

  s = np.multiply(1-y,np.log(1-a))

  

  J = -np.sum(f+s)/m            #cost over all the training examples

  grad = (1/m)*np.dot(X.T,a-y)  #gradients over one iteration

  

  return J,grad
#defining the optimization function



def optimizeTheta(initial_theta,X,y):

  

  final_theta,a,b = opt.fmin_tnc(func=computeCost, x0=initial_theta, args=(X,y))

  

  return final_theta
#defining predict function



def predict(theta,X):

  m = X.shape[0]

  p = sigmoid(np.dot(X,theta)).round()

  p = p.reshape(m,1)

  return p
#defining accuracy



def acc(p,y):

  

  return np.mean(p==y)*100
#defining logistic model



def logistic_model(X_train, y_train, X_test, y_test):

  

  #convert X from (m,n) to (m,n+1) by adding a column of all zeros

  

  X_train = initialize(X_train)

  X_test= initialize(X_test)

  

  #get initial_theta

  n = X_train.shape[1]

  initial_theta = np.zeros(n)

  print("Initial theta : " + str(initial_theta))

  

  #compute cost and gradient

  cost, grad = computeCost(initial_theta, X_train, y_train)

  print("Initial cost : " + str(cost))

  print("Initial gradient : " + str(grad))

  

  #use an optimization function to obtain optimal theta

  final_theta = optimizeTheta(initial_theta, X_train, y_train)

  print("Final theta : " + str(final_theta))

  

  #make predictions

  p = predict(final_theta, X_test)

  

  #check the accuracy

  accuracy = acc(p,y_test)

  

  return accuracy,p
y_train = y_train.values.reshape(181,1)

y_test = y_test.values.reshape(122,1)



y_train.shape, y_test.shape
acc, p = logistic_model(X_train,y_train,X_test,y_test)
acc
p