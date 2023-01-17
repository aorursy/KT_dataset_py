# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

print("We drop rows with more than 0 NA entries\n")

train_data.info()
train_data = train_data.drop(columns = "Cabin")

train_data = train_data.dropna()

train_data.info()
train_data.head()
Y = train_data["Survived"]

plt.hist(Y)

plt.show()
df = train_data.drop(columns = ["Survived", "Name", "Ticket"])

df["female"] = pd.get_dummies(df['Sex'])["female"]

df.drop(columns = ["Sex","PassengerId"], inplace = True)

one_hot = pd.get_dummies(df["Embarked"])[["C", "Q"]]

df = df.join(one_hot).drop(columns = "Embarked")
import seaborn as sns



myBasicCorr = df.corr()

sns.heatmap(myBasicCorr)

print(myBasicCorr)
from sklearn.decomposition import PCA



df_subset = df.copy()

data_subset = df_subset.to_numpy()

pca = PCA(n_components=3)

pca_result = pca.fit_transform(data_subset)

df_subset['pca-one'] = pca_result[:,0]

df_subset['pca-two'] = pca_result[:,1] 

df_subset['pca-three'] = pca_result[:,2]

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))



print(pd.DataFrame(pca.components_,columns=df.columns,index = ['PC-1','PC-2', 'PC-3']))

import matplotlib.pyplot as plt

%matplotlib inline



fare1 = train_data.loc[train_data.Survived == 1]["Fare"]

fare0 = train_data.loc[train_data.Survived == 0]["Fare"]



plt.hist(fare1)

plt.hist(fare0)

plt.show()
from sklearn.model_selection import train_test_split

train_df, valid_df, train_Y, valid_Y = train_test_split(df, Y, test_size=0.2, random_state=42)

print(train_df.shape, valid_df.shape)
train_X = train_df["Fare"]

valid_X = valid_df["Fare"]



def logit(x):

    return 1/(1+np.exp(-x))



# Initialize weights randomly

np.random.seed(42)

theta0, theta1 = np.random.randn(2)[0], np.random.randn(2)[1]



# Normalising X

train_X = (train_X - np.mean(train_X))/np.std(train_X)

valid_X = (valid_X - np.mean(valid_X))/np.std(valid_X)



# Substitute outliers

train_X[train_X > 3*np.std(train_X)] = 3*np.std(train_X)

valid_X[valid_X > 3*np.std(valid_X)] = 3*np.std(valid_X)



# Learning Rate 0.00001

alpha = 10e-3



# Number of iterations

N = 10001

# Initialize cost

train_cost = np.zeros(N)

accuracy = np.zeros(N)

valid_cost = np.zeros(N)

for i in range(N):

    diff_train = logit(theta0 + theta1*train_X) - train_Y

    diff_valid = logit(theta0 + theta1*valid_X) - valid_Y

    train_cost[i] = 1/len(train_X) * -np.sum(train_Y * np.log(logit(theta0 + theta1*train_X)) + (1 - train_Y) * np.log(1 - logit(theta0 + theta1*train_X)))

    valid_cost[i] = 1/len(valid_X) * -np.sum(valid_Y * np.log(logit(theta0 + theta1*valid_X)) + (1 - valid_Y) * np.log(1 - logit(theta0 + theta1*valid_X)))

    accuracy[i] = 1 - np.sum(np.abs(diff_train))/len(diff_train)

    theta0 = theta0 - alpha*(np.sum(diff_train))

    theta1 = theta1 - alpha*(np.sum(diff_train.dot(train_X)))/len(diff_train)

    if i % 250 == 0:

        print("Iteration %d, training_cost: %2.2f, validation_cost: %2.2f, accuracy: %2.2f, theta0: %2.2f, theta1: %2.2f" % (i, train_cost[i], valid_cost[i], accuracy[i], theta0, theta1))

        #print(np.sum(train_Y))

        #print(np.sum(logit(theta0 + theta1*train_X)))

        #print(np.sum(diff_train))

plt.figure(figsize=(7,6))

lo = plt.scatter(np.arange(N), train_cost, marker='x', color='r')

ll = plt.scatter(np.arange(N), valid_cost, marker='o', color='g')



plt.legend((lo, ll, ),

           ('Training', 'Validate'),

           scatterpoints=1,

           loc='lower right',

           ncol=3,

           fontsize=15)

plt.show()
from sklearn.linear_model import LogisticRegression

train_X = train_df["Fare"]



# Normalising X

train_X = (train_X - np.mean(train_X))/np.std(train_X)



# Substitute outliers

train_X[train_X > 3*np.std(train_X)] = 3*np.std(train_X)



sk_train_X = np.c_[np.ones(len(train_X)), train_X  ]  # Add bias

clf = LogisticRegression(random_state=0, solver = "sag").fit(sk_train_X, train_Y)

accuracy = clf.score(sk_train_X, train_Y)

print("theta0: %2.2f, theta1: %2.2f, accuracy: %2.2f" %(clf.coef_[0][0], clf.coef_[0][1], accuracy))
X = train_X

Y = train_Y

cost = valid_cost

z = logit(theta0 + theta1*X)

errors = Y - z



Rsq = np.sum(errors**2)/np.sum((Y-np.mean(Y))**2)

print("R-squared: ", Rsq)

#For multiple logistic regression

Fstat = Rsq*(len(Y)-2)/1



print("F-stat: ", Fstat)

plt.figure(figsize = (10,5))

plt.plot()



plt.subplot(1,2,1)

plt.scatter(X, Y)

plt.scatter(X, z)

plt.title("Errors vs predicted")

plt.subplot(1,2,2)

plt.scatter(z, errors)

meanline = np.mean(errors) * np.ones(len(z))

plt.plot(z, meanline, 'r')

plt.title("Cost function")

plt.show()

def replaceOutliers(df):

    for col in df.columns.to_list():

        mean = df[col].mean()

        std = df[col].std()

        df[col] = np.where(df[col] >3*std, 3*std,df[col])

        #outliers = (df[col] - mean).abs() > 3*std

        #df[outliers] = np.nan

        #df[col].fillna(3*std, inplace=True)

    return df
train_X = train_df[["Fare", "Age"]]

valid_X = valid_df[["Fare", "Age"]]



def logit(x):

    return 1/(1+np.exp(-x))



# Initialize weights randomly

np.random.seed(42)

theta0, theta1, theta2 = np.random.randn(3)[0], np.random.randn(3)[1], np.random.randn(3)[2]

# Normalising X

train_X = (train_X - np.mean(train_X, axis = 0))/np.std(train_X, axis = 0)

valid_X = (valid_X - np.mean(valid_X, axis = 0))/np.std(valid_X, axis = 0)

# Substitute outliers

train_X = replaceOutliers(train_X)

valid_X = replaceOutliers(valid_X)

# Learning Rate 0.00001

alpha = 10e-3



# Number of iterations

N = 10001

# Initialize cost

train_cost = np.zeros(N)

accuracy = np.zeros(N)

valid_cost = np.zeros(N)

for i in range(N):

    train_z = theta0 + theta1*train_X["Fare"].values + theta2*train_X["Age"].values

    valid_z = theta0 + theta1*valid_X["Fare"].values + theta2*valid_X["Age"].values

    diff_train = logit(train_z) - train_Y.values

    diff_valid = logit(valid_z) - valid_Y.values

    train_cost[i] = 1/len(train_X) * -np.sum(train_Y * np.log(logit(train_z)) + (1 - train_Y) * np.log(1 - logit(train_z)))

    valid_cost[i] = 1/len(valid_X) * -np.sum(valid_Y * np.log(logit(valid_z)) + (1 - valid_Y) * np.log(1 - logit(valid_z)))

    accuracy[i] = 1 - np.sum(np.abs(diff_train))/len(diff_train)

    theta0 = theta0 - alpha*(np.sum(diff_train))

    theta1 = theta1 - alpha*(np.sum(diff_train.dot(train_X["Fare"].values)))/len(diff_train)

    theta2 = theta2 - alpha*(np.sum(diff_train.dot(train_X["Age"].values)))/len(diff_train)

    if i % 250 == 0:

        print("I %d, t_cost: %2.2f, v_cost: %2.2f, accuracy: %2.2f, theta0: %2.2f, theta1: %2.2f, theta2: %2.2f" % (i, train_cost[i], valid_cost[i], accuracy[i], theta0, theta1, theta2))

        
plt.figure(figsize=(7,6))

lo = plt.scatter(np.arange(N), train_cost, marker='x', color='r')

ll = plt.scatter(np.arange(N), valid_cost, marker='o', color='g')



plt.legend((lo, ll, ),

           ('Training', 'Validate'),

           scatterpoints=1,

           loc='lower right',

           ncol=3,

           fontsize=15)

plt.show()
train_X = train_df[["Fare", "Age"]]

valid_X = valid_df[["Fare", "Age"]]



# Normalising X

train_X = (train_X - np.mean(train_X, axis = 0))/np.std(train_X, axis = 0)

valid_X = (valid_X - np.mean(valid_X, axis = 0))/np.std(valid_X, axis = 0)



# Substitute outliers

train_X = replaceOutliers(train_X)

valid_X = replaceOutliers(valid_X)



sk_train_X = np.c_[np.ones(len(train_X)), train_X  ]  # Add bias

clf = LogisticRegression(random_state=0, solver = "sag").fit(sk_train_X, train_Y)

accuracy = clf.score(sk_train_X, train_Y)

print("theta0: %2.2f, theta1: %2.2f, theta2: %2.2f, accuracy: %2.2f" %(clf.coef_[0][0], clf.coef_[0][1], clf.coef_[0][2], accuracy))
X = train_X

Y = train_Y

z = logit(theta0 + theta1*X["Fare"].values + theta2*X["Age"].values)

errors = Y - z



Rsq = np.sum(errors**2)/np.sum((Y-np.mean(Y))**2)

AdjsutedRsq = 1 - (1-Rsq)*(len(Y) - 1)/(len(Y) - 2-1)

print("R-squared: ", Rsq)

print("Adj R-squared: ", AdjsutedRsq)

#For multiple logistic regression

Fstat = Rsq*(len(Y)-3)/2

print("F-stat: ", Fstat)

plt.figure(figsize = (10,5))

plt.plot()



plt.subplot(1,2,1)

plt.scatter(Y, z)

plt.title("Errors vs predicted")

plt.subplot(1,2,2)

plt.scatter(z, errors)

meanline = np.mean(errors) * np.ones(len(z))

plt.plot(z, meanline, 'r')

plt.title("Cost function")

plt.show()

from sklearn.linear_model import LogisticRegression

train_X = train_df["Fare"]



# Normalising X

train_X = (train_X - np.mean(train_X))/np.std(train_X)



# Substitute outliers

train_X[train_X > 3*np.std(train_X)] = 3*np.std(train_X)



sk_train_X = np.c_[np.ones(len(train_X)), train_X  ]  # Add bias

clf = LogisticRegression(random_state=0, solver = "sag").fit(sk_train_X, train_Y)

accuracy = clf.score(sk_train_X, train_Y)

print("theta0: %2.2f, theta1: %2.2f, accuracy: %2.2f" %(clf.coef_[0][0], clf.coef_[0][1], accuracy))
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

X_test = test_data["Fare"]

# Normalising X

X_test = (X_test - np.nanmean(X_test))/np.nanstd(X_test)



# Substitute outliers

X_test[X_test > 3*np.nanstd(X_test)] = 3*np.nanstd(X_test)



# Replace NAN

X_test = X_test.fillna(np.nanmean(X_test))



sk_X_test = np.c_[np.ones(len(X_test)), X_test  ]  # Add bias

predictions = clf.predict(sk_X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")