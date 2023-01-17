import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
path = os.getcwd()+'../input/train.csv'
data = pd.read_csv('../input/train.csv')
print(data.head())
data.describe()
data.isnull().sum()
# percent of missing "Age" 
print('Percent of missing "Age" records is %.2f%%' %((data['Age'].isnull().sum()/data.shape[0])*100))
ax = data["Age"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)
data["Age"].plot(kind='density', color='teal')
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()
# mean age
print('The mean of "Age" is %.2f' %(data["Age"].mean(skipna=True)))
# median age
print('The median of "Age" is %.2f' %(data["Age"].median(skipna=True)))
print('Percent of missing "Cabin" records is %.2f%%' %((data['Cabin'].isnull().sum()/data.shape[0])*100))
# percent of missing "Embarked" 
print('Percent of missing "Embarked" records is %.2f%%' %((data['Embarked'].isnull().sum()/data.shape[0])*100))
import seaborn as sns
print('Boarded passengers grouped by port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton):')
print(data['Embarked'].value_counts())
sns.countplot(x='Embarked', data=data, palette='Set2')
plt.show()
print('The most common boarding port of embarkation is %s.' %data['Embarked'].value_counts().idxmax())

data["Age"].fillna(data["Age"].median(skipna=True), inplace=True)
data["Embarked"].fillna(data['Embarked'].value_counts().idxmax(), inplace=True)
data.drop('Cabin', axis=1, inplace=True)
data.isnull().sum()
data.head()
plt.figure(figsize=(15,8))
ax = data["Age"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)
data["Age"].plot(kind='density', color='teal')
ax.legend(['Raw Age'])
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()
## Create categorical variable for traveling alone
data['TravelAlone']=np.where((data["SibSp"]+data["Parch"])>0, 0, 1)
data.drop('SibSp', axis=1, inplace=True)
data.drop('Parch', axis=1, inplace=True)
#create categorical variables and drop some variables
data=pd.get_dummies(data, columns=["Pclass","Embarked","Sex"])
data.drop('Sex_female', axis=1, inplace=True)
data.drop('PassengerId', axis=1, inplace=True)
data.drop('Name', axis=1, inplace=True)
data.drop('Ticket', axis=1, inplace=True)


data.head()
path = os.getcwd()+'../input/test.csv'
test_df = pd.read_csv('../input/test.csv')
test_data = test_df.copy()
test_data["Age"].fillna(data["Age"].median(skipna=True), inplace=True)
test_data["Fare"].fillna(data["Fare"].median(skipna=True), inplace=True)
test_data.drop('Cabin', axis=1, inplace=True)

test_data['TravelAlone']=np.where((test_data["SibSp"]+test_data["Parch"])>0, 0, 1)

test_data.drop('SibSp', axis=1, inplace=True)
test_data.drop('Parch', axis=1, inplace=True)

testing = pd.get_dummies(test_data, columns=["Pclass","Embarked","Sex"])
testing.drop('Sex_female', axis=1, inplace=True)
testing.drop('PassengerId', axis=1, inplace=True)
testing.drop('Name', axis=1, inplace=True)
testing.drop('Ticket', axis=1, inplace=True)

final_test = testing
final_test.head()
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))
cols = list(data.columns.values) #Make a list of all of the columns in the df
cols.pop(cols.index('Survived')) #Remove b from list
data = data[cols+['Survived']] #Create new dataframe with columns in the order you want
data.head()
# add a ones column - this makes the matrix multiplication work out easier
data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)




# convert to numpy arrays and initalize the parameter array theta

cols = data.shape[1]
cols
X = data.iloc[:,0:cols-1]

X
Y = data.iloc[:,cols-1:cols]


Y.shape
X.shape
theta = np.zeros(11)
theta.shape
X.shape, theta.shape, Y.shape
cost(theta, X, Y)
def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    
    error = sigmoid(X * theta.T) - y
    
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term) / len(X)
    
    return grad
theta.shape
import scipy.optimize as opt
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, Y))
cost(result[0], X, Y)

theta_min = np.matrix(result[0])
X= np.matrix(X)
Y=np.matrix(Y)
X.shape,theta.shape, result[0].shape,theta_min.shape,theta_min.T.shape
def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

theta_min = np.matrix(result[0])
predictions = predict(theta_min, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, Y)]
temp = sum(map(int,correct))
accuracy_test = temp/ len(correct)
print ('accuracy_test = {0}%'.format(accuracy_test*100))

Y.shape
final_test
final_test.insert(0, 'Ones', 1)
final_test.insert(11, 'Survived', 1)
X_test = final_test.iloc[:,0:cols-1]
X_test = np.matrix(X_test)
Y_test = final_test.iloc[:,cols-1:cols]
Y_test = np.matrix(Y_test)
final_test
predict(theta_min,X_test)
Survived_test = predict(theta_min,X_test)
Survived_test= pd.DataFrame(Survived_test)
Survived_test.head()
final_test['Survived'] = Survived_test
final_test['Survived'].count()

test_df['PassengerId'].count()
final_test['Survived'].head()
df1 = pd.DataFrame(test_df['PassengerId'])
df2= pd.DataFrame(final_test['Survived'])
concat = pd.merge(df1,df2, left_index=True, right_index = True)
concat.head()
concat.head()
concat.to_csv('concat.csv',index=False)

