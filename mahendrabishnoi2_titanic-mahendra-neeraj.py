import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
#train.head()
#train.describe()
#train.shape
#train.info()
#test.head()
#test.describe()
#test.shape
#test.info()
"Delete the Column name Cabin because it has little information as of the values ar NaN"
train.drop("Cabin", axis=1, inplace=True)
test.drop("Cabin", axis=1, inplace=True)
#sns.countplot(x='Survived', data=train)
#sns.countplot(x='Pclass', data=train)
#sns.countplot(x='Sex', data=train)
#plt.scatter(range(train.shape[0]), np.sort(train['Age']))
train['Age'] = train['Age'].fillna(train['Age'].median())
test['Age'] = test['Age'].fillna(test['Age'].median())
train.loc[train['Sex']=='male', 'Sex'] = 1
train.loc[train['Sex']=='female', 'Sex'] = 0

test.loc[test['Sex']=='male', 'Sex'] = 1
test.loc[test['Sex']=='female', 'Sex'] = 0
train.loc[train['Embarked']=='C', 'Embarked'] = 0
train.loc[train['Embarked']=='Q', 'Embarked'] = 1
train.loc[train['Embarked']=='S', 'Embarked'] = 2

test.loc[test['Embarked']=='C', 'Embarked'] = 0
test.loc[test['Embarked']=='Q', 'Embarked'] = 1
test.loc[test['Embarked']=='S', 'Embarked'] = 2
train['Embarked'] = train['Embarked'].fillna(0)
test['Embarked'] = test['Embarked'].fillna(0)
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
#train.head()
#train.shape
#train.describe()
#train.info()
#test.head()
#test.info()
#test.describe()
X_train = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
Y_train = train['Survived']
X_test = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
from sklearn.cross_validation import train_test_split
X,x,Y,y = train_test_split(X_train, Y_train, test_size=0.2)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def hypothesis(w, b, X):
    z = np.matmul(X, w) + b
    return sigmoid(z)
def cost_grads(m, X, Y, hx, lambd, w):
    J = (-1 / m) * (np.matmul(Y.T, np.log(hx)) + np.matmul((1 - Y).T, np.log((1 - hx)))) + ((lambd / (2 * m) ) * np.sum(np.square(w)))
    dw = (1 / m) * np.matmul(X.T, (hx - Y)) + ((lambd / m) * w)
    db = (1 / m) * np.sum((hx - Y))
    grads = {'dw': dw,
            'db': db
    }
    return J, grads
    
# Initializing the parameters
#b = 0
#w = np.zeros((X.shape[1], 1))
def optimize(X, Y, learning_rate, num_iterations, lambd, print_cost=False):
    costs = []
    m = X.shape[0]
    Y = np.array(Y).reshape(Y.shape[0], 1)
    b = 0
    w = np.zeros((X.shape[1], 1))
    for i in range(num_iterations):
        hx = hypothesis(w, b, X)
        J, grads = cost_grads(m, X, Y, hx, w, lambd)
        dw = grads['dw']
        db = grads['db']
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        if i % 100 == 0:
            costs.append(J)
        
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}:{}".format(i, J))
            
    parameters = {'w': w,
                  'b': b
    }
    
    grads = {'dw': dw,
             'db': db
    }
    
    return costs, parameters, grads
        
costs, parameters, grads = optimize(X, Y, 0.0044, 50000, 0, print_cost=False)
#w = parameters['w']
#b = parameters['b']
output_cv = hypothesis(parameters['w'], parameters['b'], x)
output_train = hypothesis(parameters['w'], parameters['b'], X)
def output_as_bool(output):
    result = []
    for out in range(output.shape[0]):
        if output[out]>0.60:
            result.append(1)
        else:
            result.append(0)
    return result
def compare(x, y, result):
    count = 0
    y = list(y)
    for i in range(x.shape[0]):
        if result[i]==y[i]:
            count += 1
        else:
            pass
    return count
def accuracy(x, count):
    Accuracy = (count / x.shape[0]) * 100
    return Accuracy
result_cv = output_as_bool(output_cv)
result_train = output_as_bool(output_train)
count_cv = compare(x, y, result_cv)
count_train = compare(X, Y, result_train)
aaccuracy_cv = accuracy(x, count_cv)
aaccuracy_train = accuracy(X,count_train)
print('CV Accuracy: {:.{}f}'.format(aaccuracy_cv, 2))
print('Train Accuracy: {:.{}f}'.format(aaccuracy_train, 2))
output_test = hypothesis(parameters['w'], parameters['b'], X_test)
result_test = output_as_bool(output_test)
sns.countplot(result_test)
abc = {'PassengerId': test['PassengerId'],
         'Survived': result_test}
abc = pd.DataFrame(abc)
abc.to_csv('Submission.csv', index = False)
