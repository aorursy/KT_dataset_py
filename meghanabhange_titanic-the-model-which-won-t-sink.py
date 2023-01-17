# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

from sklearn import datasets
import numpy as np
from sklearn import cross_validation
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
def initiate_data(in_loc):
    df = pd.read_csv(in_loc)
    df = df.drop('Name', 1)
    df = df.drop('Ticket', 1)
    a = df[['Fare']]
    df['Sex'] = df['Sex'].map( {'female': 1, 'male': 2} ).astype(int)
    
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Embarked'] = df['Embarked'].map( {'C': 1, 'S': 2,'Q': 3} )
    
    age_avg = df['Age'].mean()
    age_std = df['Age'].std()
    age_null_count = df['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    df['Age'][np.isnan(df['Age'])] = age_null_random_list
    df['Age'] = df['Age'].astype(int)
    
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df["Fare"] = df["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
    
    df = df.drop('Cabin', 1)
    return df



    
df = initiate_data('../input/train.csv') 
df = df.dropna()
'''df['Class_Age'] = df.loc[:,'Age']*df.loc[:,'Pclass']
df['Class_Age'] = df['Class_Age'].fillna(df['Class_Age'].median())
df['Sex_Age']   = df.loc[:,'Sex']*df.loc[:,'Age']
df['Sex_Age'] = df['Sex_Age'].fillna(df['Sex_Age'].median())'''
df.head()
def sigmoid(z):
    '''
    Sigmoid: It takes in the score and give the probabality of the outcome. 
    It gives the class probabality of the output. 
    
    '''
    z=z.astype(float)
    return (1.0/(1.0+np.exp(-1.0*z)))

def log_likelihood(target, features,weights):
    '''
    This is a loss funtion. The log-likelihood can be viewed as a sum over all the training data.
    
    '''
    score=np.dot(features,weights)
    score=score.astype(float)
    return(sum(target*score)-np.log(1+np.exp(score)))
def gradient_log_likelihood(features,predicted,target):
    return(np.dot(features.T,np.subtract(target,predicted)))
def logistic_regression(learning_rate=0.0003,steps=100000, add_intercept=False):
    ind = 0
    tot_features = np.array(df.loc[:,('Pclass','Sex','SibSp','Parch','Age','Fare','Embarked')])
    tot_target   = np.array(df.loc[:,('Survived')])
    weights = np.zeros(tot_features.shape[1])
    for i in range(steps):
        if(ind>=701):
            ind=0
        features=tot_features[ind:ind+10]
        target=tot_target[ind:ind+10]
        count = 0
        scores    = np.dot(features,weights)
        predicted = sigmoid(scores)
        scores=scores.astype(float)
        loss      = log_likelihood(target,features,weights)
        
        weights   = np.add(gradient_log_likelihood(features, predicted, target)*learning_rate,weights)
        difference= abs(predicted-target)
        count     = len(difference)-sum(difference)
        ind       = ind+11
    return(weights)

weights = logistic_regression() 

test_data = initiate_data('../input/test.csv')
test_data.head()
def testing_data(test_data): 
    '''
    Function to test the accuracy on unknown data. 
    
    Input:
    
    out_loc- Location of the targets for CSV
    test_data- Dataframe of the test data. Genrally test.csv
    '''
    predicted = []
    features  = np.array(test_data.loc[:,('Pclass','Sex','SibSp','Parch','Age','Fare','Embarked')])
    scores    = np.dot(features,weights)
    scores    = scores.astype(float)
    predicted = np.round(sigmoid(scores),0)
    output_data = []
    newdf     = pd.DataFrame({'PassengerId':np.array(test_data['PassengerId']),'Survived':predicted})
    return (newdf)
                   
output_data = testing_data(test_data)                   
output_data.head()

df = initiate_data('../input/train.csv')
df.loc[0,:]

df['Class_Age'] = df.loc[:,'Age']*df.loc[:,'Pclass']
df['Class_Age'] = df['Class_Age'].fillna(df['Class_Age'].median())
df['Sex_Age']   = df.loc[:,'Sex']*df.loc[:,'Age']
df['Sex_Age'] = df['Sex_Age'].fillna(df['Sex_Age'].median())
df.head()
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    df.loc[:, ('Pclass','Sex','SibSp','Parch','Age','Fare','Embarked','Class_Age','Sex_Age')], df.loc[:, ('Survived')], test_size=0.20, random_state=6)
'''
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(df)
df_normalized = pd.DataFrame(np_scaled)
'''
#df = df_normalized
Model = RandomForestClassifier()
Model.fit(X_train, y_train)
Predicted = Model.predict(X_test)
accuracy = accuracy_score(y_test, Predicted)
print('RandomForestClassifier: ', accuracy)

Model = LogisticRegression()
Model.fit(X_train, y_train)
Predicted = Model.predict(X_test)
accuracy = accuracy_score(y_test, Predicted)
print('Logistic Regression: ', accuracy)
Model = DecisionTreeClassifier()
Model.fit(X_train,y_train)
Predicted = Model.predict(X_test)
accuracy = accuracy_score(y_test,Predicted)
print('Decison Tree: ',accuracy)
#KNeighborsClassifier
Model = KNeighborsClassifier()
Model.fit(X_train,y_train)
Predicted = Model.predict(X_test)
accuracy = accuracy_score(y_test,Predicted)
print('KNeighborsClassifier: ',accuracy)
Model = SVC()
Model.fit(X_train, y_train)
Predicted = Model.predict(X_test)
accuracy = accuracy_score(y_test, Predicted)
print('SVM: ', accuracy)
Model = MLPClassifier(solver='lbfgs', alpha=0.03,hidden_layer_sizes=(12,17,8,), activation='relu',learning_rate='adaptive')
Model.fit(X_train, y_train)
Predicted = Model.predict(X_test)
accuracy = accuracy_score(y_test, Predicted)
print('NN: ', accuracy)

