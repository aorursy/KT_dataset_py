import pandas as pd

from tqdm import tqdm

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
test_df = pd.read_csv('../input/test.csv')

train_df = pd.read_csv('../input/train.csv')

gender_submission_df = pd.read_csv('../input/genderclassmodel.csv')
combined = [test_df,train_df]
print('Train data shape',train_df.shape)

print('Test data shape',test_df.shape)

print('Gender Submission data shape',gender_submission_df.shape)
print(train_df.columns.values)
train_df.head()
train_df.Embarked.unique()
train_df.info()
train_df.describe()
train_df.describe(include=['O'])
train_df[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Parch','Survived']].groupby(['Parch'],as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(train_df, col='Survived', row='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
grid = sns.FacetGrid(train_df,col='Survived',row='Pclass',size=2.2,aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

combine = [train_df, test_df]
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train_df['Title'], train_df['Sex'])
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



train_df.head()
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

combine = [train_df, test_df]

train_df.shape, test_df.shape
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



train_df.head()
guess_ages = np.zeros((2,3))

guess_ages
for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex'] == i) & \

                                  (dataset['Pclass'] == j+1)]['Age'].dropna()



            # age_mean = guess_df.mean()

            # age_std = guess_df.std()

            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)



            age_guess = guess_df.median()



            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)



train_df.head()
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']

train_df.head()
train_df = train_df.drop(['AgeBand'], axis=1)

combine = [train_df, test_df]

train_df.head()
for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [train_df, test_df]



train_df.head()
for dataset in combine:

    dataset['Age*Class'] = dataset.Age * dataset.Pclass



train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
freq_port = train_df.Embarked.dropna().mode()[0]

freq_port
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



train_df.head()
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

test_df.head()
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train_df = train_df.drop(['FareBand'], axis=1)

combine = [train_df, test_df]

    

train_df.head(10)
print(train_df.shape)

print(test_df.shape)
test_df.head()
#testing = pd.concat([test_df,gender_submission_df],axis=1)

#testing = testing.drop(['PassengerId'],axis=1)
print(testing.shape)

print(testing.shape)

train_df.head()
train_df.head()
target_field = ['Survived']

train_outputs,train_feature = train_df[target_field],train_df.drop(['Survived'],axis=1)

#testing_output,testing_features = testing[target_field],testing.drop(['Survived'],axis=1)
print(train_outputs.shape)

print(train_feature.shape)

#print(testing_output.shape)

#print(testing_features.shape)

        
print(train_feature.head())
print(train_outputs.head())
#check when no hidden layer is passed

#currently only work for single output

class MiniFlow():

     #input - actual input, output - actual output , hidden layer - array of size of hidden layers

     # preprocess_function - it will return train_data,train_output

     # learning_rate = how fast learning rate should be.

        

    def __init__(self,inputs,outputs,hidden_layers,preprocess_function,learning_rate = 0.1):

        

        np.random.seed(0)

        self.inputs,self.outputs = preprocess_function(inputs,outputs)

        self.initialization(inputs,outputs,hidden_layers,learning_rate)

        

    def initialization(self,inputs,outputs,hidden_layers,learning_rate):

        

        self.learning_rate = learning_rate

        self.number_of_hidden_layers = len(hidden_layers)

        self.weights =  dict()

        self.output_size = outputs.shape[1]

        row = inputs.shape[1]

        col = hidden_layers[0]

        

        counter = 0

        for i in range(self.number_of_hidden_layers+1):

            print('row',row,'col',col)

            self.weights.update({i:np.random.rand(row,col)})

            row = col

            if i + 1 >= self.number_of_hidden_layers:

                col = self.output_size

            else:

                col = hidden_layers[counter+1]

            counter = counter+1

            

    def get_weights(self):

        return self.weights                

            

    def train(self,number_of_epoch,batch_size):

        

        #currently we not using epoch and batch size

        for j in tqdm(range(number_of_epoch)):

            batch = np.random.choice(len(self.inputs), size=batch_size)

            X,Y = self.inputs[batch],self.outputs[batch]       

            for i in range(len(X)):

                self.feed_forward_backward(X[i],Y[i],i)                  

        pass

    def feed_forward(self,inputs):

        

        result = inputs

        self.temp_inputs = dict()

        for i in range(self.number_of_hidden_layers+1):

            self.temp_inputs.update({i:np.asmatrix(result)})

            result = np.array(np.dot(result,self.weights[i]),dtype=np.float32)

        result = self.sigmoid(result)

        return result



    def feed_forward_backward(self,inputs,outputs,counter_value):

        

        prediction = self.feed_forward(inputs)



        #last layer error and delta

        error = self.error_calculation(outputs,prediction)

        error_delta = error * self.sigmoid_derivative(prediction)        

        error_delta = error_delta.reshape(1,1)



        #actual back propogation

        for i in reversed(range(self.number_of_hidden_layers+1)):



            current_weight = self.weights[i]

            temp = (self.temp_inputs[i].T.dot(error_delta)*self.learning_rate)

            

            self.weights[i] = self.weights[i] + temp

            error = np.asmatrix(error_delta.dot(current_weight.T))

            error_delta = error

        

        

    def error_calculation(self,actual,predicted):

        return actual - predicted

        

    def sigmoid(self,data):

        return 1.0/(1.0+np.exp(-data))

    

    def sigmoid_derivative(self,data):

        return data * (1.0- data)

           

def process_data(inputs,outputs):

    return inputs,outputs
neuralNetwork = MiniFlow(inputs=train_feature.values,outputs=train_outputs.values,

                         hidden_layers=[5,3],learning_rate=0.01,preprocess_function=process_data)
neuralNetwork.train(10000,64)
correct = 0 

arr = np.empty_like(gender_submission_df.values);

for i in range(len(test_df.values)):

    passenger_id = test_df.PassengerId.values[i]

    r = neuralNetwork.feed_forward(test_df.drop('PassengerId',axis=1).values[i])

    temp = 0;

    if r > 0.50:

        temp =  1 

    actual_result = gender_submission_df.loc[np.where(gender_submission_df['PassengerId'] == passenger_id)[0][0]]['Survived'] 



    arr[i][0] = passenger_id

    arr[i][1] = actual_result

    if(temp == actual_result):

        correct = correct+1

        

print(correct/len(test_df)*100)       

        

        
output_df = pd.DataFrame(data=arr)
output_df.head()
output_df.columns = ['PassengerId','Survived']
output_df.head()
output_df.to_csv('output.csv',encoding='utf-8',columns=['PassengerId','Survived'],index=False)