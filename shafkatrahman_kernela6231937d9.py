# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_dir = '/kaggle/input/titanic/train.csv'

test_dir = '/kaggle/input/titanic/test.csv'

train_set = pd.read_csv(train_dir)

test_set= pd.read_csv(test_dir)
train_set.head()

train_set.isnull().sum()
dic = {}

for data in train_set['Cabin']:

    #print(type(data))

    if isinstance(data, str):

        dic[data[0]]= True

print(dic.keys())
dic['Unknown'] = True

cabin_list =sorted(list(dic.keys()))

print(cabin_list)

def search_substring(big_string, substring_list):

    for substring in substring_list:

        if substring in big_string:

            return substring

    return substring_list[-1]
# replace passenger's name with his/her title (Mr, Mrs, Miss, Master)

def get_title(string):

    import re

    regex = re.compile(r'Mr|Don|Major|Capt|Jonkheer|Rev|Col|Dr|Mrs|Countess|Dona|Mme|Ms|Miss|Mlle|Master', re.IGNORECASE)

    results = regex.search(string)

    if results != None:

        return(results.group().lower())

    else:

        return(str(np.nan))

title_dictionary = {

    "capt":"Officer", 

    "col":"Officer", 

    "major":"Officer", 

    "dr":"Officer",

    "jonkheer":"Royalty",

    "rev":"Officer",

    "countess":"Royalty",

    "dona":"Royalty",

    "lady":"Royalty",

    "don":"Royalty",

    "mr":"Mr",

    "mme":"Mrs",

    "ms":"Mrs",

    "mrs":"Mrs",

    "miss":"Miss",

    "mlle":"Miss",

    "master":"Master",

    "nan":"Mr"

}
train_set['Deck'] = train_set['Cabin'].map(lambda x:search_substring(str(x), cabin_list))

test_set['Deck'] = test_set['Cabin'].map(lambda x:search_substring(str(x), cabin_list))

train_set.drop('Cabin',1,inplace = True)

test_set.drop('Cabin',1,inplace = True)



train_set['Title'] = train_set['Name'].apply(get_title)

test_set['Title'] = test_set['Name'].apply(get_title)

train_set.drop('Name',1,inplace = True)

test_set.drop('Name',1,inplace = True)



train_set['Title'] = train_set['Title'].map(title_dictionary)

test_set['Title'] = test_set['Title'].map(title_dictionary)
means_title = train_set.groupby('Title')['Age'].mean()

title_list = ['Mr','Miss','Mrs','Master', 'Royalty', 'Officer']

def age_nan_replace(means, dframe, title_list):

    for title in title_list:

        temp = dframe['Title'] == title #extract indices of samples with same title

        dframe.loc[temp, 'Age'] = dframe.loc[temp, 'Age'].fillna(means[title]) # replace nan values for mean
age_nan_replace(means_title,train_set,title_list)

age_nan_replace(means_title,test_set,title_list)
train_set['Embarked'].fillna('S',inplace=True)

test_set['Embarked'].fillna('S',inplace=True)
train_set.head()
import matplotlib.pyplot as plt

%matplotlib inline

index = train_set['Survived'].unique() # get the number of bars

grouped_data = train_set.groupby(['Survived', 'Sex']) 

temp = grouped_data.size()

print(temp)

temp = temp.unstack() 

print(temp)

women_stats = (temp.iat[0,0], temp.iat[1,0])

men_stats = (temp.iat[0,1], temp.iat[1,1])

p1 = plt.bar(index, women_stats)

p2 = plt.bar(index, men_stats, bottom=women_stats)

plt.xticks(index, ('No', 'Yes'))

plt.ylabel('Number of People')

plt.xlabel('Survival')

plt.title('Survival of passengers')

plt.legend((p1[0], p2[0]), ('Women', 'Men'))

plt.tight_layout()



train_set.pivot_table('Survived',index='Sex',columns='Pclass').plot(kind='bar')

train_set.pivot_table('Survived',index='Title',columns='Pclass').plot(kind='bar')

age_intervals = pd.qcut(train_set['Age'], 3)

train_set.pivot_table('Survived', ['Sex', age_intervals], 'Pclass').plot(kind='bar')
parch_intervals = pd.cut(train_set['Parch'], [0,1,2,3])

sibsp_intervals = pd.cut(train_set['SibSp'], [0,1,2,3])

train_set.pivot_table('Survived', parch_intervals, 'Sex').plot(kind='bar')
train_set.pivot_table('Survived', sibsp_intervals, 'Sex').plot(kind='bar')
train_set['Family Size'] = train_set['Parch'] + train_set['SibSp']

test_set['Family Size'] = test_set['Parch'] + test_set['SibSp']

train_set.drop('Parch',axis=1,inplace=True)

train_set.drop('SibSp',axis=1,inplace=True)

test_set.drop('Parch',axis=1,inplace=True)

test_set.drop('SibSp',axis=1,inplace=True)
train_set.drop('Ticket',axis=1,inplace=True)

test_set.drop('Ticket',axis=1,inplace=True)
train_set.head()
from sklearn.preprocessing import StandardScaler 



numericals_list = ['Fare','Age']



for column in numericals_list:

    sc = StandardScaler(with_mean = True, with_std = True)

    sc.fit(train_set[column].values.reshape(-1,1))

    train_set[column] = sc.transform(train_set[column].values.reshape(-1,1))

    test_set[column] = sc.transform(test_set[column].values.reshape(-1,1))
from sklearn.preprocessing import LabelEncoder



categorical_list = ['Sex','Embarked','Deck','Title']



for column in categorical_list:

 

    le = LabelEncoder()

    le.fit(train_set[column])

    train_set[column]=le.transform(train_set[column])

    test_set[column]=le.transform(test_set[column])

train_set.head()

test_set.head()
train_set = pd.get_dummies(train_set, columns =  ['Pclass','Embarked','Deck','Title'])

test_set = pd.get_dummies(test_set, columns =  ['Pclass','Embarked','Deck','Title'])
train_set, test_set = train_set.align(test_set, axis=1)

test_set.drop(['Survived'],axis=1,inplace = True)

test_set.fillna(0,inplace=True)
y = train_set['Survived'].values

X = train_set.drop(['Survived','PassengerId'],axis=1).values

X_test = test_set.drop('PassengerId', axis=1).values
print(X.shape)

print(y.shape)

print(X_test.shape)
import sys





def progress(prefix,count,total,name1,value1,name2,value2,suffix=''):

    bar_len = 60

    filled_len = int(round(bar_len * count / float(total)))



    percents = round(100.0 * count / float(total), 1)

    bar = '=' * filled_len + '-' * (bar_len - filled_len)



    sys.stdout.write('[%s] %s : %s%s..%s %.5f %s %.5f%s %s:\r' % (bar,prefix, percents, '%',name1,value1,name2,value2,'%', suffix))

    sys.stdout.flush()  # As suggested by Rom Ruben


import os

import torch

from torch.utils.data import DataLoader

import random

import scipy.stats as stats

import torch.optim as optim

import torch.nn as nn





class Model(nn.Module):

    def __init__(self):

        super(Model,self).__init__()

        self.fc1 = nn.Linear(25,18) # 18 got best yet

      

        self.fc2 = nn.Linear(18,2)



        self.drop_layer = nn.Dropout(p=0.2)

      #  self.relu = nn.ReLU()



    def forward(self,x):

        #print(x.shape)

        x = self.drop_layer(torch.tanh(self.fc1(x)))



        x = self.fc2(x)



        return x

class Dataset(torch.utils.data.Dataset):

    def __init__(self,X,y = None,mode = None):

        self.Data = X

        self.labels = y

        self.mode = mode

        

    def __len__(self):

        return self.Data.shape[0]

    def __getitem__(self,ix): 

        d=self.Data[ix]

       

        if self.mode=='test':

            d = torch.from_numpy(d).float()

            return d

        

        l=self.labels[ix]

        #d=d.reshape(-1,1)

        #l=l.reshape(-1)

        d = torch.from_numpy(d).float()

        

        return d,l
def tarin_model(model,dataset,criterion,max_epochs):



    model.train()

    loss_list = []

    accuracy_list = []

    

    

    for epoch in range(max_epochs):

        iteration = 0

        total_loss = 0

        total_accuracy = 0

        for data in dataset:

            x,lb = data 

            y_predict = model(x)

            loss = criterion(y_predict,lb)



            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            

            iteration+=1

            total_loss+= loss.item()

            

            accuracy = 0

            with torch.no_grad():

                y_predict = y_predict[:,1]

                y_predict = (y_predict >= 0.5)

                #print(y_predict.shape)

                #print(lb.shape)

                correct = (y_predict == lb).float().sum()

                accuracy = correct / x.shape[0] *100

                total_accuracy += accuracy

            progress('epoch : {}: '.format(epoch),iteration,len(dataset),'loss=',loss,'accuracy=',accuracy)

        

        total_accuracy/=iteration

        total_loss/=iteration

        loss_list.append(total_loss)

        accuracy_list.append(accuracy.item())

        

        

      #  print('[Info]...epoch:{}'.format(epoch))

       

    return loss_list,accuracy_list
def test_model(model,dataset,criterion):



    model.eval()

    

    y_pred = []

    

    for data in dataset:

        x = data 

        y_predict = model(x)

    

        y_predict = y_predict[:,1]

        y_predict = (y_predict >= 0.5)

        

        for i in range(y_predict.shape[0]):

            y_pred.append(int(y_predict[i].item()))

        

    return y_pred

        

use_cuda = torch.cuda.is_available()

device = torch.device("cuda:0" if use_cuda else "cpu")



params = {'batch_size': 891,

          'shuffle': True,

         }

params_test = {'batch_size': X_test.shape[0],

               'shuffle': False,

              }



training_set = Dataset(X,y,mode='train')

testing_set = Dataset(X_test,mode='test')





training_generator = torch.utils.data.DataLoader(training_set, **params)

test_generator = torch.utils.data.DataLoader(testing_set, **params_test)



model = Model()

parameters_2_optimize = list(model.parameters())

optimizer = optim.Adam(parameters_2_optimize, lr=0.01)

criterion = nn.CrossEntropyLoss()









loss_list,accuracy_list = tarin_model(model,training_generator,criterion,2000)
test_prediction = test_model(model,test_generator,criterion)
test_set_new= pd.read_csv(test_dir)

df = test_set_new['PassengerId']

IDs = df.tolist()

df = {  

        'PassengerId' : IDs,

        'Survived' : test_prediction



     }

df = pd.DataFrame(df)

df.head()
df.to_csv('submission.csv',index=False)
plt.plot(  loss_list,color='red',linewidth=3,label='loss',marker='o', linestyle='dashed')

plt.xlabel("Epoch #")

plt.ylabel("loss")

plt.legend()

plt.ylim(0,max(loss_list))



plt.xlim(0,7000)

plt.rcParams["figure.figsize"] = (30,30)

plt.show()

print(min(loss_list))

plt.clf()

plt.plot( accuracy_list,color='green',linewidth=1,label='accuracy',marker='', linestyle='dashed')

plt.xlabel("Epoch #")

plt.ylabel("accuracy")

plt.legend()

plt.ylim(min(accuracy_list),100)



plt.xlim(0,7000)

plt.rcParams["figure.figsize"] = (30,30)

plt.show()

print(max(accuracy_list))
#kaggle competitions submit -c titanic -f submission.csv -m "wish me luck"