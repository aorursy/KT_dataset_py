path = '../input/apy.csv'
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
df = pd.read_csv(path)
df.dropna(inplace=True)
area  = df['Area']
                
from sklearn.preprocessing import LabelEncoder

seasonl = LabelEncoder()

districtl = LabelEncoder()

statel = LabelEncoder()





Season = seasonl.fit_transform(df['Season'].str.lower())

District_Name = districtl.fit_transform(df['District_Name'].str.lower())

State_Name =statel.fit_transform(df['State_Name'].str.lower())

Crop = LabelEncoder().fit_transform(df['Crop'])

                


Season = np.asarray(Season)

District_Name = np.asarray(District_Name)

State_Name = np.asarray(State_Name)

Crop = np.asarray(Crop)
Crop
Production = np.asarray(df['Production'])
area =  np.asarray(df['Area'])
y = Crop
#from sklearn.preprocessing import MinMaxScaler
#area = np.asarray(area)
#Production = np.reshape(Production,(len(Production),1))

#area = np.reshape(area,(len(Production),1))
#Production = MinMaxScaler().fit_transform(Production)

#area = MinMaxScaler().fit_transform(area)
#Production = np.reshape(Production,(len(Production),))
#area = np.reshape(area,(len(Production),))
xtrain = np.stack((State_Name,District_Name,Season,Production,area),axis=1)


#Production




    
penalty = ['l1', 'l2']



C = np.logspace(0, 4, 10)



hyperparameters = dict(C=C, penalty=penalty)
from sklearn.model_selection import GridSearchCV

from sklearn import linear_model

param_grid = { 

    'n_estimators': [200, 500],

    'max_features': ['sqrt', 'log2'],

    'criterion' :['gini', 'entropy']

}
import sklearn
from sklearn.ensemble import RandomForestClassifier

regressor = RandomForestClassifier()  
#clf = GridSearchCV(regressor, param_grid, verbose=2)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(xtrain, Crop, test_size=0.1, random_state=42)
xtrain
rfc1=RandomForestClassifier( max_features='log2',verbose=1,n_jobs=2 ,n_estimators= 20, max_depth=None, criterion='gini')
predict = rfc1.fit(x_train,y_train)
seasonArray = seasonl.classes_

districtArray = districtl.classes_

stateArray = statel.classes_
stateArray
districtArray
len('Autumn')
len(seasonArray[0])
seasons = {}

district = {}

state = {}

for i in range(0,len(seasonArray)-1):

    seasons.update({i:seasonArray[i][:-5]})

for i in range(0,len(districtArray)-1):

    district.update({i:districtArray[i]})

for i in range(0,len(stateArray)-1):

    state.update({i:stateArray[i]})

    
def get_key(my_dict,val): 

    for key, value in my_dict.items(): 

         if val == value : 

             return key 

  

    return "key doesn't exist"
get_key(state,'Assam')
x_test
x_train[0]


ypred = rfc1.predict(x_test)
import pickle





with open('model.pickle', 'wb') as f:

    pickle.dump(rfc1 ,f)
ypred[61]
def score(y,yhat):

    count = 0

    for i in range(len(y)):

        if(y[i]==yhat[i]):

            count += 1

    score = count/len(y)

    return score*100
score(y_test,ypred)
le = LabelEncoder()

Crop = le.fit_transform(df['Crop'])
le.classes_
print('Enter 5 variables which are StateName,District_name,Season,Production,area i.e  0 1 3 200 157')

string_data = input()

string_data = str(string_data)
string_data = string_data.split()
ypred = rfc1.predict([string_data])

ypred =le.inverse_transform(ypred)
print('The crop we recommend on this soil is :'+ ypred)
x_train
 

