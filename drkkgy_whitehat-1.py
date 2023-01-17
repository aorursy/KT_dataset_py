# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random

n = random.randint(0,22)

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import os

for dirname, _, filenames in os.walk('/kaggle/output'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
## Loading data in a data frame 

data = pd.read_csv("/kaggle/input/titanic/train.csv")

 

print(data)



## Exploring the columns



#data['Pclass'].hist() ##Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)

#data['Sex'].hist()

#data['Age'].hist()

#data['SibSp'].hist()  ##Number of Siblings/Spouses Aboard.

#data['Parch'].hist() ##Number of Parents/Children Aboard

#data['Fare'].hist() 

#data['Embarked'].hist() ## Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)



## Reading Test Data

data1 = pd.read_csv("/kaggle/input/titanic/test.csv")

data11 = pd.read_csv("/kaggle/input/titanic/test.csv")





## Analysisng missing values 

def miss_column(data):

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):

            print("Percentage of Missing Values Column Wise:",data.isnull().sum()*100/(data.isnull().sum() + data.count()))

            print("Data details:",data.describe())

            

miss_column(data)



## droping columns beacuse of Nan Values



data = data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

data1 = data1.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
## extracting data and removing incompatibile features

#data3 = pd.read_csv("/kaggle/working/output/whitehat30.csv")



def pre_process(data):

    ## Performing Normalization 

    data.loc[:,'Age'] = (data.loc[:,'Age'])/max(data.loc[:,'Age'])

    data.loc[:,'Fare'] = (data.loc[:,'Fare'])/max(data.loc[:,'Fare'])



    ## Performing feature transformation

    ismale = []

    isfemale = []

    inclass1 = []

    inclass2 = []

    inclass3 = []

    inembarkC = []

    inembarkS = []

    inembarkQ = []

#------------------------------------

    age_avg = data['Age'].mean()

    fare_avg = data['Fare'].mean()

    iter = 0

    for x in data.loc[:,'Age']:

        if(str(x) == 'nan'):

            data.loc[iter:,'Age'] = age_avg

        iter = iter+1

            

    iter = 0

    for x in data.loc[:,'Fare']:

        if(str(x) == 'nan'):

            data.loc[iter:,'Fare'] = fare_avg

        iter = iter+1

    

  #-------------------------------------------          

    for x in data.loc[:,'Sex']:

        if x == 'male':

            ismale.append(1)

            isfemale.append(0)

        else:

            ismale.append(0)

            isfemale.append(1)

    for x in data.loc[:,'Pclass']:

        if x == 1:

            inclass1.append(1)

            inclass2.append(0)

            inclass3.append(0)

        elif x == 2:

            inclass1.append(0)

            inclass2.append(1)

            inclass3.append(0)

        else:

            inclass1.append(0)

            inclass2.append(0)

            inclass3.append(1)

        

    for x in data.loc[:,'Embarked']:

        if x == 'C':

            inembarkC.append(1)

            inembarkS.append(0)

            inembarkQ.append(0)

        elif x == 'S':

            inembarkC.append(0)

            inembarkS.append(1)

            inembarkQ.append(0)

        elif x == 'Q':

            inembarkC.append(0)

            inembarkS.append(0)

            inembarkQ.append(1)

        else:

            inembarkC.append(1)

            inembarkS.append(1)

            inembarkQ.append(1)

        

                      

    ## Adding new features

    data['ismale'] = ismale

    data['isfemale'] = isfemale

    data['inclass1'] = inclass1

    data['inclass2'] = inclass2

    data['inclass3'] = inclass3

    data['inembarkC'] = inembarkC

    data['inembarkS'] = inembarkS

    data['inembarkQ'] = inembarkQ

    data = data.drop(['Sex','Pclass','Embarked'],axis=1)

    

    return data



# processeing data with custom function

train_data = pre_process(data)

test_data = pre_process(data1)





X_train = train_data.loc[:,['Age','SibSp','Parch','Fare','ismale','isfemale','inclass1','inclass2','inclass3'

                           ,'inembarkC','inembarkS','inembarkQ']]



y_train = train_data.loc[:,'Survived']





X_test = test_data.loc[:850,['Age','SibSp','Parch','Fare','ismale','isfemale','inclass1','inclass2','inclass3'

                           ,'inembarkC','inembarkS','inembarkQ']]



X_test1 = data.loc[850:891,['Age','SibSp','Parch','Fare','ismale','isfemale','inclass1','inclass2','inclass3'

                           ,'inembarkC','inembarkS','inembarkQ']]

y_test = data.iloc[850:891]['Survived'].values



'''

### Layer Values

layers = [1,37,78,74] #1 37 78 74

###

#Training NL classifier

#clf = MLPClassifier(random_state=1,learning_rate_init=0.0001,hidden_layer_sizes=(10,25,25,10),max_iter=1000000 ).fit(X_train, y_train)

from sklearn.neural_network import MLPClassifier



clf = MLPClassifier(random_state=1,learning_rate_init=0.0001,hidden_layer_sizes=(layers[0],layers[1],layers[2],layers[3]),max_iter=1000000 ).fit(X_train, y_train)



result_df = clf.predict(X_test)



#--------------------------------------------------------------

layers = [97,0,1]

# training Ensemble modles

from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(max_depth= layers[0], random_state= 0,n_estimators =layers[2])



clf.fit(X_train, y_train)



result_df = clf.predict(X_test)



from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=8)

neigh.fit(X_train, y_train)

result_df = neigh.predict(X_test)



from sklearn.ensemble import AdaBoostClassifier



clf = AdaBoostClassifier(learning_rate = 0.1,n_estimators=1000, random_state=0)

clf.fit(X_train, y_train)

result_df = clf.predict(X_test)



### Try this out tomorrow ###

from sklearn.linear_model import LogisticRegression

layers = [100,0,1]

clf = LogisticRegression(solver = "liblinear" , max_iter = 10000000,class_weight="balanced")

# Fit our model to the training data

clf.fit(X_train, y_train)

# Predict on the test data

result_df = clf.predict(X_test)

'''

layers = [100,0,1]

from sklearn.ensemble import GradientBoostingClassifier

#clf = GradientBoostingClassifier(random_state=0,learning_rate = 0.1,max_depth = 3,n_estimators = 80) Best Scorer

clf = GradientBoostingClassifier(random_state=0,learning_rate = 0.1,max_depth = 3,n_estimators = 180)

clf.fit(X_train, y_train)

result_df = clf.predict(X_test)

#-------------------------

## Processing Submission CSV

Survival = list(result_df)



result_predict = list(data11.loc[:,'PassengerId'])

dataX = zip(result_predict,Survival)

result_predict1 = pd.DataFrame(dataX, columns = ['PassengerId', 'Survived']) 



#print(result_predict1)



##--------------------------

outname = 'whitehat59.csv'



outdir = './output'

if not os.path.exists(outdir):

    os.mkdir(outdir)



fullname = os.path.join(outdir, outname)    







##--------------------------



result_predict1.to_csv(fullname,index=False)



print(clf.score(X_test1, y_test))



score1 = clf.score(X_test1, y_test)

##Automating training to get approx good model

break_count = 0

prev_score = 0

while(score1 < 90 ):

   # clf = MLPClassifier(random_state=1,learning_rate_init=0.0001,hidden_layer_sizes=(layers[0],layers[1],layers[2],layers[3]),max_iter=1000000 ).fit(X_train, y_train)

    #  clf = RandomForestClassifier(max_depth=layers[0], random_state=0,n_estimators = layers[2])

   # clf.fit(X_train, y_train)

   # clf = LogisticRegression(solver = "liblinear" , max_iter = layers[0],class_weight="balanced")

    #clf.fit(X_train, y_train)

    clf = GradientBoostingClassifier(random_state=0,learning_rate = 0.01,max_depth = layers[0])

    clf.fit(X_train, y_train) 

    score1 = clf.score(X_test1, y_test)

    print(str(score1) + " " + str(break_count) + " " + str(layers) )

    layers[0] = random.randint(1,50)

    layers[1] = random.randint(1,100)

    layers[2] = random.randint(1,400)



    if(break_count < 10):

        break_count = break_count + 1

    else:

        break

    

print(layers)

    

#83 85 8 47

# 37 47 71 72

# 18 54 43 68

# 71 30 61 6

# 12 29 22 85 :- 86

#12 18 41 75 :- 87

for x in data.loc[:,'Age']:

    if(str(x) == 'nan'):

        data.loc[1:,'Age'] = 0.0

print(data)

   

data = pd.read_csv("/kaggle/input/titanic/train.csv")

print(data)
y_val=data.iloc[800:891]['Survived'].values

print(y_val)
clf.get_params()