import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import style

import math

%matplotlib inline
raw_data = pd.read_csv('../input/train.csv', index_col = 0);

raw_data.head()
raw_data.Embarked.isnull().any()
raw_data.Embarked.fillna('N', inplace = True);

dataEmb = raw_data.groupby(['Embarked','Sex']).mean().Survived.unstack().sort_values(by = 'male');

dataEmb.plot.bar();
raw_data.Pclass.isnull().any()
dataPclass = raw_data.groupby(['Pclass','Sex']).mean().Survived.unstack().sort_values(by = 'male');

ax= dataPclass.plot.bar();

raw_data.Age.nunique()
raw_data.Age.isnull().any()
AgeBins =np.array([-1,0,5,14,20,35,60,max(raw_data.Age)])

AgeLabel = ['Unknown','Baby','Child','Teenager','Young Adult','Adult','Senior']

AgeData = raw_data.Age.fillna(-0.5)

AgeSorted = pd.cut(AgeData, bins = AgeBins, labels = AgeLabel)

raw_data['AgeRank'] = AgeSorted
raw_data.groupby(['AgeRank','Sex']).Survived.mean().unstack().sort_values(by = 'male').plot.bar();
raw_data.Cabin.isnull().any()
raw_data.Cabin.unique()
raw_data.Cabin = raw_data.Cabin.fillna('N');

new_Cabins = []

for cabinString in raw_data.Cabin:

    new_Cabins.append(cabinString[0]);

raw_data['NewCabins'] = new_Cabins

raw_data.groupby(['NewCabins','Sex']).Survived.mean().unstack().sort_values(by = 'male').plot.bar();
raw_data[['Fare','Pclass']].corr()
raw_data.Fare.isnull().any()
raw_data.Fare.describe()[3:8]
FareQR =np.array(list(map(int,list(raw_data.Fare.describe()[3:8]))));

FareBins = np.concatenate((np.array([-1]),FareQR));

FareBins[-1] +=1;

FareLabel = ['Unknown','1st','2nd','3rd','4th'];

FareData = raw_data.Fare;

FareSorted = pd.cut(FareData, bins = FareBins, labels = FareLabel);

raw_data['NewFare'] = FareSorted;

raw_data.groupby(['NewFare','Sex']).Survived.mean().multiply(100).unstack().sort_values(by = 'male').plot.bar(title = 'Survivability by Fare group');

raw_data.Name.head(10)
Names = list(raw_data.Name)

LNames = []

for name in Names:

    LName = name.split(',')[1].split()[0]

    LNames.append(LName)



raw_data['LName'] = LNames
raw_data.groupby(['LName','Pclass']).Survived.agg(['mean', 'count'])
raw_data.groupby(['LName']).mean().Survived.sort_values().plot.bar();
def AgePreProcess(raw_data):

    AgeBins =np.array([-1,0,5,14,20,35,60,max((raw_data.Age))+1]);

    AgeLabel = ['Unknown','Baby','Child','Teenager','Young Adult','Adult','Senior'];

    if raw_data.Age.isnull().any() == True:

        AgeData = raw_data.Age.fillna(-0.5);

    else:

        AgeData = raw_data.Age

    AgeSorted = pd.cut(AgeData, bins = AgeBins, labels = AgeLabel);

    raw_data.Age = AgeSorted;



def CabinPreProcess(raw_data):

    raw_data.Cabin = raw_data.Cabin.fillna('N');

    new_Cabins = [];

    for cabinString in raw_data.Cabin:

        new_Cabins.append(cabinString[0]);

    raw_data.Cabin = new_Cabins

    

def FarePreProcess(raw_data):

    a = np.array(list(map(int,list(raw_data.Fare.describe()[4:]))));

    b = np.array([-1,0]);

    FareBins =np.concatenate((b,a));

    FareBins[-1] = FareBins[-1] + 1;

    FareLabel = ['Unknown','1st','2nd','3rd','4th'];

    if raw_data.Fare.isnull().any() == True:

        FareData = raw_data.Fare.fillna(-0.5);

    else:

        FareData = raw_data.Fare;

    FareSorted = pd.cut(FareData, bins = FareBins, labels = FareLabel);

    raw_data.Fare = FareSorted;



def LNamesPreProcess(raw_data):

    Names = list(raw_data.Name);

    LNames = [];

    for name in Names:

        LName = name.split(',')[1].split()[0];

        LNames.append(LName);

    raw_data['LName'] = LNames;



def EmbarkedPreProcess(raw_data):

    raw_data.Embarked = raw_data.Embarked.fillna('N');



def DataPreProcess(raw_data):

    AgePreProcess(raw_data);

    CabinPreProcess(raw_data);

    FarePreProcess(raw_data);

    LNamesPreProcess(raw_data);

    EmbarkedPreProcess(raw_data);

    raw_data = raw_data.reset_index();

    raw_data.drop(['Name','Ticket',], axis = 1, inplace = True);

    return raw_data
data_train = pd.read_csv('../input/train.csv', index_col = 0);

data_train = DataPreProcess(data_train)
data_train.head()
def changing_string(df, List = None):

    features = ['Fare', 'Cabin', 'Age','Embarked'];

    LAR = {};

    #The if is necessary since this algorithm should run also for the test training

    if 'Survived' in df.columns: 

        for feature in features:

            new_list = list(df.groupby([feature,'Sex']).mean().Survived.unstack().sort_values(by = 'male').index);

            new_replace = list(map(int,list(range(len(new_list)))));

            LAR[feature] = [new_list,new_replace];

            df[feature].replace(to_replace = new_list,value = new_replace, inplace = True);



        #For the rest of the features (LName and Sex), we will do it manually.

        LNameList = list(df.groupby(['LName']).mean().Survived.sort_values().index);

        LNameValue = list(map(int,list(range(len(LNameList)))));

        LAR['LName'] = [LNameList,LNameValue];

        df.LName.replace(to_replace = LNameList, value = LNameValue, inplace = True);

        df.Sex.replace(to_replace = ('male','female'),value = (0,1), inplace = True);

        return df, LAR

    else:

        if List == None:

            print('Need to run train set first through. If done, include the List name into the changing_string');

            exit();

        else:

            for feature in features:

                df[feature].replace(to_replace = List[feature][0], value = List[feature][1], inplace =True);

            df.Sex.replace(to_replace = ('male','female'),value = (0,1), inplace = True);

            df.LName.replace(to_replace = List['LName'][0], value = List['LName'][1], inplace = True);

            return df

            
train, List = changing_string(data_train)
def LearningAlgorithm(What = None, iterations = 50):

    from sklearn.model_selection import train_test_split

    X_all = train.drop(['Survived', 'PassengerId'], axis=1);

    y_all = train['Survived'];

    if What != None:

        X_all = X_all.drop(What, axis=1);

    svms = np.array([]);

    gnbs = np.array([]);

    SGDs = np.array([]);



    for i in range(iterations):

        num_test = 0.40;

        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test);

        from sklearn import svm

        clf = svm.SVC();

        clf.fit(X_train, y_train);

        y_predicted = clf.predict(X_test);

        z1 = clf.score(X_test, y_test);

        svms = np.append(svms,z1);



        from sklearn.naive_bayes import GaussianNB

        gnb = GaussianNB();

        gnb.fit(X_train, y_train);

        y_predict = gnb.predict(X_test);

        z2 = gnb.score(X_test,y_test)

        gnbs = np.append(gnbs,z2);





    print("SVM " + str(round(svms.mean(),4)));

    print("GNB " + str(round(gnbs.mean(),4)));
LearningAlgorithm()
LearningAlgorithm(['Fare','Pclass'])

print('\n')

LearningAlgorithm('LName')

print('\n')

LearningAlgorithm('Cabin')
data_train = pd.read_csv('../input/train.csv', index_col = 0);

data_train = DataPreProcess(data_train);

train, List = changing_string(data_train);
data_test = pd.read_csv('../input/test.csv', index_col = 0);

data_test = DataPreProcess(data_test);

data_test.LName.unique()
test = changing_string(data_test,List)

test.head()
from sklearn import svm

from sklearn.model_selection import train_test_split

clf = svm.SVC();

X_all = train.drop(['Survived', 'PassengerId','LName'], axis=1);

y_all = train['Survived'];

num_test = 0.40;

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test);

clf.fit(X_train, y_train);





testPrediction = clf.predict(data_test.drop(['PassengerId','LName'],axis =1));

ids = data_test['PassengerId'];

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': testPrediction })

output.to_csv('titanic-predictions.csv', index = False)