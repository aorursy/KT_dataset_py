# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing as pp

#Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.cross_validation import KFold

from sklearn.model_selection import train_test_split

import re





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Analysis can be found here

#https://public.tableau.com/profile/shashank.gupta1991#!/vizhome/TitanicPredictions/PclassvsCNTSurvived
train_df = pd.read_csv("../input/train.csv")

Y = train_df.Survived

test_df = pd.read_csv("../input/test.csv")

merged = train_df.append(test_df)

#merged.drop(['Survived','PassengerId'],axis=1,inplace=True)

#print(merged.columns)

merged.head()
def fillMissingValues(data):

      

    #Filling average of the missing fare pclass

    Pclasses = merged.loc[merged.Fare.isnull()].Pclass.values

    for Pclass in Pclasses:

        df = merged.groupby('Pclass',as_index=False)['Fare'].mean()

        val = df.loc[df.Pclass == Pclass].Fare.values[0]

        data.Fare.fillna(val,inplace=True)

        

    #Partially Filling Missing Cabins

    data = fillMissingCabin(data)

    return data

   

def fillMissingCabin(data):

    grp = data.groupby(['PreTkt','TktNum'])['CabinCode'].unique()

    grp = grp[grp.apply(lambda x: len(x)>1)]

    df = pd.DataFrame(grp)

    #print(df.index.values.tolist())

    df.CabinCode.fillna(np.nan,inplace=True)

    index = df.index.values.tolist()

    CabinCode = df.CabinCode.values.tolist()

    idxMap = {}

    for idx, cc in zip(index,CabinCode):

        if idx[0] != "NA":

            if pd.isnull(cc[0]) and pd.notnull(cc[1]):

                idxMap[idx] = cc[1]

            elif pd.isnull(cc[1]) and pd.notnull(cc[0]):

                idxMap[idx] = cc[0]

            if idx in idxMap:

                #print("here")

                data.loc[((data.PreTkt==idx[0]) & (data.TktNum == idx[1]) & (pd.isnull(data.CabinCode))),'CabinCode'] = idxMap[idx]

                

    data.CabinCode.fillna('U',inplace=True)

    data = pd.concat([data,pd.get_dummies(data.CabinCode)],axis=1)

    return data  

    

def processTicket(data):

    regex = re.compile("(.*)?( [0-9]+)")

    data.Ticket = data.Ticket.map(lambda x : x if type(x) != str else x.replace(".","").replace("/",""))

    splitTkt = data.Ticket.map(lambda x: x if type(x)!=str else [y.strip().upper().replace(' ',"") for y in regex.split(x) if y])

    preTkt = [val[0] if len(val)>1 else "NA" for val in splitTkt.values]

    TktNum = [val[1] if len(val) > 1 else val[0] for val in splitTkt.values]

    data['PreTkt'] = pd.Series(preTkt)

    data['TktNum'] = pd.Series(TktNum)

    data.loc[(data.TktNum == 'LINE'),'TktNum'] = '111111'

    data = pd.concat([data,pd.get_dummies(data.PreTkt)],axis=1)

    return data

    

def processSex(data):

    print('Starting sex feature ....')

    #Using OneHotEncoder for Sex

    data = pd.concat([data,pd.get_dummies(data.Sex)],axis=1)

    print('Ending sex feature ....')

    return data

    

def processEmbarked(data):

    print('Starting Embarked feature ....')

    df = data.Embarked.dropna()

    #Filling Most Frequent Value for Embarked

    data.Embarked.fillna(df.value_counts().idxmax(),inplace=True)

    

    #Using OneHotEncoder for Embarked

    columns = ['SE','CE','QE']

    data = pd.concat([data,pd.get_dummies(data.Embarked,columns= columns)],axis=1)

    data = data.rename(columns={'S':'SE','C':'CE','Q':'QE'})

    print('Ending Embarked feature ....')

    return data

    

def processAge(data):

    print('Starting Age feature ....')

    data.Age.fillna(data.Age.median(),inplace=True)

    bins = [0,1,4,12,18,35,55,data.Age.max()]

    group_names = ['Infant','Toddler','Child','Teen','YAdult','MAdult','OAdult']

    data['AgeCat'] = pd.cut(data.Age,bins,labels=group_names)

    data = pd.concat([data,pd.get_dummies(data.AgeCat)],axis=1)

    print('Ending Embarked feature ....')

    return data



def processFamily(data):

    data['Family'] = data.Parch + data.SibSp

    return data



def processCabinCode(data):

    data['CabinCode'] = data.Cabin.map(lambda x: x if type(x)!=str else x[0])

    #print(data.CabinCode.value_counts()[data.CabinCode.value_counts() > 0])

    return data

    

def missingValuePct(data):

    print('Starting Missing Value count...')

    #data.replace('NaN',np.NaN)

    columns = list(data.columns)

    

    for col in columns:

        print("column : {0} --> missing count : {1}".format(col,data[col].isnull().sum()))

    

# Choose the type of classifier. 

def getClassifier(X_train, y_train):

    clf = RandomForestClassifier()



    # Choose some parameter combinations to try

    parameters = {'n_estimators': [4, 6, 9], 

                  'max_features': ['log2', 'sqrt','auto'], 

                  'criterion': ['entropy', 'gini'],

                  'max_depth': [2, 3, 5, 10], 

                  'min_samples_split': [2, 3, 5],

                  'min_samples_leaf': [1,5,8]

                 }



    # Type of scoring used to compare parameter combinations

    acc_scorer = make_scorer(accuracy_score)



    # Run the grid search

    grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)

    grid_obj = grid_obj.fit(X_train, y_train)



    # Set the clf to the best combination of parameters

    clf = grid_obj.best_estimator_



    # Fit the best algorithm to the data. 

    clf.fit(X_train, y_train)

    return clf



#10 Folds cross validation

def run_kfold(clf,X,Y):

    kf = KFold(X.shape[0],n_folds=10)

    outcome = []

    fold = 0

    

    for train_index, test_index in kf:

        fold += 1

        X_train, X_test = X.values[train_index], X.values[test_index]

        y_train, y_test = Y.values[train_index], Y.values[test_index]

        clf.fit(X_train, y_train)

        new_predictions = clf.predict(X_test)

        accuracy = accuracy_score(y_test, new_predictions)

        outcome.append(accuracy)

        print("Fold {0} accuracy: {1}".format(fold, accuracy))

        

    mean_outcome = np.mean(outcome)

    std_outcome = np.std(outcome)

    print("Mean Accuracy: {0}".format(mean_outcome))

    print("STD of Accuracy: {0}".format(std_outcome))

    

def split_data(X,Y,test_size):

    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=test_size,random_state=23)

    return X_train, X_test, y_train, y_test



def processAll(data):

    data = processEmbarked(data)

    data = processFamily(data)

    data = processCabinCode(data)

    data = processTicket(data)

    data = fillMissingValues(data)

    data = processSex(data)

    data = processAge(data)

    return data
#Missing Values analysis

missingValuePct(merged)
merged = processAll(merged)

train = pd.DataFrame(merged.head(len(train_df)))

test = pd.DataFrame(merged.iloc[len(train_df):])

print(merged.columns)
X = train[['Family', 'Pclass', 'female', 'male', 'CE',

       'QE', 'SE', 'Infant', 'Toddler', 'Child', 'Teen', 'YAdult',

       'MAdult', 'OAdult', 'A4', 'A5', 'AS', 'C', 'CA', 'CASOTON', 'FA', 'FC', 'FCC',

       'NA', 'PC', 'PP', 'PPP', 'SC', 'SCA4', 'SCAH', 'SCAHBASLE', 'SCOW',

       'SCPARIS', 'SOC', 'SOP', 'SOPP', 'SOTONO2', 'SOTONOQ', 'SP', 'STONO2',

       'SWPP', 'WC', 'WEP', 'A', 'B', 'C', 'D', 'E',

       'F', 'G', 'T', 'U',]]



Y = train.Survived



sub_test = test[['Family', 'Pclass', 'female', 'male', 'CE',

       'QE', 'SE', 'Infant', 'Toddler', 'Child', 'Teen', 'YAdult',

       'MAdult', 'OAdult', 'A4', 'A5', 'AS', 'C', 'CA', 'CASOTON', 'FA', 'FC', 'FCC',

       'NA', 'PC', 'PP', 'PPP', 'SC', 'SCA4', 'SCAH', 'SCAHBASLE', 'SCOW',

       'SCPARIS', 'SOC', 'SOP', 'SOPP', 'SOTONO2', 'SOTONOQ', 'SP', 'STONO2',

       'SWPP', 'WC', 'WEP', 'A', 'B', 'C', 'D', 'E',

       'F', 'G', 'T', 'U',]]
X_train, X_test, y_train, y_test = split_data(X,Y,30)

clf = getClassifier(X_train, y_train)

#run_kfold(clf,X,Y)
#clf.fit(X_train,y_train)

new_predictions = clf.predict(sub_test)

test1_df = pd.DataFrame()

test1_df['PassengerId'] = test.PassengerId

test1_df['Survived'] = pd.Series(new_predictions)

test1_df.Survived = test1_df.Survived.astype(int)



test1_df.to_csv("submission.csv",index=False)

print(test1_df.head())