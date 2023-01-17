# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# import the relevant libraries

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

#from pandas.tools.plotting import scatter_matrix

import pandas_profiling

import matplotlib.pyplot as plt

import seaborn as sns



# for data preprocessing

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn import datasets



#for ML

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors    import KNeighborsClassifier

from sklearn.svm          import SVC

from sklearn.tree         import DecisionTreeClassifier

from sklearn.linear_model import RidgeClassifierCV



# for validation

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV
# load data for the project

data = pd.read_csv('../input/titanic/train.csv')

validation = pd.read_csv('../input/titanic/test.csv')

df = data.copy(deep =True)

validation["Survived"]=int(0)
# first inspection of  data

# uncomment below to see the results

#df.profile_report()

df.dtypes

#first round of cleaning



def cleaner_titanic(data):

    # create a male binary section has they have a high death rate

    data['Male']=data.Sex.replace({'female':0,'male':1})

    

    # replace missing ages by median

    data.Age.fillna(data.Age.median(),inplace=True)

    

    # convert cabine number into deck people without cabine are assigned to deck U for unknown as it is a meaningful

    # empty value.

    data['Deck']=data.Cabin.str[0]

    data['Deck']=data.Deck.apply(lambda x: 'U' if (pd.isna(x)) else x)

    

    # create a family size column

    data['Familysize']=data.SibSp+data.Parch

    

    # correct ticket price for the number of menber in the family

    data['Farperperson']= data.Fare/(data.Familysize+1)

    

    # extract titles from the name

    data['Title'] = data.Name.apply(lambda x: x[x.find(',')+2:x.find('.')+1])

    

    # creat different categories from titles

    data['Noble'] = data.Title.apply(lambda x: 1 if (x=='Don.' or x=='Jonkheer.' or x=='the Countess.' or x=='Lady.' or x=='Sir.' or x=='Master.') else 0)

    data['Uclass']= data.Title.apply(lambda x: 1 if (x=='Major.' or x=='Col.' or x=='Capt.' or x=='Dr.' or x=='Rev.') else 0)

    

    # Create a creww class as they have a very low survival rate

    data['Crew']=data.Fare.apply(lambda x: 1 if (x==0.0) else 0)

    

    

    # split categorial columns into binary for the learning agorithm

    deckhot= OneHotEncoder(dtype=np.int, sparse=True)

    # Deck and embarked will be split

    # first change the two missing values in embarked by the most common one

    data.Embarked.fillna('S',inplace=True)

    data.Embarked = data.Embarked.replace({'S':'So','C':'Ch','Q':'Qu'})

    columns_embark= ['So','Ch','Qu']

    columns_deck = ['U','C','E','G','D','A','B','F','T']

    for i in columns_embark:

        data[i]=data['Embarked'].apply(lambda x: 1 if (x==i) else 0)

        #print(i)

    for i in columns_deck:

        data[i]=data['Deck'].apply(lambda x: 1 if (x==i) else 0)

        #print(i)

    

    #Deck=pd.DataFrame(deckhot.fit_transform(data[['Embarked','Deck']]).toarray(),columns=columns_deck)

    #for i in columns_deck:

    #    data[i]=Deck[i]

    

    return data.copy()





#preprocess the data

# only keep the relevant columns

def preprocess_titanic(dfc):

    # remove all irrelevant columns

    drop_columns = ['Fare','SibSp','Parch','Name','Cabin','Sex','Ticket','PassengerId','Embarked','Title','Deck']

    dfc.drop(drop_columns,axis=1,inplace=True)

    # replace all empty values

    dfc.Farperperson.fillna(dfc.Farperperson.median(),inplace=True)

    # scale continuous data

    process = dfc[['Age','Farperperson']].values

    process = StandardScaler().fit_transform(process)

    y = dfc.Survived.values

    X= dfc.drop(['Age','Farperperson','Survived'],axis=1).values

    X = np.concatenate((X,process),axis=1)

    return (X,y)

dfc = cleaner_titanic(df)

#dfc.profile_report()
#plot of parameters

h=sns.FacetGrid(dfc, row= 'Male', col='Embarked', hue='Survived')

h.map(plt.hist,'Farperperson',bins=4)

h.add_legend()
# preprocess the training and learning data 

(X,y)= preprocess_titanic(dfc)

validation.shape

val = cleaner_titanic(validation)

(Xv,yv)=preprocess_titanic(val)
# cross validation of a logistic regression

parameters = {'penalty':('l1','l2'),'C':(1.0,0.5,0.2,0.1,0.01)}

LR = LogisticRegression()

clf=GridSearchCV(LR,parameters)

clf.fit(X,y)

print('Best score is:')

print(clf.best_score_)

print('With parameters:')

print(clf.best_params_)

#clf.cv_results_



#scores = cross_val_score(LR, X, y, cv=5)

#print(scores )

#print('Give a mean score of: ' + str(np.mean(scores)) )

#compare different models by looping over an array of ML models

MLA = [LogisticRegression(),KNeighborsClassifier(),SVC(),DecisionTreeClassifier()]

C=(1.0,0.9,0.5,0.1,0.01,0.001)

MLA_param = [ {'penalty':('l1','l2'),'C':C},

             {'n_neighbors':(1,3,5,10),'weights':('uniform','distance')},

             {'C':C,'kernel':('rbf','linear','poly'),'degree':(2,3,4)},

             {'criterion':('gini','entropy'),'max_depth':(1,5,10)}]

MLA_score = [0.0,0.0,0.0,0.0]

MLA_best_param = [{},{},{},{}]

for i, alg in enumerate(MLA):

    clf=GridSearchCV(alg,MLA_param[i],cv=5)

    clf.fit(X,y)

    MLA_score[i]=clf.best_score_

    MLA_best_param[i]=clf.best_params_

    

    

print(MLA_score)

print(MLA_best_param)
# use the best parameters to learn on entire dataset and make a prediction from test

ML = SVC(C=1,kernel='rbf',degree=2)

ML.fit(X,y)



yv = ML.predict(Xv)

validation['predicted']=yv
validation.head()
!ls ../working
# create results for submission

'../output/submission.csv'

validation[['PassengerId','predicted']].to_csv('../working/submission.csv',index=False)