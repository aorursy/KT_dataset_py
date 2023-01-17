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
#transform them into numerical features
import pandas as pd

adult_utc = pd.read_csv("../input/adult_utc.csv")

adult_utc
#save

adult_utc.to_csv('adult1.csv',index=False)

adult_utc.columns = [

    "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",

    "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",

    "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"

]



adult_utc
adult_utc.to_csv('adult2.csv',index=False)

adult_utc.Income.unique()

adult_utc["Income"] = adult_utc["Income"].map({ " <=50K": -1, " >50K": 1 })

adult_utc
adult_utc.Gender.unique()

adult_utc["Gender"] = adult_utc["Gender"].map({ " Male": 1, " Female": 0 })

adult_utc


print (adult_utc[['NativeCountry','Income']].groupby(['NativeCountry']).mean())
import numpy as np # linear algebra



#fill missing value  as '?'

print (adult_utc.shape)

adult_utc['NativeCountry'] = adult_utc['NativeCountry'].replace(' ?',np.nan)

adult_utc['WorkClass'] = adult_utc['WorkClass'].replace(' ?',np.nan)

adult_utc['Occupation'] = adult_utc['Occupation'].replace(' ?',np.nan)



adult_utc.iloc[:,0:13] = adult_utc.iloc[:,0:13].apply(lambda x: x.fillna(x.value_counts().index[0]))

def h(x):

    if (x['MaritalStatus'] == ' Divorced') :

        return 'Single'

    if (x['MaritalStatus'] == ' Married-spouse-absent') :

            return 'Single'

    if (x['MaritalStatus'] == ' Never-married'):

            return 'Single'



    if (x['MaritalStatus'] == ' Separated'):

            return 'Single'

    if (x['MaritalStatus'] == ' Widowed'):

            return 'Single'



    if (x['MaritalStatus'] == ' Married-AF-spouse'):

            return 'Couple'

    if (x['MaritalStatus'] == ' Married-civ-spouse'):

            return 'Couple'



   
adult_utc['MaritalStatus']=adult_utc.apply(h, axis=1)

#adult_utc['MaritalStatus'] = adult_utc['MaritalStatus'].replace([' Divorced',' Married-spouse-absent',' Never-married',' Separated',' Widowed'],'Single')

#adult_utc['MaritalStatus'] = adult_utc['MaritalStatus'].replace([' Married-AF-spouse',' Married-civ-spouse'],'Couple')
adult_utc
adult_utc.MaritalStatus.unique()

adult_utc['MaritalStatus'] = adult_utc['MaritalStatus'].map({'Couple':1,'Single':0})

adult_utc
adult_utc.Relationship.unique()

#convert  Relationship into integer into  0 1 2 3 4 5



rel_map = {' Unmarried':0,' Wife':1,' Husband':2,' Not-in-family':3,' Own-child':4,' Other-relative':5}



adult_utc['Relationship'] = adult_utc['Relationship'].map(rel_map)
adult_utc
adult_utc.Race.unique()

race_map={' White':0,' Amer-Indian-Eskimo':1,' Asian-Pac-Islander':2,' Black':3,' Other':4}

adult_utc['Race']= adult_utc['Race'].map(race_map)
adult_utc


adult_utc.WorkClass.unique()

# function to convert to numeric



def f(x):

    if (x['WorkClass'] == ' Federal-gov') :

        return 'govt'

    if (x['WorkClass'] == ' Local-gov') :

            return 'govt'

    if (x['WorkClass'] ==' State-gov'):

            return 'govt'



    if (x['WorkClass'] == ' Private'):

            return 'private'

    if (x['WorkClass'] == ' Self-emp-inc'):

            return 'self_employed'



    if (x['WorkClass'] == ' Self-emp-not-inc'):

            return 'self_employed'

    else:

            return 'without_pay'

    

        

    
adult_utc
adult_utc['WorkClass']=adult_utc.apply(f, axis=1)

adult_utc
adult_utc.NativeCountry.unique()

def g(y):

    if y['NativeCountry'] == ' United-States' : return 'US'

   

    else: return 'Non-US'
adult_utc['NativeCountry']=adult_utc.apply(g, axis=1)

adult_utc
#Convert country in integer

adult_utc.NativeCountry.unique()

adult_utc['NativeCountry'] = adult_utc['NativeCountry'].map({'US':1,'Non-US':0}).astype(int)

adult_utc
y_all = adult_utc["Income"].values

y_all


adult_utc.Occupation.unique()





adult_utc.Education.unique()

adult_utc
adult_utc.fnlwgt.unique()

# : Normalise fnlwgt values to a new range from 1 to 100



a, b = 1, 100

x, y = adult_utc.fnlwgt.min(), adult_utc.fnlwgt.max()

adult_utc['fnlwgt'] = (adult_utc.fnlwgt - x) / (y - x) * (b - a) + a



print(adult_utc['fnlwgt'])
#: Perform discretization on fnlwgt feature in Adult dataset.

bins = [0, 1, 5, 10, 25, 50, 100]

adult_utc['Binned'] = pd.cut(adult_utc['fnlwgt'].astype(float), bins)
adult_utc
adult_utc.Binned.unique()

# we drop the columns  fnlwgt as it is not meaningful 

# and CapitalGain  CapitalLoss

adult_utc.drop(labels=['fnlwgt','CapitalGain','CapitalLoss'],axis=1,inplace=True)

adult_utc
adult_utc.WorkClass.unique()

#convert  WorkClass into integer into  0 1 2 3 

WorkClass = {'govt':0,'private':1,'self_employed':2,'without_pay':3}

adult_utc['WorkClass'] = adult_utc['WorkClass'].map(WorkClass)
adult_utc
adult_utc.Education.unique()

#convert  Education into integer into  0 1 2 3 

Education = {' Preschool':0,' 1st-4th':1,' 5th-6th':2,' 7th-8th':3,' 9th':4,' 10th':5,' 11th':6,' 12th':7,' Prof-school':8,

           ' Assoc-acdm':9,' Assoc-voc':10,' Some-college':11,' HS-grad':12,' Bachelors':13,' Masters':14, ' Doctorate':15}

adult_utc['Education'] = adult_utc['Education'].map(Education)
adult_utc.Binned.unique()

#mapping = {'(0, 1]' : 1 , '(1, 5]': 2 , '(5, 10]': 3, '(10, 25]': 4, '(25, 50]': 5, '(50, 100]': 6}

#adult_utc['binned'] = adult_utc['binned'].map(mapping)

#adult_utc = adult_utc.replace ({'binned': mapping  })



#mapping = {'(0,1]' : 1 , '(1, 5]': 2 }

#adult_utc = adult_utc.replace ({'binned': mapping})



mp = {'(0, 1]': 1, '(1, 5]': 2, '(5, 10]': 3, '(10, 25]': 4, '(25, 50]': 5, '(50, 100]': 6}

adult_utc = adult_utc.replace ({'Binned':mp})

#adult_utc['Binned'] = adult_utc['Binned'].map(Binned)





def Bined(y):

    if y['Binned'] == '(0, 1]' : return '1'

    if y['Binned'] == '(1, 5]' : return '2'

    if y['Binned'] == '(5, 10]' : return '3'

    if y['Binned'] == '(10, 25]' : return '4'

    if y['Binned'] == '(25, 50]' : return '5'

    if y['Binned'] == '(50, 100]' : return '6'

    

#adult_utc['Binned']=adult_utc.apply(Bined, axis=1)

adult_utc.Occupation.unique()

def O(y):

    if y['Occupation'] == ' Exec-managerial' : return '1'

    if y['Occupation'] == ' Handlers-cleaners' : return '2'

    if y['Occupation'] == ' Prof-specialty' : return '3'

    if y['Occupation'] == ' Other-service' : return '4'

    if y['Occupation'] == ' Adm-clerical' : return '5'

    if y['Occupation'] == ' Craft-repair' : return '6'

    if y['Occupation'] == ' Transport-moving' : return '7' 

    if y['Occupation'] == ' Farming-fishing' : return '8'  

    if y['Occupation'] == ' Machine-op-inspct' : return '9' 

    if y['Occupation'] == ' Tech-support' : return '10'

    if y['Occupation'] == ' Protective-serv' : return '11'

    if y['Occupation'] == ' Armed-Forces' : return '12'

    if y['Occupation'] == ' Priv-house-serv' : return '13'



    

adult_utc['Occupation']=adult_utc.apply(O, axis=1)
adult_utc
mp = {'(0, 1]': 1, '(1, 5]': 2, '(5, 10]': 3, '(10, 25]': 4, '(25, 50]': 5, '(50, 100]': 6}

adult_utc = adult_utc.replace ({'Binned':mp})
#mapping = {'(0, 1]' : 1 , '(1, 5]': 2 , '(5, 10]': 3, '(10, 25]': 4, '(25, 50]': 5, '(50, 100]': 6}

#adult_utc['Binned'] = adult_utc['Binned'].map(mapping)
"""

def B(y):

    if y['Binned'] == '(0, 1]' : return '1'

    if y['Binned'] == '(1, 5]' : return '2'

    if y['Binned'] == '(5, 10]' : return '3'

    if y['Binned'] == '(10, 25]' : return '4'

    if y['Binned'] == '(25, 50]' : return '5'

    if y['Binned'] == '(50, 100]' : return '6'



    

adult_utc['Binned']=adult_utc.apply(B, axis=1)

"""
adult_utc
adult_utc.to_csv('adult_numerical1.csv',index=False)
