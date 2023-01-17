#importing useful packages

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import re #Regrex Python



from subprocess import check_output
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test_df    = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )



#test_df.head(2)

data=pd.concat([train,test_df],axis=0).reset_index()

data.head()
data['Survived'][1]
pd.crosstab(index=train["Survived"],  # Make a crosstab

                              columns="count")  

#plt.hist(data["Survived"])
Title = []

for i in data['Name']:

    Title.append(re.sub(r'(.*, )|(\..*$)', "",i))

data['Title'] = Title

pd.crosstab(data['Sex'],data['Title'])
#Regroup rare titles

for i in range(0,len(data['Title'])):

    if data['Title'][i] in ['Capt','Col','Don','Dr','Jonkheer','Major','Rev','the Countess','Dona']:

        data['Title'][i] = 'rare title'

    if data['Title'][i] in ['Lady','Mlle','Ms',]:

        data['Title'][i] = 'Miss'

    if data['Title'][i] in ['Mme']:

        data['Title'][i] = 'Mrs'

    if data['Title'][i] in ['Sir']:

        data['Title'][i] = 'Mr'

pd.crosstab(data['Sex'],data['Title'])
Title2=[]

for i in data['Name']:

    if '(' in i:

        Title2.append(1)

    else:

        Title2.append(0)

data['Title2'] = Title2

pd.crosstab(data['Title2'],data['Sex'])

sns.factorplot('Title2','Survived', data=data,size=4,aspect=3)
Last_Name = []

for i in data['Name']:

    Last_Name.append(re.sub(r'(.*\. )|(\(.*$)', "",i))

data['Last_Name'] = Last_Name
data['Fsize'] = data['SibSp'] + data['Parch'] + 1

data['Fsize'][2]
FsizeD = []

for i in data['Fsize']:

    if i == 1:

        FsizeD.append('singleton')

    elif i < 5:

        FsizeD.append('small')

    else:

        FsizeD.append('large')

data['FsizeD']=FsizeD

data.head()
Cabin1=[]

for i in data['Cabin']:

    if pd.isnull(i):

        Cabin1.append('-1')

    else:

        Cabin1.append(i[0])

data['Cabin1']=Cabin1

pd.crosstab(index=data["Cabin1"],columns="count") 
data['Embarked'][61]='C'

data['Embarked'][829]='C'