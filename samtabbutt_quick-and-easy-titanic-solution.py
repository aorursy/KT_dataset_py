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

        file_path = os.path.join(dirname, filename)

        if 'test' in filename:

            test_df = pd.read_csv(file_path)

        elif 'train' in filename:

            train_df = pd.read_csv(file_path)



train_df.head()
test_df.head()
from sklearn.preprocessing import OneHotEncoder
train_df = train_df.set_index('PassengerId')

test_df = test_df.set_index('PassengerId')
Y = pd.DataFrame()

Y['Survived'] = train_df['Survived']

Y['PassengerId'] = train_df.index

Y = Y.set_index('PassengerId')



stacked_df = pd.concat([train_df.drop('Survived',axis=1),test_df])
stacked_df.head()
lowCardColumns = [col for col in stacked_df.columns if stacked_df[col].nunique()<10]

highCardColumns = [col for col in stacked_df.columns if stacked_df[col].nunique()>=10]
for col in lowCardColumns:

    print(col,stacked_df[col].unique())
stacked_df['Embarked']=stacked_df['Embarked'].fillna('0')
enc = OneHotEncoder(handle_unknown='ignore',sparse=False)

OHCols = pd.DataFrame(enc.fit_transform(stacked_df[lowCardColumns]))

OHCols_index = stacked_df.index

stacked_df = pd.concat([OHCols,stacked_df.drop(lowCardColumns,axis=1)],axis=1)

stacked_df = stacked_df.drop(0)

stacked_df.head()
stacked_df['Age'] = stacked_df['Age'].fillna(stacked_df['Age'].mean())

highCardColumns = highCardColumns.remove('Age')

stacked_df['Cabin'].unique()
import seaborn as sns

from matplotlib import pyplot as plt
train_df['Cabin Filled']= train_df['Cabin'].fillna('A')

train_df['Cabin Trunc'] = train_df['Cabin Filled'].apply(lambda x:x[0])
fig,ax = plt.subplots(1,2)

sns.despine(left=True)



sns.countplot(x='Cabin Trunc',data=train_df[train_df['Survived']==1],ax=ax[0])

sns.countplot(x='Cabin Trunc',data=train_df[train_df['Survived']==0],ax=ax[1])

ax[0].title.set_text('Survived: 1')

ax[1].title.set_text('Survived: 0')

plt.show()
def createHeatMap(col1,col2):

    rowDict = {col:i for i,col in enumerate(col1.unique())}

    colDict = {col:i for i,col in enumerate(col2.unique())}

    heatMap = np.zeros(shape=(col2.nunique(),col1.nunique()))

    for j,i in enumerate(col1):

        heatMap[int(colDict[col2.iloc[j]])][int(rowDict[i])]+=1

    return heatMap
temp_df = train_df.dropna().copy()

highCardColumns = [col for col in stacked_df.columns if stacked_df[col].nunique()>=10]

fig,ax = plt.subplots(len(highCardColumns),1)

for i,col in enumerate(highCardColumns):

    heatMapMatrix = createHeatMap(temp_df['Cabin Trunc'],temp_df[col])

    sns.heatmap(heatMapMatrix,ax=ax[i])

plt.show()
stacked_df = stacked_df.drop('Cabin',axis=1)
stacked_df['Fare'].fillna(stacked_df['Fare'].median(), inplace=True)

stacked_df['Name'] = stacked_df['Name'].fillna('Missed name').apply(lambda x:x.split(' ')[1])

stacked_df = stacked_df.drop('Ticket',axis=1)

stacked_df.head()
from sklearn.preprocessing import LabelEncoder
label_encode = LabelEncoder()

stacked_df['Name'] = label_encode.fit_transform(stacked_df['Name'])

stacked_df.head()
train_X = stacked_df.loc[:len(Y),:]

test_X = stacked_df.loc[len(Y)+1:,:]



print('train_X length',len(train_X),'Y length',len(Y))

print('test_X length',len(test_X),'original test length',len(test_df))
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix





X_train, X_valid, y_train, y_valid = train_test_split(train_X, Y,

                                                      train_size=0.8, test_size=0.2,

                                                      random_state=0)



# function for comparing different approaches

def score_dataset(X_train, X_valid, y_train, y_valid):

    model = RandomForestRegressor(n_estimators=100, random_state=0)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    rounded_preds = [round(x) for x in preds]

    cf_matrix = confusion_matrix(y_valid, rounded_preds)

    sns.heatmap(cf_matrix)

    plt.show()

    return accuracy_score(y_valid, rounded_preds)
score_dataset(X_train, X_valid, y_train, y_valid)