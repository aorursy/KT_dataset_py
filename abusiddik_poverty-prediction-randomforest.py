
#import necessary packages
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

import os
print(os.listdir("../input"))

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

train_data.shape
test_data.shape
train_data.describe(include='object')
train_data.describe()
train_data.dtypes
pd.set_option('display.max_rows',200)
pd.set_option('display.max_columns',200)
train_data.head()
### Data Preprocessing 

#### Removing Duplicate and Constant features

for i,j in enumerate(train_data.dtypes):
    if j == 'O':
        print(i,j)
#drop id and household id from dataset
train_data.drop(['Id','idhogar'],axis=1,inplace=True)
test_data.drop(['idhogar'],axis=1,inplace=True)

#total number of rows
row_train,_ = train_data.shape
row_test,_ = test_data.shape
print(row_train,row_test)

# Replace yes with 1 and no with 0 in  dependency, edjefe, edjefa in TrainData
for i in range(0,row_train):
    if train_data['dependency'][i] == "no":
        train_data.set_value(i,'dependency',0)
    elif train_data['dependency'][i] == "yes":
        train_data.set_value(i,'dependency',1)
    if train_data['edjefe'][i] == "no":
        train_data.set_value(i,'edjefe',0)
    elif train_data['edjefe'][i] == "yes":
        train_data.set_value(i,'edjefe',1)
    if train_data['edjefa'][i] == "no":
        train_data.set_value(i,'edjefa',0)
    elif train_data['edjefa'][i] == "yes":
        train_data.set_value(i,'edjefa',1)
# Replace yes with 1 and no with 0 in  dependency, edjefe, edjefa in TestData
for i in range(0,row_test):
    if test_data['dependency'][i] == "no":
        test_data.set_value(i,'dependency',0)
    elif test_data['dependency'][i] == "yes":
        test_data.set_value(i,'dependency',1)
    if test_data['edjefe'][i] == "no":
        test_data.set_value(i,'edjefe',0)
    elif test_data['edjefe'][i] == "yes":
        test_data.set_value(i,'edjefe',1)
    if test_data['edjefa'][i] == "no":
        test_data.set_value(i,'edjefa',0)
    elif test_data['edjefa'][i] == "yes":
        test_data.set_value(i,'edjefa',1)
Counter(train_data['edjefa'])

for i,j in enumerate(train_data.dtypes):
    if j == 'object':
        print(i,j)
test_column = []
for each in test_data.columns:
    if each not in ['Id']:
        test_column.append(each)
len(test_column)
train_data[train_data.columns] = train_data[train_data.columns].apply(pd.to_numeric)


test_data[test_column] = test_data[test_column].apply(pd.to_numeric)

#Contstant Features
colsToRemove = []
for col in train_data.columns:
    if train_data[col].std() == 0: 
        colsToRemove.append(col)
print("Columns to Remove", colsToRemove)

# remove constant columns in the training set
train_data.drop(colsToRemove, axis=1, inplace=True)

# remove constant columns in the test set
test_data.drop(colsToRemove, axis=1, inplace=True) 


#Remove Duplicate Columns
colsToRemove = []
colsScaned = []
dupList = {}

columns = train_data.columns

for i in range(len(columns)-1):
    v = train_data[columns[i]].values
    dupCols = []
    for j in range(i+1,len(columns)):
        if np.array_equal(v, train_data[columns[j]].values):
            colsToRemove.append(columns[j])
            if columns[j] not in colsScaned:
                dupCols.append(columns[j]) 
                colsScaned.append(columns[j])
                dupList[columns[i]] = dupCols
print("Duplicate Columns are", colsToRemove)

# remove duplicate columns in the training set
train_data.drop(colsToRemove, axis=1, inplace=True) 

# remove duplicate columns in the testing set
test_data.drop(colsToRemove, axis=1, inplace=True)

train_data.shape
#Missing data in Train set
missing_df = train_data.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df['missing_ratio']=      missing_df['missing_count']/train_data.shape[0]
missing_df.ix[missing_df['missing_ratio']>0.7]
#Missing data in Test set
missing_df_test = test_data.isnull().sum(axis=0).reset_index()
missing_df_test.columns = ['column_name', 'missing_count']
missing_df_test['missing_ratio']=      missing_df_test['missing_count']/test_data.shape[0]
missing_df_test.ix[missing_df_test['missing_ratio']>0.7]
ind = np.arange(missing_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12,18))
rects = ax.barh(ind, missing_df.missing_count.values, color='blue')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
plt.show()

train_data.drop(['rez_esc'],axis=1,inplace=True)


test_data.drop(['rez_esc'],axis=1,inplace=True)
# Rend - v2a1
#tipovivi1 is Own house. The one who have own house dont need to pay rend, so convert it as 0
for i in range(0,row_train):
    if train_data['tipovivi1'][i] == 1:
        train_data.set_value(i,'v2a1',0)
for i in range(0,row_test):
    if test_data['tipovivi1'][i] == 1:
        test_data.set_value(i,'v2a1',0)
        
train_data.loc[train_data['tipovivi1'].isnull(), 'v2a1'] = train_data['v2a1'].median()
test_data.loc[test_data['tipovivi1'].isnull(), 'v2a1'] = test_data['v2a1'].median()

#v18q1 - Number of tablets 
for i in range(0,row_train):
    if train_data['v18q'][i] == 0:
        train_data.set_value(i,'v18q1',0)
for i in range(0,row_test):
    if test_data['v18q'][i] == 0:
        test_data.set_value(i,'v18q1',0)
train_data['v2a1'].replace('', np.nan, inplace=True)
test_data['v2a1'].replace('', np.nan, inplace=True)

train_data['v2a1'] = train_data['v2a1'].fillna(train_data['v2a1'].median())
test_data['v2a1'] = test_data['v2a1'].fillna(test_data['v2a1'].median())


# the following columns have some missing value, replace these missing value by mode.
#meaneduc
#SQBmeaned
train_data['meaneduc'].replace(' ', np.nan, inplace=True)
train_data['SQBmeaned'].replace(' ', np.nan, inplace=True)
test_data['meaneduc'].replace(' ', np.nan, inplace=True)
test_data['SQBmeaned'].replace(' ', np.nan, inplace=True)

train_data.dropna(subset=['meaneduc'],inplace=True)
train_data.dropna(subset=['SQBmeaned'],inplace=True)
test_data.dropna(subset=['meaneduc'],inplace=True)
test_data.dropna(subset=['SQBmeaned'],inplace=True)
train_data.isnull().any().values
### Feature Engineering

### Wall_Materials

'''
Create a new column called 'Wall_Material'
Values will range from 0 to 7 to indicate differenct materials
0 -  if predominant material on the outside wall is block or brick
1 -  if predominant material on the outside wall is socket (wood,  zinc or absbesto"
2 -  if predominant material on the outside wall is prefabricated or cement
3 -  if predominant material on the outside wall is waste material
4 -  if predominant material on the outside wall is wood
5 -  if predominant material on the outside wall is zink
6 -  if predominant material on the outside wall is natural fibers
7 -  if predominant material on the outside wall is other
'''

for i in train_data.index:
    if train_data['paredblolad'][i] == 1:
        train_data.set_value(i,'Wall_Material',0)
    elif train_data['paredzocalo'][i] == 1:
        train_data.set_value(i,'Wall_Material',1)
    elif train_data['paredpreb'][i] == 1:
        train_data.set_value(i,'Wall_Material',2)
    elif train_data['pareddes'][i] == 1:
        train_data.set_value(i,'Wall_Material',3)
    elif train_data['paredmad'][i] == 1:
        train_data.set_value(i,'Wall_Material',4)
    elif train_data['paredzinc'][i] == 1:
        train_data.set_value(i,'Wall_Material',5)
    elif train_data['paredfibras'][i] == 1:
        train_data.set_value(i,'Wall_Material',6)
    elif train_data['paredother'][i] == 1:
        train_data.set_value(i,'Wall_Material',7)
#for test set
for i in test_data.index:
    if test_data['paredblolad'][i] == 1:
        test_data.set_value(i,'Wall_Material',0)
    elif test_data['paredzocalo'][i] == 1:
        test_data.set_value(i,'Wall_Material',1)
    elif test_data['paredpreb'][i] == 1:
        test_data.set_value(i,'Wall_Material',2)
    elif test_data['pareddes'][i] == 1:
        test_data.set_value(i,'Wall_Material',3)
    elif test_data['paredmad'][i] == 1:
        test_data.set_value(i,'Wall_Material',4)
    elif test_data['paredzinc'][i] == 1:
        test_data.set_value(i,'Wall_Material',5)
    elif test_data['paredfibras'][i] == 1:
        test_data.set_value(i,'Wall_Material',6)
    elif test_data['paredother'][i] == 1:
        test_data.set_value(i,'Wall_Material',7)
train_data.Wall_Material.value_counts()
### Floor_Material

'''
Create a new column called 'Floor_Material'
Values will range from 0 to 5 to indicate differenct floor materials

0 -  if no floor at the household
1 -  if predominant material on the floor is mosaic,  ceramic,  terrazo"
2 -  if predominant material on the floor is cement
3 -  if predominant material on the floor is  natural material
4 -  if predominant material on the floor is wood
5 -  if predominant material on the floor is other

'''
for i in train_data.index:
    if train_data['pisonotiene'][i] == 1:
        train_data.set_value(i,'Floor_Material',0)
    elif train_data['pisomoscer'][i] == 1:
        train_data.set_value(i,'Floor_Material',1)
    elif train_data['pisocemento'][i] == 1:
        train_data.set_value(i,'Floor_Material',2)
    elif train_data['pisonatur'][i] == 1:
        train_data.set_value(i,'Floor_Material',3)
    elif train_data['pisomadera'][i] == 1:
        train_data.set_value(i,'Floor_Material',4)
    elif train_data['pisoother'][i] == 1:
        train_data.set_value(i,'Floor_Material',5)

print(train_data.Floor_Material.value_counts())

#test set
for i in test_data.index:
    if test_data['pisonotiene'][i] == 1:
        test_data.set_value(i,'Floor_Material',0)
    elif test_data['pisomoscer'][i] == 1:
        test_data.set_value(i,'Floor_Material',1)
    elif test_data['pisocemento'][i] == 1:
        test_data.set_value(i,'Floor_Material',2)
    elif test_data['pisonatur'][i] == 1:
        test_data.set_value(i,'Floor_Material',3)
    elif test_data['pisomadera'][i] == 1:
        test_data.set_value(i,'Floor_Material',4)
    elif test_data['pisoother'][i] == 1:
        test_data.set_value(i,'Floor_Material',5)
'''
Create a new column called 'Roof_Material'
Values will range from 0 to 3 to indicate differenct Roof_Material

0 -  if predominant material on the roof is metal foil or zink
1 -  if predominant material on the roof is fiber cement,  mezzanine "
2 -  if predominant material on the roof is natural fibers
3 -  if predominant material on the roof is other
'''

for i in train_data.index:
    if train_data['techozinc'][i] == 1:
        train_data.set_value(i,'Roof_Material',0)
    elif train_data['techoentrepiso'][i] == 1:
        train_data.set_value(i,'Roof_Material',1)
    elif train_data['techocane'][i] == 1:
        train_data.set_value(i,'Roof_Material',2)
    elif train_data['techootro'][i] == 1:
        train_data.set_value(i,'Roof_Material',3)

#for test_data
for i in test_data.index:
    if test_data['techozinc'][i] == 1:
        test_data.set_value(i,'Roof_Material',0)
    elif test_data['techoentrepiso'][i] == 1:
        test_data.set_value(i,'Roof_Material',1)
    elif test_data['techocane'][i] == 1:
        test_data.set_value(i,'Roof_Material',2)
    elif test_data['techootro'][i] == 1:
        test_data.set_value(i,'Roof_Material',3)
'''
Create a new column called 'Water_Provision'
Values will range from 0 to 2 to indicate differenct Water_Provision

0 -  if no water provision
1 -  if water provision inside the dwelling
2 -  if water provision outside the dwelling

'''

for i in train_data.index:
    if train_data['abastaguano'][i] == 1:
        train_data.set_value(i,'Water_Provision',0)
    elif train_data['abastaguadentro'][i] == 1:
        train_data.set_value(i,'Water_Provision',1)
    elif train_data['abastaguafuera'][i] == 1:
        train_data.set_value(i,'Water_Provision',2)
        

#for test_data
for i in test_data.index:
    if test_data['abastaguano'][i] == 1:
        test_data.set_value(i,'Water_Provision',0)
    elif test_data['abastaguadentro'][i] == 1:
        test_data.set_value(i,'Water_Provision',1)
    elif test_data['abastaguafuera'][i] == 1:
        test_data.set_value(i,'Water_Provision',2)
        

test_data['Water_Provision'].value_counts()
'''

Create a new column called 'Toilet_Provision'
Values will range from 0 to 4 to indicate differenct Toilet_Provision

0 - no toilet in the dwelling
1 -  toilet connected to sewer or cesspool
2 - toilet connected to  septic tank
3 - toilet connected to black hole or letrine
4 - toilet connected to other system
'''

for i in train_data.index:
    if train_data['sanitario1'][i] == 1:
        train_data.set_value(i,'Toilet_Provision',0)
    elif train_data['sanitario2'][i] == 1:
        train_data.set_value(i,'Toilet_Provision',1)
    elif train_data['sanitario3'][i] == 1:
        train_data.set_value(i,'Toilet_Provision',2)
    elif train_data['sanitario5'][i] == 1:
        train_data.set_value(i,'Toilet_Provision',3)
    elif train_data['sanitario6'][i] == 1:
        train_data.set_value(i,'Toilet_Provision',4)
       
        
        

#test_data
for i in test_data.index:
    if test_data['sanitario1'][i] == 1:
        test_data.set_value(i,'Toilet_Provision',0)
    elif test_data['sanitario2'][i] == 1:
        test_data.set_value(i,'Toilet_Provision',1)
    elif test_data['sanitario3'][i] == 1:
        test_data.set_value(i,'Toilet_Provision',2)
    elif test_data['sanitario5'][i] == 1:
        test_data.set_value(i,'Toilet_Provision',3)
    elif test_data['sanitario6'][i] == 1:
        test_data.set_value(i,'Toilet_Provision',4)

test_data['Toilet_Provision'].value_counts()
'''
Create a new column called 'Cooking_Provision'
Values will range from 0 to 3 to indicate differenct Cooking_Provision

0 -  no main source of energy used for cooking (no kitchen)
1 -  main source of energy used for cooking electricity
2 -  main source of energy used for cooking gas
3 -  main source of energy used for cooking wood charcoal
'''

for i in train_data.index:
    if train_data['energcocinar1'][i] == 1:
        train_data.set_value(i,'Cooking_Provision',0)
    elif train_data['energcocinar2'][i] == 1:
        train_data.set_value(i,'Cooking_Provision',1)
    elif train_data['energcocinar3'][i] == 1:
        train_data.set_value(i,'Cooking_Provision',2)
    elif train_data['energcocinar4'][i] == 1:
        train_data.set_value(i,'Cooking_Provision',3)

#for test_data

for i in test_data.index:
    if test_data['energcocinar1'][i] == 1:
        test_data.set_value(i,'Cooking_Provision',0)
    elif test_data['energcocinar2'][i] == 1:
        test_data.set_value(i,'Cooking_Provision',1)
    elif test_data['energcocinar3'][i] == 1:
        test_data.set_value(i,'Cooking_Provision',2)
    elif test_data['energcocinar4'][i] == 1:
        test_data.set_value(i,'Cooking_Provision',3)

test_data.Cooking_Provision.value_counts()
'''
Create a new column called 'Rubbish_Disposal'
Values will range from 0 to 4 to indicate different methods of Rubbish_Disposal

0 -  if rubbish disposal mainly by tanker truck
1 -  if rubbish disposal mainly by botan hollow or buried
2 -  if rubbish disposal mainly by burning
3 -  if rubbish disposal mainly by throwing in an unoccupied space
4-  if rubbish disposal mainly other

'''
for i in train_data.index:
    if train_data['elimbasu1'][i] == 1:
        train_data.set_value(i,'Rubbish_Disposal',0)
    elif train_data['elimbasu2'][i] == 1:
        train_data.set_value(i,'Rubbish_Disposal',1)
    elif train_data['elimbasu3'][i] == 1:
        train_data.set_value(i,'Rubbish_Disposal',2)
    elif train_data['elimbasu4'][i] == 1:
        train_data.set_value(i,'Rubbish_Disposal',3)
    elif train_data['elimbasu6'][i] == 1:
        train_data.set_value(i,'Rubbish_Disposal',4)


#for test_data
for i in test_data.index:
    if test_data['elimbasu1'][i] == 1:
        test_data.set_value(i,'Rubbish_Disposal',0)
    elif test_data['elimbasu2'][i] == 1:
        test_data.set_value(i,'Rubbish_Disposal',1)
    elif test_data['elimbasu3'][i] == 1:
        test_data.set_value(i,'Rubbish_Disposal',2)
    elif test_data['elimbasu4'][i] == 1:
        test_data.set_value(i,'Rubbish_Disposal',3)
    elif test_data['elimbasu6'][i] == 1:
        test_data.set_value(i,'Rubbish_Disposal',4)

'''
Create a new column called 'Wall_Quality'
Values will range from 0 to 2 to indicate the quality of the wall


0 -  if walls are bad
1 -  if walls are regular
2 -  if walls are good

'''
for i in train_data.index:
    if train_data['epared1'][i] == 1:
        train_data.set_value(i,'Wall_Quality',0)
    elif train_data['epared2'][i] == 1:
        train_data.set_value(i,'Wall_Quality',1)
    elif train_data['epared3'][i] == 1:
        train_data.set_value(i,'Wall_Quality',2)

#for test_data

for i in test_data.index:
    if test_data['epared1'][i] == 1:
        test_data.set_value(i,'Wall_Quality',0)
    elif test_data['epared2'][i] == 1:
        test_data.set_value(i,'Wall_Quality',1)
    elif test_data['epared3'][i] == 1:
        test_data.set_value(i,'Wall_Quality',2)
'''
Create a new column called 'Roof_Quality'
Values will range from 0 to 2 to indicate the quality of the roof

0 -  if roof are bad
1 -  if roof are regular
2 -  if roof are good

'''
for i in train_data.index:
    if train_data['etecho1'][i] == 1:
        train_data.set_value(i,'Roof_Quality',0)
    elif train_data['etecho2'][i] == 1:
        train_data.set_value(i,'Roof_Quality',1)
    elif train_data['etecho3'][i] == 1:
        train_data.set_value(i,'Roof_Quality',2)

#for test_data
for i in test_data.index:
    if test_data['etecho1'][i] == 1:
        test_data.set_value(i,'Roof_Quality',0)
    elif test_data['etecho2'][i] == 1:
        test_data.set_value(i,'Roof_Quality',1)
    elif test_data['etecho3'][i] == 1:
        test_data.set_value(i,'Roof_Quality',2)

train_data.Roof_Quality.value_counts()

'''
Create a new column called 'Floor_Quality'
Values will range from 0 to 2 to indicate the quality of the floor

0 -  if floor are bad
1 -  if floor are regular
2 -  if floor are good

'''
for i in train_data.index:
    if train_data['eviv1'][i] == 1:
        train_data.set_value(i,'Floor_Quality',0)
    elif train_data['eviv2'][i] == 1:
        train_data.set_value(i,'Floor_Quality',1)
    elif train_data['eviv3'][i] == 1:
        train_data.set_value(i,'Floor_Quality',2)

#for test_data

for i in test_data.index:
    if test_data['eviv1'][i] == 1:
        test_data.set_value(i,'Floor_Quality',0)
    elif test_data['eviv2'][i] == 1:
        test_data.set_value(i,'Floor_Quality',1)
    elif test_data['eviv3'][i] == 1:
        test_data.set_value(i,'Floor_Quality',2)
### Gender

'''
Create a new column called 'Gender'


0 -  male
1 -  female

'''
for i in train_data.index:
    if train_data['male'][i] == 1:
        train_data.set_value(i,'Gender',0)
    elif train_data['female'][i] == 1:
        train_data.set_value(i,'Gender',1)

#for test_data

for i in test_data.index:
    if test_data['male'][i] == 1:
        test_data.set_value(i,'Gender',0)
    elif test_data['female'][i] == 1:
        test_data.set_value(i,'Gender',1)

train_data.Gender.value_counts()
'''

Create a new column called 'Familiy_Status'
Values will range from 0 to 6 to indicate various status of a family members

estadocivil1
0  - if less than 10 years old
1  - if free or coupled uunion
2  - if married
3  - if divorced
4  - if separated
5  - if widow/er
6  - if single

'''

for i in train_data.index:
    if train_data['estadocivil1'][i] == 1:
        train_data.set_value(i,'Familiy_Status',0)
    elif train_data['estadocivil2'][i] == 1:
        train_data.set_value(i,'Familiy_Status',1)
    elif train_data['estadocivil3'][i] == 1:
        train_data.set_value(i,'Familiy_Status',2)
    elif train_data['estadocivil4'][i] == 1:
        train_data.set_value(i,'Familiy_Status',3)
    elif train_data['estadocivil5'][i] == 1:
        train_data.set_value(i,'Familiy_Status',4)
    elif train_data['estadocivil6'][i] == 1:
        train_data.set_value(i,'Familiy_Status',5)
    elif train_data['estadocivil7'][i] == 1:
        train_data.set_value(i,'Familiy_Status',6)

#for test_data

for i in test_data.index:
    if test_data['estadocivil1'][i] == 1:
        test_data.set_value(i,'Familiy_Status',0)
    elif test_data['estadocivil2'][i] == 1:
        test_data.set_value(i,'Familiy_Status',1)
    elif test_data['estadocivil3'][i] == 1:
        test_data.set_value(i,'Familiy_Status',2)
    elif test_data['estadocivil4'][i] == 1:
        test_data.set_value(i,'Familiy_Status',3)
    elif test_data['estadocivil5'][i] == 1:
        test_data.set_value(i,'Familiy_Status',4)
    elif test_data['estadocivil6'][i] == 1:
        test_data.set_value(i,'Familiy_Status',5)
    elif test_data['estadocivil7'][i] == 1:
        test_data.set_value(i,'Familiy_Status',6)
'''
Create a new column called 'Relationship'
Values will range from 0 to 11 to indicate the relationship.

parentesco1
0  -  if household head
1  -  if spouse/partner
2  -  if son/doughter
3  -  if stepson/doughter
4  -  if son/doughter in law
5  -  if grandson/doughter
6  -  if mother/father
7  -  if father/mother in law
8  -  if brother/sister
9  -  if brother/sister in law
10  -  if other family member
11  -  if other non family member

'''

for i in train_data.index:
    if train_data['parentesco1'][i] == 1:
        train_data.set_value(i,'Relationship',0)
    elif train_data['parentesco2'][i] == 1:
        train_data.set_value(i,'Relationship',1)
    elif train_data['parentesco3'][i] == 1:
        train_data.set_value(i,'Relationship',2)
    elif train_data['parentesco4'][i] == 1:
        train_data.set_value(i,'Relationship',3)
    elif train_data['parentesco5'][i] == 1:
        train_data.set_value(i,'Relationship',4)
    elif train_data['parentesco6'][i] == 1:
        train_data.set_value(i,'Relationship',5)
    elif train_data['parentesco7'][i] == 1:
        train_data.set_value(i,'Relationship',6)
    elif train_data['parentesco8'][i] == 1:
        train_data.set_value(i,'Relationship',7)
    elif train_data['parentesco9'][i] == 1:
        train_data.set_value(i,'Relationship',8)
    elif train_data['parentesco10'][i] == 1:
        train_data.set_value(i,'Relationship',9)
    elif train_data['parentesco11'][i] == 1:
        train_data.set_value(i,'Relationship',10)
    elif train_data['parentesco12'][i] == 1:
        train_data.set_value(i,'Relationship',11)

#for test_data
for i in test_data.index:
    if test_data['parentesco1'][i] == 1:
        test_data.set_value(i,'Relationship',0)
    elif test_data['parentesco2'][i] == 1:
        test_data.set_value(i,'Relationship',1)
    elif test_data['parentesco3'][i] == 1:
        test_data.set_value(i,'Relationship',2)
    elif test_data['parentesco4'][i] == 1:
        test_data.set_value(i,'Relationship',3)
    elif test_data['parentesco5'][i] == 1:
        test_data.set_value(i,'Relationship',4)
    elif test_data['parentesco6'][i] == 1:
        test_data.set_value(i,'Relationship',5)
    elif test_data['parentesco7'][i] == 1:
        test_data.set_value(i,'Relationship',6)
    elif test_data['parentesco8'][i] == 1:
        test_data.set_value(i,'Relationship',7)
    elif test_data['parentesco9'][i] == 1:
        test_data.set_value(i,'Relationship',8)
    elif test_data['parentesco10'][i] == 1:
        test_data.set_value(i,'Relationship',9)
    elif test_data['parentesco11'][i] == 1:
        test_data.set_value(i,'Relationship',10)
    elif test_data['parentesco12'][i] == 1:
        test_data.set_value(i,'Relationship',11)
'''

Create a new column called 'Education_Level'
Values will range from 0 to 7 to indicate the education level.


instlevel1

0   -  no level of education
1   -  incomplete primary
2   -  complete primary
3   -  incomplete academic secondary level
4   -  complete academic secondary level
5   -  incomplete technical secondary level
6   -  complete technical secondary level
7   -  undergraduate and higher education
8   -  postgraduate higher education


'''
for i in train_data.index:
    if train_data['instlevel1'][i] == 1:
        train_data.set_value(i,'Education_Level',0)
    elif train_data['instlevel2'][i] == 1:
        train_data.set_value(i,'Education_Level',1)
    elif train_data['instlevel3'][i] == 1:
        train_data.set_value(i,'Education_Level',2)
    elif train_data['instlevel4'][i] == 1:
        train_data.set_value(i,'Education_Level',3)
    elif train_data['instlevel5'][i] == 1:
        train_data.set_value(i,'Education_Level',4)
    elif train_data['instlevel6'][i] == 1:
        train_data.set_value(i,'Education_Level',5)
    elif train_data['instlevel7'][i] == 1:
        train_data.set_value(i,'Education_Level',6)
    elif train_data['instlevel8'][i] == 1:
        train_data.set_value(i,'Education_Level',7)
    elif train_data['instlevel9'][i] == 1:
        train_data.set_value(i,'Education_Level',8)



#for test_data

for i in test_data.index:
    if test_data['instlevel1'][i] == 1:
        test_data.set_value(i,'Education_Level',0)
    elif test_data['instlevel2'][i] == 1:
        test_data.set_value(i,'Education_Level',1)
    elif test_data['instlevel3'][i] == 1:
        test_data.set_value(i,'Education_Level',2)
    elif test_data['instlevel4'][i] == 1:
        test_data.set_value(i,'Education_Level',3)
    elif test_data['instlevel5'][i] == 1:
        test_data.set_value(i,'Education_Level',4)
    elif test_data['instlevel6'][i] == 1:
        test_data.set_value(i,'Education_Level',5)
    elif test_data['instlevel7'][i] == 1:
        test_data.set_value(i,'Education_Level',6)
    elif test_data['instlevel8'][i] == 1:
        test_data.set_value(i,'Education_Level',7)
    elif test_data['instlevel9'][i] == 1:
        test_data.set_value(i,'Education_Level',8)

train_data.Education_Level.value_counts()

'''

Create a new column called 'House_Status'
Values will range from 0 to 3 to indicate the house status.


tipovivi1
0   -  own and fully paid house
1   -  own,  paying in installments"
2   -  rented
3   -  precarious
4   -  other(assigned,  borrowed)"

'''

for i in train_data.index:
    if train_data['tipovivi1'][i] == 1:
        train_data.set_value(i,'House_Status',0)
    elif train_data['tipovivi2'][i] == 1:
        train_data.set_value(i,'House_Status',1)
    elif train_data['tipovivi3'][i] == 1:
        train_data.set_value(i,'House_Status',2)
    elif train_data['tipovivi4'][i] == 1:
        train_data.set_value(i,'House_Status',3)
    elif train_data['tipovivi5'][i] == 1:
        train_data.set_value(i,'House_Status',4)

#for test_data
for i in test_data.index:
    if test_data['tipovivi1'][i] == 1:
        test_data.set_value(i,'House_Status',0)
    elif test_data['tipovivi2'][i] == 1:
        test_data.set_value(i,'House_Status',1)
    elif test_data['tipovivi3'][i] == 1:
        test_data.set_value(i,'House_Status',2)
    elif test_data['tipovivi4'][i] == 1:
        test_data.set_value(i,'House_Status',3)
    elif test_data['tipovivi5'][i] == 1:
        test_data.set_value(i,'House_Status',4)

train_data.House_Status.value_counts()
'''
Create a new column called 'Region'
Values will range from 0 to 5 to indicate the different Regions. 

lugar1
0  - region Central
1  - region Chorotega
2  - region PacÃƒÂ­fico central
3  - region Brunca
4  - region Huetar AtlÃƒÂ¡ntica
5  - region Huetar Norte
'''


for i in train_data.index:
    if train_data['lugar1'][i] == 1:
        train_data.set_value(i,'Region',0)
    elif train_data['lugar2'][i] == 1:
        train_data.set_value(i,'Region',1)
    elif train_data['lugar3'][i] == 1:
        train_data.set_value(i,'Region',2)
    elif train_data['lugar4'][i] == 1:
        train_data.set_value(i,'Region',3)
    elif train_data['lugar5'][i] == 1:
        train_data.set_value(i,'Region',4)
    elif train_data['lugar6'][i] == 1:
        train_data.set_value(i,'Region',5)

#for test_data

for i in test_data.index:
    if test_data['lugar1'][i] == 1:
        test_data.set_value(i,'Region',0)
    elif test_data['lugar2'][i] == 1:
        test_data.set_value(i,'Region',1)
    elif test_data['lugar3'][i] == 1:
        test_data.set_value(i,'Region',2)
    elif test_data['lugar4'][i] == 1:
        test_data.set_value(i,'Region',3)
    elif test_data['lugar5'][i] == 1:
        test_data.set_value(i,'Region',4)
    elif test_data['lugar6'][i] == 1:
        test_data.set_value(i,'Region',5)

test_data.Region.value_counts()
### Area

'''
Create a new column called 'Area'

0  - Urban
1  - Rural


''' 

for i in train_data.index:
    if train_data['area1'][i] == 1:
        train_data.set_value(i,'Area',0)
    elif train_data['area2'][i] == 1:
        train_data.set_value(i,'Area',1)

#for test_data


for i in test_data.index:
    if test_data['area1'][i] == 1:
        test_data.set_value(i,'Area',0)
    elif test_data['area2'][i] == 1:
        test_data.set_value(i,'Area',1)

test_data.Area.value_counts()
#Remove unwanted columns
col_to_del = ['area1','area2','lugar1','lugar2','lugar3','lugar4','lugar5','lugar6','tipovivi1', \
              'tipovivi2','tipovivi3','tipovivi4','tipovivi5','instlevel1','instlevel2','instlevel3', \
              'instlevel4','instlevel5','instlevel6','instlevel7','instlevel8','instlevel9','parentesco1', \
              'parentesco2','parentesco3','parentesco4','parentesco5','parentesco6','parentesco7','parentesco8', \
              'parentesco9','parentesco10','parentesco11','parentesco12','estadocivil1','estadocivil2','estadocivil3',\
              'estadocivil4','estadocivil5','estadocivil6','estadocivil7','male','female','eviv1','eviv2','eviv3', \
              'etecho1','etecho2','etecho3','epared1','epared2','epared3','elimbasu1','elimbasu2','elimbasu3', \
              'elimbasu4','elimbasu6','energcocinar1','energcocinar2','energcocinar3','energcocinar4',\
              'sanitario1','sanitario2','sanitario3','sanitario5','sanitario6','abastaguadentro','abastaguafuera',\
              'abastaguano','techozinc','techoentrepiso','techocane','techootro','pisomadera','pisonotiene','pisonatur',\
              'pisoother','pisocemento','pisomoscer','paredother','paredfibras','paredzinc','paredmad','pareddes',\
              'paredpreb','paredzocalo','paredblolad'             
             ]

len(col_to_del)


train_data.drop(col_to_del,axis=1,inplace=True)

test_data.drop(col_to_del,axis=1,inplace=True)

print(train_data.shape)

test_data.shape
#Check whether the class is imbalanced or not
plt.figure(figsize=(6,6))
sns.countplot(x="Target", data=train_data)
plt.ylabel('Count', fontsize=12)
plt.xlabel('target', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Class Imbalance Checking", fontsize=15)
plt.show()
s = train_data.isnull().any().reset_index()
s.loc[s[0]==True]
ts = test_data.isnull().any().reset_index()
ts.loc[ts[0]==True]
#drop rows which contains null values
train_data.dropna(subset=['Roof_Material'],inplace=True)
train_data.dropna(subset=['Education_Level'],inplace=True)
train_data.dropna(subset=['Rubbish_Disposal'],inplace=True)

test_data.dropna(subset=['Roof_Material'],inplace=True)
test_data.dropna(subset=['Education_Level'],inplace=True)
test_data.dropna(subset=['Rubbish_Disposal'],inplace=True)
#finding Correlation
import seaborn as sns
corr = train_data.corr()
sns.heatmap(corr)
y = train_data.Target

X = train_data.drop(['Target'],axis=1)

print(X.shape)

print(y.shape)
#Apply SMOTE for oversampling (to avoid class imbalance)

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)
# Apply Standard Scalar
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)
print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '2': {} \n".format(sum(y_train==2)))
print("Before OverSampling, counts of label '3': {} \n".format(sum(y_train==3)))
print("Before OverSampling, counts of label '4': {} \n".format(sum(y_train==4)))
# Applying SMOTE
sm = SMOTE(random_state=0)
X_train_res, y_train_res = sm.fit_sample(X_train_sc, y_train.ravel())
print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '2': {} \n".format(sum(y_train_res==2)))
print("After OverSampling, counts of label '3': {} \n".format(sum(y_train_res==3)))
print("After OverSampling, counts of label '4': {} \n".format(sum(y_train_res==4)))
Counter(y_train_res)
#Here we got balanced target variables
X_train_res = pd.DataFrame(X_train_res)
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
### Logistic Regression

kfold = KFold(n_splits=10, random_state=0)
logestic = LogisticRegression()
logestic.fit(X_train_res,y_train_res)
scoring = 'accuracy'
results = cross_val_score(logestic,X_train_res, y_train_res, cv=kfold, scoring=scoring)
acc_log = results.mean()
log_std = results.std()
acc_log
### Decision Tree

kfold = KFold(n_splits=10, random_state=0)
dTree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dTree.fit(X_train_res,y_train_res)
scoring = 'accuracy'
results = cross_val_score(dTree,X_train_res,y_train_res, cv=kfold, scoring=scoring)
acc_dt = results.mean()
dt_std = results.std()
acc_dt
### KNN

kfold = KFold(n_splits=10, random_state=0)
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(X_train_res,y_train_res)
scoring = 'accuracy'
results = cross_val_score(knn,X_train_res,y_train_res, cv=kfold, scoring=scoring)
acc_knn = results.mean()
knn_std = results.std()
acc_knn
### Random Forest

kfold = KFold(n_splits=10, random_state=0)
randomForest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
randomForest.fit(X_train_res,y_train_res)
scoring = 'accuracy'
results = cross_val_score(randomForest,X_train_res,y_train_res, cv=kfold, scoring=scoring)
acc_rf = results.mean()
rf_std = results.std()
acc_rf


### SVC

kfold = KFold(n_splits=10, random_state=0)
svc = SVC()
svc.fit(X_train_res,y_train_res)
scoring = 'accuracy'
results = cross_val_score(svc,X_train_res,y_train_res, cv=kfold, scoring=scoring)
acc_svc = results.mean()
svc_std = results.std()
models = pd.DataFrame({'Model': ['LogisticRegression','SVC','KNN', 'Decision Tree','Random Forest'],
                       'Score': [acc_log, acc_svc, acc_knn, acc_dt, acc_rf],
                       'Std.':[log_std,svc_std,knn_std,dt_std,rf_std]
                      })
models.sort_values(by='Score', ascending=False)
y_prediction = randomForest.predict(X_test_sc)
print(Counter(y_prediction))
Counter(y_test)
from sklearn.metrics import accuracy_score

# Evaluate accuracy
print(accuracy_score(y_test, y_prediction))
#Submission

test_data_without_id = test_data.copy()
test_data_without_id.drop(['Id'],axis = 1,inplace = True)
y_pred_final = randomForest.predict(test_data_without_id)
Counter(y_pred_final)
submission_file = pd.DataFrame({"Id":test_data.Id,"Target":y_pred_final})

submission_file.to_csv("poverty_prediction_submission.csv",index=False)
