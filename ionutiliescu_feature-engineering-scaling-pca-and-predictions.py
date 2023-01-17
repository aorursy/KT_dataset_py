# data analysis

import pandas as pd

import numpy as np



# visualisation

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# metrics and algorithm validation

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_squared_error

from sklearn.model_selection import KFold, cross_val_score



# encoding and modeling

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from scipy.stats import skew
import os

mingw_path = 'C:\Program Files\mingw-w64\x86_64-7.1.0-posix-seh-rt_v5-rev0\mingw64\bin'



os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

import xgboost as xgb
# import the dataset

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
df = train.copy()
df.head()
df.describe()
# viewing the correlation between columns

plt.figure(figsize=(18,15))

sns.heatmap(df.corr())
df.columns.tolist()
df.info()
df.SalePrice.describe()
# eliminating all columns with a price over 350,000

to_el = df[df.SalePrice > 350000].index.tolist()
df.drop(to_el,inplace=True)
# creating PUD column



# indexes where PUD = 1

pud_i = df[df.MSSubClass.isin([120,150,160,180])].index.tolist()



# create PUD

df['PUD'] = 0



# add ones in PUD

for i in pud_i:

    df.set_value(i,'PUD',1)

    

# test set

# indexes where PUD = 1

pud_i = test[test.MSSubClass.isin([120,150,160,180])].index.tolist()



# create PUD

test['PUD'] = 0



# add ones in PUD

for i in pud_i:

    test.set_value(i,'PUD',1)
# dropping the MSSubClass

df.drop('MSSubClass',axis=1,inplace=True)

test.drop('MSSubClass',axis=1,inplace=True)
df.HouseStyle.value_counts()
for i in df.index:

    if df.HouseStyle[i] == '1Story':

        df.set_value(i,'HouseStyle',0)

    elif df.HouseStyle[i] in ['1.5Fin','1.5Unf']:

        df.set_value(i,'HouseStyle',1)

    elif df.HouseStyle[i] in ['2Story','SLvl','SFoyer']:

        df.set_value(i,'HouseStyle',2)

    elif df.HouseStyle[i] in ['2.5Unf','2.5Fin']:

        df.set_value(i,'HouseStyle',3)

        

for i in test.index:

    if test.HouseStyle[i] == '1Story':

        test.set_value(i,'HouseStyle',0)

    elif test.HouseStyle[i] in ['1.5Fin','1.5Unf']:

        test.set_value(i,'HouseStyle',1)

    elif test.HouseStyle[i] in ['2Story','SLvl','SFoyer']:

        test.set_value(i,'HouseStyle',2)

    elif test.HouseStyle[i] in ['2.5Unf','2.5Fin']:

        test.set_value(i,'HouseStyle',3)
# add for OneHot

hot = ['HouseStyle']
df.HouseStyle = pd.to_numeric(df.HouseStyle)

test.HouseStyle = pd.to_numeric(test.HouseStyle)
# some empty values

ind = test[test.MSZoning.isnull()].index.tolist()
for i in ind:

    test.set_value(i,'MSZoning',1)
df.MSZoning.value_counts()
test.MSZoning.value_counts()
for i in df.index:

    if df.MSZoning[i] == 'C (all)':

        df.set_value(i,'MSZoning',0)

    elif df.MSZoning[i] == 'RL':

        df.set_value(i,'MSZoning',1)

    elif df.MSZoning[i] == 'RM':

        df.set_value(i,'MSZoning',2)

    elif df.MSZoning[i] == 'RH':

        df.set_value(i,'MSZoning',3)

    elif df.MSZoning[i] == 'FV':

        df.set_value(i,'MSZoning',4)
for i in test.index:

    if test.MSZoning[i] == 'C (all)':

        test.set_value(i,'MSZoning',0)

    elif test.MSZoning[i] == 'RL':

        test.set_value(i,'MSZoning',1)

    elif test.MSZoning[i] == 'RM':

        test.set_value(i,'MSZoning',2)

    elif test.MSZoning[i] == 'RH':

        test.set_value(i,'MSZoning',3)

    elif test.MSZoning[i] == 'FV':

        test.set_value(i,'MSZoning',4)
# add for HotEncoder

hot.append('MSZoning')
df.LotArea.describe()
df.LotArea.hist()
to_drop = df.LotArea[df.LotArea > 60000].keys().tolist()

df.drop(to_drop, inplace=True)
skw = ['LotArea']
df.drop('Street',axis=1,inplace=True)

test.drop('Street',axis=1,inplace=True)
df.drop('Alley',axis=1,inplace=True)

test.drop('Alley',axis=1,inplace=True)
for i in df.index:

    if df.LotShape[i] == 'Reg':

        df.set_value(i,'LotShape',1)

    else:

        df.set_value(i,'LotShape',0)



for i in test.index:

    if test.LotShape[i] == 'Reg':

        test.set_value(i,'LotShape',1)

    else:

        test.set_value(i,'LotShape',0)
df.LandContour.value_counts()
for i in df.index:

    if df.LandContour[i] == 'Lvl':

        df.set_value(i,'LandContour', 0)

    elif df.LandContour[i] == 'Bnk':

        df.set_value(i,'LandContour', 1)

    elif df.LandContour[i] == 'HLS':

        df.set_value(i,'LandContour', 2)

    elif df.LandContour[i] == 'Low':

        df.set_value(i,'LandContour', 3)
for i in test.index:

    if test.LandContour[i] == 'Lvl':

        test.set_value(i,'LandContour', 0)

    elif test.LandContour[i] == 'Bnk':

        test.set_value(i,'LandContour', 1)

    elif test.LandContour[i] == 'HLS':

        test.set_value(i,'LandContour', 2)

    elif test.LandContour[i] == 'Low':

        test.set_value(i,'LandContour', 3)
hot.append('LandContour')
df.Utilities.value_counts()
test.Utilities.value_counts()
df.drop('Utilities',axis=1,inplace=True)

test.drop('Utilities',axis=1,inplace=True)
df.LotConfig.value_counts()
test.LotConfig.value_counts()
for i in df.index:

    if df.LotConfig[i] == 'Inside':

        df.set_value(i,'LotConfig', 0)

    elif df.LotConfig[i] == 'Corner':

        df.set_value(i,'LotConfig', 1)

    elif df.LotConfig[i] == 'CulDSac':

        df.set_value(i,'LotConfig', 2)

    elif df.LotConfig[i] in ['FR2','FR3']:

        df.set_value(i,'LotConfig', 3)
for i in test.index:

    if test.LotConfig[i] == 'Inside':

        test.set_value(i,'LotConfig', 0)

    elif test.LotConfig[i] == 'Corner':

        test.set_value(i,'LotConfig', 1)

    elif test.LotConfig[i] == 'CulDSac':

        test.set_value(i,'LotConfig', 2)

    elif test.LotConfig[i] in ['FR2','FR3']:

        test.set_value(i,'LotConfig', 3)
hot.append('LotConfig')
df.LandSlope.value_counts()
test.LandSlope.value_counts()
df.drop('LandSlope',axis=1,inplace=True)

test.drop('LandSlope',axis=1,inplace=True)
encoder = LabelEncoder()
# first fitting and transforming the df column

df.Neighborhood = encoder.fit_transform(df.Neighborhood)
# then fitting the test column

test.Neighborhood = encoder.transform(test.Neighborhood)
hot.append('Neighborhood')
df.Condition1.value_counts()
df.Condition2.value_counts()
# classifying ones that appear in both columns and are not Norm

df[df.Condition2 != 'Norm'][['Condition1', 'Condition2']]



# 9 - Neg

# 29 - Neg

# 63 - Neg

# 88 - Neg

# 184 - Neg

# 523 - Pos

# 531 - Normal

# 548 - Neg

# 583 - Pos

# 589 - Neg

# 974 - Neg

# 1003 - Neg

# 1186 - Neg

# 1230 - Neg
# positive and negative categories

pos_c = ['PosN', 'RRNn', 'RRNe', 'PosA']

neg_c = ['Feedr', 'Artery', 'RRAe']
# finding the indexes of those in Condition1 column

negs = []

pos = []

for i in df.index:

    if df['Condition1'][i] in pos_c:

        pos.append(i)

    elif df['Condition1'][i] in neg_c:

        negs.append(i)
# adding the extra ones in Condition2 in their corresponding categories

negs.append([9,29,63,88,184,548,589,974,1003,1186,1230])

pos.append([523,583])
# creating the new column with Normal as standard

df['Conditions'] = 1
# replacing the corresponding rows with the categories

for i in negs:

    df.set_value(i,'Conditions',0)

    

for i in pos:

    df.set_value(i,'Conditions',2)
# same for test set

test.Condition1.value_counts()
test.Condition2.value_counts()
# classifying ones that appear in both columns and are not Norm

test[test.Condition2 != 'Norm'][['Condition1', 'Condition2']]



# 81 - Neg

# 203 - Pos

# 245 - Pos

# 486 - Pos

# 593 - Neg

# 650 - Neg

# 778 - Neg

# 807 - Pos

# 940 - Neg

# 995 - Neg

# 1138 - Pos

# 1258 - Neg

# 1336 - Neg

# 1342 - Neg
negs = []

pos = []

for i in test.index:

    if test['Condition1'][i] in pos_c:

        pos.append(i)

    elif test['Condition1'][i] in neg_c:

        negs.append(i)
negs.append([81,593,650,778,940,995,1258,1336,1342])

pos.append([203,245,486,807,1138])
test['Conditions'] = 1
for i in negs:

    test.set_value(i,'Conditions',0)

    

for i in pos:

    test.set_value(i,'Conditions',2)
# dropping them once done

df.drop('Condition1',axis=1,inplace=True)

test.drop('Condition1',axis=1,inplace=True)

df.drop('Condition2',axis=1,inplace=True)

test.drop('Condition2',axis=1,inplace=True)
df.BldgType.value_counts()
test.BldgType.value_counts()
for i in df.index:

    if df.BldgType[i] in ['1Fam', '2fmCon']:

        df.set_value(i,'BldgType',1)

    else:

        df.set_value(i,'BldgType',0)

        

for i in test.index:

    if test.BldgType[i] in ['1Fam', '2fmCon']:

        test.set_value(i,'BldgType',1)

    else:

        test.set_value(i,'BldgType',0)
df.OverallQual.value_counts()
good = [10,9,8,7]

average = [6,5,4]

bad = [3,2,1]
# train

for i in df.index:

    if df['OverallQual'][i] in good:

        df.set_value(i,'OverallQual',2)

    elif df['OverallQual'][i] in average:

        df.set_value(i,'OverallQual',1)

    elif df['OverallQual'][i] in bad:

        df.set_value(i,'OverallQual',0)
# test

for i in test.index:

    if test['OverallQual'][i] in good:

        test.set_value(i,'OverallQual',2)

    elif test['OverallQual'][i] in average:

        test.set_value(i,'OverallQual',1)

    elif test['OverallQual'][i] in bad:

        test.set_value(i,'OverallQual',0)
# train

for i in df.index:

    if df['OverallCond'][i] in good:

        df.set_value(i,'OverallCond',2)

    elif df['OverallCond'][i] in average:

        df.set_value(i,'OverallCond',1)

    elif df['OverallCond'][i] in bad:

        df.set_value(i,'OverallCond',0)

        

# test

for i in test.index:

    if test['OverallCond'][i] in good:

        test.set_value(i,'OverallCond',2)

    elif test['OverallCond'][i] in average:

        test.set_value(i,'OverallCond',1)

    elif test['OverallCond'][i] in bad:

        test.set_value(i,'OverallCond',0)
for i in df.index:

    if df['YearBuilt'][i] == df['YearRemodAdd'][i]:

        df.set_value(i,'YearRemodAdd',1)

    else:

        df.set_value(i,'YearRemodAdd',0)

        

for i in test.index:

    if test['YearBuilt'][i] == test['YearRemodAdd'][i]:

        test.set_value(i,'YearRemodAdd',1)

    else:

        test.set_value(i,'YearRemodAdd',0)
df.YearBuilt.describe()
df.YearBuilt.hist(bins=6)
test.YearBuilt.describe()
test.YearBuilt.hist()
# train

for i in df.index:

    if df['YearBuilt'][i] > 0 and df['YearBuilt'][i] <= 1895:

        df.set_value(i,'YearBuilt', 0)

    elif df['YearBuilt'][i] > 1895 and df['YearBuilt'][i] <= 1918:

        df.set_value(i,'YearBuilt', 1)

    elif df['YearBuilt'][i] > 1918 and df['YearBuilt'][i] <= 1941:

        df.set_value(i,'YearBuilt', 2)

    elif df['YearBuilt'][i] > 1941 and df['YearBuilt'][i] <= 1964:

        df.set_value(i,'YearBuilt', 3)

    elif df['YearBuilt'][i] > 1964 and df['YearBuilt'][i] <= 1987:

        df.set_value(i,'YearBuilt', 4)

    elif df['YearBuilt'][i] > 1987:

        df.set_value(i,'YearBuilt', 5)
# test

for i in test.index:

    if test['YearBuilt'][i] > 0 and test['YearBuilt'][i] <= 1895:

        test.set_value(i,'YearBuilt', 0)

    elif test['YearBuilt'][i] > 1895 and test['YearBuilt'][i] <= 1918:

        test.set_value(i,'YearBuilt', 1)

    elif test['YearBuilt'][i] > 1918 and test['YearBuilt'][i] <= 1941:

        test.set_value(i,'YearBuilt', 2)

    elif test['YearBuilt'][i] > 1941 and test['YearBuilt'][i] <= 1964:

        test.set_value(i,'YearBuilt', 3)

    elif test['YearBuilt'][i] > 1964 and test['YearBuilt'][i] <= 1987:

        test.set_value(i,'YearBuilt', 4)

    elif test['YearBuilt'][i] > 1987:

        test.set_value(i,'YearBuilt', 5)
hot.append('YearBuilt')
df.RoofStyle.value_counts()
test.RoofStyle.value_counts()
for i in df.index:

    if df.RoofStyle[i] == 'Gable':

        df.set_value(i,'RoofStyle',0)

    elif df.RoofStyle[i] == 'Hip':

        df.set_value(i,'RoofStyle',1)

    else:

        df.set_value(i,'RoofStyle',2)
for i in test.index:

    if test.RoofStyle[i] == 'Gable':

        test.set_value(i,'RoofStyle',0)

    elif test.RoofStyle[i] == 'Hip':

        test.set_value(i,'RoofStyle',1)

    else:

        test.set_value(i,'RoofStyle',2)
hot.append('RoofStyle')
df.RoofMatl.value_counts()
test.RoofMatl.value_counts()
df.drop('RoofMatl',axis=1,inplace=True)

test.drop('RoofMatl',axis=1,inplace=True)
df['SalePrice'].groupby(df['Exterior1st']).mean()
df['SalePrice'].groupby(df['Exterior2nd']).mean()
# getting all the combinations

combs = []

for i in df.index:

    combs.append((df.Exterior1st[i], df.Exterior2nd[i]))
# eliminating duplicates

combs = set(combs)
# different combos from the training set

test_diff = []

count = 0

for i in test.index:

    found = 0

    for j in combs:

        if (test.Exterior1st[i], test.Exterior2nd[i]) == j or (test.Exterior2nd[i], test.Exterior1st[i]) == j:

            found = 1

    if found == 0:

        test_diff.append((test.Exterior1st[i], test.Exterior2nd[i]))

        count += 1
# eliminating duplicates

test_diff = set(test_diff)
# rows with different fields in test set

count
# creating a new column

df['Exterior'] = ''
# adding combinations

for i in df.index:

    for j in combs:

        if (df.Exterior1st[i], df.Exterior2nd[i]) == j or (df.Exterior2nd[i], df.Exterior1st[i]) == j:

            df.set_value(i,'Exterior',j)
# mode values grouped by combination

# chosen mode as it brigns the mean closest to the overall one

grp = pd.DataFrame(df.SalePrice.groupby(df.Exterior).agg(lambda x:x.value_counts().index[0]).sort_values(ascending=False))
# number of occurances

grp2 = pd.DataFrame(df.Exterior.value_counts())
# joining them

new = grp.join(grp2)

new.head()
new.describe()
new.SalePrice.hist(bins=5)
# value categories split by combinations

zeros = new[new.SalePrice <= 134600]['Exterior'].keys().tolist()

ones = new[(new.SalePrice > 134600) & (new.SalePrice <= 187200)]['Exterior'].keys().tolist()

twos = new[(new.SalePrice > 187200) & (new.SalePrice <= 239800)]['Exterior'].keys().tolist()

threes = new[new.SalePrice > 239800]['Exterior'].keys().tolist()
# replace on training set

for i in df.index:

    if df.Exterior[i] in zeros:

        df.set_value(i,'Exterior',0)

    elif df.Exterior[i] in ones:

        df.set_value(i,'Exterior',1)

    elif df.Exterior[i] in twos:

        df.set_value(i,'Exterior',2)

    elif df.Exterior[i] in threes:

        df.set_value(i,'Exterior',3)
# same for test set

test['Exterior'] = ''



# adding combinations

for i in test.index:

    for j in combs:

        if (test.Exterior1st[i], test.Exterior2nd[i]) == j or (test.Exterior2nd[i], test.Exterior1st[i]) == j:

            test.set_value(i,'Exterior',j)
# replace on test set

for i in test.index:

    if test.Exterior[i] in zeros:

        test.set_value(i,'Exterior',0)

    elif test.Exterior[i] in ones:

        test.set_value(i,'Exterior',1)

    elif test.Exterior[i] in twos:

        test.set_value(i,'Exterior',2)

    elif test.Exterior[i] in threes:

        test.set_value(i,'Exterior',3)

    else:

        test.set_value(i,'Exterior',1)
# dropping Exterior1st and Exterior2nd

df.drop('Exterior1st',axis=1,inplace=True)

df.drop('Exterior2nd',axis=1,inplace=True)

test.drop('Exterior1st',axis=1,inplace=True)

test.drop('Exterior2nd',axis=1,inplace=True)
df.MasVnrType.value_counts()
ind = df.MasVnrType[df.MasVnrType.isnull()].index.tolist()
for i in ind:

    df.set_value(i,'MasVnrType',0)
ind = test.MasVnrType[test.MasVnrType.isnull()].index.tolist()



for i in ind:

    test.set_value(i,'MasVnrType',0)
for i in df.index:

    if df.MasVnrType[i] in ['BrkCmn', 'None']:

        df.set_value(i,'MasVnrType', 0)

    elif df.MasVnrType[i] == 'BrkFace':

        df.set_value(i,'MasVnrType', 1)

    elif df.MasVnrType[i] == 'Stone':

        df.set_value(i,'MasVnrType', 2)
for i in test.index:

    if test.MasVnrType[i] in ['BrkCmn', 'None']:

        test.set_value(i,'MasVnrType', 0)

    elif test.MasVnrType[i] == 'BrkFace':

        test.set_value(i,'MasVnrType', 1)

    elif test.MasVnrType[i] == 'Stone':

        test.set_value(i,'MasVnrType', 2)
df.MasVnrArea.describe()
ind = df.MasVnrArea[df.MasVnrArea.isnull()].index.tolist()

# same empty columns as above

for i in ind:

    df.set_value(i,'MasVnrArea',0)
ind = test.MasVnrArea[test.MasVnrArea.isnull()].index.tolist()



for i in ind:

    test.set_value(i,'MasVnrArea',0)
df[df.MasVnrArea != 0]['MasVnrArea'].hist()
df[df.MasVnrArea != 0]['MasVnrArea'].sort_values(ascending=False).head()
# drop outlier

df.drop(297,inplace=True)
skw.append('MasVnrArea')
df.ExterQual.value_counts()
# train

for i in df.index:

    if df.ExterQual[i] in ['Gd', 'Ex']:

        df.set_value(i,'ExterQual', 2)

    elif df.ExterQual[i] == 'TA':

        df.set_value(i,'ExterQual', 1)

    elif df.ExterQual[i] in ['Fa', 'Po']:

        df.set_value(i,'ExterQual', 0)
# test

for i in test.index:

    if test.ExterQual[i] in ['Gd', 'Ex']:

        test.set_value(i,'ExterQual', 2)

    elif test.ExterQual[i] == 'TA':

        test.set_value(i,'ExterQual', 1)

    elif test.ExterQual[i] in ['Fa', 'Po']:

        test.set_value(i,'ExterQual', 0)
# train

for i in df.index:

    if df.ExterCond[i] in ['Gd', 'Ex']:

        df.set_value(i,'ExterCond', 2)

    elif df.ExterCond[i] == 'TA':

        df.set_value(i,'ExterCond', 1)

    elif df.ExterCond[i] in ['Fa', 'Po']:

        df.set_value(i,'ExterCond', 0)

        

# test

for i in test.index:

    if test.ExterCond[i] in ['Gd', 'Ex']:

        test.set_value(i,'ExterCond', 2)

    elif test.ExterCond[i] == 'TA':

        test.set_value(i,'ExterCond', 1)

    elif test.ExterCond[i] in ['Fa', 'Po']:

        test.set_value(i,'ExterCond', 0)
df.Foundation.value_counts()
for i in df.index:

    if df.Foundation[i] == 'BrkTil':

        df.set_value(i,'Foundation',0)

    elif df.Foundation[i] == 'CBlock':

        df.set_value(i,'Foundation',1)

    elif df.Foundation[i] == 'PConc':

        df.set_value(i,'Foundation',2)

    else:

        df.set_value(i,'Foundation',3)
for i in test.index:

    if test.Foundation[i] == 'BrkTil':

        test.set_value(i,'Foundation',0)

    elif test.Foundation[i] == 'CBlock':

        test.set_value(i,'Foundation',1)

    elif test.Foundation[i] == 'PConc':

        test.set_value(i,'Foundation',2)

    else:

        test.set_value(i,'Foundation',3)
hot.append('Foundation')
df.BsmtQual.value_counts()
# creating new column

df['HasBsmt'] = 1
# empty indexes

idx = df.BsmtQual[df.BsmtQual.isnull()].index.tolist()
# adding the empty ones

for i in idx:

    df.set_value(i,'HasBsmt',0)
test['HasBsmt'] = 1

# empty indexes

idx = test.BsmtQual[test.BsmtQual.isnull()].index.tolist()

# adding the empty ones

for i in idx:

    test.set_value(i,'HasBsmt',0)
for i in df.index:

    if df.BsmtQual[i] in ['Po', 'Fa']:

        df.set_value(i,'BsmtQual',1)

    elif df.BsmtQual[i] == 'TA':

        df.set_value(i,'BsmtQual',2)

    elif df.BsmtQual[i] in ['Gd', 'Ex']:

        df.set_value(i,'BsmtQual',3)

    else:

        df.set_value(i,'BsmtQual',0)
for i in test.index:

    if test.BsmtQual[i] in ['Po', 'Fa']:

        test.set_value(i,'BsmtQual',1)

    elif test.BsmtQual[i] == 'TA':

        test.set_value(i,'BsmtQual',2)

    elif test.BsmtQual[i] in ['Gd', 'Ex']:

        test.set_value(i,'BsmtQual',3)

    else:

        test.set_value(i,'BsmtQual',0)
df.BsmtCond.value_counts()
for i in df.index:

    if df.BsmtCond[i] in ['Po', 'Fa']:

        df.set_value(i,'BsmtCond',1)

    elif df.BsmtCond[i] == 'TA':

        df.set_value(i,'BsmtCond',2)

    elif df.BsmtCond[i] in ['Gd', 'Ex']:

        df.set_value(i,'BsmtCond',3)

    else:

        df.set_value(i,'BsmtCond',0)

        

for i in test.index:

    if test.BsmtCond[i] in ['Po', 'Fa']:

        test.set_value(i,'BsmtCond',1)

    elif test.BsmtCond[i] == 'TA':

        test.set_value(i,'BsmtCond',2)

    elif test.BsmtCond[i] in ['Gd', 'Ex']:

        test.set_value(i,'BsmtCond',3)

    else:

        test.set_value(i,'BsmtCond',0)
df.BsmtExposure.value_counts()
for i in df.index:

    if df.BsmtExposure[i] in ['Gd', 'Av']:

        df.set_value(i,'BsmtExposure',0)

    elif df.BsmtExposure[i] == 'Mn':

        df.set_value(i,'BsmtExposure',1)

    elif df.BsmtExposure[i] == 'No':

        df.set_value(i,'BsmtExposure',2)

    else:

        df.set_value(i,'BsmtExposure',3)

        

for i in test.index:

    if test.BsmtExposure[i] in ['Gd', 'Av']:

        test.set_value(i,'BsmtExposure',0)

    elif test.BsmtExposure[i] == 'Mn':

        test.set_value(i,'BsmtExposure',1)

    elif test.BsmtExposure[i] == 'No':

        test.set_value(i,'BsmtExposure',2)

    else:

        test.set_value(i,'BsmtExposure',3)
hot.append('BsmtExposure')
df.BsmtFinType1.value_counts()
#Type1

for i in df.index:

    if df.BsmtFinType1[i] == 'Unf':

        df.set_value(i,'BsmtFinType1',1)

    elif df.BsmtFinType1[i] in ['BLQ', 'LwQ']:

        df.set_value(i,'BsmtFinType1',2)

    elif df.BsmtFinType1[i] in ['GLQ', 'ALQ', 'Rec']:

        df.set_value(i,'BsmtFinType1',3)

    else:

        df.set_value(i,'BsmtFinType1',0)

        

for i in test.index:

    if test.BsmtFinType1[i] == 'Unf':

        test.set_value(i,'BsmtFinType1',1)

    elif test.BsmtFinType1[i] in ['BLQ', 'LwQ']:

        test.set_value(i,'BsmtFinType1',2)

    elif test.BsmtFinType1[i] in ['GLQ', 'ALQ', 'Rec']:

        test.set_value(i,'BsmtFinType1',3)

    else:

        test.set_value(i,'BsmtFinType1',0)
# Type2

for i in df.index:

    if df.BsmtFinType2[i] == 'Unf':

        df.set_value(i,'BsmtFinType2',1)

    elif df.BsmtFinType2[i] in ['BLQ', 'LwQ']:

        df.set_value(i,'BsmtFinType2',2)

    elif df.BsmtFinType2[i] in ['GLQ', 'ALQ', 'Rec']:

        df.set_value(i,'BsmtFinType2',3)

    else:

        df.set_value(i,'BsmtFinType2',0)

        

for i in test.index:

    if test.BsmtFinType2[i] == 'Unf':

        test.set_value(i,'BsmtFinType2',1)

    elif test.BsmtFinType2[i] in ['BLQ', 'LwQ']:

        test.set_value(i,'BsmtFinType2',2)

    elif test.BsmtFinType2[i] in ['GLQ', 'ALQ', 'Rec']:

        test.set_value(i,'BsmtFinType2',3)

    else:

        test.set_value(i,'BsmtFinType2',0)
df.BsmtFinSF1.hist()
df.BsmtFinSF2.hist()
# empty value

test.set_value(test.BsmtFinSF1[test.BsmtFinSF1.isnull()].index[0],'BsmtFinSF1',0)

print('---')
# empty value

test.set_value(test.BsmtFinSF2[test.BsmtFinSF2.isnull()].index[0],'BsmtFinSF2',0)

print('---')
skw.append(['BsmtFinSF1', 'BsmtFinSF2'])
no_b = df.HasBsmt[df.HasBsmt == 0].index.tolist()
unf = df.BsmtUnfSF[df.BsmtUnfSF == 0].index.tolist()
df['BsmtFinished'] = 0

for i in df.index:

    if df.BsmtUnfSF[i] == 0 and i in unf and i not in no_b:

        df.set_value(i,'BsmtFinished',1)
df.BsmtUnfSF.hist()
test.set_value(test.BsmtUnfSF[test.BsmtUnfSF.isnull()].index[0],'BsmtUnfSF',0)



no_b = test.HasBsmt[test.HasBsmt == 0].index.tolist()

unf = test.BsmtUnfSF[test.BsmtUnfSF == 0].index.tolist()



test['BsmtFinished'] = 0

for i in test.index:

    if test.BsmtUnfSF[i] == 0 and i in unf and i not in no_b:

        test.set_value(i,'BsmtFinished',1)
skw.append('BsmtUnfSF')
df.TotalBsmtSF.hist()
# dropping outliers

elim = df[df.TotalBsmtSF > 2300].index.tolist()
df.drop(elim,inplace=True)
test.set_value(test.TotalBsmtSF[test.TotalBsmtSF.isnull()].index[0],'TotalBsmtSF',0)

print('---')
test.TotalBsmtSF.hist()
df.Heating.value_counts()
test.Heating.value_counts()
df.drop('Heating', axis=1, inplace=True)

test.drop('Heating', axis=1, inplace=True)
df.HeatingQC.value_counts()
for i in df.index:

    if df.HeatingQC[i] in ['Po', 'Fa']:

        df.set_value(i,'HeatingQC',0)

    elif df.HeatingQC[i] == 'TA':

        df.set_value(i,'HeatingQC',1)

    elif df.HeatingQC[i] == 'Gd':

        df.set_value(i,'HeatingQC',2)

    elif df.HeatingQC[i] == 'Ex':

        df.set_value(i,'HeatingQC',3)
for i in test.index:

    if test.HeatingQC[i] in ['Po', 'Fa']:

        test.set_value(i,'HeatingQC',0)

    elif test.HeatingQC[i] == 'TA':

        test.set_value(i,'HeatingQC',1)

    elif test.HeatingQC[i] == 'Gd':

        test.set_value(i,'HeatingQC',2)

    elif test.HeatingQC[i] == 'Ex':

        test.set_value(i,'HeatingQC',3)
encoder = LabelEncoder()



# train

df.CentralAir = encoder.fit_transform(df.CentralAir)

# test

test.CentralAir = encoder.transform(test.CentralAir)
df.Electrical.value_counts()
test.Electrical.value_counts()
for i in df.index:

    if df.Electrical[i] in ['Mix','FuseP','FuseF']:

        df.set_value(i,'Electrical',0)

    elif df.Electrical[i] == 'FuseA':

        df.set_value(i,'Electrical',1)

    elif df.Electrical[i] == 'SBrkr':

        df.set_value(i,'Electrical',2)
for i in test.index:

    if test.Electrical[i] in ['Mix','FuseP','FuseF']:

        test.set_value(i,'Electrical',0)

    elif test.Electrical[i] == 'FuseA':

        test.set_value(i,'Electrical',1)

    elif test.Electrical[i] == 'SBrkr':

        test.set_value(i,'Electrical',2)
df.set_value(df.Electrical[df.Electrical.isnull()].index[0],'Electrical',2)

print('---')
hot.append('Electrical')
df['1stFlrSF'].hist()
outs = df[df['1stFlrSF'] > 2300].index.tolist()
df.drop(outs,inplace=True)
test['1stFlrSF'].hist()
skw.append('1stFlrSF')
df['2ndFlrSF'].hist()
test['2ndFlrSF'].hist()
df.LowQualFinSF.value_counts()
df.drop('LowQualFinSF', axis=1,inplace=True)

test.drop('LowQualFinSF', axis=1,inplace=True)
df.GrLivArea.hist()
test.GrLivArea.hist()
skw.append('GrLivArea')
df.BsmtFullBath.value_counts()
test.BsmtFullBath.value_counts()
# 3 to 2

df.set_value(df.BsmtFullBath[df.BsmtFullBath == 3].index[0],'BsmtFullBath',2)

test.set_value(test.BsmtFullBath[test.BsmtFullBath == 3].index[0],'BsmtFullBath',2)

print('---')
# filling in some empty values

ind = test.BsmtFullBath[test.BsmtFullBath.isnull()].index.tolist()



for i in ind:

    test.set_value(i,'BsmtFullBath',0)
df.BsmtHalfBath.value_counts()
test.BsmtHalfBath.value_counts()
# 2 to 1

df.set_value(df.BsmtHalfBath[df.BsmtHalfBath == 2].index[0],'BsmtHalfBath',1)

test.set_value(test.BsmtHalfBath[test.BsmtHalfBath == 2].index[0],'BsmtHalfBath',1)

df.set_value(df.BsmtHalfBath[df.BsmtHalfBath == 2].index[0],'BsmtHalfBath',1)

test.set_value(test.BsmtHalfBath[test.BsmtHalfBath == 2].index[0],'BsmtHalfBath',1)

print('---')
# filling in some nulls

ind = test.BsmtHalfBath[test.BsmtHalfBath.isnull()].index.tolist()



for i in ind:

    test.set_value(i,'BsmtHalfBath',0)
df['BsmtBath'] = df['BsmtHalfBath']/2 + df['BsmtFullBath']

test['BsmtBath'] = test['BsmtHalfBath']/2 + test['BsmtFullBath']
df.drop('BsmtHalfBath',axis=1,inplace=True)

df.drop('BsmtFullBath',axis=1,inplace=True)

test.drop('BsmtHalfBath',axis=1,inplace=True)

test.drop('BsmtFullBath',axis=1,inplace=True)
df.FullBath.value_counts()
test.FullBath.value_counts()
for i in test.index:

    if test.FullBath[i] == 4:

        test.set_value(i,'FullBath',3)    
df.HalfBath.value_counts()
test.HalfBath.value_counts()
# correlations before

df.corr()['FullBath']['SalePrice']
df.corr()['HalfBath']['SalePrice']
df['Bath'] = df['HalfBath']/2 + df['FullBath']

test['Bath'] = test['HalfBath']/2 + test['FullBath']
# correlation after

df.corr()['Bath']['SalePrice']
df.BedroomAbvGr.value_counts()
test.BedroomAbvGr.value_counts()
df.set_value(df.BedroomAbvGr[df.BedroomAbvGr == 8].index[0],'BedroomAbvGr',6)

print('---')
df.KitchenAbvGr.value_counts()
test.KitchenAbvGr.value_counts()
df.set_value(df[df.KitchenAbvGr == 0].index[0],'KitchenAbvGr',1)

print('---')
df.KitchenQual.value_counts()
test.KitchenQual.value_counts()
for i in df.index:

    if df.KitchenQual[i] in ['Ex','Gd']:

        df.set_value(i,'KitchenQual',2)

    elif df.KitchenQual[i] == 'TA':

        df.set_value(i,'KitchenQual',1)

    else:

        df.set_value(i,'KitchenQual',0)
for i in test.index:

    if test.KitchenQual[i] in ['Ex','Gd']:

        test.set_value(i,'KitchenQual',2)

    elif test.KitchenQual[i] == 'TA':

        test.set_value(i,'KitchenQual',1)

    else:

        test.set_value(i,'KitchenQual',0)
df.TotRmsAbvGrd.value_counts()
test.TotRmsAbvGrd.value_counts()
hot.append('TotRmsAbvGrd')
df.Functional.value_counts()
test.Functional.value_counts()
for i in df.index:

    if df.Functional[i] == 'Typ':

        df.set_value(i,'Functional',0)

    elif df.Functional[i] in ['Min1','Min2','Mod']:

        df.set_value(i,'Functional',1)

    elif df.Functional[i] in ['Maj1','Maj2','Sal','Sev']:

        df.set_value(i,'Functional',2)
for i in test.index:

    if test.Functional[i] == 'Typ':

        test.set_value(i,'Functional',0)

    elif test.Functional[i] in ['Min1','Min2','Mod']:

        test.set_value(i,'Functional',1)

    elif test.Functional[i] in ['Maj1','Maj2','Sal','Sev']:

        test.set_value(i,'Functional',2)
# empty values

ind = test.Functional[test.Functional.isnull()].index.tolist()



for i in ind:

    test.set_value(i,'Functional',0)
hot.append('Functional')
df.Fireplaces.value_counts()
test.Fireplaces.value_counts()
for i in df.index:

    if df.Fireplaces[i] == 3:

        df.set_value(i,'Fireplaces',2)



for i in test.index:

    if test.Fireplaces[i] in [3,4]:

        test.set_value(i,'Fireplaces',2)
hot.append('Fireplaces')
df.FireplaceQu.value_counts()
test.FireplaceQu.value_counts()
for i in df.index:

    if df.FireplaceQu[i] in ['Ex','Gd']:

        df.set_value(i,'FireplaceQu',3)

    elif df.FireplaceQu[i] == 'TA':

        df.set_value(i,'FireplaceQu',2)

    elif df.FireplaceQu[i] in ['Po','Fa']:

        df.set_value(i,'FireplaceQu',1)

    else:

        df.set_value(i,'FireplaceQu',0)
for i in test.index:

    if test.FireplaceQu[i] in ['Ex','Gd']:

        test.set_value(i,'FireplaceQu',3)

    elif test.FireplaceQu[i] == 'TA':

        test.set_value(i,'FireplaceQu',2)

    elif test.FireplaceQu[i] in ['Po','Fa']:

        test.set_value(i,'FireplaceQu',1)

    else:

        test.set_value(i,'FireplaceQu',0)
df.GarageType.value_counts()
test.GarageType.value_counts()
for i in df.index:

    if df.GarageType[i] in ['Attchd', 'BuiltIn', 'Basment']:

        df.set_value(i,'GarageType',1)

    elif df.GarageType[i] in ['Detchd', 'CarPort', '2Types']:

        df.set_value(i,'GarageType',2)

    else:

        df.set_value(i,'GarageType',0)
for i in test.index:

    if test.GarageType[i] in ['Attchd', 'BuiltIn', 'Basment']:

        test.set_value(i,'GarageType',1)

    elif test.GarageType[i] in ['Detchd', 'CarPort', '2Types']:

        test.set_value(i,'GarageType',2)

    else:

        test.set_value(i,'GarageType',0)
df.GarageYrBlt.describe()
df.GarageYrBlt.hist(bins=5)
test.GarageYrBlt.describe()
# assume they meant 2007 instead of 2207

test.set_value(test.GarageYrBlt[test.GarageYrBlt > 2010].index[0],'GarageYrBlt', 2007)

print('---')
# empty values as 0

df.GarageYrBlt = df.GarageYrBlt.apply(lambda x: np.nan_to_num(x))

test.GarageYrBlt = test.GarageYrBlt.apply(lambda x: np.nan_to_num(x))
# train

for i in df.index:

    if df['GarageYrBlt'][i] <= 1918:

        df.set_value(i,'GarageYrBlt', 0)

    elif df['GarageYrBlt'][i] > 1918 and df['GarageYrBlt'][i] <= 1941:

        df.set_value(i,'GarageYrBlt', 1)

    elif df['GarageYrBlt'][i] > 1941 and df['GarageYrBlt'][i] <= 1964:

        df.set_value(i,'GarageYrBlt', 2)

    elif df['GarageYrBlt'][i] > 1964 and df['GarageYrBlt'][i] <= 1987:

        df.set_value(i,'GarageYrBlt', 3)

    elif df['GarageYrBlt'][i] > 1987:

        df.set_value(i,'GarageYrBlt', 4)
# test

for i in test.index:

    if test['GarageYrBlt'][i] <= 1918:

        test.set_value(i,'GarageYrBlt', 0)

    elif test['GarageYrBlt'][i] > 1918 and test['GarageYrBlt'][i] <= 1941:

        test.set_value(i,'GarageYrBlt', 1)

    elif test['GarageYrBlt'][i] > 1941 and test['GarageYrBlt'][i] <= 1964:

        test.set_value(i,'GarageYrBlt', 2)

    elif test['GarageYrBlt'][i] > 1964 and test['GarageYrBlt'][i] <= 1987:

        test.set_value(i,'GarageYrBlt', 3)

    elif test['GarageYrBlt'][i] > 1987:

        test.set_value(i,'GarageYrBlt', 4)
hot.append('GarageYrBlt')
df.GarageFinish.value_counts()
test.GarageFinish.value_counts()
for i in df.index:

    if df.GarageFinish[i] == 'Unf':

        df.set_value(i,'GarageFinish',0)

    elif df.GarageFinish[i] == 'RFn':

        df.set_value(i,'GarageFinish',1)

    elif df.GarageFinish[i] == 'Fin':

        df.set_value(i,'GarageFinish',2)

    else:

        df.set_value(i,'GarageFinish',3)
for i in test.index:

    if test.GarageFinish[i] == 'Unf':

        test.set_value(i,'GarageFinish',0)

    elif test.GarageFinish[i] == 'RFn':

        test.set_value(i,'GarageFinish',1)

    elif test.GarageFinish[i] == 'Fin':

        test.set_value(i,'GarageFinish',2)

    else:

        test.set_value(i,'GarageFinish',3)
hot.append('GarageFinish')
df.GarageCars.value_counts()
test.GarageCars.value_counts()
# moving from 5 to 4

test.set_value(test.GarageCars[test.GarageCars == 5].index[0],'GarageCars',4)

print('---')
# filling empty column

test.set_value(test.GarageCars[test.GarageCars.isnull()].index[0],'GarageCars',0)

print('---')
hot.append('GarageCars')
df.GarageArea.hist()
test.GarageArea.hist()
test.set_value(test.GarageArea[test.GarageArea.isnull()].index[0],'GarageArea',0)

print('---')
df.GarageQual.value_counts()
test.GarageQual.value_counts()
for i in df.index:

    if df.GarageQual[i] in ['Gd', 'Ex']:

        df.set_value(i,'GarageQual',3)

    elif df.GarageQual[i] == 'TA':

        df.set_value(i,'GarageQual',2)

    elif df.GarageQual[i] in ['Fa', 'Po']:

        df.set_value(i,'GarageQual',1)

    else:

        df.set_value(i,'GarageQual',0)
for i in test.index:

    if test.GarageQual[i] in ['Gd', 'Ex']:

        test.set_value(i,'GarageQual',3)

    elif test.GarageQual[i] == 'TA':

        test.set_value(i,'GarageQual',2)

    elif test.GarageQual[i] in ['Fa', 'Po']:

        test.set_value(i,'GarageQual',1)

    else:

        test.set_value(i,'GarageQual',0)
hot.append('GarageQual')
df.GarageCond.value_counts()
test.GarageCond.value_counts()
# train

for i in df.index:

    if df.GarageCond[i] in ['Gd', 'Ex']:

        df.set_value(i,'GarageCond',3)

    elif df.GarageCond[i] == 'TA':

        df.set_value(i,'GarageCond',2)

    elif df.GarageCond[i] in ['Fa', 'Po']:

        df.set_value(i,'GarageCond',1)

    else:

        df.set_value(i,'GarageCond',0)

        

# test

for i in test.index:

    if test.GarageCond[i] in ['Gd', 'Ex']:

        test.set_value(i,'GarageCond',3)

    elif test.GarageCond[i] == 'TA':

        test.set_value(i,'GarageCond',2)

    elif test.GarageCond[i] in ['Fa', 'Po']:

        test.set_value(i,'GarageCond',1)

    else:

        test.set_value(i,'GarageCond',0)
hot.append('GarageCond')
df.PavedDrive.value_counts()
test.PavedDrive.value_counts()
# train

for i in df.index:

    if df.PavedDrive[i] == 'Y':

        df.set_value(i,'PavedDrive',1)

    else:

        df.set_value(i,'PavedDrive',0)

        

# test

for i in test.index:

    if test.PavedDrive[i] == 'Y':

        test.set_value(i,'PavedDrive',1)

    else:

        test.set_value(i,'PavedDrive',0)
# WoodDeckSF

# train

wood_tr = {'id':[]}

for i in df.index:

    if df.WoodDeckSF[i] > 0:

        wood_tr['id'].append(i)

        

wood_tr['class'] = 5

wood_tr['name'] = 'WoodDeckSF'



# test

wood_ts = {'id':[]}

for i in df.index:

    if df.WoodDeckSF[i] > 0:

        wood_ts['id'].append(i)

        

wood_ts['class'] = 5

wood_ts['name'] = 'WoodDeckSF'
# OpenPorchSF

# train

open_tr = {'id':[]}

for i in df.index:

    if df.OpenPorchSF[i] > 0:

        open_tr['id'].append(i)



open_tr['class'] = 0

open_tr['name'] = 'OpenPorchSF'



# test

open_ts = {'id':[]}

for i in test.index:

    if test.OpenPorchSF[i] > 0:

        open_ts['id'].append(i)



open_ts['class'] = 0

open_ts['name'] = 'OpenPorchSF'
# EnclosedPorch

# train

encl_tr = {'id':[]}

for i in df.index:

    if df.EnclosedPorch[i] > 0:

        encl_tr['id'].append(i)

        

encl_tr['class'] = 1

encl_tr['name'] = 'EnclosedPorch'

        

# test

encl_ts = {'id':[]}

for i in test.index:

    if test.EnclosedPorch[i] > 0:

        encl_ts['id'].append(i)

        

encl_ts['class'] = 1

encl_ts['name'] = 'EnclosedPorch'
# 3SsnPorch

# train

sn3_tr = {'id':[]}

for i in df.index:

    if df['3SsnPorch'][i] > 0:

        sn3_tr['id'].append(i)

        

sn3_tr['class'] = 2

sn3_tr['name'] = '3SsnPorch'

        

# test

sn3_ts = {'id':[]}

for i in test.index:

    if test['3SsnPorch'][i] > 0:

        sn3_ts['id'].append(i)

        

sn3_ts['class'] = 2

sn3_ts['name'] = '3SsnPorch'
# ScreenPorch

# train

scp_tr = {'id':[]}

for i in df.index:

    if df['ScreenPorch'][i] > 0:

        scp_tr['id'].append(i)

        

scp_tr['class'] = 3

scp_tr['name'] = 'ScreenPorch'

        

# test

scp_ts = {'id':[]}

for i in test.index:

    if test['ScreenPorch'][i] > 0:

        scp_ts['id'].append(i)



scp_ts['class'] = 3

scp_ts['name'] = 'ScreenPorch'
df['PorchSF'] = 0

df['PorchType'] = 4
k = [open_tr,encl_tr,sn3_tr,scp_tr,wood_tr]

for i in df.index:

    f = [0] * 5

    v_max = 0

    name = ''

    s = 0

    if i in open_tr['id']:

        f[0] = 1

    if i in encl_tr['id']:

        f[1] = 1

    if i in sn3_tr['id']:

        f[2] = 1

    if i in scp_tr['id']:

        f[3] = 1

    if i in wood_tr['id']:

        f[4] = 1

    if sum(f) > 1:

        for j in range(0,4):

            if f[j] != 0 and df[k[j]['name']][i] > v_max:

                v_max = df[k[j]['name']][i]

                name = k[j]['class']

        for j in range(0,5):      

            s += df[k[j]['name']][i]

        df.set_value(i,'PorchSF',s)

        df.set_value(i,'PorchType',name)

    else:

        for w,z in enumerate(f):

            if f[w] == 1:

                df.set_value(i,'PorchSF',df[k[w]['name']][i])

                df.set_value(i,'PorchType',k[w]['class'])
test['PorchSF'] = 0

test['PorchType'] = 4
k = [open_ts,encl_ts,sn3_ts,scp_ts,wood_ts]

for i in test.index:

    f = [0] * 5

    v_max = 0

    name = ''

    s = 0

    if i in open_ts['id']:

        f[0] = 1

    if i in encl_ts['id']:

        f[1] = 1

    if i in sn3_ts['id']:

        f[2] = 1

    if i in scp_ts['id']:

        f[3] = 1

    if i in wood_ts['id']:

        f[4] = 1

    if sum(f) > 1:

        for j in range(0,4):

            if f[j] != 0 and test[k[j]['name']][i] > v_max:

                v_max = test[k[j]['name']][i]

                name = k[j]['class']

        for j in range(0,5):      

            s += test[k[j]['name']][i]

        test.set_value(i,'PorchSF',s)

        test.set_value(i,'PorchType',name)

    else:

        for w,z in enumerate(f):

            if f[w] == 1:

                test.set_value(i,'PorchSF',test[k[w]['name']][i])

                test.set_value(i,'PorchType',k[w]['class'])
df.corr()['PorchSF']['SalePrice']
df.corr()['PorchType']['SalePrice']
df.OpenPorchSF = df.OpenPorchSF.apply(lambda x: 0 if x == 0 else 1)

df.EnclosedPorch = df.EnclosedPorch.apply(lambda x: 0 if x == 0 else 1)

test.OpenPorchSF = df.OpenPorchSF.apply(lambda x: 0 if x == 0 else 1)

test.EnclosedPorch = df.EnclosedPorch.apply(lambda x: 0 if x == 0 else 1)
# some empty ones I missed

ind = test.OpenPorchSF[test.OpenPorchSF.isnull()].index.tolist()

for i in ind:

    test.set_value(i,'OpenPorchSF',0)
ind = test.EnclosedPorch[test.EnclosedPorch.isnull()].index.tolist()

for i in ind:

    test.set_value(i,'EnclosedPorch',0)
hot.append('PorchType')
df.PoolArea.value_counts()
df.PoolQC.value_counts()
df.drop(['PoolArea','PoolQC'],axis=1,inplace=True)

test.drop(['PoolArea','PoolQC'],axis=1,inplace=True)
df.Fence.value_counts()
test.Fence.value_counts()
df.Fence = df.Fence.apply(lambda x: 1 if x in ['GdPrv','MnPrv','GdWo','MnWw'] else 0)

test.Fence = test.Fence.apply(lambda x: 1 if x in ['GdPrv','MnPrv','GdWo','MnWw'] else 0)
test.Fence.value_counts()
df.MiscFeature.value_counts()
test.MiscFeature.value_counts()
df.drop('MiscFeature',axis=1,inplace=True)

test.drop('MiscFeature',axis=1,inplace=True)
df.MiscVal.hist()
df.drop('MiscVal',axis=1,inplace=True)

test.drop('MiscVal',axis=1,inplace=True)
df.MoSold.value_counts()
df.MoSold.hist()
hot.append('MoSold')
df.YrSold.value_counts()
test.YrSold.value_counts()
encoder = LabelEncoder()
df.YrSold = encoder.fit_transform(df.YrSold)
test.YrSold = encoder.transform(test.YrSold)
df.SaleType.value_counts()
for i in df.index:

    if df.SaleType[i] in ['WD','CWD','VWD']:

        df.set_value(i,'SaleType',0)

    elif df.SaleType[i] in ['Con','ConLw','ConLI','ConLD','Oth']:

        df.set_value(i,'SaleType',1)

    elif df.SaleType[i] == 'New':

        df.set_value(i,'SaleType',2)

    elif df.SaleType[i] == 'COD':

        df.set_value(i,'SaleType',3)
for i in test.index:

    if test.SaleType[i] in ['WD','CWD','VWD']:

        test.set_value(i,'SaleType',0)

    elif test.SaleType[i] in ['Con','ConLw','ConLI','ConLD','Oth']:

        test.set_value(i,'SaleType',1)

    elif test.SaleType[i] == 'New':

        test.set_value(i,'SaleType',2)

    elif test.SaleType[i] == 'COD':

        test.set_value(i,'SaleType',3)
# missed one row

test.set_value(test.SaleType[test.SaleType.isnull()].index[0],'SaleType',0)

print('---')
hot.append('SaleType')
df.SaleCondition.value_counts()
test.SaleCondition.value_counts()
encoder = LabelEncoder()
df.SaleCondition = encoder.fit_transform(df.SaleCondition)
test.SaleCondition = encoder.transform(test.SaleCondition)
df = df.apply(pd.to_numeric)

test = test.apply(pd.to_numeric)
# use values mostly correlated with LotFrontage for X

X = df[~df.LotFrontage.isnull()][['LotArea','BldgType','1stFlrSF','GrLivArea','TotRmsAbvGrd']].values

y = df[~df.LotFrontage.isnull()]['LotFrontage'].values



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)



from sklearn.linear_model import LinearRegression

regr = LinearRegression()

regr.fit(X_train,y_train)

prd = regr.predict(X_test)

print('Roor Mean squared error: ',mean_squared_error(y_test,prd)**0.5)



to_pred = df[df.LotFrontage.isnull()][['LotArea','BldgType','1stFlrSF','GrLivArea','TotRmsAbvGrd']].values

predictions = regr.predict(to_pred)

idx = df[df.LotFrontage.isnull()].index.tolist()



x = 0

for i in idx:

    df.set_value(i,'LotFrontage',round(predictions[x]))

    x += 1
# same for test

X = test[~test.LotFrontage.isnull()][['LotArea','BldgType','1stFlrSF','GrLivArea','TotRmsAbvGrd']].values

y = test[~test.LotFrontage.isnull()]['LotFrontage'].values



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)



from sklearn.linear_model import LinearRegression

regr = LinearRegression()

regr.fit(X_train,y_train)

prd = regr.predict(X_test)

print('Root Mean squared error: ',mean_squared_error(y_test,prd)**0.5)



to_pred = test[test.LotFrontage.isnull()][['LotArea','BldgType','1stFlrSF','GrLivArea','TotRmsAbvGrd']].values

predictions = regr.predict(to_pred)

idx = test[test.LotFrontage.isnull()].index.tolist()



x = 0

for i in idx:

    test.set_value(i,'LotFrontage',round(predictions[x]))

    x += 1
df.drop('Id',axis=1,inplace=True)

test.drop('Id',axis=1,inplace=True)
test_c = test.copy()

df_c = df.copy()
from scipy.stats import skew
skw
print('Train: ',skew(df.LotArea))

print('Test: ',skew(test.LotArea))
(np.log1p(df.LotArea)).hist()

print('New skew: ',skew(np.log1p(df.LotArea)))
(np.log1p(test.LotArea)).hist()

print('New skew: ',skew(np.log1p(test.LotArea)))
df.LotArea = np.log1p(df.LotArea)

test.LotArea = np.log1p(test.LotArea)
print('Train: ',skew(df.MasVnrArea))

print('Test: ',skew(test.MasVnrArea))
(np.log1p(df.MasVnrArea)).hist()

print('New skew: ',skew(np.log1p(df.MasVnrArea)))
(np.log1p(test.MasVnrArea)).hist()

print('New skew: ',skew(np.log1p(test.MasVnrArea)))
df.MasVnrArea = np.log1p(df.MasVnrArea)

test.MasVnrArea = np.log1p(test.MasVnrArea)
print('Train: ',skew(df.BsmtFinSF1))

print('Test: ',skew(test.BsmtFinSF1))
(np.sqrt(df.BsmtFinSF1)).hist()

print('New skew: ',skew(np.sqrt(df.BsmtFinSF1)))
(np.sqrt(test.BsmtFinSF1)).hist()

print('New skew: ',skew(np.sqrt(test.BsmtFinSF1)))
print('Train: ',skew(df.BsmtFinSF2))

print('Test: ',skew(test.BsmtFinSF2))
(np.log1p(df.BsmtFinSF2)).hist()

print('New skew: ',skew(np.log1p(df.BsmtFinSF2)))
(np.log1p(test.BsmtFinSF2)).hist()

print('New skew: ',skew(np.log1p(test.BsmtFinSF2)))
df.BsmtFinSF2 = np.log1p(df.BsmtFinSF2)

test.BmstFinSF2 = np.log1p(test.BsmtFinSF2)
print('Train: ',skew(df.BsmtUnfSF))

print('Test: ',skew(test.BsmtUnfSF))
(np.sqrt(df.BsmtUnfSF)).hist()

print('New skew: ',skew(np.sqrt(df.BsmtUnfSF)))
(np.sqrt(test.BsmtUnfSF)).hist()

print('New skew: ',skew(np.sqrt(test.BsmtUnfSF)))
df.BsmtUnfSF = np.log1p(df.BsmtUnfSF)

test.BsmtUnfSF = np.log1p(test.BsmtUnfSF)
print('Train: ',skew(df['1stFlrSF']))

print('Test: ',skew(test['1stFlrSF']))
(np.sqrt(df['1stFlrSF'])).hist()

print('New skew: ',skew(np.sqrt(df['1stFlrSF'])))
(np.sqrt(test['1stFlrSF'])).hist()

print('New skew: ',skew(np.sqrt(test['1stFlrSF'])))
df['1stFlrSF'] = np.sqrt(df['1stFlrSF'])

test['1stFlrSF'] = np.sqrt(test['1stFlrSF'])
print('Train: ',skew(df.GrLivArea))

print('Test: ',skew(test.GrLivArea))
(np.sqrt(df.GrLivArea)).hist()

print('New skew: ',skew(np.sqrt(df.GrLivArea)))
(np.sqrt(test.GrLivArea)).hist()

print('New skew: ',skew(np.sqrt(test.GrLivArea)))
df.GrLivArea = np.sqrt(df.GrLivArea)

test.GrLivArea = np.sqrt(test.GrLivArea)
add_f = df.corr()['SalePrice'].sort_values(ascending=False).head(11).keys().tolist()[1:]
# for j,i in enumerate(add_f):

#     df[i+'_2'] = df[i] ** 2

#     df[i+'_3'] = df[i] ** 3

#     df[i+'_sqrt'] = df[i] ** 0.5
# for j,i in enumerate(add_f):

#     test[i+'_2'] = test[i] ** 2

#     test[i+'_3'] = test[i] ** 3

#     test[i+'_sqrt'] = test[i] ** 0.5
df_c = df.copy()

df_c.drop('SalePrice',axis=1,inplace=True)
cols = []

for i,j in enumerate(df_c.columns.tolist()):

    cols.append((i,j))
for i,w in enumerate(hot):

    for j,k in cols:

        if w == k:

            hot[i] = (j,w)           
# arranging them in order

gh = hot[0]

for i in range(1,5):

    hot[i-1] = hot[i] 

hot[4] = gh
ar = hot[-3]

for i in range(19,21):

    hot[i-1] = hot[i]

hot[-1] = ar
X = df_c.values
shpr = X.shape[1]
# training set

for i,j in hot:

    shp = X.shape[1] - shpr

    hot_encoder = OneHotEncoder(categorical_features=[i+shp])

    X = hot_encoder.fit_transform(X).toarray()

    X = X[:,1:]
tst = test.copy()
tst = test.values
# test prediction set

shpr = tst.shape[1]

for i,j in hot:

    shp = tst.shape[1] - shpr

    hot_encoder = OneHotEncoder(categorical_features=[i+shp])

    tst = hot_encoder.fit_transform(tst).toarray()

    tst = tst[:,1:]
y = df.SalePrice.values
# splitting the train and test sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# feature scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)

tst = sc_X.transform(tst)
from sklearn.decomposition import PCA

# leave it as None initially to explore the variance first, then change to the choosen number from explained_variance

pca = PCA(n_components=125)



# fitting and transforming the training set and transforming the test set

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)

tst = pca.transform(tst)



# cumulated explained vairance of the principal components

explained_variance = pca.explained_variance_ratio_

explained_variance
sum(explained_variance.tolist()[0:125])
len(explained_variance)
# importing libraries



from sklearn.linear_model import ElasticNet as EN

from sklearn.linear_model import Lasso as LS



# cross validation



algorithms = []



algorithms.append(('XGB', xgb.XGBRegressor()))

algorithms.append(('ElasticNet', EN()))

algorithms.append(('Lasso', LS()))



results = []

names = []

scoring = 'r2'



for name, model in algorithms:

    cv_results = cross_val_score(model, X_train, y_train, cv=10, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = '%s: %f (%f)' %(name, cv_results.mean(), cv_results.std())

    print(msg)

    

fig = plt.figure(figsize=(22,5))

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)
regr = xgb.XGBRegressor()
regr.fit(X_train,y_train)
pred = regr.predict(X_test)
mean_squared_error(y_test,pred)**0.5
pred = regr.predict(tst)
regressor = LS(alpha=0.001, max_iter=50000)
regressor.fit(X_train,y_train)
prediction = regressor.predict(X_test)
mean_squared_error(y_test,prediction)**0.5
predi = regressor.predict(tst)
pp = (pred + predi)/2
# # exporting the final result

# import csv

# with open('output.csv','w') as resultFile:

#      wr = csv.writer(resultFile, dialect='excel')

#      wr.writerow(pp)