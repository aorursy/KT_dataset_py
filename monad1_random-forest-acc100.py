# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

data.head()
data.shape
data = data.loc[:, data.columns.notnull()]
data.shape
print(data.columns[data.isna().any()].tolist())

len(data.columns[data.isna().any()].tolist())
data = data.fillna(data.mean())

print(data.columns[data.isna().any()].tolist())

len(data.columns[data.isna().any()].tolist())
#data.isnull().sum()



# create a list of NaN column

null_val_col = data.columns[data.isna().any()].tolist()



for n in null_val_col:

    print(n , 'missing_value : ', len(data) - data[n].count())  





# count NaN value for single columns

# len(data) - data['Alley'].count()

data = data.drop(['MiscFeature', 'Fence','PoolQC','FireplaceQu','Alley'], axis = 1)

data.head(10)
# create a list of NaN column

null_val_col = data.columns[data.isna().any()].tolist()



for n in null_val_col:

    data = data[data[n].notna()]

    



null_val_col = data.columns[data.isna().any()].tolist()



for n in null_val_col:

    print(n , 'missing_value : ', len(data) - data[n].count())  

#Check remaining missing values if any 

all_data_na = (data.isnull().sum() / len(data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

missing_data.head()
data.shape

data.columns
data.dtypes





data['MSSubClass'] = data['MSSubClass'].apply(str)





#Changing OverallCond into a categorical variable

data['OverallCond'] = data['OverallCond'].astype(str)





#Year and month sold are transformed into categorical features.

data['YrSold'] = data['YrSold'].astype(str)

data['MoSold'] = data['MoSold'].astype(str) 

from sklearn.preprocessing import LabelEncoder

cols = ( 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')

# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(data[c].values)) 

    data[c] = lbl.transform(list(data[c].values))



# shape        

print('Shape all_data: {}'.format(data.shape))
data = pd.get_dummies(data)

print(data.shape)
import seaborn as sns



#histogram

sns.distplot(data['SalePrice']);
print(data.columns)



# drop Id columns cause there are no need of Id for predictions

data = data.drop(['Id'], axis = 1)



data.tail()
X_train = data.drop("SalePrice", axis=1)

Y_train = data["SalePrice"]

X_test  = data.drop("SalePrice", axis=1).copy()


from sklearn.ensemble import RandomForestClassifier



# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)



Y_prediction = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

print(round(acc_random_forest,2,), "%")





importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})

importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head(10)


sns.set(rc={'figure.figsize':(100,20)})

importances.plot.bar()
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)

random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)



acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

print(round(acc_random_forest,2,), "%")

print("oob score:", round(random_forest.oob_score_, 4)*100, "%")

# Fit the model to the training data

random_forest.fit(X_train, Y_train)



# Generate test predictions

preds_test = random_forest.predict(X_test)



# Save predictions in format used for competition scoring

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)