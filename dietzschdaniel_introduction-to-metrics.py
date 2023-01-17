# Data Processing

import numpy as np 

import pandas as pd 





# Data Visualization

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
# Loading the data

df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_test = pd.read_csv("/kaggle/input/titanic/test.csv")





# Handling NaN values

df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())

df_test['Age'] = df_test['Age'].fillna(df_test['Age'].mean())



df_train['Cabin'] = df_train['Cabin'].fillna("Missing")

df_test['Cabin'] = df_test['Cabin'].fillna("Missing")



df_train = df_train.dropna()



df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].mean())



# Cleaning the data

df_train = df_train.drop(columns=['Name'], axis=1)

df_test = df_test.drop(columns=['Name'], axis=1)



sex_mapping = {

    'male': 0,

    'female': 1

}

df_train.loc[:, "Sex"] = df_train['Sex'].map(sex_mapping)

df_test.loc[:, "Sex"] = df_test['Sex'].map(sex_mapping)



df_train = df_train.drop(columns=['Ticket'], axis=1)

df_test = df_test.drop(columns=['Ticket'], axis=1)



df_train = df_train.drop(columns=['Cabin'], axis=1)

df_test = df_test.drop(columns=['Cabin'], axis=1)



df_train = pd.get_dummies(df_train, prefix_sep="__",

                              columns=['Embarked'])

df_test = pd.get_dummies(df_test, prefix_sep="__",

                              columns=['Embarked'])
df_train.head()
df_train.shape
df_test.head()
df_test.shape
# Everything except the target variable

X = df_train.drop("Survived", axis=1)



# Target variable

y = df_train['Survived'].values
# Random seed for reproducibility

np.random.seed(42)



# Splitting the data into train & test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# Setting up RandomForestClassifier()

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc.fit(X_train, y_train)



# Predicting values

y_pred = rfc.predict(X_test)
# Importing confusion_matrix

from sklearn.metrics import confusion_matrix



# Computing the confusion_matrix

confusion_matrix(y_test, y_pred)
# Importing plot_confusion_matrix

from sklearn.metrics import plot_confusion_matrix



# Plot plot_confusion_matrix

plot_confusion_matrix(rfc, X_test, y_test);
# Importing accuracy_score

from sklearn.metrics import accuracy_score



# Computing the accuracy_score

accuracy_score(y_test, y_pred)
# Importing precision_score

from sklearn.metrics import precision_score



# Computing the precision_score

precision_score(y_test, y_pred)
# Importing recall_score

from sklearn.metrics import recall_score



# Computing the recall_score

recall_score(y_test, y_pred)
# Importing f1_score

from sklearn.metrics import f1_score



# Computing the f1_score

f1_score(y_test, y_pred)
# Importing roc_curve

from sklearn.metrics import roc_curve



# Plot ROC curve

fpr, tpr, _ = roc_curve(y_test, y_pred)

plt.plot(fpr, tpr)



# Importing roc_auc_score

from sklearn.metrics import roc_auc_score



# Computing the roc_auc_score

roc_auc_score(y_test, y_pred)
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")



train = train.fillna(train.median())



train['Alley'] = train['Alley'].fillna('None')

train = train.drop(['Utilities'], axis=1)

train['MasVnrType'] = train['MasVnrType'].fillna('Missing')

train['BsmtQual'] = train['BsmtQual'].fillna('None')

train['BsmtCond'] = train['BsmtCond'].fillna('None')

train['BsmtExposure'] = train['BsmtExposure'].fillna('None')

train['BsmtFinType1'] = train['BsmtFinType1'].fillna('None')

train['BsmtFinType2'] = train['BsmtFinType2'].fillna('None')

train['Electrical'] = train['Electrical'].fillna('None')

train['FireplaceQu'] = train['FireplaceQu'].fillna('None')

train['GarageType'] = train['GarageType'].fillna('None')

train['GarageFinish'] = train['GarageFinish'].fillna('None')

train['GarageQual'] = train['GarageQual'].fillna('None')

train['GarageCond'] = train['GarageCond'].fillna('None')

train = train.drop(['PoolQC'], axis=1)

train['Fence'] = train['Fence'].fillna('None')

train['MiscFeature'] = train['MiscFeature'].fillna('None')

train['SaleType'] = train['SaleType'].fillna('None')



from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')

# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(train[c].values)) 

    train[c] = lbl.transform(list(train[c].values))

    

train = pd.get_dummies(train)







# Everything except the target variable

X = train.drop("SalePrice", axis=1)



# Target variable

y = train['SalePrice'].values





# Random seed for reproducibility

np.random.seed(42)



# Splitting the data into train & test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)





# Setting up GradientBoostingRegressor()

from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor()

model.fit(X_train, y_train)



# Predicting values

y_pred = model.predict(X_test)
# Importing mean_absolute_error

from sklearn.metrics import mean_absolute_error



# Computing the mean_absolute_error

mean_absolute_error(y_test, y_pred)
# Importing mean_squared_error

from sklearn.metrics import mean_squared_error



# Computing the mean_squared_error

mean_squared_error(y_test, y_pred)
# Importing mean_squared_error and numpy

from sklearn.metrics import mean_squared_error

import numpy as np



# Computing the root of mean_squared_error

mean_squared_error(y_test, y_pred)

np.sqrt(mean_squared_error(y_test, y_pred))
# Importing mean_squared_log_error and numpy

from sklearn.metrics import mean_squared_log_error

import numpy as np



# Computing the root of mean_squared_error

np.sqrt(mean_squared_log_error(y_test, y_pred))
# Importing r2_score

from sklearn.metrics import r2_score



# Computing the root of r2_score

r2_score(y_test, y_pred)