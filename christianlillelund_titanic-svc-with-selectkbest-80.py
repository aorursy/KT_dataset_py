import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



pd.reset_option('^display.', silent=True)



X_train = pd.read_csv("/kaggle/input/titanic/train.csv")

X_test = pd.read_csv("/kaggle/input/titanic/test.csv")



y_train = X_train.Survived

num_train = len(X_train)

X_train.drop(['Survived'], axis=1, inplace=True)



df = pd.concat([X_train, X_test], ignore_index=True)



df.head()
print("Total training samples:", len(X_train), "\n")

print("Partial data\n", X_train.iloc[0:4, 0:6], "\n")

print("Samples per sex\n", X_train.groupby('Sex')['Sex'].count())
X_train.describe()
# Print a survivor as a sample

sample_index = 25

print(X_train.iloc[sample_index])
# Show the column types we are dealing with



df.dtypes.value_counts()

categorical_columns = df.select_dtypes('object').columns

print(len(df.columns)-len(df.select_dtypes('object').columns),'numerical columns:')

print([i for i in list(df.columns) if i not in list(df.select_dtypes('object').columns)], '\n')

print(len(df.select_dtypes('object').columns),'categorical columns:')

print(list(df.select_dtypes('object').columns))
pd.set_option('mode.chained_assignment', None)



# Delete unused variable Ticket

df = df.drop(['Ticket'],axis=1)



# Use mean variable for missing Embarked

df.Embarked[df.Embarked.isnull()] = 'S'



# Remove integer suffix values for Cabin

df.Cabin[~df.Cabin.isnull()] = df.Cabin[~df.Cabin.isnull()].map(lambda x: x[0])



# Encode cabins

df.Cabin[df.Cabin=='A'] = 1

df.Cabin[df.Cabin=='B'] = 2

df.Cabin[df.Cabin=='C'] = 3

df.Cabin[df.Cabin=='D'] = 3

df.Cabin[df.Cabin=='E'] = 3

df.Cabin[df.Cabin=='F'] = 4

df.Cabin[df.Cabin=='T'] = 5

df.Cabin[df.Cabin=='G'] = 6



cabins = df.groupby(['Pclass']).Cabin

f = lambda x: x.fillna(round(x.median()))

df.Cabin = cabins.transform(f)



# Encode the titles of the passengers, some are VIP

# Thanks https://www.kaggle.com/rushikeshdudhat/accuracy-80-using-xgboost-titanic-ml-challenge

vip_names = ['Mlle', 'Master', 'Dr', 'Rev', 'Col', 'Major', 'Dona', 'Capt', 'Lady', 'Jonkheer', 'Countess', 'Don', 'Sir']

df.Name = df.Name.str.extract('([A-Za-z]+)\.',expand = False)

df.Name = df.Name.replace('Mlle','Miss') 

df.Name = df.Name.replace('Ms','Mrs') 

df.Name = df.Name.replace('Mme','Mrs')

df.Name = df.Name.replace(vip_names, 'VIP')



# Encode the group that passengers travel in

#d.sibsp = sibling-spouse, parch=parent

df = df.assign(Member=0)

df.Member[(df.SibSp == 0) & (df.Parch == 0)] = 1

df.Member[(df.SibSp > 0) & (df.Parch == 0)] = 2

df.Member[((df.SibSp == 2) & (df.Parch > 0)) | (df.SibSp > 2)] = 3

df.Member[((df.SibSp > 0) & (df.Parch > 1)) | ((df.SibSp == 0) & (df .Parch > 1))] = 4

df.Member[((df.SibSp < 2) & (df.Parch == 1))] = 5



# Fill missing values for age with median

ages = df.groupby(['Sex','Member']).Age

f = lambda x: x.fillna(x.median())

df.Age = ages.transform(f)



# Fill missing values for fare with median

fares = df.groupby(['Pclass','Embarked']).Fare

f = lambda x: x.fillna(x.median())

df.Fare = fares.transform(f)



# Make a rank of the passengers's fares

df['Farerank'] = df.Fare.rank() / len(df.Fare)



# Combine the family features

df['Pvar'] = df.Parch+1 * df.SibSp+1

df = df.drop(['Parch', 'SibSp'], axis=1)



# Capitalize the Sex

df.Sex = df.Sex.str.capitalize()
# Check for remaining null values

print(df.isnull().values.any())
# Split age into groups and separate by survival rate

# Thanks to https://www.kaggle.com/sid2412/a-simple-and-effective-approach-to-ml

df_surr = pd.concat([df, y_train], axis=1)

df_surr['AgeGroup'] = pd.cut(df_surr['Age'],5)

df_surr[['AgeGroup', 'Survived']].groupby('AgeGroup', as_index=False).mean().sort_values('Survived', ascending=False)
# Encode the age group of passengers based on above tableau

df.Age[df.Age <= 16] = 4

df.Age[(df.Age > 32) & (df.Age <= 48)] = 3

df.Age[(df.Age > 48) & (df.Age <= 64)] = 2

df.Age[(df.Age > 16) & (df.Age <= 32)] = 1

df.Age[df.Age > 64] = 0
from sklearn.preprocessing import OneHotEncoder



def encode_df(df, object_cols):

    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

    df_enc = pd.DataFrame(ohe.fit_transform(df[object_cols]))

    df_enc.columns = ohe.get_feature_names(object_cols)

    df_enc.index = df.index

    return df_enc



# Use OH encoder to encode cat cols

object_cols = ['Name', 'Embarked']

df_enc = encode_df(df, object_cols)

num_df = df.drop(object_cols, axis=1)

df = pd.concat([num_df, df_enc], axis=1)
# Split the df into train and test set



X_train = df.iloc[:num_train,:]

X_test = df.iloc[num_train:,:]
X_train.head()
# Split training set by male/female



Fmask = (X_train.Sex=='Female')

FX_train = X_train.loc[Fmask]

MX_train = X_train.loc[~Fmask]

FX_train.reset_index(inplace=True)

MX_train.reset_index(inplace=True)

FY_train = y_train.loc[Fmask]

MY_train = y_train.loc[~Fmask]

FX_train = FX_train.drop(['Sex'], axis=1)

MX_train = MX_train.drop(['Sex'], axis=1)



# Split test set by male/female



Fmask = (X_test.Sex=='Female')

FX_test = X_test.loc[Fmask]

MX_test = X_test.loc[~Fmask]

FX_test.reset_index(inplace=True)

MX_test.reset_index(inplace=True)

FX_test = FX_test.drop(['Sex'], axis=1)

MX_test = MX_test.drop(['Sex'], axis=1)
print(f'Females that survived/died: \n{FY_train.value_counts()}\n')

print(f'Males that died/survived: \n{MY_train.value_counts()}')
# Scale train and test data separately

from sklearn.preprocessing import StandardScaler



Fsc = StandardScaler()

FX_train_sc = Fsc.fit_transform(FX_train)

FX_test_sc = Fsc.transform(FX_test)



Msc = StandardScaler()

MX_train_sc = Msc.fit_transform(MX_train)

MX_test_sc = Msc.transform(MX_test)
# Feature selection and find best K features

from sklearn.feature_selection import SelectKBest, f_regression

from sklearn import feature_selection

import warnings

warnings.filterwarnings('ignore')



k=9 

F_selector = feature_selection.SelectKBest(feature_selection.f_regression, k)

FX_train_kb = F_selector.fit_transform(FX_train_sc, FY_train)

FX_test_kb = F_selector.transform(FX_test_sc)



M_selector = feature_selection.SelectKBest(feature_selection.f_regression, k)

MX_train_kb = M_selector.fit_transform(MX_train_sc, MY_train)

MX_test_kb = M_selector.transform(MX_test_sc)



f_feature_names = [df.columns[i] for i in F_selector.get_support(indices=True)]

m_feature_names = [df.columns[i] for i in M_selector.get_support(indices=True)]



print(f"Best {k} features for female: {f_feature_names}")

print(f"Best {k} features for male: {m_feature_names}")



F_indices = np.argsort(F_selector.scores_)[::-1]

M_indices = np.argsort(M_selector.scores_)[::-1]



fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(10,6))

axs[0].bar(f_feature_names, F_selector.scores_[F_indices[range(k)]], color='r', align='center')

axs[0].set_title('Best features for female')

axs[0].set_ylabel('Scores')

fig.suptitle(f'Best features by SelectKBest for K={k}', fontsize=16)



axs[1].bar(m_feature_names, M_selector.scores_[M_indices[range(k)]], color='b', align='center')

axs[1].set_title('Best features for male')

axs[1].set_ylabel('Scores')



plt.show()
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV



param_grid = {'C': [2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3], 

              'kernel': ['rbf'],

              'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}



sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)



F_model = SVC()

F_grs = GridSearchCV(F_model, param_grid=param_grid, cv=sss, n_jobs=5)

F_grs.fit(np.array(FX_train_kb), np.array(FY_train))

F_predictions = F_grs.predict(FX_test_kb).tolist()



M_model = SVC()

M_grs = GridSearchCV(M_model, param_grid=param_grid, cv=sss, n_jobs=5)

M_grs.fit(np.array(MX_train_kb), np.array(MY_train))

M_predictions = M_grs.predict(MX_test_kb).tolist()



print("Best parameters for F classifier: " + str(F_grs.best_params_))

F_gpd = pd.DataFrame(F_grs.cv_results_)

print("Estimated accuracy of this model for unseen data: {0:1.4f}"

      .format(F_gpd['mean_test_score'][F_grs.best_index_]))

print()

print("Best parameters for M classifier: " + str(M_grs.best_params_))

M_gpd = pd.DataFrame(M_grs.cv_results_)

print("Estimated accuracy of this model for unseen data: {0:1.4f}"

      .format(M_gpd['mean_test_score'][M_grs.best_index_]))

num_preds = len(F_predictions) + len(M_predictions)

print()

print(f"Total number of predictions: {num_preds}")
# Collect female and male predictions and save as CSV



predtot = pd.Series([0]*418)



Finds = [x-891 for x in Fmask[Fmask==True].index.tolist()]

Minds = [x-891 for x in Fmask[Fmask==False].index.tolist()]



predtot[Finds] = F_predictions

predtot[Minds] = M_predictions



pid = pd.Series(range(892,1310))

predfinal = pd.DataFrame({'PassengerID': pid, 'Survived': predtot})



# Save to CSV

predfinal.to_csv('submission.csv', index=False)