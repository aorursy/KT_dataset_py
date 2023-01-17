import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import re



pd.options.display.max_columns=500



import warnings

warnings.filterwarnings('ignore')
df1 = pd.read_csv('../input/data.csv', index_col=0) # wehave first unnamed col and we remove it

print(df1.shape)

df1.head()
# create a second dataframe which we will use for further computations and avoid reading data from drive again

fifa = df1.copy()
fifa.set_index('ID', inplace=True)

# fifa.describe(include='all')
colnew = []



for col in fifa.columns:

    colnew.append(col.replace(' ', ''))

    

fifa.columns = colnew
# removing the columns that do not provide any additional information

fifa.drop(['Name','Photo','Flag','LoanedFrom','ClubLogo','JerseyNumber','ReleaseClause', 

           'Joined','ContractValidUntil','Height','Weight','Special','RealFace'], inplace=True, axis=1)
# split 'Work Rate' column into two columns which can be used in computation



df_workrate = fifa['WorkRate'].str.split('/ ', expand=True)

fifa['WorkrateAttack'], fifa['WorkrateDefense'] = df_workrate[0], df_workrate[1]

fifa.drop('WorkRate', axis=1, inplace=True)
# split the player position attributes at '+' and keep the existing attribute value for 'LS' to 'RB' columns

for col_name in fifa.loc[:,'LS':'RB'].columns:

    fifa[col_name] = fifa[col_name].str.split('+', expand=True)[0]
# combine the position attributes as per pitch area

fifa['Forward'] = fifa.loc[:,'LS':'RW'].astype('float64').mean(axis=1)

fifa['Midfield'] = fifa.loc[:,'LAM':'RM'].astype('float64').mean(axis=1)

fifa['Defense'] = fifa.loc[:,'LWB':'RB'].astype('float64').mean(axis=1)

fifa['GoalKeeper'] = fifa.loc[:,'GKDiving':'GKReflexes'].astype('float64').mean(axis=1)



# drop the columns that are replaced above

fifa.drop(fifa.loc[:,'LS':'RB'].columns.tolist(), axis=1, inplace=True)

fifa.drop(fifa.loc[:,'GKDiving':'GKReflexes'].columns.tolist(), axis=1, inplace=True)
# drop the rows that have 90% missing values

fifa.dropna(thresh=12, axis=0, inplace=True)
fwd = ['Crossing', 'Finishing', 'ShortPassing', 'Dribbling', 'Curve', 'FKAccuracy', 'Acceleration', 

       'SprintSpeed', 'ShotPower', 'BallControl', 'Penalties']

mid = ['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 

       'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 

       'Balance','ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions', 

       'Positioning', 'Vision', 'Penalties']

des = ['Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'Jumping', 'Stamina', 'Strength', 

       'LongShots', 'HeadingAccuracy']



x = fifa[fwd][fifa['Forward'].isna()].mean(axis=1).astype('int64')

fifa.Forward[fifa['Forward'].isna()] = x



x = fifa[mid][fifa['Midfield'].isna()].mean(axis=1).astype('int64')

fifa.Midfield[fifa['Midfield'].isna()] = x



x = fifa[des][fifa['Defense'].isna()].mean(axis=1).astype('int64')

fifa.Defense[fifa['Defense'].isna()] = x
# imputing data in Position column based on Player attributes that can help us identify which position is suitable for the player 



from sklearn.tree import DecisionTreeClassifier



col = fifa.loc[:,'Crossing':'SlidingTackle'].columns

# filter data for model building

X = fifa[col][fifa['Position'].notna()]

y = fifa.Position[fifa['Position'].notna()]



# create test data with unknown fuelType fields

xt = fifa[col][fifa['Position'].isna()]

        

# build a decision tree model

dtree = DecisionTreeClassifier().fit(X,y)



# predict on test data

pos_pred = dtree.predict(xt)



# fill the missing values with the predicted values

fifa.Position[fifa['Position'].isna()] = pos_pred

# fifa.Position.isna().sum()
# players with no club details can be considered to be free agent.

fifa['Club'][fifa['Club'].isna()] = 'Free Agent'
# Correction in Value and Wage column.



ind = fifa.index

for i in ind:

    if ('M' in fifa.loc[i,'Value']) and ('.' in fifa.loc[i,'Value']):

        fifa.loc[i,'Value'] = fifa.loc[i,'Value'].replace('€','').replace('.','').replace('M','00000')

    else:

        fifa.loc[i,'Value'] = fifa.loc[i,'Value'].replace('€','').replace('K','000').replace('M','000000')

    fifa.loc[i,'Wage'] = fifa.loc[i,'Wage'].replace('€','').replace('K','000')

    

fifa.Value = fifa.Value.astype('int64')

fifa.Wage = fifa.Wage.astype('int64')
fifa.loc[:,'Crossing':'SlidingTackle'] = fifa.loc[:,'Crossing':'SlidingTackle'].astype('int64')

fifa.loc[:,'Forward':'GoalKeeper'] = fifa.loc[:,'Forward':'GoalKeeper'].astype('int64')
fifa.BodyType[fifa.BodyType=='Messi'] = 'Lean'

fifa.BodyType[fifa.BodyType=='C. Ronaldo'] = 'Normal'

fifa.BodyType[fifa.BodyType=='Neymar'] = 'Lean'

fifa.BodyType[fifa.BodyType=='Courtois'] = 'Normal'

fifa.BodyType[fifa.BodyType=='PLAYER_BODY_TYPE_25'] = 'Normal'

fifa.BodyType[fifa.BodyType=='Shaqiri'] = 'Normal'

fifa.BodyType[fifa.BodyType=='Akinfenwa'] = 'Lean'
fifa.head()
fifa.shape
corr_2 = fifa.corr(method='pearson')

cols = corr_2.nlargest(30, 'Value')['Value'].index

cm = np.corrcoef(fifa[cols].values.T)

sns.set(font_scale=1.25)

plt.subplots(figsize=(20,15))

# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, 

#                  yticklabels=cols.values, xticklabels=cols.values)



sns.heatmap(cm, fmt='.2f', cmap=['y','y','b','g','r'], annot=True, square=True, annot_kws={'size':10},

           yticklabels=cols.values, xticklabels=cols.values)

plt.show()
k=20

corr_1 = fifa.corr()

corr_1.nlargest(k,'Value')['Value']
col = fifa.loc[:,'Crossing':'SlidingTackle']

fifa.skew()
a = fifa.Forward

# a = np.power(fifa.Forward,3)
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, PolynomialFeatures

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

from statsmodels.tools.tools import add_constant

from sklearn.model_selection import train_test_split, cross_val_score

from scipy.stats import chi2_contingency, chi2
X = add_constant(fifa)

catcol = ['Nationality','BodyType','WorkrateAttack', 'WorkrateDefense','PreferredFoot','WeakFoot','Club','Position']

X.drop(['SlidingTackle','BallControl','StandingTackle','Dribbling','Finishing','ShortPassing',

       'LongShots','SprintSpeed','Positioning','Curve','Interceptions','Volleys','Marking'], axis=1, inplace=True)

X.drop(catcol, axis=1, inplace=True)

# X = pd.get_dummies(X)

pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
col = fifa.loc[:,'Crossing':'SlidingTackle']

X = fifa.drop(col, axis=1)

X.drop(['Value'], axis=1, inplace=True)

y = pd.DataFrame(fifa.Value)



mms = MinMaxScaler()

y = mms.fit_transform(y)

le = LabelEncoder()

X.Nationality = le.fit_transform(X.Nationality).astype('str')

X.Club = le.fit_transform(X.Club).astype('str')

X.PreferredFoot = le.fit_transform(X.PreferredFoot).astype('str')

X.Position = le.fit_transform(X.Position).astype('str')

X.BodyType = le.fit_transform(X.BodyType).astype('str')

X.WorkrateAttack = le.fit_transform(X.WorkrateAttack).astype('str')

X.WorkrateDefense = le.fit_transform(X.WorkrateDefense).astype('str')



X = mms.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=12)



lin_mod_1 = LinearRegression()

lin_mod_1.fit(X_train, y_train)



y_pred = lin_mod_1.predict(X_test)



# y_pred = mms.inverse_transform(y_pred)



print('Train r2_score:',lin_mod_1.score(X_train, y_train))

print('Test r2_score:',lin_mod_1.score(X_test, y_test))



print(mean_squared_error(y_true=y_test, y_pred=y_pred))
# X = fifa.drop(['Value'], axis=1)



col = fifa.loc[:,'Crossing':'SlidingTackle']

X = fifa.drop(col, axis=1)

X.drop(['Value'], axis=1, inplace=True)

y = pd.DataFrame(fifa.Value)



y = MinMaxScaler().fit_transform(y)



le = LabelEncoder()

X.Nationality = le.fit_transform(X.Nationality).astype('str')

X.Club = le.fit_transform(X.Club).astype('str')

X.PreferredFoot = le.fit_transform(X.PreferredFoot).astype('str')

X.Position = le.fit_transform(X.Position).astype('str')

X.BodyType = le.fit_transform(X.BodyType).astype('str')

X.WorkrateAttack = le.fit_transform(X.WorkrateAttack).astype('str')

X.WorkrateDefense = le.fit_transform(X.WorkrateDefense).astype('str')



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=12)



ridge = Ridge(alpha=100)

ridge.fit(X_train, y_train)



y_pred = ridge.predict(X_test)



print('Train r2_score:',ridge.score(X_train, y_train))

print('Test r2_score:',ridge.score(X_test, y_test))



print(mean_squared_error(y_true=y_test, y_pred=y_pred))
X = fifa.drop(['Value'], axis=1)

y = pd.DataFrame(fifa.Value)



y = MinMaxScaler().fit_transform(y)



le = LabelEncoder()

X.Nationality = le.fit_transform(X.Nationality).astype('str')

X.Club = le.fit_transform(X.Club).astype('str')

X.PreferredFoot = le.fit_transform(X.PreferredFoot).astype('str')

X.Position = le.fit_transform(X.Position).astype('str')

X.BodyType = le.fit_transform(X.BodyType).astype('str')

X.WorkrateAttack = le.fit_transform(X.WorkrateAttack).astype('str')

X.WorkrateDefense = le.fit_transform(X.WorkrateDefense).astype('str')



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=12)



lasso = Lasso(alpha=0.01)

lasso.fit(X_train, y_train)



y_pred = lasso.predict(X_test)



print('Train r2_score:',lasso.score(X_train, y_train))

print('Test r2_score:',lasso.score(X_test, y_test))



print(mean_squared_error(y_true=y_test, y_pred=y_pred))
col = fifa.loc[:,'Crossing':'SlidingTackle']

X = fifa.drop(col, axis=1)

X.drop(['Value'], axis=1, inplace=True)

y = pd.DataFrame(fifa.Value)



mms = MinMaxScaler()

y = mms.fit_transform(y)

le = LabelEncoder()

X.Nationality = le.fit_transform(X.Nationality).astype('str')

X.Club = le.fit_transform(X.Club).astype('str')

X.PreferredFoot = le.fit_transform(X.PreferredFoot).astype('str')

X.Position = le.fit_transform(X.Position).astype('str')

X.BodyType = le.fit_transform(X.BodyType).astype('str')

X.WorkrateAttack = le.fit_transform(X.WorkrateAttack).astype('str')

X.WorkrateDefense = le.fit_transform(X.WorkrateDefense).astype('str')



X = mms.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=12)



linreg = LinearRegression()



mse_all = cross_val_score(X=X_train, y=y_train, cv=5, estimator=linreg)



linreg.fit(X_train, y_train)

y_pred = linreg.predict(X_test)



# y_pred = mms.inverse_transform(y_pred)



print('Train r2_score:',lin_mod_1.score(X_train, y_train))

print('Test r2_score:',lin_mod_1.score(X_test, y_test))



print(mean_squared_error(y_true=y_test, y_pred=y_pred))

print('crossval mse: ', mse_all.mean())
X = add_constant(fifa.loc[:,['Overall','Forward','GoalKeeper','Defense','Potential','Wage',

                              'Midfield','InternationalReputation','Jumping','Strength']])

X.drop(['Forward'], axis=1, inplace=True)

# X = pd.get_dummies(X)

print(pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns))



corr_X = X.corr(method='pearson')

cols = X.columns

sns.set(font_scale=1.25)

plt.subplots(figsize=(8,6))



sns.heatmap(corr_X, fmt='.2f', cmap=['y','y','b','g','r'], annot=True, square=True, annot_kws={'size':10},

           yticklabels=cols.values, xticklabels=cols.values)

plt.show()
table1 = pd.crosstab(fifa.InternationalReputation, fifa.WorkrateDefense)

stat, p, dof, exp = chi2_contingency(table1)

p
X = X.drop('const', axis=1)

y = fifa.Value



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)



linreg = LinearRegression().fit(X_train, y_train)

y_pred = linreg.predict(X_test)

x_pred = linreg.predict(X_train)

print(linreg.score(X_train, y_train))

print(linreg.score(X_test, y_test))

print('mse on train:',mean_squared_error(y_true=(y_train), y_pred=(x_pred)))

print('mse on test_:',mean_squared_error(y_true=(y_test), y_pred=(y_pred)))