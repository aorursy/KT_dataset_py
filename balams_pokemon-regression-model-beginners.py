#Import Basic Libraries
import numpy as np 
import pandas as pd 

#Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

#Scaling Data
from sklearn.preprocessing import MinMaxScaler

#Preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, KFold

#Model
from sklearn.linear_model import LinearRegression, ElasticNet,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

import os
print(os.listdir("../input"))
pokemon = pd.read_csv('../input/pokemon.csv')
pokemon.head(5)
#Here First Column name is #. 
pokemon = pokemon.rename(columns = {'#':"ID"})
pokemon.head(5)
#Read Combat
combat = pd.read_csv('../input/combats.csv')
combat.head(5)
#Get Shape of two dataframes
print('Pokemon shape is ' + str(pokemon.shape))
print('Combat shape is ' + str(combat.shape))
#Get Info
print('Pokemon info :')
pokemon.info()
print('\nCombat info :')
combat.info()
#Get Missing value
pokemon.isnull().sum()

combat.isnull().sum()
#Total Number Match of each pockemon
FirstCombat = combat.First_pokemon.value_counts().reset_index(name = 'FirstCombat')
SecondCombat = combat.Second_pokemon.value_counts().reset_index(name = 'SecondCombat')
TotalCombat = pd.merge(FirstCombat, SecondCombat, how = 'left', on = 'index')
TotalCombat['TotalMatch'] = TotalCombat['FirstCombat']+TotalCombat['SecondCombat']

TotalCombat.sort_values('index').head()
#Match winning details
FirstWin = combat['First_pokemon'][combat['First_pokemon'] == combat['Winner']].value_counts().reset_index(name = 'FirstWin')
SecondWin = combat['Second_pokemon'][combat['Second_pokemon'] == combat['Winner']].value_counts().reset_index(name = 'SecondWin')
TotalWin = pd.merge(FirstWin, SecondWin, how  = 'left', on = 'index')
TotalWin['TotalWin'] = TotalWin['FirstWin']+ TotalWin['SecondWin']
TotalWin.head(5)
#Here we have 3 data frame. Let's combine all
result = pd.merge(pokemon, TotalCombat, how = 'left', left_on= 'ID', right_on = 'index')
result = pd.merge(result, TotalWin, how = 'left', on = 'index')
result = result.drop(['index'], axis = 1)
result.head(10)
#Winning Percentage
pd.set_option('precision', 0)
result['WinningPercentage'] = (result.TotalWin / result.TotalMatch) * 100
result.head(5)
#Some Pokemon don't have Type2 char. So we can replace it with null char
result['Type 2'].fillna('Not Applicable', inplace = True)
result.head(10)
categ = ['Type 1','Type 2','Generation','Legendary']
conti = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']
#Univarient Analysis
plt.figure(figsize= (7,40))
i = 0
for cat in categ:
    plt.subplot(8,2,i+1)
    sns.countplot(x = cat, data = result);
    plt.xticks(rotation = 90)
    i+=1
for cont in conti:
    plt.subplot(8,2,i+1)
    sns.distplot(result[cont])
    i+=1
plt.show()
#Now Visulaize how char related with WinningPercentage 
plt.figure(figsize = (8,30))
i =0
for cat in categ:
    plt.subplot(8,2,i+1)    
    sns.barplot(x = cat, y = 'WinningPercentage', data = result);
    plt.tight_layout()
    plt.xticks(rotation = 90)
    i+=1

for cont in conti:
    plt.subplot(8,2,i+1)
    sns.scatterplot(x = 'WinningPercentage', y = cont, data = result)
    i+=1
plt.show()


result.info()
#drop na values in our dataframe
result = result.dropna()
result.info()
result.loc[result['Type 2'] != 'Not Applicable', 'Char'] = 'Both_Char'
result.loc[result['Type 2'] == 'Not Applicable', 'Char'] = 'Only_One_Char'
result.head(5)
pd.set_option('display.float_format', '{:.2f}'.format)
Scaleing_result = result

from sklearn.preprocessing import StandardScaler

col_name = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','Generation','FirstWin','SecondWin','TotalWin']
scale = StandardScaler()
Scaleing_result[col_name] = scale.fit_transform(Scaleing_result[col_name])
Scaleing_result.head(5)
#Let's drop ID, Name Column
Encoding_result = Scaleing_result.drop(['ID','Name','FirstCombat','SecondCombat','TotalMatch'],axis =1)
Encoding_result['Legendary'] = Encoding_result['Legendary'].astype(str)
Encoding_result = pd.get_dummies(Encoding_result, drop_first = True)
Encoding_result.head(5)
#Correlation Matrix
plt.figure(figsize = (5,5))
sns.heatmap(Encoding_result.corr(), cmap = 'Greens')
plt.show()
#Split Dependent and Target Variable
WinningPercentage = Encoding_result['WinningPercentage']
Encoding_result.drop(['WinningPercentage'], axis =1, inplace = True)
#Split Dataset
x_train, x_test, y_train, y_test = train_test_split(Encoding_result,WinningPercentage, test_size = 0.2, random_state = 10)
#Let's Create Model
models = []
models.append(('LR',LinearRegression()))
models.append(('EN', ElasticNet()))
models.append(('Lasso', Lasso()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('GB', GradientBoostingRegressor()))
models.append(('Ada', AdaBoostRegressor()))
model_results = []
names = []
for name, model in models:
    kfold = KFold(n_splits = 10, random_state = 10)
    cv_result = cross_val_score(model, x_train, y_train, cv = kfold, scoring = 'neg_mean_squared_error')
    model_results.append(cv_result)
    names.append(name)
    msg = '%s %f (%f)' % (name, cv_result.mean(), cv_result.std())
    print(msg)
#Visualize our result
plt.figure(figsize = (5,5))
sns.boxplot(x = names, y = model_results)
GBM = GradientBoostingRegressor()
GBM.fit(x_train, y_train)
pred = GBM.predict(x_test)
plt.figure(figsize = (7,7))
sns.regplot(y_test, pred)
plt.show()
plt.figure(figsize = (18,3))
sns.lineplot(x=y_test.index.values, y=y_test, color = 'purple')
sns.lineplot(x=y_test.index.values, y=pred, color = 'orange')
plt.show()