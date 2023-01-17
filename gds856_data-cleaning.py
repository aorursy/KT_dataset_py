# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# import necessary modules

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualisation

import matplotlib.pyplot as plt # plot

import os

import re



pd.options.display.max_columns=500



import warnings

warnings.filterwarnings('ignore')



from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, PolynomialFeatures, StandardScaler

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

from statsmodels.tools.tools import add_constant

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.decomposition import PCA

#Read the file

fifa=pd.read_csv(r'../input/data.csv',index_col=0)

pd.set_option('display.max_columns',100)

fifa.head()
#cleaning columns

cc=['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW','LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM',

       'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']



for i in cc:

    fifa[i]=fifa[i].str.split('+',n=1,expand=True)[0]

    fifa[i]=pd.to_numeric(fifa[i])
# combine the position attributes as per pitch area

fifa['Forward'] = fifa.loc[:,'LS':'RW'].astype('float64').mean(axis=1)

fifa['Midfield'] = fifa.loc[:,'LAM':'RM'].astype('float64').mean(axis=1)

fifa['Defense'] = fifa.loc[:,'LWB':'RB'].astype('float64').mean(axis=1)

fifa['GoalKeeper'] = fifa.loc[:,'GKDiving':'GKReflexes'].astype('float64').mean(axis=1)



# drop the columns that are replaced above

fifa.drop(fifa.loc[:,'LS':'RB'].columns.tolist(), axis=1, inplace=True)

fifa.drop(fifa.loc[:,'GKDiving':'GKReflexes'].columns.tolist(), axis=1, inplace=True)
fifa["Club"].fillna("Freelance", inplace = True) 
#renaming columns

colnew = []



for col in fifa.columns:

    colnew.append(col.replace(' ', ''))

    

fifa.columns = colnew
fifa.Wage=fifa.Wage.replace(['M', 'K', '€|\.'], ['000000', '000', ''], regex = True)

fifa.Wage=fifa.Wage.astype('int64')
def cleaning_value(x):

    if '€' in str(x) and 'M' in str(x):

        c = str(x).replace('€' , '')

        c = str(c).replace('M' , '')

        c = float(c) * 1000000

        

    else:

        c = str(x).replace('€' , '')

        c = str(c).replace('K' , '')

        c = float(c) * 1000

            

    return c



fn = lambda x : cleaning_value(x)



fifa['VALUE'] = fifa['Value'].apply(fn)

fifa.VALUE=fifa.VALUE.astype('int64')



fifa=fifa.drop('Value', axis=1)
# Cleaning columns

fifa['Weight']=fifa['Weight'].str.replace('lbs','')

fifa['Height']=fifa['Height'].str.replace("'",'.')



# Changing datatypes from Object to float

fifa.Height = fifa.Height.astype(float)

fifa.Weight = fifa.Weight.astype(float)
#Split the Work Rate Column in two

tempwork = fifa["WorkRate"].str.split("/ ", n = 1, expand = True) 

#Create new column for first work rate

fifa["WorkrateAttack"]= tempwork[0]   

#Create new column for second work rate

fifa["WorkrateDefense"]= tempwork[1]



fifa=fifa.drop('WorkRate', axis=1)
fifa['BodyType']=fifa['BodyType'].str.replace('Messi','Lean')

fifa['BodyType']=fifa['BodyType'].str.replace('Neymar','Normal')

fifa['BodyType']=fifa['BodyType'].str.replace('Courtois','Normal')

fifa['BodyType']=fifa['BodyType'].str.replace('C. Ronaldo','Lean')

fifa['BodyType']=fifa['BodyType'].str.replace('PLAYER_BODY_TYPE_25','Stocky')

fifa['BodyType']=fifa['BodyType'].str.replace('Akinfenwa','Stocky')

fifa['BodyType']=fifa['BodyType'].str.replace('Shaqiri','Stocky')
# removing the columns that do not provide any additional information

fifa.drop(['ID','Photo','Name','RealFace','Flag','LoanedFrom','ClubLogo','JerseyNumber','ReleaseClause', 

           'Joined','ContractValidUntil','Club','Nationality'], inplace=True, axis=1)
fifa.dropna(axis=0,how='all' ,subset=['Finishing'], inplace=True)
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
fifa.dropna(axis=0,inplace=True)
# Checking Co-relations

ad=fifa.corr()

ad.style.background_gradient(cmap='coolwarm')



# fifa=fifa.drop(['Midfield','Dribbling','BallControl','Special','LongShots','Positioning','Interceptions','SkillMoves','Forward',

#                 'Crossing','ShortPassing'], axis=1)
fifa=fifa.dropna(axis=0)
# One Hot Encoding Categorical Variables



fifa=pd.concat([pd.get_dummies(fifa['PreferredFoot']),fifa],axis=1).drop('PreferredFoot',axis=1)

fifa=pd.concat([pd.get_dummies(fifa['BodyType']),fifa],axis=1).drop('BodyType',axis=1)

fifa=pd.concat([pd.get_dummies(fifa['WorkrateAttack']),fifa],axis=1).drop('WorkrateAttack',axis=1)

fifa=pd.concat([pd.get_dummies(fifa['WorkrateDefense']),fifa],axis=1).drop('WorkrateDefense',axis=1)

fifa=pd.concat([pd.get_dummies(fifa['Position']),fifa],axis=1).drop('Position',axis=1)

#fifa=pd.concat([pd.get_dummies(fifa['Nationality']),fifa],axis=1).drop('Nationality',axis=1)

#fifa=pd.concat([pd.get_dummies(fifa['Club']),fifa],axis=1).drop('Club',axis=1)
fifa=fifa[fifa.VALUE > 0]
fifa.skew().sort_values(ascending=False)
from scipy import stats
plt.subplots(figsize=(16,10))

plt.subplot(221)

sns.distplot(fifa['VALUE'])

print('Value skew: ',fifa.VALUE.skew())



plt.subplot(223)

d=np.log1p(fifa['VALUE'])

sns.distplot(d)

print('Value skew: ',d.skew())



plt.subplot(222)

stats.probplot(fifa.VALUE, plot=plt)



plt.subplot(224)

stats.probplot(d, plot=plt)





plt.show()
plt.subplots(figsize=(16,10))

plt.subplot(221)

sns.distplot(fifa.Wage)

print('Wage skew: ',fifa.Wage.skew())



plt.subplot(223)

w=np.log1p(fifa['Wage'])

sns.distplot(w)

print('Wage skew: ',w.skew())



plt.subplot(222)

stats.probplot(fifa.Wage, plot=plt)



plt.subplot(224)

stats.probplot(w, plot=plt)





plt.show()
# removing outliers from VALUE and Wage columns

q1wage=fifa.Wage.quantile(0.25)

q3wage=fifa.Wage.quantile(0.75)

iqrwage= q3wage - q1wage



q1val=fifa.VALUE.quantile(0.25)

q3val=fifa.VALUE.quantile(0.75)

iqrval= q3val - q1val



upperwage= q3wage + iqrwage*1.5

upperval= q3val + iqrval*1.5



fifa=fifa[(fifa.Wage <= upperwage) & (fifa.VALUE <= upperval)]