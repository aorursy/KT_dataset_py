import pandas as pd, numpy as np

import sklearn

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

import warnings

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import ExtraTreesClassifier

warnings.filterwarnings('ignore')
fifa_inp=pd.read_excel('../input/fifa-19-complete-player-dataset/data.xls',index_col=[1,2])

fifa_inp.head()
fifa_inp['Weight']=fifa_inp['Weight'].str.replace('lbs','')

fifa_inp.loc[:, 'Value':'Wage'] = fifa_inp.loc[:, 'Value':'Wage'].replace({np.nan: 0})



for column in fifa_inp.loc[:, 'LS':'RB'].columns: 

    fifa_inp[column] = fifa_inp[column].str.split('+').str[0]



for col in  ['Release Clause','Value',"Wage"]:

    fifa_inp[col]=fifa_inp[col].str.replace('K','000')

    fifa_inp[col]=fifa_inp[col].str.replace('M','000000')

    fifa_inp[col]=fifa_inp[col].str.replace('€','')

    fifa_inp[col]=fifa_inp[col].str.replace('.','')
fifa_inp['Value']=fifa_inp['Value'].astype('float')

fifa_inp['Wage']=fifa_inp['Wage'].astype('float')

val_min=np.nanmin(fifa_inp['Value'].values)

wage_min=np.nanmin(fifa_inp['Wage'].values)

fifa_inp['Value'].fillna(val_min, inplace = True)

fifa_inp['Wage'].fillna(wage_min, inplace = True)
fifa_inp['Body Type']=fifa_inp['Body Type'].str.replace('Akinfenwa','Stocky')

fifa_inp['Body Type']=fifa_inp['Body Type'].str.replace('Courtois','Lean')

fifa_inp['Body Type']=fifa_inp['Body Type'].str.replace('Messi|C. Ronaldo|Neymar|PLAYER_BODY_TYPE_25|Shaqiri','Normal', regex=True)

fifa_inp['Body Type'].head()
fifa_inp=fifa_inp.drop(columns=['Photo','Flag', 'Nationality','Club Logo','Real Face','Special','Loaned From','Unnamed: 0'])
#Drop these rows - 90% column values are Nan

len_bef=len(fifa_inp)

fifa_inp = fifa_inp.dropna(subset=['International Reputation'])
print("Rows droped= ")

print(len_bef-len(fifa_inp))
fifa_inp.head()
fifa_inp.loc[:, 'LS':'RB'] = fifa_inp.loc[:, 'LS':'RB'].replace({np.nan: 0})



fifa_inp['Jersey Number'].fillna(8, inplace = True)

fifa_inp['Position'].fillna('ST', inplace = True)

fifa_inp['Club'].fillna('No Club', inplace = True)

fifa_inp['Release Clause'].fillna(0, inplace = True)
fifa_inp['Height']=fifa_inp['Height'].str.split("'").str[0].astype('int64')*12+fifa_inp['Height'].str.split("'").str[1].astype('int64')
fifa_inp['Contract Valid Until'].fillna(2019, inplace = True)

fifa_inp.loc[:, 'Joined'] = fifa_inp.loc[:, 'Joined'].replace({np.nan: fifa_inp['Joined'].max()})

fifa_inp['Contract Valid Until']=fifa_inp['Contract Valid Until'].astype('str')

fifa_inp['Contract Valid Until'] = fifa_inp['Contract Valid Until'].str.split('-').str[0]

fifa_inp['Joined'] = pd.DatetimeIndex(fifa_inp['Joined']).year

fifa_inp['Joined'].head()
fifa_inp.loc[:, 'Age':'Potential']=fifa_inp.loc[:, 'Age':'Potential'].astype('int64')

fifa_inp.loc[:, 'Value':'Wage']=fifa_inp.loc[:,  'Value':'Wage'].astype('int64')

fifa_inp.loc[:, 'International Reputation':'Skill Moves']=fifa_inp.loc[:,'International Reputation':'Skill Moves'].astype('int64')

fifa_inp.loc[:, 'Jersey Number']=fifa_inp.loc[:,'Jersey Number'].astype('int64')

fifa_inp.loc[:, 'Joined':'Release Clause']=fifa_inp.loc[:, 'Joined':'Release Clause'].astype('int64')
print("Nan in each columns" , fifa_inp.isna().sum(), sep='\n')
#fifa_inp.to_excel(r'Fifa\Fifa_clean.xls')
#fifa_inp['Name'].value_counts()

#Players with same first and last name
dataTypeSeries = fifa_inp.dtypes 

print('Data type of each column of Dataframe :')

print(dataTypeSeries)

#fifa_inp.nunique()
club_encode=LabelEncoder()

foot_encode=LabelEncoder()

workrate_encode=LabelEncoder()

body_encode=LabelEncoder()

position_encode=LabelEncoder()
# Encoding the variable

fifa_inp['Club']=club_encode.fit_transform(fifa_inp['Club'])

fifa_inp['Preferred Foot']=foot_encode.fit_transform(fifa_inp['Preferred Foot'])

fifa_inp['Work Rate']=workrate_encode.fit_transform(fifa_inp['Work Rate'])

fifa_inp['Body Type']=body_encode.fit_transform(fifa_inp['Body Type'])

fifa_inp['Position']=position_encode.fit_transform(fifa_inp['Position'])
#foot_encode.inverse_transform(fifa_inp['Preferred Foot'])

club = dict(zip(club_encode.classes_, club_encode.transform(club_encode.classes_)))

foot = dict(zip(foot_encode.classes_, foot_encode.transform(foot_encode.classes_)))

workrate = dict(zip(workrate_encode.classes_, workrate_encode.transform(workrate_encode.classes_)))

body = dict(zip(body_encode.classes_, body_encode.transform(body_encode.classes_)))

position = dict(zip(position_encode.classes_, position_encode.transform(position_encode.classes_)))

print(position)
#fifa_inp.to_excel(r'C:\Users\anand\Downloads\Data sets\Fifa\Fifa_clean.xls')
#club
foot
workrate
body
'''Function to distribution plot'''

plt.rcParams['figure.figsize'] = (10, 6)

font_size = 16

title_size = 20

sns.distplot(fifa_inp['Age'], color = 'teal')

plt.xlabel('Age', fontsize = 16)

plt.ylabel('Precentage of the Players', fontsize = 16)

plt.xticks(fontsize = 16)

plt.yticks(fontsize = 16)

plt.title('Age Distribution of Players', fontsize = 20)

plt.show()
sns.violinplot(x="International Reputation", y="Overall",data=fifa_inp, palette="Set3")

plt.rcParams['figure.figsize'] = (10, 6)

_=plt.title('Overall score among players of different Reputation', fontsize = 20)

#sns.boxplot(x="International Reputation", y="Overall",data=fifa_inp, palette="Set3")
club_reputation=fifa_inp.groupby('Club')['International Reputation'].apply(lambda x: ((x==5)|(x==4)).sum()).sort_values(ascending=False).reset_index(name='count')

top_club=list(club_reputation.head(5).Club)

data_club = fifa_inp.loc[fifa_inp['Club'].isin(top_club)]

#club_reputation.head(5)
x='Club'

y='proportion'

hue='International Reputation'

top_club=list(club_reputation.head(5).Club)

data_club = fifa_inp.loc[fifa_inp['Club'].isin(top_club)]

data_club['Club']=club_encode.inverse_transform(data_club['Club'])

plt.rcParams['figure.figsize'] = (12, 8)

ax=data_club[hue].groupby(data_club[x]).value_counts(normalize=True).rename(y).reset_index().pipe((sns.barplot, "data"), x=x, y=y, hue=hue)

ax.set_title(label = 'Players of Reputation in some top Clubs', fontsize = 20)

ax=plt.xticks(rotation = 45)
#data_club[hue].groupby(data_club[x]).value_counts(normalize=True).rename(y).reset_index()

data_club.head(2)
# Separating out the features

X = fifa_inp.copy()

X=X.drop(columns=['International Reputation'])

# Separating out the target

Y = fifa_inp.loc[:,['International Reputation']]

# Standardizing the features

X = StandardScaler().fit_transform(X)

X
from sklearn.decomposition import PCA

comp=50

columns = []

for i in range(1,comp+1):

    columns.append('PC '+str(i))

pca = PCA()

principalComponents = pca.fit_transform(X)

principalDf = pd.DataFrame(data = principalComponents)
var=pca.explained_variance_ratio_

#pca.explained_variance_
plt.rcParams['figure.figsize'] = (8,5)

ax=plt.plot(range(1,len(pca.explained_variance_)+1),np.cumsum(pca.explained_variance_ratio_))

#ax=plt.bar(range(1,len(pca.explained_variance_)+1),pca.explained_variance_)

#ax.set_title(label = 'Distribution of International Reputation in some Popular Clubs', fontsize = 20)

plt.title('PCA Components vs variance', fontsize = 16)

plt.xlabel('number of components')

_=plt.ylabel('cumulative explained variance')
# feature extraction

# Separating out the features

X = fifa_inp.copy()

X=X.drop(columns=['International Reputation','Wage','Value','Release Clause'])

X_columns=X.columns

# Separating out the target

Y = fifa_inp.loc[:,['International Reputation']]

model = ExtraTreesClassifier(n_estimators=10)

model.fit(X, Y)

Feature_importance=pd.Series(model.feature_importances_,index=X_columns)

Feature_importance=Feature_importance.sort_values(ascending=False)

Feature_importance=Feature_importance[Feature_importance>0.015]

Feature_importance
Correlation_DF = fifa_inp[list(Feature_importance.index)+['International Reputation']]

#list(Feature_importance.index)
colormap = plt.cm.inferno

plt.figure(figsize=(16,12))

plt.title('Correlation between Features related to international reputation', y=1.05, size=15)

_=sns.heatmap(Correlation_DF.corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)

ridge.fit(X,Y)


ridge_coeff=pd.Series(ridge.coef_[0],index=X_columns)

#pd.DataFrame(data={'Features':X_columns,'Coefficients':ridge.coef_})

ridge_imp=ridge_coeff[(ridge_coeff<-0.008) | (ridge_coeff>0.008)]

ridge_imp
ridge_Correlation_DF = fifa_inp[list(ridge_imp.index)+['International Reputation']]

#list(Feature_importance.index)
colormap = plt.cm.inferno

plt.figure(figsize=(16,12))

plt.title('Correlation between Features related to international reputation', y=1.05, size=15)

_=sns.heatmap(ridge_Correlation_DF.corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)