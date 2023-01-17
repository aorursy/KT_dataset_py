import numpy as np # linear algebra

import pandas as pd # data processing





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')
pd.set_option('display.max_columns', None)
df.head()
df.shape
df.isna().sum()  #we will not convert values of -1 to none ,just leave them
df[df['winPlacePerc'].isna()]
df=df.drop(2744604)            #dropping this nan

df=df.reset_index(drop=True)
df.matchId.value_counts()        #it seems the data have all players os some matches 

                                 #but actually I didn't use it
df[df.killPoints==0][df.rankPoints!=-1].shape    #getting the shape of 0 kill points and no rank
df[df.killPoints==0].shape                   #it seems all 0 kill points has no rank
df.describe()          #getting the statistical analysis of the data 
df.matchType.value_counts() 
import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline
df.head()
plt.figure(figsize=(13,7))

sns.boxplot(x='matchType',y='winPlacePerc',data=df)     

plt.xticks(rotation=300)

plt.show()





#it seems the type of match isn't playing a big role in winning
sns.lmplot(x='winPlacePerc',y='kills',data=df,fit_reg=False)

plt.show()



#A lot wins without even one kill, may be they are cheaters or have a great team beside them

#but in general more kills more wins
sns.lmplot(x='winPlacePerc',y='winPoints',data=df,fit_reg=False)

plt.show()



#win points don't give you more probabilty to win ever
sns.lmplot(x='winPlacePerc',y='killPoints',data=df,fit_reg=False)

plt.show()



#neither kill points
sns.lmplot(x='winPlacePerc',y='rankPoints',data=df,fit_reg=False)

plt.show()



#and rank is trash 
sns.lmplot(x='winPlacePerc',y='longestKill',data=df,fit_reg=False)

plt.show()



#okay it has a little tiny correlation but a lot of noise 
sns.lmplot(x='winPlacePerc',y='damageDealt',data=df)

plt.show()



#some winners didn't even make a scratch ,but you can see a slight correlation though
sns.lmplot(x='winPlacePerc',y='walkDistance',data=df)

plt.show()



#winners without even walk ,hackers
df[df['walkDistance']==0][df['winPlacePerc']==1].describe()



#and a lot of cheaters here weapons and kills without a single move
sns.lmplot(x='winPlacePerc',y='swimDistance',data=df,fit_reg=False)

plt.show()



#not interesting 
weapons_num=pd.cut(df['weaponsAcquired'],[0,2,6,12,25,50,100,200,300],

                   labels=['0-2','2-6','6-12','12-25','25-50','50-100','100-200','+200'])



#making the number of weapons acquired in categories so we can visualize it properly
plt.figure(figsize=(12,7))

sns.boxplot(x=weapons_num,y='winPlacePerc',data=df)

plt.show()



#okay having weapons from 6 and up to 25 makes your probability to win a bit more

#it's the proper number beacuse more than 100 hunderd are likely cheatings
del weapons_num
kills=pd.cut(df['kills'],[1,10,20,40,60,80],labels=['1-10','10-20','20-40','40-60','+60'])

#make kills in categories
plt.figure(figsize=(13,6))

sns.boxplot(x=kills,y=df['winPlacePerc'])

plt.show()
df.headshotKills.value_counts()



#64 headshot !
head_shot=pd.cut(df['headshotKills'],[0,1,2,3,4,7,10,16,19,35,64],

                 labels=['0','1','2','3','4-7','7-10','10-16','16-19','19-35','35-64'])



#making head shots in categories
plt.figure(figsize=(13,6))

sns.boxplot(x=head_shot,y=df['winPlacePerc'])



#if you get more than 5 head shots you are likely more to win 
groups=df.groupby('groupId').sum()



#we will make some features that take all the group in consideration
groups=groups.reset_index()
groups.head()
add_col=groups.loc[:,['groupId','kills','revives','weaponsAcquired','teamKills']]



#These are the new columns
add_col=add_col.rename(columns={'kills':'groupKills','revives':'groupRevives',

                                'weaponsAcquired':'groupWeapons','teamKills':'groupOwnKills'})



#let's give them nice names 
df2=df.merge(add_col,how='left',on=['groupId'])

#and we merge them with the data
df2.head()
del df 

del groups

del add_col

#then get rid of them 
df2.loc[:,['groupKills','groupRevives','groupWeapons','groupOwnKills']].describe()
gkills=pd.cut(df2['groupKills'],[1,10,20,40,60,80,200],

              labels=['1-10','10-20','20-40','40-60','60-80','+80'])
plt.figure(figsize=(13,6))

sns.boxplot(x=gkills,y=df2['winPlacePerc'])

plt.show()
gweapons_num=pd.cut(df2['groupWeapons'],[0,2,6,12,25,50,100,200,300,700],

                labels=['0-2','2-6','6-12','12-25','25-50','50-100','100-200','200-300','+300'])
plt.figure(figsize=(13,6))

sns.boxplot(x=gweapons_num,y=df2['winPlacePerc'])

plt.show()



#Your group should have from 12 to 25 to be close to win 

#because the game is not about collecting weapons
grevives=pd.cut(df2['groupRevives'],[0,2,4,8,15,30,50],

                labels=['0-2','2-4','4-8','8-15','15-30','+30'])
plt.figure(figsize=(13,6))

sns.boxplot(x=grevives,y=df2['winPlacePerc'])

plt.show()
df2.head()
df2['matchType']=df2.matchType.astype('category') #make it categorical to encode
df2['matchtype']=df2['matchType'].cat.codes  #encoding
df2.head()
pubg=df2.drop(columns=['Id','groupId','matchId','matchType'])  
del df2
pubg=pubg.drop(columns=['killPoints','matchDuration','maxPlace','numGroups','rankPoints'

                        ,'winPoints','groupOwnKills','teamKills','vehicleDestroys'])



#dropping the unnecessary columns
from sklearn.model_selection import train_test_split
x=pubg.drop(columns=['winPlacePerc'])

y=pubg['winPlacePerc']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
del x

del y
plt.figure(figsize=(15,10))

sns.heatmap(pubg.corr(),annot=True)
corr_matrix = pubg.corr()

corr_matrix["winPlacePerc"].sort_values(ascending=False)



#It seems that some of our new features are good correlated 
from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(x_train, y_train)
from sklearn.metrics import mean_squared_error



y_pred = lin_reg.predict(x_test)

lin_mse = mean_squared_error(y_test, y_pred)

lin_rmse = np.sqrt(lin_mse)

lin_rmse
del lin_reg

del lin_mse

del lin_rmse
from sklearn.tree import DecisionTreeRegressor



tree_reg = DecisionTreeRegressor(random_state=42)

tree_reg.fit(x_train, y_train)
y_pred = tree_reg.predict(x_test)

tree_mse = mean_squared_error(y_test, y_pred)

tree_rmse = np.sqrt(tree_mse)

tree_rmse

#great
from sklearn.model_selection import cross_val_score



scores = cross_val_score(tree_reg, x_train, y_train,

                         scoring="neg_mean_squared_error", cv=5)
-scores
del scores
from sklearn.model_selection import GridSearchCV



param_grid = [

    {'max_features': [6, 8,None]},

  ]



grid_search = GridSearchCV(tree_reg, param_grid, cv=5,

                           scoring='neg_mean_squared_error',

                           return_train_score=True)

grid_search.fit(x_train, y_train)
grid_search.best_params_

#I knew it
tree_reg = DecisionTreeRegressor(random_state=42)

tree_reg.fit(pubg.drop(columns=['winPlacePerc']), pubg['winPlacePerc'])



#Fitting the model with all the data and it's ready to go
df_test=pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/test_V2.csv')
groups=df_test.groupby('groupId').sum()

groups=groups.reset_index()
add_col=groups.loc[:,['groupId','kills','revives','weaponsAcquired','teamKills']]

add_col=add_col.rename(columns={'kills':'groupKills','revives':'groupRevives','weaponsAcquired':'groupWeapons','teamKills':'groupOwnKills'})

df2_test=df_test.merge(add_col,how='left',on=['groupId'])
df2_test.head()
df2_test['matchType']=df2_test.matchType.astype('category')

df2_test['matchtype']=df2_test['matchType'].cat.codes
final=df2_test.drop(columns=['Id','groupId','matchId','matchType'])

final=final.drop(columns=['killPoints','matchDuration','maxPlace','numGroups',

                          'rankPoints','winPoints','groupOwnKills','teamKills','vehicleDestroys'])
Id=df2_test['Id']  #for submission
winPlacePerc = tree_reg.predict(final)  #our prediction
pred =pd.concat([Id,pd.Series(winPlacePerc)],axis=1)
pred=pred.rename(columns={0:'winPlacePerc'})
pred.to_csv('submission.csv',index=False) 