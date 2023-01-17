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
nba=pd.read_csv('/kaggle/input/nba-mvp-votings-through-history/mvp_votings.csv').drop(columns='Unnamed: 0')
pd.set_option('display.max_columns', None)
nba
#set whole column to 'No', then just change to 'Yes' for mvp winners
nba['Mvp?']='No'

#for every season
for season in nba['season'].value_counts().index:
    
    #isolate data from that season
    season_df=nba[nba['season'].isin([season])]
    
    #get the index of player with most mvp points
    index=[season_df['points_won'].idxmax()]
    
    #change player's 'Mvp?' entry to yes
    nba['Mvp?'][index]='Yes'
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(20,10))
sns.heatmap(nba.corr(),annot=True,linewidth=0.5)
df=pd.DataFrame(nba.corr()['award_share']).reset_index()
df['Beat Threshold']=abs(df['award_share'])>0.45
sns.lmplot(x='index', y="award_share", data=df,hue='Beat Threshold',fit_reg=False,height=4,
           aspect=4).set_xticklabels(rotation=90)
def scatter(attribute):
    p1=sns.lmplot(x=attribute, y="award_share", data=nba,hue='Mvp?',fit_reg=False,height=8,aspect=4)
    ax = p1.axes[0,0]
    for i in range(len(nba)):
        ax.text(nba[attribute][i], nba['award_share'][i],nba['player'][i] +' '+nba['season'][i],
               fontsize='small',rotation=45)
    plt.show()
scatter('per')
scatter('bpm')
scatter('ws')
scatter('season')
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

features=['per','bpm','ws','ws_per_48']
training_seasons=['1980-81', '1981-82', '1984-85', '1982-83', '1998-99', '1996-97',
       '1990-91', '1997-98', '1988-89', '2001-02', '1985-86', '2000-01',
       '2007-08', '1991-92', '1993-94', '2006-07', '1986-87', '1995-96',
       '1987-88', '2013-14', '1999-00', '2012-13', '2004-05', '2003-04',
       '1994-95', '2011-12', '2009-10', '1983-84', '1989-90', '1992-93']
testing_seasons=['2017-18', '2010-11', '2002-03', '2014-15', '2008-09', '2005-06',
       '2016-17', '2015-16']
#training data
training_data=nba[nba['season'].isin(training_seasons)]
train_X=training_data[features]
train_y=training_data['award_share']

#testing data
testing_data=nba[nba['season'].isin(testing_seasons)]
val_X=testing_data[features]
val_y=testing_data['award_share']

basic_model = DecisionTreeRegressor(random_state=1)
basic_model.fit(train_X, train_y)
predictions=basic_model.predict(val_X)
df=pd.DataFrame(val_X)
df['prediction']=predictions
df['award_share']=val_y
df['season']=[nba['season'][index] for index in df.reset_index()['index']]
df['player']=[nba['player'][index] for index in df.reset_index()['index']]
df['Mvp?']=[nba['Mvp?'][index] for index in df.reset_index()['index']]
df=df[['per','bpm','ws','ws_per_48','player','season','award_share','Mvp?','prediction']]
df
#create column indicating whether player actually won the mvp
df['mvp prediction']='No'
for season in df['season'].value_counts().index:
    season_df=df[df['season'].isin([season])]
    index=season_df['prediction'].idxmax()
    mvp=df['player'][index]
    
    #will only change for the mvp winner, otherwise all others players will be 'no'
    df['mvp prediction'][index]='Yes'
pd.set_option('display.max_rows', None)
df
ax=plt.gca()
ax.plot(df['player'],df['award_share'],'o',color='red',label = 'Actual Values')
plt.xticks(rotation=90)

ax.plot(df['player'],df['prediction'],'X',color='yellow',label = 'Predicted Values')

for i in df.reset_index()['index']:
    ax.text(df['player'][i], df['award_share'][i],df['season'][i],fontsize='small',rotation=45)
    ax.text(df['player'][i], df['prediction'][i],df['season'][i],fontsize='small',rotation=45)

        
ax.set_xlabel('Player')
ax.set_ylabel('Award share')
ax.set_title('Actual and Preicted MVP Award Shares')
ax.legend(loc = 'upper right')

ax.figure.set_size_inches(20, 8)

plt.show()
fig, axes = plt.subplots(nrows=2, ncols=4,figsize=(20, 10))
fig.subplots_adjust(hspace=1)

#fig.autofmt_xdate(rotation=90)

seasons=df['season'].value_counts().index
for season,ax in zip(seasons,axes.flatten()):
    frame=df[df['season'].isin([season])]
    ax.scatter(frame['player'],frame['award_share'],marker='v',color='red',label = 'Actual Values')
    mvp_id=frame['award_share'].idxmax()
    ax.annotate('mvp',(frame['player'][mvp_id], frame['award_share'][mvp_id]))
    #,bbox=dict(boxstyle="circle")
    circle_rad = 10  # This is the radius, in points
    ax.plot(frame['player'][mvp_id], frame['award_share'][mvp_id], 'o',
        ms=circle_rad * 2, mec='r', mfc='none', mew=2)
    

    ax.scatter(frame['player'],frame['prediction'],marker='x',color='yellow',label = 'Predicted Values')
    predicted_id=frame['prediction'].idxmax()
    ax.annotate("prediction", (frame['player'][predicted_id], frame['prediction'][predicted_id]))
#     for i in df.reset_index()['index']:
#         ax.text(df['player'][i], df['award_share'][i],df['season'][i],fontsize='small',rotation=45)
#         ax.text(df['player'][i], df['prediction'][i],df['season'][i],fontsize='small',rotation=45)
    
    #circle1 = plt.Circle((frame['player'][predicted_id], frame['prediction'][predicted_id]), 0.1, color='b',fill=False)
    #ax.add_artist(circle1)
    ax.plot(frame['player'][predicted_id], frame['prediction'][predicted_id], 'o',
        ms=circle_rad * 2, mec='y', mfc='none', mew=2)
    
    ax.set_xlabel('Player')
    ax.set_ylabel('Award share')
    ax.set_title(season)
    ax.legend(loc = 'upper right')
    ax.set_xticklabels(labels=frame['player'],rotation=90)

   # plt.xticks(rotation=90)
nba['binary']=pd.Series(nba['Mvp?']=='Yes').astype(int)
#training data
training_data=nba[nba['season'].isin(training_seasons)]
train_X=training_data[features]
train_y=training_data['binary']

#testing data
testing_data=nba[nba['season'].isin(testing_seasons)]
val_X=testing_data[features]
val_y=testing_data['binary']
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()

parameters = {'learning_rate': [0.01,0.02,0.03],
              'subsample'    : [0.9, 0.5, 0.2],
              'n_estimators' : [100,500,1000],
              'max_depth'    : [4,6,8]
             }

#grid = GridSearchCV(estimator=model, param_grid = parameters, cv = 2, n_jobs=-1)
#grid.fit(train_X, train_y)

gd_sr = GridSearchCV(estimator=model,
                     param_grid=parameters,
                     scoring='accuracy',
                     cv=5,
                     n_jobs=-1)

gd_sr.fit(train_X, train_y)

best_parameters = gd_sr.best_params_
print(best_parameters)
best_result = gd_sr.best_score_
print(best_result)
pred=gd_sr.best_estimator_.predict(val_X)
frame=val_X.join(val_y)
frame['pred']=pred
frame['pred'].value_counts()
frame['binary'].value_counts()
frame[(frame['binary']==1) & (frame['pred']==1)]