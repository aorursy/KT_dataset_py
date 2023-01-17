# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as ex

import plotly.figure_factory as ff

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')
s_data = pd.read_csv('/kaggle/input/all-space-missions-from-1957/Space_Corrected.csv')

s_data = s_data[s_data.columns[2:]].copy()

s_data.head(3)
s_data['Datum'] = pd.to_datetime(s_data['Datum'])

rockes_status_dict = {'StatusRetired':1,'StatusActive':2}

status_mission_dict = {'Prelaunch Failure':1,'Partial Failure':3,'Failure':2,'Success':4}



s_data['Status Rocket'].replace(rockes_status_dict,inplace=True)

s_data['Status Mission'].replace(status_mission_dict,inplace=True)

#creating the day of launch

s_data['LaunchDay'] = s_data['Datum'].apply(lambda x: x.weekday())

#creating the hour of launch

s_data['LaunchTime'] = s_data['Datum'].apply(lambda x: x.hour)

#creating the year of launch

s_data['LaunchYear'] = s_data['Datum'].apply(lambda x: x.year)

#creating the month of launch

s_data['LaunchMonth'] = s_data['Datum'].apply(lambda x: x.month)



s_data['Country'] = s_data['Location'].apply(lambda x: x.split(',')[-1])

s_data = s_data[s_data['Country'].isin(s_data['Country'].value_counts().index[:17])]
s_data
#extract data with no missing values in the rocket price data

s_data_n = s_data[~s_data[' Rocket'].isna()].copy()

s_data_n[' Rocket'] = s_data_n[' Rocket'].replace('5,000.0 ',500)

s_data_n[' Rocket'] = s_data_n[' Rocket'].replace('1,160.0 ',116)

s_data[' Rocket'] = s_data[' Rocket'].replace('5,000.0 ',500)

s_data[' Rocket'] = s_data[' Rocket'].replace('1,160.0 ',116)



s_data_n[' Rocket'] = s_data_n[' Rocket'].astype('float64')

s_data[' Rocket'] = s_data[' Rocket'].astype('float64')



s_data_n
plt.figure(figsize=(20,11))

ax = sns.countplot(s_data['Company Name'],order=s_data['Company Name'].value_counts().index)

ax.set_xticklabels(ax.get_xticklabels(),rotation=90,fontsize=15)

ax.set_xlabel(ax.get_xlabel(),fontsize=16)

ax.set_ylabel(ax.get_ylabel(),fontsize=16)

ax.set_title('Distribution Of Different Companies And Amount Of Space Mission Laucned By Them',fontsize=17)

plt.show()
fig = ff.create_distplot([s_data[' Rocket']],['Rocket'],curve_type='kde',bin_size=50)

fig.update_layout(title='Distribution Of Different Rocket Prices',height=900)

fig.show()

fig = ff.create_distplot([s_data['LaunchTime']],['LaunchTime'],curve_type='kde',bin_size=2)

fig.update_layout(title='distribution Of Different Space Mission Launch Times',height=900)

fig.show()

plt.figure(figsize=(20,11))

ax = sns.countplot(s_data['LaunchDay'],palette='mako')

#ax.set_xticklabels(ax.get_xticklabels(),rotation=90,fontsize=15)

#ax.set_xlabel(ax.get_xlabel(),fontsize=16)

#ax.set_ylabel(ax.get_ylabel(),fontsize=16)

ax.set_title('Distribution Of Different Space Mission Launch Days Of Week',fontsize=17)

ax.patches[2].set_fc('r')

plt.show()
plt.figure(figsize=(20,11))

ax = sns.countplot(s_data['LaunchMonth'],palette='mako')

#ax.set_xticklabels(ax.get_xticklabels(),rotation=90,fontsize=15)

#ax.set_xlabel(ax.get_xlabel(),fontsize=16)

#ax.set_ylabel(ax.get_ylabel(),fontsize=16)

ax.set_title('Distribution Of Different Space Mission Launch Months',fontsize=17)

ax.patches[5].set_fc('r')

ax.patches[11].set_fc('r')

plt.show()
plt.figure(figsize=(20,11))

ax = sns.countplot(s_data['Country'],palette='mako',order=s_data['Country'].value_counts().index)

#ax.set_xticklabels(ax.get_xticklabels(),rotation=90,fontsize=15)

#ax.set_xlabel(ax.get_xlabel(),fontsize=16)

#ax.set_ylabel(ax.get_ylabel(),fontsize=16)

ax.set_title('Distribution Of Different Space Mission Launch Country',fontsize=17)

ax.patches[0].set_fc('r')

ax.patches[1].set_fc((0.6,0,0))

plt.show()
plt.figure(figsize=(20,11))

ax = sns.countplot(s_data['Country'],palette='mako',order=s_data['Country'].value_counts().index,hue=s_data['Status Mission'])

ax.set_xticklabels(ax.get_xticklabels(),rotation=90,fontsize=15)

#ax.set_xlabel(ax.get_xlabel(),fontsize=16)

#ax.set_ylabel(ax.get_ylabel(),fontsize=16)

ax.set_title('Distribution Of Different Space Mission Statuses',fontsize=17)

plt.legend(labels=status_mission_dict.keys(),loc=1,prop={'size':20})

plt.show()
plt.figure(figsize=(20,11))

ax = sns.countplot(s_data[s_data['Status Mission'] == 2]['Country'],palette='mako')

ax.set_xticklabels(ax.get_xticklabels(),rotation=90,fontsize=15)

#ax.set_xlabel(ax.get_xlabel(),fontsize=16)

#ax.set_ylabel(ax.get_ylabel(),fontsize=16)

ax.set_title('Distribution Of Failed Space Launches In Different Countries',fontsize=17)

#plt.legend(labels=status_mission_dict.keys(),loc=1,prop={'size':20})

plt.show()
plt.figure(figsize=(20,11))

ax = sns.countplot(s_data['LaunchYear'],palette='mako',hue=s_data['Status Mission'])

ax.set_xticklabels(ax.get_xticklabels(),rotation=90,fontsize=15)

#ax.set_xlabel(ax.get_xlabel(),fontsize=16)

#ax.set_ylabel(ax.get_ylabel(),fontsize=16)

ax.set_title('Distribution Of Different Space Mission Statuses',fontsize=17)

plt.legend(labels=status_mission_dict.keys(),loc=1,prop={'size':20})

plt.show()
from wordcloud import WordCloud, STOPWORDS 

stopwords = set(STOPWORDS)

comment_words = '' 

for val in s_data.Detail: 

    val = str(val) 

    tokens = val.split() 

    for i in range(len(tokens)): 

        tokens[i] = tokens[i].lower() 

    comment_words += " ".join(tokens)+" "

comment_words = set(comment_words.split())

comment_words = " ".join(comment_words)

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='black', 

                stopwords = stopwords, 

                min_font_size = 10).generate(comment_words) 

  

# plot the WordCloud image                        

plt.figure(figsize = (20, 11), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 
sn_cor = s_data_n.corr()

plt.figure(figsize=(20,11))

ax = sns.heatmap(sn_cor,annot=True,cmap='mako')

ax.set_title('Correlation Between Features Where Rocket Price Is Not Missing',fontsize=19)

plt.show()
sn_cor = s_data.corr()

plt.figure(figsize=(20,11))

ax = sns.heatmap(sn_cor,annot=True,cmap='mako')

ax.set_title('Correlation Between Features Where Rocket Price Is Missing',fontsize=19)

plt.show()
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor
X = s_data_n[['Status Rocket','LaunchYear']]

y = s_data_n[' Rocket']

train_x,test_x,train_y,test_y = train_test_split(X,y)

# Linear Regression

lr_pipe = Pipeline(steps=[('Scale',StandardScaler()),('model',LinearRegression())])

lr_pipe.fit(train_x,train_y)

s = np.sqrt(-1*cross_val_score(lr_pipe,X,y,cv=5,scoring='neg_mean_squared_error'))

print("LinearRegression Mean RMSE Score Across 5 Folds = ",s.mean())
xgb_pipe = Pipeline(steps=[('Scale',StandardScaler()),('model',XGBRegressor(n_estimators=500,learning_rate=0.03))])

xgb_pipe.fit(train_x,train_y)

s = np.sqrt(-1*cross_val_score(xgb_pipe,X,y,cv=5,scoring='neg_mean_squared_error'))

print("XGB Mean RMSE Score Across 5 Folds = ",s.mean())
xgb_model = XGBRegressor(n_estimators=1000,learning_rate=0.05)

xgb_model.fit(train_x,train_y,eval_set=[(test_x[0:20],test_y[0:20])],early_stopping_rounds=5,verbose=False)

xgb_predict = xgb_model.predict(test_x)

xgb_score = mean_squared_error(xgb_predict,test_y)

xgb_score
print("XGB Mean RMSE After Parameter Tuning = ",np.sqrt(xgb_score))
Xm = s_data[s_data[' Rocket'].isna()][['Status Rocket','LaunchYear']]

xgb_predict = xgb_model.predict(Xm)

missing_index = s_data[s_data[' Rocket'].isna()][' Rocket'].index.to_list()

s_data.loc[missing_index,' Rocket'] = xgb_predict
plt.figure(figsize=(20,11))

ax = sns.distplot(s_data[' Rocket'])

#ax.set_xticklabels(ax.get_xticklabels(),rotation=90,fontsize=15)

#ax.set_xlabel(ax.get_xlabel(),fontsize=16)

#ax.set_ylabel(ax.get_ylabel(),fontsize=16)

ax.set_title('Distribution Of Different Rocket Prices After Predicting Missing Values',fontsize=17)

plt.show()
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report
X = s_data_n[[' Rocket','LaunchYear']]

y = s_data_n['Status Rocket']

train_x,test_x,train_y,test_y = train_test_split(X,y)

ada_pipe = Pipeline(steps=[('scale',StandardScaler()),('model',AdaBoostClassifier(learning_rate=0.3,random_state=42))])

ada_pipe.fit(train_x,train_y)

ada_pipe.score(test_x,test_y)



print(classification_report(ada_pipe.predict(test_x),test_y))
RF_pipe = Pipeline(steps=[('scale',StandardScaler()),('model',RandomForestClassifier(n_estimators=100,random_state=42))])

RF_pipe.fit(train_x,train_y)

RF_pipe.score(test_x,test_y)

print(classification_report(RF_pipe.predict(test_x),test_y))
DT_pipe = Pipeline(steps=[('scale',StandardScaler()),('model',DecisionTreeClassifier(max_leaf_nodes=5))])

DT_pipe.fit(train_x,train_y)

DT_pipe.score(test_x,test_y)

print(classification_report(DT_pipe.predict(test_x),test_y))
s_data.to_csv('Data_With_Predicted_Rocket_Prices_For_Missing Values.csv',index=False)
