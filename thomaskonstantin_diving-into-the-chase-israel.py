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
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')
c_data = pd.read_csv('/kaggle/input/the-chasegame-show-israel-episode-data/The_Chase__Dataset.csv')
#creating a featues which shows us the amount of player that got to the final round

def count_players(sir):

    return sir['p1_round_status']+sir['p_2_round_status']+sir['p_3_round_status']+sir['p_4_round_status']



c_data['contestants_at_finals'] = c_data.apply(count_players,axis=1)



#creating a featues which shows us did a contestant choose a 'risk' or and 'easypath'



choice_dict = {3:'Risk',2:'Same Sum',1:'Safety'}



def choice_type(r_sum,p_sum):

    if r_sum < p_sum:

        return 3

    elif r_sum == p_sum:

        return 2

    else:

        return 1



results = []

for index,row in c_data.iterrows():

    results.append(choice_type(c_data.loc[index,'p_1_fast_round_sum'],c_data.loc[index,'p_1_play_sum_chosen']))

c_data['p_1_choice'] = results



results = []

for index,row in c_data.iterrows():

    results.append(choice_type(c_data.loc[index,'p_2_fast_round_sum'],c_data.loc[index,'p_2_play_sum_chosen']))

c_data['p_2_choice'] = results



results = []

for index,row in c_data.iterrows():

    results.append(choice_type(c_data.loc[index,'p_3_fast_round_sum'],c_data.loc[index,'p_3_play_sum_chosen']))

c_data['p_3_choice'] = results





results = []

for index,row in c_data.iterrows():

    results.append(choice_type(c_data.loc[index,'p_4_fast_round_sum'],c_data.loc[index,'p_4_play_sum_chosen']))

c_data['p_4_choice'] = results

c_data.head(3)
#useful lists 

player_genders = ['p_1_gender','p_2_gender','p_3_gender','p_4_gender']

player_ages = ['p_1_age','p_2_age','p_3_age','p_4_age']
gender_dict = {'male':1,'female':0}

c_data = c_data.replace(gender_dict)
fig,axs = plt.subplots(2,2)

fig.set_figwidth(20)

fig.set_figheight(15)

sns.countplot(c_data[player_genders[0]],ax = axs[0,0],palette=['pink','teal'])

sns.countplot(c_data[player_genders[1]],ax = axs[0,1],palette=['pink','teal'])

sns.countplot(c_data[player_genders[2]],ax = axs[1,0],palette=['pink','teal'])

sns.countplot(c_data[player_genders[3]],ax = axs[1,1],palette=['pink','teal'])

axs[0,0].set_title(player_genders[0])

axs[0,1].set_title(player_genders[1])

axs[1,0].set_title(player_genders[2])

axs[1,1].set_title(player_genders[3])



gender_l = []

age_l    = []

fast_round = []

chosen_sum = []

round_status =[]

choice = []



for index,row in c_data.iterrows():

    for n in np.arange(1,4):

        gender_l.append(row['p_'+str(n)+'_gender'])

    for n in np.arange(1,4):

        age_l.append(row['p_'+str(n)+'_age'])

    for n in np.arange(1,4):

        chosen_sum.append(row['p_'+str(n)+'_play_sum_chosen'])

    for n in np.arange(1,4):

        fast_round.append(row['p_'+str(n)+'_fast_round_sum'])

    for n in np.arange(2,4):

        round_status.append(row['p_'+str(n)+'_round_status'])

    for n in np.arange(1,4):

        choice.append(row['p_'+str(n)+'_choice'])     



p_data = pd.DataFrame({'Gender':gender_l,"Age":age_l,'fast_round_sum':fast_round,'chosen_sum':chosen_sum,'choice_type':choice})
plt.figure(figsize=(20,11))

ax = sns.countplot(p_data['Gender'],palette=['pink','teal'])

ax.set_xticklabels(['Female','Male'],fontsize=14)

ax.set_ylabel('Count',fontsize=19)

ax.set_xlabel('Gender',fontsize=19)

ax.set_title('Distribution Of Gender In The 3 Season Of The Show',fontsize=19)
plt.figure(figsize=(20,11))

ax = sns.distplot(p_data[p_data['Gender']==1]['fast_round_sum'],bins=10,label='Male',color='teal',hist=False,kde_kws={'lw':5})

ax = sns.distplot(p_data[p_data['Gender']==0]['fast_round_sum'],bins=10,label='Female',color='pink',hist=False,kde_kws={'lw':5})

ax.set_ylabel('Density',fontsize=19)

ax.set_xlabel('Amount Of Money Won On The Fast Round',fontsize=19)

ax.set_title('Distribution Fast Round Win Amounts Between Genders',fontsize=19)

plt.legend(prop={'size':20})
plt.figure(figsize=(20,11))

ax = sns.countplot(p_data['choice_type'],hue=p_data['Gender'],palette=['pink','teal'])

ax.set_ylabel('Counbt',fontsize=19)

ax.set_xlabel('Choice Type',fontsize=19)

ax.set_title('Distribution Choice Types When Offered Risk/Safety',fontsize=19)

ax.set_xticklabels(['Safety','Same Sum','Risk'],fontsize=16)

plt.legend(labels = ['Female','Male'],prop={'size':20})
plt.figure(figsize=(20,11))

ax = sns.distplot(p_data[p_data['Gender']==1]['Age'],bins=10,label='Male',color='teal',hist=False,kde_kws={'lw':5})

ax = sns.distplot(p_data[p_data['Gender']==0]['Age'],bins=10,label='Female',color='pink',hist=False,kde_kws={'lw':5})

ax.set_ylabel('Density',fontsize=19)

ax.set_xlabel('Age',fontsize=19)

ax.set_title('Distribution Of Age Between Genders',fontsize=19)

plt.legend(prop={'size':20})
#creating age groups 

p_data['Age_Group'] = pd.cut(p_data['Age'],bins=10,labels=['14-19','20 - 25','26-30','31-35','36-41','42-46','47-51','52-56','57-61','62-67',])

plt.figure(figsize=(20,11))

ax = sns.countplot(p_data['choice_type'],hue=p_data['Age_Group'])

ax.set_ylabel('Counbt',fontsize=19)

ax.set_xlabel('Choice Type',fontsize=19)

ax.set_title('Distribution Choice Types When Offered Risk/Safety Between Age Groups',fontsize=19)

ax.set_xticklabels(['Safety','Same Sum','Risk'],fontsize=16)

plt.legend(prop={'size':20})
p_data
plt.figure(figsize=(20,11))

ax = sns.boxplot(x=p_data['choice_type'],y=p_data['Age'],hue=p_data['Gender'],palette=['pink','teal'],linewidth=2.2)



ax.set_ylabel('Density',fontsize=19)

ax.set_xlabel('Choice',fontsize=19)

ax.set_xticklabels(['Safety','Same Sum','Risk'],fontsize=16)

ax.set_title('Distribution Choices Across Gender And Age',fontsize=19)

plt.legend(prop={'size':20})
plt.figure(figsize=(20,11))

ax = sns.boxplot(x=c_data['contestants_at_finals'],y=c_data['team_total_sum'],hue=c_data['game_result'],palette=['red','green'])

ax.set_title('Distribution of the total sum the team has made across number of contestans in finals hued by losing or winning the game',fontsize=16)

ax.set_xlabel('Number Of Teammates In The Final',fontsize=16)

ax.set_ylabel('Total Money Earned',fontsize=16)
plt.figure(figsize=(20,11))

ax = sns.boxplot(x=c_data['contestants_at_finals'],y=c_data['question_answered'],hue=c_data['game_result'],palette=['red','green'])

ax.set_title('Distribution of the number of questions answered across number of contestans in finals hued by losing or winning the game',fontsize=16)

ax.set_xlabel('Number Of Teammates In The Final',fontsize=16)

ax.set_ylabel('Number Of Question Answered',fontsize=16)
p_cors = p_data.corr('pearson')

plt.figure(figsize=(20,11))

ax = sns.heatmap(p_cors,annot=True,cmap='mako')
c_cors = c_data.corr('pearson')

for index,row in c_cors.iterrows():

    for en,value in enumerate(row):

        if value > 0.2 or value < -0.2:

            pass

        else:

            c_cors.at[index,c_cors.columns[en]] = 0



plt.figure(figsize=(20,11))

ax = sns.heatmap(c_cors,cmap='mako',annot=True)
#end of eda
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler,PolynomialFeatures

from sklearn.metrics import classification_report,mean_squared_error

from sklearn.model_selection import cross_val_score,train_test_split

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LinearRegression
y = c_data['game_result']

X = c_data[['question_answered','contestants_at_finals']]



train_x,test_x,train_y,test_y = train_test_split(X,y)



KNN_pipe = Pipeline(steps=[('scale',StandardScaler()),('knn',KNeighborsClassifier(n_neighbors=15))])

KNN_scores = cross_val_score(KNN_pipe,X,y,cv=5,scoring='f1')

KNN_scores.mean()



ADA_pipe = Pipeline(steps=[('scale',StandardScaler()),('knn',AdaBoostClassifier(n_estimators=5,learning_rate=0.03,random_state=42))])

ADA_scores = cross_val_score(ADA_pipe,X,y,cv=5,scoring='f1')

ADA_scores.mean()



RF_pipe = Pipeline(steps=[('scale',StandardScaler()),('knn',RandomForestClassifier(n_estimators=35,random_state=42))])

RF_scores = cross_val_score(RF_pipe,X,y,cv=5,scoring='f1')

RF_scores.mean()





plt.figure(figsize=(20,11))

ax = sns.lineplot(x=np.arange(0,5),y=KNN_scores,label='Knn CV F1 Scores')

ax = sns.lineplot(x=np.arange(0,5),y=ADA_scores,label='AdaBoost CV F1 Scores')

ax = sns.lineplot(x=np.arange(0,5),y=RF_scores,label='RandomForest CV F1 Scores')

ax.set_xticks(np.arange(0,5,1))
y = c_data['game_result']

X = c_data[['p_2_age','p_3_gender','p_4_gender']]



train_x,test_x,train_y,test_y = train_test_split(X,y)



KNN_pipe = Pipeline(steps=[('scale',StandardScaler()),('knn',KNeighborsClassifier(n_neighbors=5))])

KNN_scores = cross_val_score(KNN_pipe,X,y,cv=5,scoring='f1')

KNN_scores.mean()



ADA_pipe = Pipeline(steps=[('scale',StandardScaler()),('knn',AdaBoostClassifier(n_estimators=35,learning_rate=0.3,random_state=42))])

ADA_scores = cross_val_score(ADA_pipe,X,y,cv=5,scoring='f1')

ADA_scores.mean()



RF_pipe = Pipeline(steps=[('scale',StandardScaler()),('knn',RandomForestClassifier(n_estimators=35,random_state=42))])

RF_scores = cross_val_score(RF_pipe,X,y,cv=5,scoring='f1')

RF_scores.mean()





plt.figure(figsize=(20,11))

ax = sns.lineplot(x=np.arange(0,5,1),y=KNN_scores,label='Knn CV F1 Scores')

ax = sns.lineplot(x=np.arange(0,5,1),y=ADA_scores,label='AdaBoost CV F1 Scores')

ax = sns.lineplot(x=np.arange(0,5,1),y=RF_scores,label='RandomForest CV F1 Scores')

ax.set_xlabel('Fold Number')

ax.set_xticks(np.arange(0,5,1))