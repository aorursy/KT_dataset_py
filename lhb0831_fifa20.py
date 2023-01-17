# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

'''Data Cleaning'''

df = pd.read_csv('../input/fifa-20-complete-player-dataset/players_20.csv')

df=df.dropna(subset=['club','nationality','age','short_name'])

df=df[df['value_eur']>1][df['wage_eur']>1][df['potential']>1][df['overall']>1]



'''Set the color palette'''

sns.set_style(style='darkgrid')

sns.set_context(context='poster',font_scale=0.5)

sns.set_palette(sns.color_palette(("muted")))
'''TOP n clubs or countries that have the best overall rating'''

def top_n_charts(field,n,data):

    sub_df = data[['short_name',field,'overall','player_positions']]

    # # Add a count index for the next move

    sub_df['count']=1

    ls=sub_df.groupby(field).sum()

    ls=ls[ls['count']>10]

    ls['overall_mean']=(ls['overall']/ls['count'])

    ls=ls.sort_values('overall_mean',ascending=False).reset_index()

    print(ls[:n])

    f,ax=plt.subplots(figsize=(20,7))

    sns.barplot(x=field,y='overall_mean',data=ls[:n])

    ax.set(ylim=(65,85))

    plt.show()

top_n_charts('club',10,df)

top_n_charts('nationality',10,df)
'''Extract the player's best position and we assume that the first position is the best'''

df_player=df.copy()

'''And classify and reduce them into 6 species'''

df_player['best_pos']=df_player['player_positions'].str.split(',').str[0]

df_player=df_player[df_player['best_pos']!='GK']

dict_pos={'ST':'Fwd_Center',

          'CF':'Fwd_Center',

          'LW':'Fwd_Winger',

          'RW':'Fwd_Winger',

          'LM':'Mid_Side',

          'RM':'Mid_Side',

          'CM':'Mid_Center',

          'CAM':'Mid_Center',

          'CDM':'Mid_Center',

          'CB':'Back_Center',

          'LB':'Back_Side',

          'RB':'Back_Side',

          'LWB':'Back_Winger',

          'RWB':'Back_Winger'}

df_player['best_pos']=df_player['best_pos'].map(dict_pos)

s=['Back_Center','Back_Side','Back_Winger','Fwd_Center','Fwd_Winger','Mid_Center','Mid_Side']



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report,confusion_matrix

from sklearn.neighbors import NearestNeighbors,KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB,BernoulliNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neural_network import MLPClassifier



cols=['pace','shooting','passing','dribbling','defending','physic']

for col in cols:

    df_player['n_'+col] = df_player[col]/df_player['overall']

req_col_basic=['n_'+col for col in cols]

x=df_player[req_col_basic]

y=df_player['best_pos']

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)



def display_confusion_matrix(y,y_hat,label,axs):

    res=confusion_matrix(y, y_hat)

    row_sums=res.astype(float).sum(axis=1)

    res=res/row_sums[:,np.newaxis]

    res=pd.DataFrame(res,columns=label,index=label)

    sns.heatmap(res,cmap='Blues',annot=True,ax=axs)



models=[]

models.append(("KNN", KNeighborsClassifier(n_neighbors=10)))

models.append(("GaussianNB", GaussianNB()))

models.append(("DecisionTreeGini", DecisionTreeClassifier()))

models.append(("SVM Classifier", SVC(C=10)))

models.append(("RandomForest", RandomForestClassifier(n_estimators=11, max_features=None)))

models.append(("Adaboost", AdaBoostClassifier(n_estimators=1000)))

models.append(("LogisticRegression", LogisticRegression(C=1000, tol=1e-10, solver="sag", max_iter=10000)))

models.append(("GBDT", GradientBoostingClassifier(max_depth=6, n_estimators=100)))

models.append(("NeuralNetClassifier", MLPClassifier(solver='lbfgs',alpha=1e-5,\

                                                    hidden_layer_sizes=(7,7,7),\

                                                    activation='relu',random_state=1)))

for i in range(int(len(models)/3)):

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 8))

    for j in range(3):

        clf_name,clf=models[i*3+j]

        clf.fit(X_train,Y_train)

        Y_pred=clf.predict(X_test)

        display_confusion_matrix(Y_test,Y_pred,s,ax[j])

        ax[j].set_title(clf_name)

            

#     '''The factors of being good at each position'''

#     cols=['pace','shooting','passing','dribbling','defending','physic']

#     for col in cols:

#         df_player['n_'+col] = df_player[col]/df_player['overall']

#     req_col_basic=['n_'+col for col in cols]

#     xy_combo=df_player[req_col_basic+['best_pos']]

#     xy_combo=xy_combo.groupby('best_pos').mean()

#     plt.figure()

#     sns.heatmap(xy_combo,cmap='Blues',annot=True)

    

plt.show()
cols=['attacking_crossing','attacking_finishing','attacking_heading_accuracy','attacking_short_passing',\

      'attacking_volleys','skill_dribbling','skill_curve','skill_fk_accuracy','skill_long_passing',\

      'skill_ball_control','movement_acceleration','movement_sprint_speed','movement_agility',\

      'movement_reactions','movement_balance','power_shot_power','power_jumping','power_stamina',\

      'power_strength','power_long_shots','mentality_aggression','mentality_interceptions',\

      'mentality_positioning','mentality_vision','mentality_penalties','mentality_composure',\

      'defending_marking','defending_standing_tackle','defending_sliding_tackle']

for col in cols:

    df_player['n_'+col] = df_player[col]/df_player['overall']

req_col_basic=['n_'+col for col in cols]

x=df_player[req_col_basic]

y=df_player['best_pos']

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)



for i in range(int(len(models)/3)):

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 8))

    for j in range(3):

        clf_name,clf=models[i*3+j]

        clf.fit(X_train,Y_train)

        Y_pred=clf.predict(X_test)

        display_confusion_matrix(Y_test,Y_pred,s,ax[j])

        ax[j].set_title(clf_name)
'''The factors of being good at each position'''

cols=['pace','shooting','passing','dribbling','defending','physic']

for col in cols:

    df_player['n_'+col] = df_player[col]/df_player['overall']

req_col_basic=['n_'+col for col in cols]

xy_combo=df_player[req_col_basic+['best_pos']]

xy_combo=xy_combo.groupby('best_pos').mean()

plt.figure()

sns.heatmap(xy_combo,cmap='Blues',annot=True)

plt.show()
def values_potential_relations(df):

    sub_df=df.copy()

    sub_df=sub_df[['short_name','age','overall','potential']]

    sub_df['difference']=sub_df['potential']-sub_df['overall']

    sub_df[['value_eur','wage_eur']]=df[['value_eur','wage_eur']]

    print(sub_df)

    sns.heatmap(sub_df.corr(),vmax=1,vmin=-1,cmap=sns.color_palette('RdBu',n_colors=128),annot=True)

    plt.show()

values_potential_relations(df)