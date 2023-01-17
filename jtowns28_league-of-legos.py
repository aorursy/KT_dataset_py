# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv')

df.head()








#Let's first examine the correlation in our dataset



sns.heatmap(df.corr(),cmap=sns.diverging_palette(20, 220, n=200))

#Next, let's see how correlated each of these are correlated with whether Blue team wins:

plt.figure(figsize=(14,10))

sns.barplot(y=df.corr()['blueWins'],x=df.columns)

plt.xticks(rotation=70)



#It's clear some of these are very correlated.  Let's examine these more closely, removing all the cases where correlation <0.3
#Let's just add a few intuitive features

#Having a high number of kills can offset a high number of deaths.  This will give us a relative difference between kills and deaths

#

df['blueKD']=df['blueKills']-df['blueDeaths']



#redKD is redundant. A blueKD of 5 would imply a redKD of -5

#df['redKD']=df['redKills']-df['redDeaths']











#Let's also remove redKills and redDeaths.  These are redundant, since a blue kill corresponds to a red death

df=df.drop(['redKills'],axis=1)

df=df.drop(['redDeaths'], axis=1)



#The differentials for red Team are also redundant, since it is the difference between (one is blue-red, other is red-blue)

df=df.drop(['redGoldDiff'],axis=1)

df=df.drop(['redExperienceDiff'],axis=1)

#Let's go ahead and drop those features which are not very correlated



df.corr()['blueWins']

cols=df.columns

for i,j in enumerate(df.corr()['blueWins']):

    print (j)

    if abs(j) < 0.25:

        df=df.drop([cols[i]], axis=1)





df.head()

#I would like to make a barplot to examine the correlation. Some of these values need to be discretized. Let's look at redTotalGold:



sns.distplot(df['redTotalGold'])

plt.show()

#We will discretize into bins using qcut



def truncgraph(values):

    trunc=pd.qcut(values, 8)

    return trunc





#now, let's try to graph with a barplot





sns.barplot(y=df['blueWins'],x=truncgraph(df['redTotalGold']))

plt.xticks(rotation=70)

    



'''Clearly, if red team has very little gold after the first 10 minutes, they are very likely to lose. Conversely, if they get a lot of

gold, they are very likely to win. Let's see this for the remaining features.

'''

#Here, we can examine those effects. Here, both the distributions are plotted and the barplots of each factor corresponding to the winrate

#For instance, huge gold differences are great predictors, but typically the difference in gold is centered about zero.



len_plots=len(df.columns)



plt.figure()

#fig,axs=plt.subplots(17,2)

plt.figure(figsize=(20,100))

for i,j in enumerate(df.columns):

    #If there is a strong correlation:

    if abs(df[j]).mean() > 10:

        

        

        plt.subplot(len_plots,2,(2*i)+1)

        sns.distplot((df[j]))

        

        

        plt.subplot(len_plots,2,(2*i)+2)

        sns.barplot(truncgraph(df[j]),y=df['blueWins'])

        plt.xticks(rotation=10)

        

    else:

        plt.subplot(len_plots,2,(2*i)+1)

        sns.distplot((df[j]))

        plt.subplot(len_plots,2,(2*i)+2)

        sns.barplot(x=df[j],y=df['blueWins'])

        plt.xticks(rotation=10)

        #plt.show()

              





plt.show()
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split





#Now, let's see if we can predict who is going to win the game based off the first 10 minutes of gameplay.



#Split into X and y

X=df.drop(['blueWins'], axis=1)

y=df['blueWins']



#do a 80:20 train:test split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=13)





#Scale the X values

scaler=StandardScaler().fit(X_train)

X_train_scaled=scaler.transform(X_train)

X_test_scaled=scaler.transform(X_test)









from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV





#I am going to use XGBoost for the learner. I am going to perform a 3-fold cross validation on the training set

#to determine an decent set of hyperparameters



#I have gone ahead and skipped the hyperparameter tuning part to make the notebook run faster, which has already been performed

#Feel free to uncomment it 



def hyperParameterTuning(X_train, y_train):

    param_tuning = {

        'reg_lambda': np.linspace(.01,2,10),

        

        'max_depth': [2,3,4],

        'min_child_weight': [1,3,5],

        'eta': [ 0.05 ],

        'subsample': [ 0.8,1],

        'colsample_bytree': [0.8,1],

        'n_estimators' : [200],

   

    }



    xgb_model = XGBClassifier()



    gsearch = GridSearchCV(estimator = xgb_model,

                           param_grid = param_tuning,                        

                           scoring = 'accuracy', 

                           cv = 3,

                           n_jobs = 1,

                           verbose = 4)



    gsearch.fit(X_train,y_train)



    return gsearch#.best_params_,gsearch.best_estimator



#model=hyperParameterTuning(X_train_scaled,y_train).best_estimator_

#print (model)

model=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=1, eta=0.05, gamma=0,

              gpu_id=-1, importance_type='gain', interaction_constraints='',

              learning_rate=0.0500000007, max_delta_step=0, max_depth=2,

              min_child_weight=5,  

              n_estimators=200, n_jobs=0, num_parallel_tree=1, random_state=0,

              reg_alpha=0, reg_lambda=0.01, scale_pos_weight=1, subsample=0.8,

              tree_method='exact', validate_parameters=1, verbosity=None)



model.fit(X_train_scaled,y_train)
#Here's the best estimator







from sklearn.metrics import plot_confusion_matrix

confusion_matrix=plot_confusion_matrix(model,X_test_scaled,y_test, normalize='true')
#Let's see our explanatory factors on what would lead to a victory or loss

plt.figure(figsize=(14,10))

sns.barplot(y=model.feature_importances_, x=X_train.columns)

plt.xticks(rotation=80)