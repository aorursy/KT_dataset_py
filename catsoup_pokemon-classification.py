#graphs and utilities

import os

import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



#data pre-processing and model evaluation

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder



#models

import xgboost as xgb

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import accuracy_score



def plotMissing(df):

    #will error out if there is no missing data, just for quick use

    sns.set_style("whitegrid")

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    missing.sort_values(inplace=True,ascending=False)

    missing.plot.bar()



df_full = pd.read_csv('../input/Pokemon.csv')

df_full.head()
plotMissing(df_full)
battle_stats = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']



#make a column with both types combined to make their full typing

df_full['FullType'] = df_full['Type 1'] + df_full['Type 2']

#if the pokemon only has one type, it needs fixing since nan + string = nan

df_full['FullType'].fillna(df_full['Type 1'],inplace=True)



def makeTarget(df):

    local_df = df_full.iloc[df]

    if local_df['Type 1'] == 'Electric' or local_df['Type 2'] == 'Electric':

        return 1

    return 0



#make a target which tells us if the pokemon is Electric type or not (1 = electric, 0 = not electric)

df_full['Target'] = [makeTarget(x) for x in df_full.index]



#make a column with all the battle stats added together (included on the data but just for knowledge)

#df_full['TotalStat'] = df_full[battle_stats].sum(axis=1)



#bar plot the most common first type (overlaps into one single plot)

#df_types = df_full['Type 1'].value_counts()

#df_types.plot.bar()

#df_types2 = df_full['Type 2'].value_counts()

#df_types.plot.bar()



sns.set_style("whitegrid")

fig, ax =plt.subplots(1,2,figsize=(14,5))

sns.countplot(df_full['Type 1'].sort_values(), ax=ax[0],order = df_full['Type 1'].value_counts().index)

sns.countplot(df_full['Type 2'].sort_values(), ax=ax[1], order = df_full['Type 2'].value_counts().index)

fig.autofmt_xdate()

fig.show()
#make a DF with the highest single stats

df_top_stats = pd.DataFrame(columns=df_full.columns)



for idx,battle_stat in enumerate(battle_stats):

    top_stat = df_full.nlargest(1,battle_stat).sort_values(battle_stat,ascending=False)

    df_top_stats = pd.concat([df_top_stats, top_stat], axis=0)

    

df_top_stats.head()
#show that the target (electric types = 1, other type = 0) has been applied correctly

check_df = pd.DataFrame()

check_df = check_df.append(df_full[['Type 1','Type 2','Name','Target']].iloc[[30,157,0,9]])

check_df
df_full[df_full['Legendary'] == False].nlargest(10,'Total').sort_values('Total',ascending=False)
#Tornadus is the only pure flying type pokemon (2 forms)

pure_flying = df_full.iloc[df_full.loc[df_full['FullType'] == 'Flying'].index]

pure_flying
#build up a type list and include a type for none

type_list = df_full['Type 1'].unique().tolist()

type_list.append('none')



#replace NaN for 'Type 2' with 'none' so it will work with the LabelEncoder

df_full['Type 2'].fillna('none',inplace=True)



#use the label encoder to convert the type names, e.g 'Grass' into integers

le = LabelEncoder()

le.fit(type_list)

LabelEncoder()



#'Type 1' and 'Type 2'not included in the end because this is what we are predicting

#but I'm keeping this in as an example of LabelEncoder

df_full['Type 1'] = le.transform(df_full['Type 1'])

df_full['Type 2']  = le.transform(df_full['Type 2'])



features = ['Total','HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','Generation','Legendary']

X = df_full[features]

y = df_full['Target']

train_X, val_X, train_y, val_y = train_test_split(X,y,random_state=0)
def get_score(model):

    """ Return the mean accuracy of the classifiers over the 5 folds"""

    my_pipeline = Pipeline(steps=[

    ('preprocessor', SimpleImputer()),

    ('model', model)

    ])

    

    scores = cross_val_score(my_pipeline, train_X, train_y,cv=5,scoring='accuracy')

    return scores.mean()



#list all the valid scoring methods on this cross_val_score function:

#import sklearn.metrics

#sorted(sklearn.metrics.SCORERS.keys())
#parameters shared by many of these models

n_estimators=200

seed=0



svm = SVC(random_state=0,kernel='linear',C=0.025)

rfc = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)

adc = AdaBoostClassifier(n_estimators=n_estimators,random_state=seed)

gbc = GradientBoostingClassifier(n_estimators=n_estimators,random_state=seed)

etc = ExtraTreesClassifier(n_estimators=n_estimators,random_state=seed)

xgbmodel = xgb.XGBClassifier(n_estimators= 2000,max_depth= 4,min_child_weight= 2,gamma=0.9,\

                        subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic',\

                        nthread= -1,scale_pos_weight=1)



#models = [svm,rfc,adc,gbc,etc,xgbmodel]

models = [('Svm',svm),('Random Forest',rfc),('adaboost',adc),('Gradient Boost',gbc),\

          ('Extra Trees',etc),('Extreme Gradient Boost',xgbmodel)]





df_scores = pd.DataFrame(columns=['Name','Accuracy Score'])



for idx,model in enumerate(models):

    score = get_score(model[1])

    df_scores.loc[idx] = [model[0],score]



df_scores.sort_values('Accuracy Score',inplace=True,ascending=False)

df_scores.head(6)
#trying a few of these models on the 'full' Train/Validate set

final_extra_trees_model = ExtraTreesClassifier(n_estimators=n_estimators,random_state=seed)

final_extra_trees_model.fit(train_X,train_y)

final_predictions = final_extra_trees_model.predict(val_X)

print(accuracy_score(final_predictions,val_y))



final_rfc = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)

final_rfc.fit(train_X,train_y)

final_rfc_pred = final_rfc.predict(val_X)

print(accuracy_score(final_rfc_pred,val_y))



final_xgb = xgb.XGBClassifier(n_estimators= 2000,max_depth= 4,min_child_weight= 2,gamma=0.9,\

                        subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic',\

                        nthread= -1,scale_pos_weight=1)



final_xgb.fit(train_X,train_y)

final_xgb_pred = final_xgb.predict(val_X)

print(accuracy_score(final_xgb_pred,val_y))