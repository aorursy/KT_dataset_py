

import numpy as np 

import pandas as pd 

import os

print(os.listdir("../input"))

import seaborn as sns

sns.set(rc={'figure.figsize':(18,10)})

data=pd.read_csv('../input/heart.csv')
data.head()
sns.set(rc={'figure.figsize':(18,10)})

equilibre=data['target'].value_counts()

ax=equilibre.plot.bar(title='target_count')

ax.set(xlabel='Notrhing/heart_disease', ylabel='Count')

print(equilibre)

sns.set(rc={'figure.figsize':(18,10)})

import matplotlib as plt

corr=data.corr()

ax = sns.heatmap(corr,cmap='coolwarm')

import lifelines

from lifelines import KaplanMeierFitter

from lifelines import CoxPHFitter
sns.set(rc={'figure.figsize':(18,10)})

kmf = KaplanMeierFitter()

kmf.fit(data['age'], data['target'], label="kmf.plot()")

ax=kmf.plot()

ax.set(xlabel='Age', ylabel='Probability_nothing')
cph = CoxPHFitter()

cph.fit(data, duration_col='age',  event_col='target')



estimation=cph.baseline_survival_

 

hazard=cph.baseline_cumulative_hazard_

print(cph.score_)

print(cph.summary)
sns.set(rc={'figure.figsize':(18,10)})

hazard['curve']=estimation.values

hazard['curve1']=hazard['curve']+(hazard['baseline hazard']/100)

hazard['curve2']=hazard['curve']-(hazard['baseline hazard']/100)



ax=hazard['curve'].plot(color='r',label='main_curve')

hazard['curve1'].plot(color='b',alpha=0.5,ax=ax,label='error_sup')

hazard['curve2'].plot(color='b',alpha=0.5,ax=ax,label='error_inf')

ax.set(xlabel='Age', ylabel='Probability_nothing')

ax.legend()
sns.set(rc={'figure.figsize':(18,10)})

cph.plot()
sns.set(rc={'figure.figsize':(18,10)})

total_1=data['fbs'].loc[data['target']==0].value_counts()

total_2=data['fbs'].loc[data['target']==1].value_counts()

df=pd.DataFrame({'nothing':total_2,'heart_disease':total_1})

ax=df.plot.bar(title='Target functiun of fbs',colormap='Accent')

ax.set(xlabel='Fbs', ylabel='Count')
sns.set(rc={'figure.figsize':(18,10)})

total_1=data['exang'].loc[data['target']==0].value_counts()

total_2=data['exang'].loc[data['target']==1].value_counts()

df=pd.DataFrame({'Nothing':total_2,'Heart_diseases':total_1})



a=df.plot.pie(subplots=True,colormap='Set1',autopct='%.0f%%',label='',title='angina')

total_1=data['slope'].loc[data['target']==0].value_counts()

total_2=data['slope'].loc[data['target']==1].value_counts()

df=pd.DataFrame({'Nothing':total_2,'Heart_diseases':total_1})

sns.set(rc={'figure.figsize':(18,10)})

a=df.plot.pie(subplots=True,colormap='Set1',autopct='%.0f%%',label='',title ='slope')

total_1=data['sex'].loc[data['target']==0].value_counts()

total_2=data['sex'].loc[data['target']==1].value_counts()

df=pd.DataFrame({'nothing':total_2,'Heart_diseases':total_1})



ax=df.plot.bar(colormap='Accent',label='',title ='male vs female')

ax.set(xlabel='male/female', ylabel='Count')
total_1=data[['sex','age','target']].loc[data['sex']==0]

total_2=data[['sex','age','target']].loc[data['sex']==1]



kmf = KaplanMeierFitter()

kmf.fit(total_1['age'], total_1['target'], label="kmf.plot()")

ax=kmf.plot(label='female')



kmf = KaplanMeierFitter()

kmf.fit(total_2['age'], total_2['target'], label="kmf.plot()")

kmf.plot(color='g',title='male vs female',ax=ax,label='male')



ax.set(xlabel='age', ylabel='Probability of good health')


total=data[['age','sex']].loc[data['target']==1]

total_1=data['age'].loc[data['sex']==0].value_counts()

total_2=data['age'].loc[data['sex']==1].value_counts()

sns.set(rc={'figure.figsize':(20,12)})





df=pd.DataFrame({'Male':total_2,'Female':total_1})

df.plot.bar(title='Distribution')

#total_2=data[['age','target']].loc[data['sex']==1]

#group1=total_2.groupby('age').size()

#group1.plot.bar(colormap='Accent',label='',title ='male and female by age',ax=ax,color='r',stack=True)

ax.set(xlabel='age', ylabel='Count')
data.head()

me=np.array([24,1,0,130,170,0,0,120,0,0.62,1,0,3,0])

data.loc[-1] = me
cph = CoxPHFitter()

cph.fit(data, duration_col='age',  event_col='target')

censored_subjects = data.loc[data['target'] == 0]
unconditioned_sf = cph.predict_survival_function(censored_subjects)

print(unconditioned_sf.head())
ax=unconditioned_sf[-1].plot(label='me')

unconditioned_sf[167].plot(color='r',ax=ax,label='random')

ax.set(xlabel='age', ylabel='probability of good health')

ax.legend()
from lifelines.utils import median_survival_times, qth_survival_times

predictions_75 = qth_survival_times(0.75, unconditioned_sf)

predictions_25 = qth_survival_times(0.25, unconditioned_sf)

predictions_50 = median_survival_times(unconditioned_sf)

import matplotlib.pyplot as plt



ax=unconditioned_sf[-1].plot(label='me')

unconditioned_sf[167].plot(color='y',label='random',ax=ax)



plt.axvline((predictions_75[-1].values), 0,1,color='g',label='75%')

plt.axvline((predictions_50[-1].values), 0,1,color='b',label='50%')

plt.axvline((predictions_25[-1].values), 0,1,color='r',label='25%')

ax.set(xlabel='age', ylabel='probability of good health')

ax.legend()
data_train=(data.loc[data['target']==1]).copy()

data_train.drop(['target'],1,inplace=True)

feature=[c for c in data_train.columns if c not in ['age']]

target=['age']



data_train.head()
train=data_train[:130]

val=data_train[130:]
from keras.layers import Activation, Dense, Dropout

from sklearn.metrics import mean_absolute_error

from keras.models import Sequential



NN_model = Sequential()

NN_model.add(Dense(32, kernel_initializer='normal'))

NN_model.add(Dense(1, kernel_initializer='normal'))    

NN_model.add(Activation('linear'))

NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])



history=NN_model.fit(train[feature].values, train[target].values, epochs=19, batch_size=10)





preds = NN_model.predict(val[feature].values) 

score = mean_absolute_error(val[target].values, preds)

print(score)
fig2, ax_loss = plt.subplots()

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.title('Model- Loss')

plt.legend(['Training', 'Validation'], loc='upper right')

plt.plot(history.history['loss'])

plt.plot(history.history['mean_absolute_error'])

plt.show()
me=np.array([1,0,130,170,0,0,120,0,0.62,1,0,3,0])

data_test=(data.loc[data['target']==0]).copy()

data_test.drop(['target'],1,inplace=True)

data_test.loc[-1] = me

estimation = NN_model.predict(data_test[feature].values)

estimation[-1]

print("A heart diseases will append at the age of: {}".format(estimation[-1]))
import lightgbm as lgb

feature=[c for c in data_train.columns if c not in ['target']]

target=data['target']
from sklearn.model_selection import StratifiedKFold

folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

predict = np.zeros(len(data))

feature_importance_df = pd.DataFrame()
param={

       'bagging_fraction': 0.33,

       'boost_from_average':'false',

       'boost': 'gbdt',

       'max_depth': -1,

       'metric':'auc',

       'objective': 'binary',

       'verbosity': 1

    }

from sklearn.metrics import roc_auc_score

for fold_, (trn_idx, val_idx) in enumerate(folds.split(data.values, target.values)):

    print("Fold {}".format(fold_))

    trn_data = lgb.Dataset(data.iloc[trn_idx][feature], label=target.iloc[trn_idx])

    val_data = lgb.Dataset(data.iloc[val_idx][feature], label=target.iloc[val_idx])



    num_round = 500

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 50)

    predict[val_idx] = clf.predict(data.iloc[val_idx][feature], num_iteration=clf.best_iteration)

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = feature

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    



print("CV score: {:<8.5f}".format(roc_auc_score(target, predict)))
feature_importance_df.head()
try1=feature_importance_df.groupby(['Feature'],as_index=False).mean()

try1.drop(['fold'],1,inplace=True)


sns.barplot(x="importance", y="Feature", data=try1.sort_values(by="importance", ascending=False))

plt.title('LightGBM Features (average_for_all_fold)')
me=np.array([[24,1,0,130,170,0,0,120,0,0.62,1,0,3]])

random=data.iloc[1][feature]

disease_me=clf.predict(me,num_iteration=clf.best_iteration)

disease_random=clf.predict(random,num_iteration=clf.best_iteration)

print("Your result is: {}".format(disease_me))

print("other result is: {}".format(disease_random))