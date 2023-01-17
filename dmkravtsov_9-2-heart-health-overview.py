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



def basic_analysis(df1, df2):

    '''the function compares the average values of  2 dataframes'''

    b = pd.DataFrame()

    b['First df_mean'] = round(df1.mean(),2)

    b['Second df_mean'] = round(df2.mean(),2)

    c = (b['First df_mean']/b['Second df_mean'])

    if [c<=1]:

        b['Variation, %'] = round((1-((b['First df_mean']/b['Second df_mean'])))*100)

    else:

        b['Variation, %'] = round(((b['First df_mean']/b['Second df_mean'])-1)*100)

        

    b['Influence'] = np.where(abs(b['Variation, %']) <= 9, "feature's effect on the target is not defined", 

                              "feature value affects the target")



    return b



def base_analysis(df1, df2, df3):

    '''the function compares the average values of  3  dataframes'''

    b = pd.DataFrame()

    b['First_df'] = round(df1.mean(),2)

    b['Second_df'] = round(df2.mean(),2)

    b['Third_df'] = round(df3.mean(),2)

    c = (b['First_df']/b['Third_df'])

    if [c<=1]:

        b['Variation, %'] = round((1-((b['First_df']/b['Third_df'])))*100)

    else:

        b['Variation, %'] = round(((b['First_df']/b['Third_df'])-1)*100)

        

    b['Influence'] = np.where(abs(b['Variation, %']) <= 9, "feature's value effect on survival is not defined", 

                              "feature value affects survival")



    return b





def basic_details(df):

    '''returns dataframe with feature distribution: missing values, unique values, its type'''



    b = pd.DataFrame()

    b['Missing value, %'] = round(df.isnull().sum()/df.shape[0]*100)

    b['N unique value'] = df.nunique()

    b['dtype'] = df.dtypes

    return b

## let's import dataset as df

df = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
# and checking for general info as for dataset....

df.head()
df.describe().T
basic_details(df)
survived = df.drop(df[df['DEATH_EVENT'] != 0].index)

not_survived = df.drop(df[df['DEATH_EVENT'] != 1].index)
import matplotlib.pyplot as plt

features = df.drop(['DEATH_EVENT', 'time', 'anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking'], axis=1).columns

for i in features:

    sns.boxplot(x="DEATH_EVENT",y=i,data=df)

    plt.title(i+" by "+"DEATH_EVENT")

    plt.show()
basic_analysis(survived,not_survived)
## let's drop this featute coz it's time of patient's observation before his/her removal or/and death

df=df.drop('time', axis=1)
smoking = df[df['smoking'] == 1]

non_smoking = df[df['smoking'] == 0]

basic_analysis(smoking,non_smoking)
df['platelets_cat'] = df['platelets'].map(lambda x: 0 if x <= 150000 

                                                            else (1 if 450000 >= x > 150000 

                                                                  else 2))

low_platelets = df[df['platelets_cat'] == 0]

norm_platelets = df[df['platelets_cat'] == 1]

high_platelets = df[df['platelets_cat'] == 2]  



base_analysis(low_platelets,norm_platelets,high_platelets)
import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score





X = df.drop('DEATH_EVENT', axis=1)

y = df.DEATH_EVENT



x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=10,shuffle=True, stratify=y)



d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)





params = {

        'objective':'binary:logistic',

        'max_depth':4,

        'learning_rate':0.3,

        'eval_metric':'auc',

        'min_child_weight':1,

        'subsample':0.85,

        'colsample_bytree':0.75,

        'seed':29,

        'reg_lambda':2.8,

        'reg_alpha':0,

        'gamma':0,

        'scale_pos_weight':1,

        'nthread':-1

}



watchlist = [(d_train, 'train'), (d_valid, 'valid')]



nrounds=10000  

model = xgb.train(params, d_train, nrounds, watchlist, early_stopping_rounds=600,   

  

                           maximize=True, verbose_eval=10)
accuracy = pd.DataFrame()

accuracy['predict'] = model.predict(d_valid)

accuracy['predict'] = accuracy['predict'].apply(lambda x: 1 if x>0.8 else 0)

accuracy_score(y_valid, accuracy['predict'])
xgb.plot_importance(model)
def spearman(frame, features):

    spr = pd.DataFrame()

    spr['feature'] = features

    spr['spearman'] = [frame[f].corr(frame['DEATH_EVENT'], 'spearman') for f in features]

    spr = spr.sort_values('spearman')

    plt.figure(figsize=(6, 0.25*len(features)))

    sns.barplot(data=spr, y='feature', x='spearman', orient='h')

    

features = df.columns

spearman(df, features)
xgb.to_graphviz(model)