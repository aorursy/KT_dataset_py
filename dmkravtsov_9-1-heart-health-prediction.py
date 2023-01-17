# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
## loading dataset

df = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv',encoding='latin1')

## detailed df info

df.describe().T
df.head()
df.corr()
# df feature distribution before features tuning

def basic_details(df):

    b = pd.DataFrame()

    b['Missing value, %'] = round(df.isnull().sum()/df.shape[0]*100)

    b['N unique value'] = df.nunique()

    b['dtype'] = df.dtypes

    return b

basic_details(df)
## checking for outliers

import matplotlib.pyplot as plt

features = df.drop(['DEATH_EVENT', 'time', 'anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking'], axis=1).columns

for i in features:

    sns.boxplot(x="DEATH_EVENT",y=i,data=df)

    plt.title(i+" by "+"DEATH_EVENT")

    plt.show()
## checking for outliers

features = df.columns

for i in features:

    sns.distplot(df[i])

    plt.show()
df.platelets.sort_values()
## checking for outliers

features = df.columns

for i in features:

    sns.countplot(df[i])

    plt.show()
## let's drop this featute coz it's time of patient's observation before his/her removal or/and death

df=df.drop('time', axis=1)



# df['platelets_cat'] = pd.qcut(df['platelets'],q=[0, .25, .5, .75, 1], labels=False, precision=1)





df = df.drop(df[df['serum_creatinine'] > 6].index)

df['serum_sodium_cat'] = pd.qcut(df['serum_sodium'],q=[0, .33, .66, 1], labels=False, precision=1)





## all manipulations with df['age'],

# df['ejection_fraction'],

#df['creatinine_phosphokinase'] 

#(to cat, log1p, drop outliers) can't improve model score
# df feature distribution after features tuning:

def basic_details(df):

    b = pd.DataFrame()

    b['Missing value, %'] = round(df.isnull().sum()/df.shape[0]*100)

    b['N unique value'] = df.nunique()

    b['dtype'] = df.dtypes

    return b

basic_details(df)
df.info()
#Correlation with output variable

cor = df.corr()

cor_target = (cor['DEATH_EVENT'])

#Selecting highly correlated features

relevant_features = cor_target

relevant_features.sort_values(ascending = False).head(100)
import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



y = df.DEATH_EVENT

X = df.drop('DEATH_EVENT', axis=1)



x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=10,shuffle=True, stratify=y)



d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)





params = {

        'objective':'binary:logistic',

        'max_depth':5,

        'learning_rate':0.3,

        'eval_metric':'auc',

        'min_child_weight':1,

        'subsample':0.88,

        'colsample_bytree':0.8,

        'seed':29,

        'reg_lambda':2.8,

        'reg_alpha':0,

        'gamma':0,

        'scale_pos_weight':1,

        'nthread':-1

}



watchlist = [(d_train, 'train'), (d_valid, 'valid')]



nrounds=10000  

model = xgb.train(params, d_train, nrounds, watchlist, early_stopping_rounds=500,   

  

                           maximize=True, verbose_eval=10)



import seaborn as sns

pd.set_option('display.max_rows', 1000)

import matplotlib.pyplot as plt

fig,ax = plt.subplots(figsize=(15,20))

xgb.plot_importance(model,ax=ax,max_num_features=20,height=0.8,color='g')

#plt.tight_layout()

plt.show()
xgb.to_graphviz(model)
xgb.plot_importance(model)
import eli5

eli5.show_weights(model)
sub = pd.DataFrame()



sub['DEATH_EVENT'] = model.predict(d_valid)

sub['DEATH_EVENT'] = sub['DEATH_EVENT'].apply(lambda x: 1 if x>0.5 else 0)

acc = accuracy_score(y_valid, sub['DEATH_EVENT'])

print('The accuracy of the XGBOOST is ', acc)

import xgboost as xg

xgb = xg.XGBClassifier()

fit = xgb.fit(x_train, y_train)

fit.feature_importances_
xgb_fea_imp=pd.DataFrame(list(fit.get_booster().get_fscore().items()),

columns=['feature','importance']).sort_values('importance', ascending=False)

print('',xgb_fea_imp)
from sklearn.ensemble import RandomForestClassifier



model2 = RandomForestClassifier(min_samples_split=2, random_state=29)

model2.fit(x_train,y_train)

prediction=model2.predict(x_valid)

print('The accuracy of the Random Forest Classifier is', accuracy_score(prediction,y_valid))
feat_imp = pd.DataFrame(model2.feature_importances_)

feat_imp.index = pd.Series(df.iloc[:,:-1].columns)

feat_imp = (feat_imp*100).copy().sort_values(by=0,ascending=False)

feat_imp = feat_imp.reset_index()

feat_imp
import shap



# shap_values = shap.TreeExplainer(model2).shap_values(X)

# shap.summary_plot(shap_values, X)



row_to_show = 5

data_for_prediction = x_valid.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired

data_for_prediction_array = data_for_prediction.values.reshape(1, -1)





model2.predict_proba(data_for_prediction_array)
# Create object that can calculate shap values

explainer = shap.TreeExplainer(model2)



# Calculate Shap values

shap_values = explainer.shap_values(data_for_prediction)
shap.initjs()

shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)