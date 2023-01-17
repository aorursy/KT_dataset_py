import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt

import seaborn as sns

import plotly.io as pio

import plotly.graph_objects as go

import plotly.express as px
train = pd.read_csv('/kaggle/input/analytics-vidhya-hr-analytics-challenge/train_LZdllcl.csv')

test = pd.read_csv('/kaggle/input/analytics-vidhya-hr-analytics-challenge/test_2umaH9m.csv')

train.shape, test.shape
combine = train.append(test)

combine.shape
combine.head()
columns = {"KPIs_met >80%":"KPI","awards_won?":"AwardWon"}

combine = combine.rename(columns, axis=1)

combine.head()
combine.isnull().sum()
combine.dtypes
combine['KPI'].value_counts()
combine['KPI'] = combine['KPI'].replace(0,'Bad')

combine['KPI'] = combine['KPI'].replace(1,'Good')

combine['KPI'].value_counts()
combine['age'].describe()
bins= [20,30,40,50,60]

labels = ['Age_Tier1','Age_Tier2','Age_Tier3','Age_Tier4']

combine['age'] = pd.cut(combine['age'], bins=bins, labels=labels, right=False)

combine['age'].value_counts()
combine['avg_training_score'].describe()
combine['avg_training_score'] = np.log(combine['avg_training_score'])

combine['avg_training_score'].describe()
combine['department'].value_counts()
combine['education'].value_counts()
combine['education'].fillna("Unknown", inplace=True)

combine['education'] = combine['education'].replace("Bachelor's","Bachelor")

combine['education'] = combine['education'].replace("Master's & above","Masters")

combine['education'] = combine['education'].replace("Below Secondary","Level_2")

combine['education'].value_counts()
combine['region'].value_counts().count()
combine['gender'].value_counts()
combine['gender'] = combine['gender'].replace('m',"Male")

combine['gender'] = combine['gender'].replace('f',"Female")

combine['gender'].value_counts()
combine['length_of_service'].describe()
combine['length_of_service'].value_counts()
def get_length_of_service(years):

    switcher = {

        1: "Year_1",

        2: "Year_2",

        3: "Year_3",

        4: "Year_4",

        5: "Year_5",

        6: "Year_6",

        7: "Year_7",

        8: "Year_8",

        9: "Year_9",

        10: "Year_10"

    }

    return (switcher.get(years,"Year_10_Above"))



combine['length_of_service'] = combine['length_of_service'].apply(lambda x: get_length_of_service(x))

combine['length_of_service'].value_counts()
combine['no_of_trainings'].value_counts()
def get_training_number(training):

    switcher = {

        1: "Training_1",

        2: "Training_2"

    }

    return (switcher.get(training,"Training_Above_3"))



combine['no_of_trainings'] = combine['no_of_trainings'].apply(lambda x: get_training_number(x))

combine['no_of_trainings'].value_counts()
combine['previous_year_rating'].value_counts()
combine['previous_year_rating'].fillna(0.0, inplace=True)

combine['previous_year_rating'].value_counts()
combine['recruitment_channel'].value_counts()
combine['region'].value_counts()
combine['AwardWon'].value_counts()
combine['AwardWon'] = combine['AwardWon'].replace(0, "Not_Won")

combine['AwardWon'] = combine['AwardWon'].replace(1, "Won")

combine['AwardWon'].value_counts()
train_cleaned = combine[combine['is_promoted'].isnull()!=True].drop(['employee_id'], axis=1)

fig = px.parallel_categories(train_cleaned[['education','gender','age','is_promoted']], 

                             color="is_promoted", 

                             color_continuous_scale=px.colors.sequential.Inferno)

fig.show()
fig = px.parallel_categories(train_cleaned[['no_of_trainings','AwardWon','previous_year_rating','is_promoted']], 

                             color="is_promoted", 

                             color_continuous_scale=px.colors.sequential.Inferno)

fig.show()
combine = pd.get_dummies(combine)

combine.shape
combine.head()
X = combine[combine['is_promoted'].isnull()!=True].drop(['employee_id','is_promoted'], axis=1)

y = combine[combine['is_promoted'].isnull()!=True]['is_promoted']



X_test = combine[combine['is_promoted'].isnull()==True].drop(['employee_id','is_promoted'], axis=1)



X.shape, y.shape, X_test.shape
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
from lightgbm import LGBMClassifier

model = LGBMClassifier(boosting_type='gbdt',

                       max_depth=5,

                       learning_rate=0.08,

                       n_estimators=5000,

                       min_child_weight=0.01,

                       colsample_bytree=0.5,

                       random_state=2020)



model.fit(x_train,y_train,

          eval_set=[(x_train,y_train),(x_val, y_val.values)],

          eval_metric='auc',

          early_stopping_rounds=100,

          verbose=200)



pred_y = model.predict(x_val)
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score

print(f1_score(y_val, pred_y))

confusion_matrix(y_val,pred_y)
err = []

y_pred_tot_lgm = []



from sklearn.model_selection import StratifiedKFold



fold = StratifiedKFold(n_splits=10, shuffle=True,random_state=2020)

i = 1

for train_index, test_index in fold.split(X, y):

    x_train, x_val = X.iloc[train_index], X.iloc[test_index]

    y_train, y_val = y[train_index], y[test_index]

    m = LGBMClassifier(boosting_type='gbdt',

                       max_depth=5,

                       learning_rate=0.08,

                       n_estimators=5000,

                       min_child_weight=0.01,

                       colsample_bytree=0.5,

                       random_state=2020)

    m.fit(x_train, y_train,

          eval_set=[(x_train,y_train),(x_val, y_val)],

          early_stopping_rounds=200,

          eval_metric='f1_metric',

          verbose=200)

    pred_y = m.predict(x_val)

    print("err_lgm: ",f1_score(y_val, pred_y))

    err.append(f1_score(y_val, pred_y))

    pred_test = m.predict(X_test)

    i = i + 1

    y_pred_tot_lgm.append(pred_test)
np.mean(err,0)
submission = pd.DataFrame()

submission['employee_id'] = combine[combine['is_promoted'].isnull()==True]['employee_id']

# submission['is_promoted'] = np.round(np.mean(y_pred_tot_lgm, 0), 0).astype('int')

submission['is_promoted'] = y_pred_tot_lgm[4]

submission.to_csv('rfr_lrg.csv', index=False, header=True)

submission.shape