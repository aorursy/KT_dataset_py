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

import warnings

import seaborn as sns

from colorama import Fore, Back, Style 

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score

from mlxtend.plotting import plot_confusion_matrix

from plotly.offline import plot, iplot, init_notebook_mode

import plotly.graph_objs as go

from plotly.subplots import make_subplots

import plotly.express as px

from statsmodels.formula.api import ols

import plotly.graph_objs as gobj



init_notebook_mode(connected=True)

warnings.filterwarnings("ignore")

import plotly.figure_factory as ff



%matplotlib inline



import xgboost

import pandas as pd

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold

from sklearn.metrics import roc_auc_score

from xgboost import plot_importance
df = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

print(df.shape)

df.head()
df.isnull().sum()
categorical_columns = ['sex', 'smoking', 'diabetes', 'anaemia', 'high_blood_pressure', 'DEATH_EVENT']

real_number_columns = list(set(df.columns)-set(categorical_columns))
# support function for categorical columns analysis

def get_data_for_visualize_categorical_columns(data, column):

    '''

    Params:

        data: pd.DataFrame

    '''

    death_colum = 'DEATH_EVENT'

    unique_vals = data[column].unique().tolist()

    labels = []

    death_means = []

    survived_means = []

    label_template = column + '{}_{}'

    for val in unique_vals:

        temp_data = data[data[column]==val]

        labels.append(val)

        for death_event in data[death_colum].unique():

            # death

            if death_event == 1:

                death_means.append(temp_data[temp_data[death_colum]==death_event].shape[0])

            # survived

            else:

                survived_means.append(temp_data[temp_data[death_colum]==death_event].shape[0])

            #labels.append(label_template.format(val, death_event))

#     death_std = np.zeros(len(death_means))

#     survived_std = np.zeros(len(death_means))

    return death_means, survived_means, labels

            

    

    
death_means, survived_means, labels = get_data_for_visualize_categorical_columns(df, 'sex')

print(labels, death_means, survived_means)

for i in range(len(labels)):

    if labels[i]==1:

        labels[i] = 'Male'

    else:

        labels[i]='Female'

        

fig, ax = plt.subplots(2,3, figsize=(24,10))

width=0.35   

ax[0][0].bar(labels, death_means, width, label='death')

ax[0][0].bar(labels, survived_means, width, bottom=death_means, label='survived')

ax[0][0].set_ylabel('Number of patients')

ax[0][0].set_title('Stacked bar chart about relationship between gender and death event')

ax[0][0].legend()



ax[0][1].pie(death_means,labels=labels, autopct='%1.1f%%')

ax[0][1].set_title('Pie chart: probability of gender when a death event happened')



ax[0][2].pie(survived_means,labels=labels, autopct='%1.1f%%')

ax[0][2].set_title('Pie chart: probability of gender when a death event not happen')



ax[1][0].pie([death_means[0], survived_means[0]],labels=['Death', 'Survived'], autopct='%1.1f%%')

ax[1][0].set_title('Pie chart: probability of death event when patient is male')



ax[1][1].pie([death_means[1], survived_means[1]],labels=['Death', 'Survived'], autopct='%1.1f%%')

ax[1][1].set_title('Pie chart: probability of death event when patient is female')

plt.show()
death_means, survived_means, labels = get_data_for_visualize_categorical_columns(df, 'smoking')

print(labels, death_means, survived_means)

for i in range(len(labels)):

    if labels[i]==1:

        labels[i] = 'smoke'

    else:

        labels[i]='not smoke'

print(labels, death_means, survived_means)      

fig, ax = plt.subplots(2,3, figsize=(24,10))

width=0.35   

ax[0][0].bar(labels, death_means, width, label='death')

ax[0][0].bar(labels, survived_means, width, bottom=death_means, label='survived')

ax[0][0].set_ylabel('Number of patients')

ax[0][0].set_title('Stacked bar chart about relationship between smocking and death event')

ax[0][0].legend()



ax[0][1].pie(death_means,labels=labels, autopct='%1.1f%%')

ax[0][1].set_title('Pie chart: probability of smocking when a death event happened')



ax[0][2].pie(survived_means,labels=labels, autopct='%1.1f%%')

ax[0][2].set_title('Pie chart: probability of smocking when a death event not happen')



ax[1][0].pie([death_means[0], survived_means[0]],labels=['Death', 'Survived'], autopct='%1.1f%%')

ax[1][0].set_title('Pie chart: probability of death event when patient is not smocking')



ax[1][1].pie([death_means[1], survived_means[1]],labels=['Death', 'Survived'], autopct='%1.1f%%')

ax[1][1].set_title('Pie chart: probability of death event when patient is smocking')



plt.show()
death_means, survived_means, labels = get_data_for_visualize_categorical_columns(df, 'diabetes')

print(labels, death_means, survived_means)

for i in range(len(labels)):

    if labels[i]==1:

        labels[i] = 'diabetes'

    else:

        labels[i]='not diabetes'

print(labels, death_means, survived_means)      

fig, ax = plt.subplots(2,3, figsize=(24,10))

width=0.35   

ax[0][0].bar(labels, death_means, width, label='death')

ax[0][0].bar(labels, survived_means, width, bottom=death_means, label='survived')

ax[0][0].set_ylabel('Number of patients')

ax[0][0].set_title('Stacked bar chart about relationship between diabetes and death event')

ax[0][0].legend()



ax[0][1].pie(death_means,labels=labels, autopct='%1.1f%%')

ax[0][1].set_title('Pie chart: probability of diabetes when a death event happened')



ax[0][2].pie(survived_means,labels=labels, autopct='%1.1f%%')

ax[0][2].set_title('Pie chart: probability of diabetes when a death event not happen')



ax[1][0].pie([death_means[0], survived_means[0]],labels=['Death', 'Survived'], autopct='%1.1f%%')

ax[1][0].set_title('Pie chart: probability of death event when patient is not diabetes')



ax[1][1].pie([death_means[1], survived_means[1]],labels=['Death', 'Survived'], autopct='%1.1f%%')

ax[1][1].set_title('Pie chart: probability of death event when patient is diabetes')



plt.show()
death_means, survived_means, labels = get_data_for_visualize_categorical_columns(df, 'anaemia')

print(labels, death_means, survived_means)

for i in range(len(labels)):

    if labels[i]==1:

        labels[i] = 'anaemia'

    else:

        labels[i]='not anaemia'

print(labels, death_means, survived_means)      

fig, ax = plt.subplots(2,3, figsize=(24,10))

width=0.35   

ax[0][0].bar(labels, death_means, width, label='death')

ax[0][0].bar(labels, survived_means, width, bottom=death_means, label='survived')

ax[0][0].set_ylabel('Number of patients')

ax[0][0].set_title('Stacked bar chart about relationship between anaemia and death event')

ax[0][0].legend()



ax[0][1].pie(death_means,labels=labels, autopct='%1.1f%%')

ax[0][1].set_title('Pie chart: probability of anaemia when a death event happened')



ax[0][2].pie(survived_means,labels=labels, autopct='%1.1f%%')

ax[0][2].set_title('Pie chart: probability of anaemia when a death event not happen')



ax[1][0].pie([death_means[0], survived_means[0]],labels=['Death', 'Survived'], autopct='%1.1f%%')

ax[1][0].set_title('Pie chart: probability of death event when patient is not anaemia')



ax[1][1].pie([death_means[1], survived_means[1]],labels=['Death', 'Survived'], autopct='%1.1f%%')

ax[1][1].set_title('Pie chart: probability of death event when patient is anaemia')



plt.show()
death_means, survived_means, labels = get_data_for_visualize_categorical_columns(df, 'high_blood_pressure')

print(labels, death_means, survived_means)

for i in range(len(labels)):

    if labels[i]==1:

        labels[i] = 'high_blood_pressure'

    else:

        labels[i]='not high_blood_pressure'

print(labels, death_means, survived_means)      

fig, ax = plt.subplots(2,3, figsize=(24,10))

width=0.35   

ax[0][0].bar(labels, death_means, width, label='death')

ax[0][0].bar(labels, survived_means, width, bottom=death_means, label='survived')

ax[0][0].set_ylabel('Number of patients')

ax[0][0].set_title('Stacked bar chart about relationship between high_blood_pressure and death event')

ax[0][0].legend()



ax[0][1].pie(death_means,labels=labels, autopct='%1.1f%%')

ax[0][1].set_title('Pie chart: probability of high_blood_pressure when a death event happened')



ax[0][2].pie(survived_means,labels=labels, autopct='%1.1f%%')

ax[0][2].set_title('Pie chart: probability of high_blood_pressure when a death event not happen')



ax[1][0].pie([death_means[0], survived_means[0]],labels=['Death', 'Survived'], autopct='%1.1f%%')

ax[1][0].set_title('Pie chart: probability of death event when patient is high_blood_pressure')



ax[1][1].pie([death_means[1], survived_means[1]],labels=['Death', 'Survived'], autopct='%1.1f%%')

ax[1][1].set_title('Pie chart: probability of death event when patient is not high_blood_pressure')



plt.show()
# refer https://www.kaggle.com/nayansakhiya/heart-fail-analysis-and-quick-prediction-96-rate/notebook

surv = df[df["DEATH_EVENT"]==0]["age"]

not_surv = df[df["DEATH_EVENT"]==1]["age"]

hist_data = [surv,not_surv]

group_labels = ['Survived', 'Death']

fig = ff.create_distplot(hist_data, group_labels, bin_size=0.5)

fig.update_layout(

    title_text="Analysis in Age on Survival Status")

fig.show()
# refer https://www.kaggle.com/nayansakhiya/heart-fail-analysis-and-quick-prediction-96-rate/notebook



fig = px.violin(df, y="age", x="sex", color="DEATH_EVENT", box=True, points="all", hover_data=df.columns)

fig.update_layout(title_text="Analysis in Age and Gender on Survival Status")

fig.show()

#0: female, 1: male 
# create dataset 

cols = ['sex', 'smoking', 'diabetes', 'anaemia', 'high_blood_pressure', 'age', 'ejection_fraction',

              'platelets', 'serum_creatinine', 'serum_sodium']

# cols = ['ejection_fraction','serum_creatinine', 'serum_sodium','age', 'high_blood_pressure', 'platelets', 'sex']

y = df['DEATH_EVENT']



x = df[cols]

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3, random_state=1111)



# build xgboost model 

# params = {

#     'gamma': [0.5, 1, 1.5, 2, 5],

#     'subsample': [0.6, 0.8, 1.0],

#     'colsample_bytree': [0.6, 0.8, 1.0],

#     'max_depth': [3, 4, 5],

#     'learning_rate':[0.1, 0.01, 0.001], 

#     'n_estimators':[20, 50, 100, 300],

# }



folds = 3

param_comb = 5



# skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

clf = xgboost.XGBClassifier(

    eval_metric = 'auc',

    nthread = -1,

    max_depth=5,

    subsample=0.5,

    n_estimators=4,

    gamma=0.001,

#     random_state=3

)

clf.fit(x_train,y_train)



# grid_search = GridSearchCV(clf, param_grid=params,

#                             scoring='roc_auc', n_jobs=-1, 

#                             verbose=3)

# grid_search.fit(x_train,y_train)



pred = clf.predict(x_test)

print(accuracy_score(clf.predict(x_train), y_train))

print(accuracy_score(pred, y_test))

print(roc_auc_score(pred, y_test))

plot_importance(clf)



cm = confusion_matrix(y_test, pred)

plt.figure()

plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)

plt.title("Gredient Boosting Model - Confusion Matrix")

plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.show()