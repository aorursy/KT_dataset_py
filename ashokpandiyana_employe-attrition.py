# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # for visualization

import seaborn as sns # for visualization



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

from datetime import datetime

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, learning_curve, train_test_split

from sklearn.metrics import precision_score, roc_auc_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve, accuracy_score

import xgboost as xgb

import warnings

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import plotly.figure_factory as ff



warnings.filterwarnings('ignore')
data = pd.read_csv("/kaggle/input/ibm-data/ibm.csv")
data.head()
print(f"The Data Set has {data.shape[0]} rows and {data.shape[1]} columns")
data.info()
data.describe(include='all').T
data.describe()
data.isnull().sum()
display(data.isnull().any())
data_model = data.copy()
data_model.columns
data_model['Attrition'].map(dict(Yes=1, No=0))
f,ax = plt.subplots(figsize=(20, 20))

sns.heatmap(data_model.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
data_model = data_model.drop(columns=['StandardHours','EmployeeCount','Over18'])
attrition = data_model[(data_model['Attrition'] != 0)]

no_attrition = data_model[(data_model['Attrition'] == 0)]



#------------COUNT-----------------------

trace = go.Bar(x = (len(attrition), len(no_attrition)), y = ['Yes_attrition', 'No_attrition'], orientation = 'h', opacity = 0.8, marker=dict(

        color=['gold', 'lightskyblue'],

        line=dict(color='#000000',width=1.5)))



layout = dict(title =  'Count of attrition variable')

                    

fig = dict(data = [trace], layout=layout)

py.iplot(fig)



#------------PERCENTAGE-------------------

trace = go.Pie(labels = ['No_attrition', 'Yes_attrition'], values = data_model['Attrition'].value_counts(), 

               textfont=dict(size=15), opacity = 0.8,

               marker=dict(colors=['lightskyblue','gold'], 

                           line=dict(color='#000000', width=1.5)))





layout = dict(title =  'Distribution of attrition variable')

           

fig = dict(data = [trace], layout=layout)

py.iplot(fig)
p = sns.countplot(data=data_model,

                  x = 'Attrition')#,

                  #hue = 'islong')
attrition = data_model.copy()
f, axes = plt.subplots(3, 3, figsize=(10, 8), 

                       sharex=False, sharey=False)



# Defining our colormap scheme

s = np.linspace(0, 3, 10)

cmap = sns.cubehelix_palette(start=0.0, light=1, as_cmap=True)



# Generate and plot

x = attrition['Age'].values

y = attrition['TotalWorkingYears'].values

sns.kdeplot(x, y, cmap=cmap, shade=True, cut=5, ax=axes[0,0])

axes[0,0].set( title = 'Age against Total working years')



cmap = sns.cubehelix_palette(start=0.333333333333, light=1, as_cmap=True)

# Generate and plot

x = attrition['Age'].values

y = attrition['DailyRate'].values

sns.kdeplot(x, y, cmap=cmap, shade=True, ax=axes[0,1])

axes[0,1].set( title = 'Age against Daily Rate')



cmap = sns.cubehelix_palette(start=0.666666666667, light=1, as_cmap=True)

# Generate and plot

x = attrition['YearsInCurrentRole'].values

y = attrition['Age'].values

sns.kdeplot(x, y, cmap=cmap, shade=True, ax=axes[0,2])

axes[0,2].set( title = 'Years in role against Age')



cmap = sns.cubehelix_palette(start=1.0, light=1, as_cmap=True)

# Generate and plot

x = attrition['DailyRate'].values

y = attrition['DistanceFromHome'].values

sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[1,0])

axes[1,0].set( title = 'Daily Rate against DistancefromHome')



cmap = sns.cubehelix_palette(start=1.333333333333, light=1, as_cmap=True)

# Generate and plot

x = attrition['DailyRate'].values

y = attrition['JobSatisfaction'].values

sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[1,1])

axes[1,1].set( title = 'Daily Rate against Job satisfaction')



cmap = sns.cubehelix_palette(start=1.666666666667, light=1, as_cmap=True)

# Generate and plot

x = attrition['YearsAtCompany'].values

y = attrition['JobSatisfaction'].values

sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[1,2])

axes[1,2].set( title = 'Daily Rate against distance')



cmap = sns.cubehelix_palette(start=2.0, light=1, as_cmap=True)

# Generate and plot

x = attrition['YearsAtCompany'].values

y = attrition['DailyRate'].values

sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[2,0])

axes[2,0].set( title = 'Years at company against Daily Rate')



cmap = sns.cubehelix_palette(start=2.333333333333, light=1, as_cmap=True)

# Generate and plot

x = attrition['RelationshipSatisfaction'].values

y = attrition['YearsWithCurrManager'].values

sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[2,1])

axes[2,1].set( title = 'Relationship Satisfaction vs years with manager')



cmap = sns.cubehelix_palette(start=2.666666666667, light=1, as_cmap=True)

# Generate and plot

x = attrition['WorkLifeBalance'].values

y = attrition['JobSatisfaction'].values

sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[2,2])

axes[2,2].set( title = 'WorklifeBalance against Satisfaction')



f.tight_layout()

numerical = [u'Age', u'DailyRate', u'DistanceFromHome', 

             u'Education', u'EmployeeNumber', u'EnvironmentSatisfaction',

             u'HourlyRate', u'JobInvolvement', u'JobLevel', u'JobSatisfaction',

             u'MonthlyIncome', u'MonthlyRate', u'NumCompaniesWorked',

             u'PercentSalaryHike', u'PerformanceRating', u'RelationshipSatisfaction',

             u'StockOptionLevel', u'TotalWorkingYears',

             u'TrainingTimesLastYear', u'WorkLifeBalance', u'YearsAtCompany',

             u'YearsInCurrentRole', u'YearsSinceLastPromotion',u'YearsWithCurrManager']

data = [

    go.Heatmap(

        z= attrition[numerical].astype(float).corr().values, # Generating the Pearson correlation

        x=attrition[numerical].columns.values,

        y=attrition[numerical].columns.values,

        colorscale='Viridis',

        reversescale = False,

#         text = True ,

        opacity = 1.0

        

    )

]





layout = go.Layout(

    title='Pearson Correlation of numerical features',

    xaxis = dict(ticks='', nticks=36),

    yaxis = dict(ticks='' ),

    width = 900, height = 700,

    

)





fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='labelled-heatmap')
attrition.columns
# Refining our list of numerical variables

numerical = [u'Age', u'DailyRate',  u'JobSatisfaction',

       u'MonthlyIncome', u'PerformanceRating',

        u'WorkLifeBalance', u'YearsAtCompany', u'Attrition']
categorical = []

for col, value in attrition.iteritems():

    if value.dtype == 'object':

        categorical.append(col)
numerical = attrition.columns.difference(categorical)
numerical
attrition_cat = attrition[categorical]

attrition_cat = attrition_cat.drop(['Attrition'], axis=1) # Dropping the target column
attrition_cat 
#Factorize Columns 

def Encode(data):

    for column in data.columns:

        data[column] = data[column].factorize()[0]

    return data



attrition_cat_1 = Encode(attrition_cat.copy())
attrition_cat_1
attrition_num = attrition[numerical]
attrition_num
attrition_final = pd.concat([attrition_num, attrition_cat_1], axis=1)
attrition_final
# Define a dictionary for the target mapping

target_map = {'Yes':1, 'No':0}

# Use the pandas apply method to numerically encode our attrition target variable

target = attrition["Attrition"].apply(lambda x: target_map[x])
target
# Import the train_test_split method

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit



# Split data into train and test sets as well as for validation and testing

train, test, target_train, target_val = train_test_split(attrition_final, 

                                                         target, 

                                                         train_size= 0.80,

                                                         random_state=0);

#train, test, target_train, target_val = StratifiedShuffleSplit(attrition_final, target, random_state=0);
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (accuracy_score, log_loss, classification_report)

from imblearn.over_sampling import SMOTE

import xgboost

oversampler=SMOTE(random_state=0)

smote_train, smote_target = oversampler.fit_sample(train,target_train)
seed = 0   # We set our random seed to zero for reproducibility

# Random Forest parameters

rf_params = {

    'n_jobs': -1,

    'n_estimators': 1000,

#     'warm_start': True, 

    'max_features': 0.3,

    'max_depth': 4,

    'min_samples_leaf': 2,

    'max_features' : 'sqrt',

    'random_state' : seed,

    'verbose': 0

}
rf = RandomForestClassifier(**rf_params)
rf.fit(smote_train, smote_target)
rf_predictions = rf.predict(test)
print("Accuracy score: {}".format(accuracy_score(target_val, rf_predictions)))

print("="*80)

print(classification_report(target_val, rf_predictions))
trace = go.Scatter(

    y = rf.feature_importances_,

    x = attrition_final.columns.values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 13,

        #size= rf.feature_importances_,

        #color = np.random.randn(500), #set color equal to a variable

        color = rf.feature_importances_,

        colorscale='Portland',

        showscale=True

    ),

    text = attrition_final.columns.values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Random Forest Feature Importance',

    hovermode= 'closest',

     xaxis= dict(

         ticklen= 5,

         showgrid=False,

        zeroline=False,

        showline=False

     ),

    yaxis=dict(

        title= 'Feature Importance',

        showgrid=False,

        zeroline=False,

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2010')

gb_params ={

    'n_estimators': 1500,

    'max_features': 0.9,

    'learning_rate' : 0.25,

    'max_depth': 4,

    'min_samples_leaf': 2,

    'subsample': 1,

    'max_features' : 'sqrt',

    'random_state' : seed,

    'verbose': 0

}
gb = GradientBoostingClassifier(**gb_params)

# Fit the model to our SMOTEd train and target

gb.fit(smote_train, smote_target)

# Get our predictions

gb_predictions = gb.predict(test)
print(accuracy_score(target_val, gb_predictions))

print(classification_report(target_val, gb_predictions))
trace = go.Scatter(

    y = gb.feature_importances_,

    x = attrition_final.columns.values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 13,

        #size= rf.feature_importances_,

        #color = np.random.randn(500), #set color equal to a variable

        color = gb.feature_importances_,

        colorscale='Portland',

        showscale=True

    ),

    text = attrition_final.columns.values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Gradient Boosting Model Feature Importance',

    hovermode= 'closest',

     xaxis= dict(

         ticklen= 5,

         showgrid=False,

        zeroline=False,

        showline=False

     ),

    yaxis=dict(

        title= 'Feature Importance',

        showgrid=False,

        zeroline=False,

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter')