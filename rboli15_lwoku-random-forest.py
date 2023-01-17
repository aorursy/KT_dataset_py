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



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



pd.options.display.max_columns = 100
#Load data

train = pd.read_csv("../input/learn-together/train.csv")

test = pd.read_csv("../input/learn-together/test.csv")



train.describe()

print(train.columns[train.isna().any()].tolist())

print(test.columns[test.isna().any()].tolist())

train.shape
#Check column datatypes

train.dtypes
train.head()
train.tail()
train.hist(figsize=(40,40),xrot=45)

#Create single soil field (reverse the one hot encoding)

soil_fields = ['Soil_Type'+ str(i) for i in range(1,31)]



train_soil = train[soil_fields]

train['Soil_Type'] = train_soil.idxmax(1)

train.head()
#Create single wilderness area field (reverse the one hot encoding)

Wilderness_Area_Fields = ['Wilderness_Area'+ str(i) for i in range(1,5)]



train_wilderness = train[Wilderness_Area_Fields]

train['Wilderness_Area'] = train_wilderness.idxmax(1)

train.head()
#plot categoricals

for feature in train.dtypes[train.dtypes == 'object'].index:

    plt.figure(figsize=(16, 12))

    sns.countplot(y=feature, data=train)

    sns.set()

    plt.show()
sns.catplot(hue="Wilderness_Area", y="Cover_Type", kind="count", data=train)

plt.show()
#Build contingency tables for Wilderness Area to see numerical relationship with Cover type

wild_cover_cont = pd.crosstab(index=train['Wilderness_Area'],columns=train['Cover_Type'])

wild_cover_cont

#Calculate chi-square for wilderness area and cover type

import scipy

wild_cover_c, wild_cover_p, wild_cover_dof, wild_cover_expected = scipy.stats.chi2_contingency(wild_cover_cont)

print('c: ' + str(wild_cover_c))

print('p: ' + str(wild_cover_p))

print('dof: ' + str(wild_cover_dof))

print('expected: ' + str(wild_cover_expected))
#Build contingency tables for Soil Type to see numerical relationship with Cover type

soil_cover_cont = pd.crosstab(index=train['Soil_Type'],columns=train['Cover_Type'])

soil_cover_cont
#Calculate chi-square for soil type and cover type

import scipy

soil_cover_c, soil_cover_p, soil_cover_dof, soil_cover_expected = scipy.stats.chi2_contingency(soil_cover_cont)

print('c: ' + str(soil_cover_c))

print('p: ' + str(soil_cover_p))

print('dof: ' + str(soil_cover_dof))

print('expected: ' + str(soil_cover_expected))
for i in range(1,10):

    sns.catplot(x="Cover_Type", y=train.columns[i], data=train)

    plt.title(train.columns[i])

    plt.show()
#Need to one hot encode covertype



one_hot_cov_type = pd.get_dummies(train.Cover_Type, prefix='cov_type')

one_hot_cov_type.head()





for i in range(1,10):

    for j in range(1,8):

        print(train.columns[i]+' cov_type_'+str(j) +' '+ str(scipy.stats.pointbiserialr(one_hot_cov_type['cov_type_'+str(j)], train[train.columns[i]]).correlation))
train.drop('Id', axis=1,inplace=True)

train.head()
# Import RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

#from learntools.core import *



# Create target object and call it y

y = train.Cover_Type

# Create X

X = train.drop(['Cover_Type','Soil_Type','Wilderness_Area'], axis=1)



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1, stratify=train.Cover_Type, test_size=0.2)

print(len(train_X), len(val_X), len(train_y),len(val_y))

#Build Pipelines

#pipeline = {'l1': make_pipeline(LogisticRegression(penalty='l1' , random_state=123)),

#            'l2': make_pipeline(LogisticRegression(penalty='l2' , random_state=123)),

#            'rf': make_pipeline(RandomForestClassifier(random_state=123)),

#            'gb': make_pipeline(GradientBoostingClassifier(random_state=123))

#}



pipeline = {'rf': make_pipeline(StandardScaler(), RandomForestClassifier(random_state=123))}

pipeline['rf'].get_params()
#Define hyperparameters to test



#Tested locally (crashes kaggle to run)

#rf_hyperparameters = {

#    'randomforestclassifier__n_estimators': [100, 500, 1000],

#    'randomforestclassifier__max_features': ['auto', 'sqrt', 0.33],

#    'randomforestclassifier__min_samples_split': [2, 5, 10, 15, 100],

#    'randomforestclassifier__max_depth': [5, 8, 15, 25, 30],

#    'randomforestclassifier__min_samples_leaf': [1, 2, 5, 10]

#}



#Identified best configuration

rf_hyperparameters = {

    'randomforestclassifier__n_estimators': [1000],

    'randomforestclassifier__max_features': [0.33],

    'randomforestclassifier__min_samples_split': [2],

    'randomforestclassifier__max_depth': [30],

    'randomforestclassifier__min_samples_leaf': [1]

}





hyperparameters = {'rf': rf_hyperparameters}
# Create empty dictionary called fitted_models

fitted_models = {}



# Loop through model pipelines, tuning each one and saving it to fitted_models

for name, pipeline in pipeline.items():





    # Create cross-validation object from pipeline and hyperparameters

    model = GridSearchCV(pipeline, hyperparameters[name], cv=10, n_jobs=1)

    

    # Fit model on X_train, y_train

    model.fit(train_X, train_y)

    

    # Store model in fitted_models[name] 

    fitted_models[name] = model

    

    # Print '{name} has been fitted'

    print(name, " has been fitted")
# Display best_score and parameters for each fitted model

for name, model in fitted_models.items():

    print(name, model.best_score_)

    print(name, model.best_params_)
# Retrain on full set of data

rf_model = RandomForestClassifier(random_state=0, n_estimators=1000, max_depth=30,max_features=.33,min_samples_leaf=1,min_samples_split=2)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_f1 = f1_score(val_y, rf_val_predictions, average='weighted')



#print("Validation f1 score for Random Forest Model: {:,.0f}".format(rf_val_f1))

print(rf_val_f1)