# Flattens stacked grouped columns

def flatten(dataframe):

    dataframe.columns = [' '.join(col).strip() for col in dataframe.columns.values]

    return dataframe



# Object to extend the functionality of the ML models

class SklearnHelper(object):

    def __init__(self, clf, seed=0, params=None):

        params['random_state'] = seed

        self.clf = clf(**params)

        self.__name__ = clf.__name__



    def train(self, x_train, y_train):

        self.clf.fit(x_train, y_train)



    def predict(self, x):

        return self.clf.predict(x)

    

    def fit(self,x,y):

        return self.clf.fit(x,y)

    

    def cv_score(self,x_train,y_train):

        array = cross_val_score(self.clf,x_train,y_train, cv = 5,scoring = 'accuracy')

        return array.mean()

    

    def feature_importances(self,x,y):

        return self.clf.fit(x,y).feature_importances_
# Importing datatable modules

import numpy as np

import pandas as pd



# Importing Graphing Modules

import plotly_express as px

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot



# Importing ML models/metrics

from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier

from xgboost import XGBClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

from sklearn.model_selection import cross_val_score



# Importing other useful libraries

import os

from collections import defaultdict



# Printing list of files in input folder

print(os.listdir("../input"))



# Initializing plotly offline

init_notebook_mode(connected=True)
# Loading datasets 

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

full_dataset = [train,test]
# Printing dataset information

print("Training Dataset Information:")

print(train.info())

print("\nTest Dataset Information:")

print(test.info())
# Characterizing null values

print("Training Null Values (%):")

print(train.isnull().sum()*100/train.shape[0])

print("\n")

print("Test Null Values (%):")

print(test.isnull().sum()*100/test.shape[0])
# Grouping by survival rate and standard deviation and plotting results

pclass_grouped = train[['Pclass','Survived']].groupby('Pclass', as_index=False).agg({'Survived':['mean','std']})

pclass_grouped = flatten(pclass_grouped)

fig = px.bar(pclass_grouped,x = "Pclass", y = "Survived mean", color = "Pclass", error_y = "Survived std")

fig.update_traces(error_y_color = "black")
# Looking at survival by gender

sex_grouped = train[['Sex','Survived']].groupby('Sex',as_index=False).agg({'Survived':['mean','std']})

sex_grouped = flatten(sex_grouped)

fig = px.bar(sex_grouped,x = "Sex",y = "Survived mean", error_y = "Survived std", color = "Sex")

fig.update_traces(error_y_color = "black")
# Age purely versus survival

px.histogram(train,x = "Age", opacity = 0.7, color = "Survived")
# Looking at age distribution and Pclass relationship

px.histogram(train, x = "Age", y = "Name", color = "Survived", facet_row = "Pclass", labels = dict(Name = "People"), opacity = 0.7)
# Plotting distribution of fare on a log graph

px.histogram(train, x = "Fare", log_y = True, color = "Survived", opacity = 0.7)
# Plotting distribution of fares by class

px.histogram(train, x = "Fare", log_y = True, facet_col = "Pclass", color = "Pclass")
# Plotting embarkation point by survival rate

emb_grouped = train[['Embarked','Survived']].groupby('Embarked',as_index=False).agg({'Survived':['mean','std']})

emb_grouped = flatten(emb_grouped)

fig = px.bar(emb_grouped,x = "Embarked",y = "Survived mean", error_y = "Survived std", color = "Embarked")

fig.update_traces(error_y_color = "black")
# Checking correlation between features to understand relative trends/comparisons before feature engineering

z = train.corr()

trace = go.Heatmap(

    z = z,

    x = z.columns,

    y = z.columns

)

iplot([trace])
# Combining SibSp and Parch in one column since they're highly correlated

for dataset in full_dataset:

    dataset['Family_Size'] = dataset['SibSp'] + dataset['Parch'] + 1
# Grouping by family size and plotting

grouped_family_size = train[['Family_Size','Survived','Name']].groupby(['Family_Size','Survived']).count().reset_index()

grouped_family_size.columns = ["Family_Size","Survived","Name"]

px.bar(grouped_family_size, x = "Family_Size", y = "Name", color = "Survived", barmode = "group", labels = dict(Name = "Count"))
# Cutting data

family_bins = [0, 1, 4, 20]

family_labels = [1, 2, 3]

for dataset in full_dataset:

    dataset['Family_Cat'] = pd.cut(dataset['Family_Size'], bins = family_bins, labels = family_labels, include_lowest = True)
# Re-visualizing the grouping

grouped_family_size = train[['Family_Cat','Survived','Name']].groupby(['Family_Cat','Survived']).count().reset_index()

grouped_family_size.columns = ["Family_Cat","Survived","Name"]

px.bar(grouped_family_size, x = "Family_Cat", y = "Name", color = "Survived", barmode = "group", labels = dict(Name = "Count"))
# Visualizing a few name variables

train['Name'].head(10)
# Extracting name information and storing it in the 'Title' column

for dataset in full_dataset:

    dataset['Title'] = dataset.Name.str.extract(r"([A-Za-z]+)\.", expand = False)
# Printing all unique titles

set(train['Title'].unique()) | set(test['Title'].unique())
# Replacing values with the following mappings

title_map = {'Rare': [ 'Capt', 'Col','Countess','Dr','Jonkheer','Lady','Major','Rev','Sir'],

             'Mr': ['Mr','Don'],

             'Mrs':['Mme','Mrs','Dona'],

             'Miss':['Ms', 'Miss','Mlle'],

             'Master':['Master']}



for dataset in full_dataset:

    for key, value in title_map.items():

        dataset['Title'] = dataset['Title'].replace(value,key)
# Printing new set of unique values to make sure that we didn't miss anything

unique_titles = list(set(train['Title'].unique()) | set(test['Title'].unique()))

print("Unique Titles:",unique_titles)
# Categorizing the titles into buckets and printing the mapping

title_mapping = dict(zip(unique_titles,list(range(1,6))))

print("Title Mapping:",title_mapping)

for dataset in full_dataset:

    dataset['Title'] = dataset['Title'].map(title_mapping)
# Visualizing survival and death rates by title

grouped_title = train[['Title','Survived']].groupby('Title',as_index=False).agg({"Survived":['mean','std']})

grouped_title = flatten(grouped_title).sort_values(by="Survived mean", ascending = False)

fig = px.bar(grouped_title, x = "Title", y = "Survived mean", error_y = "Survived std", color = "Title")

fig.update_traces(error_y_color = "black")
# Filling in null cabin values with U

for dataset in full_dataset:

    dataset['Cabin'].fillna("U",inplace=True)
# Extracting only deck letter from cabin

for dataset in full_dataset:

    dataset['Cabin_Deck'] = dataset['Cabin'].apply(lambda x: x.strip()[0])
# Grouping decks by class 

grouped_cabinclass = train.groupby(['Cabin_Deck','Pclass']).agg({'Name':'nunique'})

grouped_cabindeck = train.groupby(['Cabin_Deck']).agg({'Name':'nunique'})



# Normalizing the grouping

grouped_cabin = (grouped_cabinclass/grouped_cabindeck).reset_index()

fig2 = px.bar(grouped_cabin, x = "Cabin_Deck",y = "Name", color = "Pclass", labels = dict(Name = "% of Total"), title = "Normalized Distribution of Classes By Deck")

iplot(fig2)
# Grouping by cabin and plotting bar graph

grouped_cabindeck = train[['Cabin_Deck','Survived','Fare']].groupby('Cabin_Deck',as_index=False).agg({'Survived':['mean','std'],"Fare":['mean','std']})

grouped_cabindeck = flatten(grouped_cabindeck)

fig = px.bar(grouped_cabindeck, x = "Cabin_Deck", y = "Survived mean", error_y = "Survived std", color = "Cabin_Deck")

fig.update_traces(error_y_color = "black")
# Generating map for cabin deck

possible_decks = list(set(test['Cabin_Deck'].unique()) | set(train['Cabin_Deck'].unique()))

cabin_mappings = dict(zip(possible_decks,list(range(1,len(possible_decks)+1))))

print(cabin_mappings)
# Mapping cabin decks

for dataset in full_dataset:

    dataset['Cabin_Deck'] = dataset['Cabin_Deck'].map(cabin_mappings)
# Converting sex to binary mapping

for dataset in full_dataset:

    dataset['Gender'] = dataset['Sex'].map({'male':1,'female':2})
# Filling missing values for age with random integers within 1 standard deviation of the mean for combined train/test datasets

concatenated = pd.concat([train.drop('Survived', axis = 1),test])

age_avg = concatenated['Age'].mean()

age_std  = concatenated['Age'].std()



for dataset in full_dataset:

    age_null_count = dataset['Age'].isnull().sum()



    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)



    # Setting NaN 

    age_slice = dataset["Age"].copy()

    age_slice[np.isnan(age_slice)] = age_null_random_list

    dataset["Age"] = age_slice

    dataset["Age"] = dataset["Age"].astype(int)
# Creating bins to cut ages into. These bins were determined by splitting the age data into 10 quantiles of approximately equal 

# number of people. The work was done in a separate notebook 

# Please comment if you'd like to see how this was done



age_bins = [0,16,19,22,25,28,31,35,40,47,100]

ages = ["0-16","16-19","19-22","22-25","25-28","28-31","31-35","35-40",'40-47',"47+"]

age_labels = list(range(1,11))

print("Age Mappings:",dict(zip(ages,age_labels)) )
# Cutting the data by the bins

for dataset in full_dataset:

    dataset['Age_Cat'] = pd.cut(dataset['Age'], bins = age_bins, include_lowest = True, labels = age_labels)
# Plotting the cut data

grouped_agecats = train.groupby('Age_Cat', as_index=False).agg({'Survived':['mean','std']})

grouped_agecats = flatten(grouped_agecats)

fig = px.bar(grouped_agecats,x = "Age_Cat", y = "Survived mean", error_y = "Survived std", range_y = [0, 1], color = "Age_Cat")

fig.update_traces(error_y_color = "black")
# Filling missing values for train set

train[train['Embarked'].isnull()]
# Filling in missing values accordingly

train.loc[[61,829],"Embarked"] = "S"
# Converting Embarked to numerical

embarked_mapping = {"S":1,"C":2,"Q":3}

print("Embarked Mapping:",embarked_mapping)

for dataset in full_dataset:

    dataset['Embarked_Cat'] = dataset['Embarked'].map(embarked_mapping)
# Filling missing values for test set with median

test['Fare'].fillna(test['Fare'].median(),inplace=True)
# Creating bins to cut fares into. These bins were generated by analyzing quantiles of data in a separate jupyter notebook (similar 

# process as Section 6.5)

fare_bins = [0, 7.8, 10.5, 21.7, 39.7, 550]

fares = ["0-7.8","7.8-10.5","10.5-21.7","21.7-39.7","39.7+"]

fare_labels = list(range(1,6))

print("Fare Mappings:",dict(zip(fares,fare_labels)))
# Cutting the dataset according to the bins above

for dataset in full_dataset:

    dataset['Fare_Cats'] = pd.cut(dataset['Fare'], bins = fare_bins, labels = fare_labels, include_lowest = True)
grouped_fares = train[['Fare_Cats', "Survived"]].groupby("Fare_Cats",as_index=False).mean()

px.bar(grouped_fares,x = "Fare_Cats",y="Survived", color = "Fare_Cats")
# Finalized training and test models

features = ['Pclass','Title','Gender','Age_Cat','Family_Cat','Fare_Cats','Cabin_Deck','Embarked_Cat']

X_train = train[features].values

Y_train = np.array(train[['Survived']]).ravel()

X_test = test[features].values

print("X_train")

print(X_train[0:10])

print("\nX_test")

print(X_test[0:10])
# Random seed

SEED = 0



# Declaring parameters for each of the models

# Extra Trees Classifier

et_params = {

    'n_jobs': -1,

    'n_estimators':500,

    'max_features': 0.5,

    'max_depth': 8,

    'min_samples_leaf': 2,

    'verbose': 0

}



# SVC

svc_params = {

    'kernel' : 'rbf',

    'C' : 1,

    'gamma': 'auto'

}



# RandomForestClassifier

rf_params = {

    'n_jobs': -1,

    'n_estimators': 500,

    'warm_start': True, 

    'max_features': 0.2,

    'max_depth': 6,

    'min_samples_leaf': 2,

    'max_features' : 'sqrt',

    'verbose': 0

}



# XGB

xgb_params = {

    "learning_rate": 0.02,

    "n_estimators": 2000,

    "max_depth": 4,

    "min_child_weight": 2,

    "gamma":1,                        

    "subsample":0.8,

    "colsample_bytree":0.8,

    "objective": 'binary:logistic',

    "nthread": -1,

    "scale_pos_weight": 1

}



# Gradient Boosting

gb_params = {

    'n_estimators': 500,

    'max_features': 0.2,

    'max_depth': 5,

    'min_samples_leaf': 2,

    'verbose': 0

}



# AdaBoost

ada_params = {

    'n_estimators': 500,

    'learning_rate' : 0.75

}
# Initializing models using the SklearnHelper function defined in Section 2

et = SklearnHelper(clf = ExtraTreesClassifier, seed = SEED, params = et_params)

svc = SklearnHelper(clf = SVC, seed = SEED, params = svc_params)

rf = SklearnHelper(clf = RandomForestClassifier, seed = SEED, params = rf_params)

xgb = SklearnHelper(clf = XGBClassifier, seed = SEED, params = xgb_params)

gb = SklearnHelper(clf = GradientBoostingClassifier, seed = SEED, params = gb_params)

ada = SklearnHelper(clf = AdaBoostClassifier, seed = SEED, params = ada_params)

models = [ada,et,gb, rf,svc,xgb]
# Looping through scores and getting dataframe of cross validation scores

score_dict = defaultdict(list)

for model in models:

    score_dict['Model'].append(model.__name__)

    score_dict['CV_Score'].append(model.cv_score(X_train,Y_train))

scores = pd.DataFrame(score_dict).sort_values(by = "CV_Score", ascending = False)

print(scores)
# Plotting cross validation scores as a function of model

px.bar(scores, y = "Model", x = "CV_Score", color = "CV_Score", orientation = "h")
# Capturing feature importances in Plotly figures and in a dictionary

feature_imps = defaultdict(list)

figs = []

for i, model in enumerate(models):

    if model.__name__ == "SVC":

        continue

        

    trace = go.Scatter(

        y = model.feature_importances(X_train,Y_train),

        x = features,

        mode='markers',

        marker=dict(

            sizemode = 'diameter',

            sizeref = 1,

            size = 25,

            color = model.feature_importances(X_train,Y_train),

            colorscale='Portland',

            showscale=True

            )

    )

    layout = go.Layout(

            autosize= True,

            title= model.__name__,

            hovermode= 'closest',

            yaxis=dict(

                title= 'Feature Importance',

                ticklen= 5,

                gridwidth= 2

            ),

            showlegend= False

            )

    figs.append(dict(data = [trace], layout = layout))

    

    feature_imps[i].append(model.__name__)

    feature_imps[i].extend(model.feature_importances(X_train,Y_train))

feature_imps = pd.DataFrame.from_dict(feature_imps, orient = "index", columns = ['Model_Name'] + features)
# Plotting feature importances by classification model

for fig in figs:

    iplot(fig)
# Extract mean feature importance by model and plot

mean_imp = pd.concat([feature_imps.mean(axis = 0),feature_imps.std(axis = 0)], axis = 1).reset_index()

mean_imp.columns = ["Feature","Mean_Importance","Std"]

fig = px.bar(mean_imp.sort_values(by="Mean_Importance"), x = "Feature", y = "Mean_Importance", color = "Mean_Importance", error_y = "Std")

fig.update_traces(error_y_color = "black")
# Setting the parameter grid for each model

# Extra Trees

et_grid = {

    'n_estimators': [250, 500, 1000],

    'max_depth' : [2,4,8],

    'min_samples_leaf': [2,4,8]

}



# RF

rf_grid = {

    'n_estimators' : [250, 500, 1000],

    'max_depth' : [2,4,6],

    'min_samples_leaf': [2,4,8]

}





# XGB

xgb_param_grid = {

    "learning_rate": [0.01, 0.1, 1],

    "n_estimators": [1000, 2000, 4000],

    "max_depth": [2,3, 4]

}
# Setting up Grid Search Models to tune

et_gs = GridSearchCV(

    estimator = ExtraTreesClassifier(),

    param_grid = et_grid,

    cv = 5,

    scoring = 'accuracy'

)



rf_gs = GridSearchCV(

    estimator = RandomForestClassifier(),

    param_grid = rf_grid,

    cv = 5,

    scoring = 'accuracy'

)



xgb_gs = GridSearchCV(

    estimator = XGBClassifier(    

        min_child_weight = 2,

        gamma = 1,                        

        subsample = 0.8,

        colsample_bytree = 0.8,

        objective = 'binary:logistic',

        nthread = -1,

        scale_pos_weight = 1),

    param_grid = xgb_param_grid,

    cv = 5,

    scoring = 'accuracy'

)
# Determining best parameters for each model

best_params = {}

best_params['Extra Trees'] = et_gs.fit(X_train,Y_train).best_params_
best_params['RF'] = rf_gs.fit(X_train,Y_train).best_params_
best_params['XGB'] = xgb_gs.fit(X_train,Y_train).best_params_
best_params
# Creating the ensemble model with hard voting

ensemble = VotingClassifier(

    estimators = [('ET',ExtraTreesClassifier(max_depth = 8, min_samples_leaf = 4, n_estimators = 1000)),

                  ('RF',RandomForestClassifier(max_depth = 4, min_samples_leaf = 2, n_estimators = 500)), 

                  ('XGB',XGBClassifier(min_child_weight = 2,gamma = 1,  subsample = 0.8, colsample_bytree = 0.8, objective = 'binary:logistic', nthread = -1,\

                                       scale_pos_weight = 1, learning_rate = 0.01, max_depth = 4, n_estimators = 4000))],

    voting = 'hard'

)
# Fitting the ensemble to the training data

ensemble_fit = ensemble.fit(X_train,Y_train)
# Preparing submission

submission = pd.concat([test['PassengerId'],pd.Series(ensemble_fit.predict(X_test))], axis = 1,)

submission.columns = ['PassengerId','Survived']

submission.head()
submission.to_csv("submission.csv",index=False)