# basic python libraries
import pandas as pd
import numpy as np
import matplotlib
import plotly
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from matplotlib import cm
from collections import OrderedDict
matplotlib.rcParams['figure.dpi'] = 200
matplotlib.rcParams['figure.figsize'] = (15, 5)
from scipy.stats import norm, shapiro
from scipy import stats


# sklearn libraries
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.model_selection import GridSearchCV

# feature selection library
from mlxtend.feature_selection import SequentialFeatureSelector

# model building libraries
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier,VotingClassifier,StackingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.svm import SVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier

# warning library
import warnings
warnings.filterwarnings("ignore")

# setting basic options
pd.set_option('display.max_columns', None)
%matplotlib inline
# color maps
greens = sns.light_palette("green", as_cmap=True)
purples = sns.light_palette("purple", as_cmap=True)
blues = sns.light_palette("blue", as_cmap=True)
# Define color sets of paintings
night_colors = ['rgb(56, 75, 126)', 'rgb(18, 36, 37)', 'rgb(34, 53, 101)',
                'rgb(36, 55, 57)', 'rgb(6, 4, 4)']
sunflowers_colors = ['rgb(177, 127, 38)', 'rgb(205, 152, 36)', 'rgb(99, 79, 37)',
                     'rgb(129, 180, 179)', 'rgb(124, 103, 37)']
irises_colors = ['rgb(33, 75, 99)', 'rgb(79, 129, 102)', 'rgb(151, 179, 100)',
                 'rgb(175, 49, 35)', 'rgb(36, 73, 147)']
cafe_colors =  ['rgb(146, 123, 21)', 'rgb(177, 180, 34)', 'rgb(206, 206, 40)',
                'rgb(175, 51, 21)', 'rgb(35, 36, 21)']
# importing dataset
data = pd.read_csv(r'../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv', error_bad_lines = False)
print("Shape of the data is {}.".format(data.shape))
data.head().style.background_gradient(cmap=purples)
# getting the death event distribution
labels = data.DEATH_EVENT.value_counts(normalize = True)*100 
fig = px.pie(labels, values= 'DEATH_EVENT', names = ['Alive', 'Dead'], title='Target Distribution across whole dataset')
fig.show()
# Stratified Shuffle Train Test Split
sss = StratifiedShuffleSplit(n_splits = 1, test_size = 0.15 ,random_state = 42) #2-fold cross validation
for train_index, test_index in sss.split(data.iloc[:, :-1], data.iloc[:,-1]):
    print("Intersection of train index and test index:", list(set(train_index) & set(test_index)))
    train = data.iloc[train_index, :]
    test = data.iloc[test_index, :]
    print("Shape of train data is {}".format(train.shape))
    print("Shape of test data is {}".format(test.shape))
# getting the death event distribution in train set
train_labels = train.DEATH_EVENT.value_counts(normalize = True)*100 
test_labels = test.DEATH_EVENT.value_counts(normalize = True)*100 

# Create subplots, using 'domain' type for pie charts
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                    subplot_titles=['Train Target', 'Test Target'])
labels = ["Alive", "Dead"]

# Define pie charts
fig.add_trace(go.Pie(labels=labels, values=train_labels, name="Train target", scalegroup='one',
                     marker_colors=night_colors), 1, 1)
fig.add_trace(go.Pie(labels=labels, values=test_labels, name='Test Target', scalegroup='one',
                     marker_colors=cafe_colors), 1, 2)

# Tune layout and hover info
fig.update_traces(hoverinfo='label+percent+name', textinfo='percent+label')
fig.update(layout_title_text='Distribution of target values across train and test set respectively',
          layout_showlegend=False)

fig = go.Figure(fig)
fig.show()
# checking for null values in train set
train.isnull().sum()
# checking for null values in test set
test.isnull().sum()
# distribution of column values
f,ax = plt.subplots(6,2,figsize=(15,25))
sns.distplot(train['age'].dropna(),ax=ax[0,0],kde=True,color='b')
sns.distplot(train['serum_creatinine'].dropna(),ax=ax[0,1],kde=True,color='g')
sns.distplot(train['creatinine_phosphokinase'].dropna(),ax=ax[1,0],kde=True,color='r')
sns.distplot(train['platelets'].dropna(),ax=ax[1,1],kde=True,color='m')
sns.distplot(train['ejection_fraction'].dropna(),ax=ax[2,0],kde=True,color='burlywood')
sns.distplot(train['serum_sodium'].dropna(),ax=ax[2,1],kde=True,color='chartreuse')
sns.countplot('anaemia',data=train,ax=ax[3,0], palette='husl')
sns.countplot('diabetes',data=train,ax=ax[3,1], palette='RdBu')
sns.countplot('high_blood_pressure',data=train,ax=ax[4,0], palette=sns.color_palette("muted"))
sns.countplot('sex',data=train,ax=ax[4,1], palette=sns.color_palette("ch:2.5,-.2,dark=.3"))
sns.countplot('smoking',data=train,ax=ax[5,0], palette=sns.color_palette("RdBu", n_colors=7))
sns.distplot(train['time'],kde = True,ax=ax[5,1], color = '#0FFF7C')
# describing age
train.age.describe().to_frame().style.background_gradient(cmap=greens)
sum_Age = train[["age", "DEATH_EVENT"]].groupby(['age'],as_index=False).sum()
avg_Age = train[["age", "DEATH_EVENT"]].groupby(['age'],as_index=False).mean()

# plotting the dataframes
fig, (axis1,axis2,axis3) = plt.subplots(3,1,figsize=(20,10))
sns.barplot(x='age', y='DEATH_EVENT', data=sum_Age, ax = axis1)
sns.barplot(x='age', y='DEATH_EVENT', data=avg_Age, ax = axis2)
sns.pointplot(x = 'age', y = 'DEATH_EVENT', data=train, ax = axis3)
# bin the age according to the quantiles
train['age_bin'] = pd.qcut(train['age'], q=4, labels = [0,1,2,3])
test['age_bin'] = pd.qcut(test['age'], q=4, labels = [0,1,2,3])
print("Shape of train is: ", train.shape)
print("Shape of test is: ", test.shape)
# getting the distribution
age_dist = train.groupby(['age_bin', 'DEATH_EVENT'])['DEATH_EVENT'].count().unstack().fillna(0).reset_index()
age_dist['per_death'] = age_dist[1]/(age_dist[0]+age_dist[1])*100
age_dist.plot(kind = 'bar')
# distribution of anaemia with age
anaemia_grp = train.groupby(['age_bin', 'anaemia'])['DEATH_EVENT'].mean().unstack().reset_index()
anaemia_grp.plot(kind = 'bar')
# distribution of anamenia and DEATH_EVENT
anaemia_grp = train.groupby(['anaemia', 'DEATH_EVENT'])['DEATH_EVENT'].count().unstack()
anaemia_grp.plot(kind='bar')
# getting the distribution
dist = train.groupby(['anaemia','diabetes', 'DEATH_EVENT'])['DEATH_EVENT'].count().unstack().fillna(0)
dist['per_death'] = dist[1]/(dist[0]+dist[1])*100
dist[['per_death']].plot(kind='bar')
dist[[0,1]].plot(kind='bar', stacked=True)
# getting the distribution
dist = train.groupby(['anaemia','age_bin', 'DEATH_EVENT'])['DEATH_EVENT'].count().unstack().fillna(0)
dist['per_death'] = dist[1]/(dist[0]+dist[1])*100
dist[['per_death']].plot(kind='bar')
dist[[0,1]].plot(kind='bar', stacked=True)
# description of creatinine_phosphokinase
train.creatinine_phosphokinase.describe().to_frame().style.background_gradient(cmap='viridis')
# making a column for having normal range
train['normal_creatinine_phosphokinase'] = train['creatinine_phosphokinase'].apply(lambda x: 1 if (x > 10 and x <=120) else 0)
test['normal_creatinine_phosphokinase'] = test['creatinine_phosphokinase'].apply(lambda x: 1 if (x > 10 and x <=120) else 0)
# quantile binning
train['creatinine_phosphokinase_bin'] = pd.qcut(train['creatinine_phosphokinase'], q=4, labels = [0,1,2,3])
test['creatinine_phosphokinase_bin'] = pd.qcut(test['creatinine_phosphokinase'], q=4, labels = [0,1,2,3])
# getting the distribution
dist = train.groupby(['creatinine_phosphokinase_bin', 'DEATH_EVENT'])['DEATH_EVENT'].count().unstack().fillna(0).reset_index()
dist['per_death'] = dist[1]/(dist[0]+dist[1])*100
dist.plot(kind = 'bar')
# getting the distribution
dist = train.groupby(['normal_creatinine_phosphokinase', 'DEATH_EVENT'])['DEATH_EVENT'].count().unstack().fillna(0).reset_index()
dist['per_death'] = dist[1]/(dist[0]+dist[1])*100
dist
# getting the distribution
dist = train.groupby(['creatinine_phosphokinase_bin','smoking', 'DEATH_EVENT'])['DEATH_EVENT'].count().unstack().fillna(0)
dist.plot(kind='bar', stacked = True)
# getting the distribution
dist = train.groupby(['normal_creatinine_phosphokinase','smoking', 'DEATH_EVENT'])['DEATH_EVENT'].count().unstack().fillna(0)
dist['per_death'] = dist[1]/(dist[0]+dist[1])*100
dist[['per_death']].plot(kind='bar')
# getting the distribution
dist = train.groupby(['normal_creatinine_phosphokinase','anaemia', 'DEATH_EVENT'])['DEATH_EVENT'].count().unstack().fillna(0)
dist['per_death'] = dist[1]/(dist[0]+dist[1])*100
dist[['per_death']].plot(kind='bar')
dist[[0,1]].plot(kind='bar', stacked=True)
# getting the distribution
dist = train.groupby(['normal_creatinine_phosphokinase','age_bin', 'DEATH_EVENT'])['DEATH_EVENT'].count().unstack().fillna(0)
dist['per_death'] = dist[1]/(dist[0]+dist[1])*100
dist[['per_death']].plot(kind='bar')
dist[[0,1]].plot(kind='bar', stacked=True)
# getting the distribution
dist = train.groupby(['normal_creatinine_phosphokinase','diabetes', 'DEATH_EVENT'])['DEATH_EVENT'].count().unstack().fillna(0)
dist['per_death'] = dist[1]/(dist[0]+dist[1])*100
dist[['per_death']].plot(kind='bar')
dist[[0,1]].plot(kind='bar', stacked=True)
# getting the distribution
dist = train.groupby(['diabetes', 'DEATH_EVENT'])['DEATH_EVENT'].count().unstack().fillna(0)
dist['per_death'] = dist[1]/(dist[0]+dist[1])*100
dist[['per_death']].plot(kind='bar')
dist[[0,1]].plot(kind='bar', stacked=True)
# getting the distribution
dist = train.groupby(['diabetes','smoking', 'DEATH_EVENT'])['DEATH_EVENT'].count().unstack().fillna(0)
dist['per_death'] = dist[1]/(dist[0]+dist[1])*100
dist[['per_death']].plot(kind='bar')
dist[[0,1]].plot(kind='bar', stacked=True)
# getting the distribution
dist = train.groupby(['diabetes', 'sex'])['sex'].count().unstack().fillna(0)
dist['per_death'] = dist[1]/(dist[0]+dist[1])*100
dist[['per_death']].plot(kind='bar')
dist[[0,1]].plot(kind='bar', stacked=True)
# getting the distribution
dist = train.groupby(['diabetes','sex', 'DEATH_EVENT'])['DEATH_EVENT'].count().unstack().fillna(0)
dist['per_death'] = dist[1]/(dist[0]+dist[1])*100
dist[['per_death']].plot(kind='bar')
dist[[0,1]].plot(kind='bar', stacked=True)
dist = train.groupby(['diabetes','high_blood_pressure', 'DEATH_EVENT'])['DEATH_EVENT'].count().unstack().fillna(0)
dist['per_death'] = dist[1]/(dist[0]+dist[1])*100
dist[['per_death']].plot(kind='bar')
dist[[0,1]].plot(kind='bar', stacked=True)
# getting the distribution
dist = train.groupby(['diabetes', 'smoking','anaemia', 'normal_creatinine_phosphokinase','high_blood_pressure', 'DEATH_EVENT'])['DEATH_EVENT'].count().unstack().fillna(0)
dist['per_death'] = dist[1]/(dist[0]+dist[1])*100
dist[['per_death']].plot(kind='bar')
dist[[0,1]].plot(kind='bar', stacked=True)
# describe
train.ejection_fraction.describe().to_frame().style.background_gradient(cmap='PuBuGn')
# binning
train['EF_bin'] = pd.cut(train['ejection_fraction'], bins=[0,36,40,55,81,101], labels = [0,1,2,3,4])
test['EF_bin'] = pd.cut(test['ejection_fraction'], bins=[0,36,40,55,81,101], labels = [0,1,2,3,4])
# getting the distribution
dist = train.groupby(['EF_bin', 'DEATH_EVENT'])['DEATH_EVENT'].count().unstack().fillna(0)
dist['per_death'] = dist[1]/(dist[0]+dist[1])*100
dist[['per_death']].plot(kind='bar')
dist[[0,1]].plot(kind='bar', stacked=True)
# getting the distribution
dist = train.groupby(['diabetes','smoking','EF_bin', 'DEATH_EVENT'])['DEATH_EVENT'].count().unstack().fillna(0)
dist['per_death'] = dist[1]/(dist[0]+dist[1])*100
dist[['per_death']].plot(kind='bar')
dist[[0,1]].plot(kind='bar', stacked=True)
# serum_creatinine
train.serum_creatinine.describe().to_frame().style.background_gradient(cmap='twilight_shifted')
# serum_sodium
train.serum_sodium.describe().to_frame().style.background_gradient(cmap='twilight')
train.columns.tolist()
# parallel plot for continuous variable
fig = px.parallel_coordinates(train, color="DEATH_EVENT",
                              dimensions=['age',
 'creatinine_phosphokinase',
 'ejection_fraction',
 'platelets',
 'serum_creatinine',
 'serum_sodium',
                                          'DEATH_EVENT'
                                         ],
                              color_continuous_scale=px.colors.diverging.Tealrose,
                              color_continuous_midpoint=2)
fig.show()
sns.heatmap(train.corr())
# getting the train columns
train.columns.tolist()
continuous_variables = [
 'age',
 'creatinine_phosphokinase',
 'ejection_fraction',
 'platelets',
 'serum_creatinine',
 'serum_sodium'
]

target = ['DEATH_EVENT']

categorical_variables = [
 'anaemia',
 'diabetes',
 'high_blood_pressure',
 'sex',
 'smoking',
 'time',
 'age_bin',
 'normal_creatinine_phosphokinase',
 'creatinine_phosphokinase_bin',
 'EF_bin'
]
# scaling the continuous variables
autoscaler = PowerTransformer()
train[continuous_variables] = autoscaler.fit_transform(train[continuous_variables])
test[continuous_variables] = autoscaler.transform(test[continuous_variables])
# visualising the final transformed continuous variables.
# Group data together
hist_data = [train.age, train.creatinine_phosphokinase, train.ejection_fraction, 
             train.platelets, train.serum_creatinine, train.serum_sodium]

group_labels = ['Age', 'Creatinine Phosphokinase', 'Ejection Fraction', 'Platelets', 'Serum Creatinine', 
                'Serum Sodium']

colors = ['#FA93A0', '#19FFFA', '#51F04B',
                '#F06318', '#A018F0', '#3222E3']

# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels, bin_size=.4, colors = colors, show_hist=False)
fig.show()
# remove highly correlated features
# Create correlation matrix
corr_matrix = train[continuous_variables].corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.70
to_drop = [column for column in upper.columns if any(upper[column] > 0.70)]

# print the columns to be dropped off
to_drop
# train datatypes
train.dtypes
# changing to int type
train['age_bin'] = train['age_bin'].astype(int)
train['creatinine_phosphokinase_bin'] = train['creatinine_phosphokinase_bin'].astype(int)
train['EF_bin'] = train['EF_bin'].astype(int)

test['age_bin'] = test['age_bin'].astype(int)
test['creatinine_phosphokinase_bin'] = test['creatinine_phosphokinase_bin'].astype(int)
test['EF_bin'] = test['EF_bin'].astype(int)
features = [
 'age',
 'creatinine_phosphokinase',
 'ejection_fraction',
 'platelets',
 'serum_creatinine',
 'serum_sodium',
 'anaemia',
 'diabetes',
 'high_blood_pressure',
 'sex',
 'smoking',
 'time',
 'age_bin',
 'normal_creatinine_phosphokinase',
 'creatinine_phosphokinase_bin',
 'EF_bin'
]
# Choosing best base model for trainset.
ensembles = []
ensembles.append(('GBM', GradientBoostingClassifier(n_estimators = 300)))
ensembles.append(('RF', RandomForestClassifier(n_estimators = 300)))
ensembles.append(('ET', ExtraTreesClassifier()))
ensembles.append(('XGB', XGBClassifier(n_estimators = 300, n_jobs=-1)))
ensembles.append(('LR', LogisticRegressionCV(cv = 5)))
ensembles.append(('GNB', GaussianNB()))
ensembles.append(('BNB', BernoulliNB()))
ensembles.append(('KNN', KNeighborsClassifier(n_neighbors = 3)))
ensembles.append(('DT', DecisionTreeClassifier()))
ensembles.append(('LGMB', LGBMClassifier(n_estimators = 300, class_weight = 'balanced', n_jobs = -1)))
#ensembles.append(('CBC', CatBoostClassifier(cat_features = categorical_variables, class_weights = 'SqrtBalanced')))

train_set = train[features]
test_set = test[features]

results = []
names = []
for name, model in ensembles:
    kfold = StratifiedKFold(n_splits=3, shuffle = True, random_state=42)
    cv_results = cross_val_score(model, train_set, train[target],
                                 cv=kfold, scoring='roc_auc')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    model = model.fit(train_set, train[target])
    # validating the training results on validation sets.
    predicted_result = model.predict(test_set)
    print("Confusion Matrix: ", confusion_matrix(test[target], predicted_result), "\n")
    print("Classification Report: \n", classification_report(test[target], predicted_result), "\n")
# lets lokk at how catboost classifier is doing on this small dataset.
model = CatBoostClassifier(cat_features = categorical_variables,
                          loss_function = 'Logloss',
                           custom_metric = ['AUC', 'F1'],
                           eval_metric = 'AUC',
                           bootstrap_type = 'Bernoulli',
                           use_best_model = True,
                           leaf_estimation_method = 'Newton',
                           auto_class_weights = 'SqrtBalanced',
                           boosting_type = 'Ordered'
                          )
model = model.fit(train_set, train[target], eval_set = (test_set, test[target]))
# validating the training results on validation sets.
predicted_result = model.predict(test_set)
print("Confusion Matrix: ", confusion_matrix(test[target], predicted_result), "\n")
print("Classification Report: \n", classification_report(test[target], predicted_result), "\n")
# lets try voting classifier by ensembling some moderately and best performing models.
# voting ensembler soft and hard for dataset
ensembles = []
ensembles.append(('RF', RandomForestClassifier(n_estimators = 300)))
ensembles.append(('XGB', XGBClassifier(n_estimators = 300, n_jobs=-1)))
ensembles.append(('LGMB', LGBMClassifier(n_estimators = 300, class_weight = 'balanced', n_jobs = -1)))
ensembles.append(('LR', LogisticRegressionCV(cv = 3)))

# Voting classifier soft
vc_soft = VotingClassifier(estimators=ensembles, voting='soft', flatten_transform=True)
vc_soft.fit(train_set, train[target])
# validating the training results on validation sets.
print("###### Voting Classifier - Soft #######")
predicted_result = model.predict(test_set)
print("Confusion Matrix: ", confusion_matrix(test[target], predicted_result), "\n")
print("Classification Report: \n", classification_report(test[target], predicted_result), "\n")

# Voting classifier hard
print("###### Voting Classifier - Hard #######")
vc_hard = VotingClassifier(estimators=ensembles, voting='hard')
vc_hard.fit(train_set, train[target])
predicted_result = model.predict(test_set)
print("Confusion Matrix: ", confusion_matrix(test[target], predicted_result), "\n")
print("Classification Report: \n", classification_report(test[target], predicted_result), "\n")
# forward selections
# getting the features from forward selection methods
print("**************************************************")
# initialising the SequentialFeatureSelector
kfold = StratifiedKFold(n_splits=3, shuffle = True, random_state=42)
sfs = SequentialFeatureSelector(LGBMClassifier(n_estimators = 300, class_weight = 'balanced', n_jobs = -1), 
           k_features=5, 
           forward=True, 
           floating=False,
           scoring='roc_auc',
           cv=kfold)

# fit the object to the training data.
sfs.fit(train_set, train[target])

# print the selected features.
selected_features = train_set.columns[list(sfs.k_feature_idx_)]
print(selected_features)

# print the final prediction score.
print(sfs.k_score_)
print("**************************************************")
sfs.subsets_
filtered = [
   'ejection_fraction',
   'serum_creatinine',
   'sex',
   'time',
   'creatinine_phosphokinase_bin'
]
# trying embedded methods also
# Embedded Feature Selection usinf RF classifier.

model = LGBMClassifier(n_estimators = 300, class_weight = 'balanced', n_jobs = -1)

# fit the model to start training.
model.fit(train_set[filtered], train[target])

# get the importance of the resulting features.
importances = model.feature_importances_

# create a data frame for visualization.
final_df = pd.DataFrame({"Features": train_set[filtered].columns, "Importances":importances})
final_df.set_index('Importances')

# sort in ascending order to better visualization.
final_df = final_df.sort_values('Importances')

# plot the feature importances in bars.
final_df.plot.bar(x = 'Features') 

# predict the result
predicted_result = model.predict(test_set[filtered])
print("Confusion Matrix: ", confusion_matrix(test[target], predicted_result), "\n")
print("Classification Report: \n", classification_report(test[target], predicted_result), "\n")
# training the hypertuned model
model = LGBMClassifier(boosting_type = 'gbdt',
                       class_weight = 'balanced',
                       learning_rate = 0.1,
                       n_estimators = 300,
                       n_jobs = -1,
                       num_leaves = 31,
                       objective = 'binary',
                       reg_alpha = 0,
                       reg_lambda = 0)

model = model.fit(train_set[filtered], train[target])
# predict the result
predicted_result = model.predict(test_set[filtered])
print("Confusion Matrix: ", confusion_matrix(test[target], predicted_result), "\n")
print("Classification Report: \n", classification_report(test[target], predicted_result), "\n")
test['predicted_result'] = predicted_result
test['predicted_result'].head().to_frame().style.background_gradient(cmap='OrRd_r')