import pandas as pd

from scipy.io import arff

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

from sklearn import preprocessing

import numpy as np



from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_curve, auc

from sklearn.model_selection import train_test_split

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 

                              GradientBoostingClassifier, ExtraTreesClassifier)

from sklearn.model_selection import RandomizedSearchCV

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

import xgboost as xgb

from sklearn import metrics

from sklearn.metrics import mean_squared_error



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls





FOLDS =10

%matplotlib inline
# Function to graph number of people by age

def cont_age(field):

    plt.figure()

    g = None

    if field == "Age":

        df_query_mri = df[df["Age"] > 0]

        g = sns.countplot(df_query_mri["Age"])

        g.figure.set_size_inches(18.5, 10.5)

    else:

        g = sns.countplot(df[field])

        g.figure.set_size_inches(18.5, 10.5)

    

sns.despine()
# Function to graph number of people per state [Demented, Nondemented]

def cont_Dementes(field):

    plt.figure()

    g = None

    if field == "Group":

        df_query_mri = df[df["Group"] >= 0]

        g = sns.countplot(df_query_mri["Group"])

        g.figure.set_size_inches(18.5, 10.5)

    else:

        g = sns.countplot(df[field])

        g.figure.set_size_inches(18.5, 10.5)

    

sns.despine()
# 0 = F y 1= M

def bar_chart(feature):

    Demented = df[df['Group']==1][feature].value_counts()

    Nondemented = df[df['Group']==0][feature].value_counts()

    df_bar = pd.DataFrame([Demented,Nondemented])

    df_bar.index = ['Demented','Nondemented']

    df_bar.plot(kind='bar',stacked=True, figsize=(8,5))
def report_performance(model):



    model_test = model.predict(X_test)



    print("Confusion Matrix")

    print("{0}".format(metrics.confusion_matrix(y_test, model_test)))

    print("")

    print("Classification Report")

    print(metrics.classification_report(y_test, model_test))
data = '/kaggle/input/mri-and-alzheimers/oasis_longitudinal.csv'

df = pd.read_csv (data)

df.head()
df.describe()
nu = pd.DataFrame(df['Group']=='Nondemented')

nu["Group"].value_counts() 
f, ax = plt.subplots(figsize=(10, 8)) 

corr = df.corr(method = 'pearson') 

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), 

            square=True, ax=ax) 
df.corr(method = 'pearson') 
pd.scatter_matrix(df, alpha = 0.3, figsize = (14,8), diagonal = 'kde'); 
g = sns.PairGrid(df, vars=['Visit','MR Delay','M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF'],

                 hue='Group', palette='RdBu_r')

g.map(plt.scatter, alpha=0.8)

g.add_legend();
import seaborn as sb

sb.factorplot('M/F',data=df,hue='Group',kind="count")
facet= sns.FacetGrid(df,hue="Group", aspect=3)

facet.map(sns.kdeplot,'MMSE',shade= True)

facet.set(xlim=(0, df['MMSE'].max()))

facet.add_legend()

plt.xlim(12.5)
cont_age("Age")
df['Group'] = df['Group'].replace(['Converted'], ['Demented'])

df.head(3)
df.drop(['Subject ID'], axis = 1, inplace = True, errors = 'ignore')

df.drop(['MRI ID'], axis = 1, inplace = True, errors = 'ignore')

df.drop(['Visit'], axis = 1, inplace = True, errors = 'ignore')

#for this study the CDR we eliminated it

df.drop(['CDR'], axis = 1, inplace = True, errors = 'ignore')

df.head(3)
# 1 = Demented, 0 = Nondemented

df['Group'] = df['Group'].replace(['Converted'], ['Demented'])



df['Group'] = df['Group'].replace(['Demented', 'Nondemented'], [1,0])    

df.head(3)
# 1= M, 0 = F



df['M/F'] = df['M/F'].replace(['M', 'F'], [1,0])  

df.head(3)
from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()

encoder.fit(df.Hand.values)

list(encoder.classes_)

#Transoformamos

encoder.transform(df.Hand.values)

df[['Hand']]=encoder.transform(df.Hand.values)

encoder2=LabelEncoder()

encoder2.fit(df.Hand.values)

list(encoder2.classes_)
data_na = (df.isnull().sum() / len(df)) * 100

data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Lost proportion (%)' :round(data_na,2)})

missing_data.head(20)
from sklearn.impute  import SimpleImputer

# We perform it with the most frequent value 

imputer = SimpleImputer ( missing_values = np.nan,strategy='most_frequent')



imputer.fit(df[['SES']])

df[['SES']] = imputer.fit_transform(df[['SES']])



# We perform it with the median

imputer = SimpleImputer ( missing_values = np.nan,strategy='median')



imputer.fit(df[['MMSE']])

df[['MMSE']] = imputer.fit_transform(df[['MMSE']])
from sklearn.impute  import SimpleImputer

# We perform it with the median

imputer = SimpleImputer ( missing_values = np.nan,strategy='median')



imputer.fit(df[['MMSE']])

df[['MMSE']] = imputer.fit_transform(df[['MMSE']])
from sklearn.preprocessing import StandardScaler

df_norm = df

scaler = StandardScaler()

df_norm[['Age','MR Delay','M/F','Hand','EDUC','SES','MMSE','eTIV','nWBV','ASF']]=scaler.fit_transform(df[['Age','MR Delay','M/F','Hand','EDUC','SES','MMSE','eTIV','nWBV','ASF']])
df_norm.head(3)
df.drop(['Hand'], axis = 1, inplace = True, errors = 'ignore')

df.drop(['MR Delay'], axis = 1, inplace = True, errors = 'ignore')
df.head()
data_test = df
X = data_test.drop(["Group"],axis=1)

y = data_test["Group"].values

X.head(3)
# We divide our data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 0)
print("{0:0.2f}% Train".format((len(X_train)/len(data_test.index)) * 100))

print("{0:0.2f}% Test".format((len(X_test)/len(data_test.index)) * 100))
print("Original Demented : {0} ({1:0.2f}%)".format(len(df_norm.loc[df_norm['Group'] == 1]), 100 * (len(df_norm.loc[df_norm['Group'] == 1]) / len(df_norm))))

print("Original Nondemented : {0} ({1:0.2f}%)".format(len(df_norm.loc[df_norm['Group'] == 0]), 100 * (len(df_norm.loc[df_norm['Group'] == 0]) / len(df_norm))))

print("")

print("Training Demented : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 1]), 100 * (len(y_train[y_train[:] == 1]) / len(y_train))))

print("Training Nondemented : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 0]), 100 * (len(y_train[y_train[:] == 0]) / len(y_train))))

print("")

print("Test Demented : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 1]), 100 * (len(y_test[y_test[:] == 1]) / len(y_test))))

print("Test Nondemented : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 0]), 100 * (len(y_test[y_test[:] == 0]) / len(y_test))))
# Number of trees in random forest

n_estimators = range(10,250)

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = range(1,40)

# Minimum number of samples required to split a node

min_samples_split = range(3,60)
# Create the random grid

parametro_rf = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split}
model_forest = RandomForestClassifier(n_jobs=-1)

forest_random = RandomizedSearchCV(estimator = model_forest, param_distributions = parametro_rf, n_iter = 100, cv = FOLDS, 

                               verbose=2, random_state=42, n_jobs = -1, scoring='neg_mean_absolute_error')

forest_random.fit(X_train, y_train)
forest_random.best_params_
model_rf = forest_random.best_estimator_

model_rf =  RandomForestClassifier(n_estimators=60,min_samples_split=8,max_features='sqrt',max_depth= 37)

model_rf.fit(X_train,y_train)
test_score = cross_val_score(model_rf, X_train, y_train, cv=FOLDS, scoring='roc_auc').mean()

test_score
test_score = cross_val_score(model_rf, X_train, y_train, cv=FOLDS, scoring='accuracy').mean()

test_score
Predicted_rf= model_rf.predict(X_test)

test_recall = recall_score(y_test, Predicted_rf, pos_label=1)

fpr, tpr, thresholds = roc_curve(y_test, Predicted_rf, pos_label=1)

test_auc = auc(fpr, tpr)
# Number of trees in random forest

n_estimators = range(50,280)

# Maximum number of levels in tree

max_depth =  range(1,40)

# Minimum number of samples required to split a node

min_samples_leaf = [3,4,5,6,7,8,9,10,15,20,30,40,50,60]
# Create the random grid

parametro_Et = {'n_estimators': n_estimators,

               'max_depth': max_depth,

               'min_samples_leaf': min_samples_leaf}
model_et = ExtraTreesClassifier(n_jobs=-1)

et_random = RandomizedSearchCV(estimator = model_et, param_distributions = parametro_rf, n_iter = 100, cv = FOLDS, 

                               verbose=2, random_state=42, n_jobs = -1, scoring='roc_auc')

et_random.fit(X_train, y_train)
et_random.best_params_
n_estimators = range(10,200)



learning_rate = [0.0001, 0.001, 0.01, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,1]
# Create the random grid

parametros_ada = {'n_estimators': n_estimators,

               'learning_rate': learning_rate}
model_ada = AdaBoostClassifier()



ada_random = RandomizedSearchCV(estimator = model_ada, param_distributions = parametros_ada, n_iter = 100, cv = FOLDS, 

                               verbose=2, random_state=42, n_jobs = -1, scoring='roc_auc')

ada_random.fit(X_train, y_train)
ada_random.best_params_
parametros_gb = {

    "loss":["deviance"],

    "learning_rate": [0.01, 0.025, 0.005,0.5, 0.075, 0.1, 0.15, 0.2,0.3,0.8,0.9],

    "min_samples_split": [0.01, 0.025, 0.005,0.4,0.5, 0.075, 0.1, 0.15, 0.2,0.3,0.8,0.9],

    "min_samples_leaf": [1,2,3,5,8,10,15,20,40,50,55,60,65,70,80,85,90,100],

    "max_depth":[3,5,8,10,15,20,25,30,40,50],

    "max_features":["log2","sqrt"],

    "criterion": ["friedman_mse",  "mae"],

    "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],

    "n_estimators":range(1,100)

    }
model_gb= GradientBoostingClassifier()





gb_random = RandomizedSearchCV(estimator = model_gb, param_distributions = parametros_gb, n_iter = 100, cv = FOLDS, 

                               verbose=2, random_state=42, n_jobs = -1, scoring='roc_auc')

gb_random.fit(X_train, y_train)
gb_random.best_params_
C = [0.001, 0.10, 0.1, 10, 25, 50,65,70,80,90, 100, 1000,2000,10000,20000,25000,30000,40000]



kernel =  ['rbf']

    

gamma =[1e-2, 1e-3, 1e-4, 1e-5,1e-6,1e-7,1e-8,1]
# Create the random grid

parametros_svm = {'C': C,

            'gamma': gamma,

             'kernel': kernel}
model_svm = SVC()

from sklearn.model_selection import GridSearchCV

svm_random = GridSearchCV(model_svm, parametros_svm,  cv = 20, 

                               verbose=2, n_jobs = -1, scoring='roc_auc')

svm_random.fit(X, y)
param_xgb = {

        'silent': [False],

        'max_depth': [6, 10, 15, 20],

        'learning_rate': [0.001, 0.01, 0.1, 0.2, 0,3],

        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],

        'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],

        'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],

        'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],

        'gamma': [0, 0.25, 0.5, 1.0],

        'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],

        'n_estimators': [50,100,120]}
from sklearn.model_selection import GridSearchCV



model_xgb = xgb.XGBClassifier()

xgb_random = RandomizedSearchCV(estimator = model_xgb, param_distributions = param_xgb, n_iter = 100, cv = FOLDS, 

                               verbose=2, random_state=42, n_jobs = -1, scoring='roc_auc')

xgb_random.fit(X_train.values, y_train)
xgb_random.best_params_
parametro_rf = forest_random.best_params_



parametro_et = et_random.best_params_



parametro_ada = ada_random.best_params_



parametro_gb = gb_random.best_params_



parametro_svm = svm_random.best_params_



parametro_xgb= xgb_random.best_params_

model_rf = forest_random.best_estimator_



model_et = et_random.best_estimator_



model_ada = ada_random.best_estimator_



model_gb = gb_random.best_estimator_



model_svc = svm_random.best_estimator_



model_xgb= xgb_random.best_estimator_

kf = KFold(n_splits=FOLDS, random_state = 0, shuffle = True)

for i, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):

    Xtrain, Xval = X_train.values[train_index], X_train.values[val_index]

    ytrain, yval = y_train[train_index], y_train[val_index]

    

    model_rf.fit(Xtrain, ytrain)

    model_et.fit(Xtrain, ytrain)

    model_ada.fit(Xtrain, ytrain)

    model_gb.fit(Xtrain, ytrain)

    model_svc.fit(Xtrain, ytrain)

    model_xgb.fit(Xtrain, ytrain)

    
rf_feature = model_rf.feature_importances_

ada_feature = model_ada.feature_importances_

gb_feature = model_gb.feature_importances_

et_feature = model_et.feature_importances_

xbg_feature = model_xgb.feature_importances_
cols = X.columns.tolist()

# Create a dataframe with features

feature_dataframe = pd.DataFrame( {'features': cols,

     'Random Forest feature importances': rf_feature,

      'AdaBoost feature importances': ada_feature,

    'Gradient Boost feature importances': gb_feature,

    'Extra Trees  feature importances': et_feature,

    'Xgboost feature importances': xbg_feature,

    })
xbg_feature
# Scatter plot 

trace = go.Scatter(

    y = feature_dataframe['Random Forest feature importances'].values,

    x = feature_dataframe['features'].values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 25,

#       size= feature_dataframe['AdaBoost feature importances'].values,

        #color = np.random.randn(500), #set color equal to a variable

        color = feature_dataframe['Random Forest feature importances'].values,

        colorscale='Portland',

        showscale=True

    ),

    text = feature_dataframe['features'].values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Random Forest Feature Importance',

    hovermode= 'closest',

#     xaxis= dict(

#         title= 'Pop',

#         ticklen= 5,

#         zeroline= False,

#         gridwidth= 2,

#     ),

    yaxis=dict(

        title= 'Feature Importance',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2010')



# Scatter plot 

trace = go.Scatter(

    y = feature_dataframe['Extra Trees  feature importances'].values,

    x = feature_dataframe['features'].values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 25,

#       size= feature_dataframe['AdaBoost feature importances'].values,

        #color = np.random.randn(500), #set color equal to a variable

        color = feature_dataframe['Extra Trees  feature importances'].values,

        colorscale='Portland',

        showscale=True

    ),

    text = feature_dataframe['features'].values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Extra Trees Feature Importance',

    hovermode= 'closest',

#     xaxis= dict(

#         title= 'Pop',

#         ticklen= 5,

#         zeroline= False,

#         gridwidth= 2,

#     ),

    yaxis=dict(

        title= 'Feature Importance',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2010')



# Scatter plot 

trace = go.Scatter(

    y = feature_dataframe['AdaBoost feature importances'].values,

    x = feature_dataframe['features'].values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 25,

#       size= feature_dataframe['AdaBoost feature importances'].values,

        #color = np.random.randn(500), #set color equal to a variable

        color = feature_dataframe['AdaBoost feature importances'].values,

        colorscale='Portland',

        showscale=True

    ),

    text = feature_dataframe['features'].values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'AdaBoost Feature Importance',

    hovermode= 'closest',

#     xaxis= dict(

#         title= 'Pop',

#         ticklen= 5,

#         zeroline= False,

#         gridwidth= 2,

#     ),

    yaxis=dict(

        title= 'Feature Importance',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2010')



# Scatter plot 

trace = go.Scatter(

    y = feature_dataframe['Gradient Boost feature importances'].values,

    x = feature_dataframe['features'].values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 25,

#       size= feature_dataframe['AdaBoost feature importances'].values,

        #color = np.random.randn(500), #set color equal to a variable

        color = feature_dataframe['Gradient Boost feature importances'].values,

        colorscale='Portland',

        showscale=True

    ),

    text = feature_dataframe['features'].values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Gradient Boosting Feature Importance',

    hovermode= 'closest',

#     xaxis= dict(

#         title= 'Pop',

#         ticklen= 5,

#         zeroline= False,

#         gridwidth= 2,

#     ),

    yaxis=dict(

        title= 'Feature Importance',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2010')



trace = go.Scatter(

    y = feature_dataframe['Xgboost feature importances'].values,

    x = feature_dataframe['features'].values,

    mode='markers',

    marker=dict(

        sizemode = 'diameter',

        sizeref = 1,

        size = 25,

#       size= feature_dataframe['AdaBoost feature importances'].values,

        #color = np.random.randn(500), #set color equal to a variable

        color = feature_dataframe['Xgboost feature importances'].values,

        colorscale='Portland',

        showscale=True

    ),

    text = feature_dataframe['features'].values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'XgboostFeature Importance',

    hovermode= 'closest',

#     xaxis= dict(

#         title= 'Pop',

#         ticklen= 5,

#         zeroline= False,

#         gridwidth= 2,

#     ),

    yaxis=dict(

        title= 'Feature Importance',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2010')

# Create the new column that contains the average of the values.

feature_dataframe['mean'] = feature_dataframe.mean(axis= 1) # axis = 1 computes the mean row-wise

feature_dataframe.head(3)
y = feature_dataframe['mean'].values

x = feature_dataframe['features'].values

data = [go.Bar(

            x= x,

             y= y,

            width = 0.5,

            marker=dict(

               color = feature_dataframe['mean'].values,

            colorscale='Portland',

            showscale=True,

            reversescale = False

            ),

            opacity=0.6

        )]



layout= go.Layout(

    autosize= True,

    title= 'Barplots of Mean Feature Importance',

    hovermode= 'closest',

#     xaxis= dict(

#         title= 'Pop',

#         ticklen= 5,

#         zeroline= False,

#         gridwidth= 2,

#     ),

    yaxis=dict(

        title= 'Feature Importance',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='bar-direct-labels')
Predicted_rf= model_rf.predict(X_test)

Predicted_ada = model_ada.predict(X_test)

Predicted_gb = model_gb.predict(X_test)

Predicted_et = model_et.predict(X_test)

Predicted_svm= model_svc.predict(X_test)

Predicted_xgb= model_xgb.predict(X_test.values)
base_predictions_train = pd.DataFrame( {'RandomForest': Predicted_rf.ravel(),

      'AdaBoost': Predicted_ada.ravel(),

      'GradientBoost': Predicted_gb.ravel(),

      'ExtraTrees': Predicted_et.ravel(),

      'SVM': Predicted_svm.ravel(),

      'XGB': Predicted_xgb.ravel(),

     'Real value': y_test                                

                                        

    })

base_predictions_train.head(10)
acc = [] # list to store all performance metric
model='Random Forest'

test_score = cross_val_score(model_rf, X_train, y_train, cv=FOLDS, scoring='accuracy').mean() # Get recall for each parameter setting

test_recall = recall_score(y_test, Predicted_rf, pos_label=1)

fpr, tpr, thresholds = roc_curve(y_test, Predicted_rf, pos_label=1)

test_auc = auc(fpr, tpr)

acc.append([model,test_score, test_recall, test_auc, fpr, tpr, thresholds])



model='AdaBoost'

test_score = cross_val_score(model_ada, X_train, y_train, cv=FOLDS, scoring='accuracy').mean() # Get recall for each parameter setting

test_recall = recall_score(y_test, Predicted_ada, pos_label=1)

fpr, tpr, thresholds = roc_curve(y_test, Predicted_ada, pos_label=1)

test_auc = auc(fpr, tpr)

acc.append([model, test_score,test_recall, test_auc, fpr, tpr, thresholds])



model='Gradient Boosting'

test_score = cross_val_score(model_gb, X_train, y_train, cv=FOLDS, scoring='accuracy').mean() # Get recall for each parameter setting

test_recall = recall_score(y_test, Predicted_gb, pos_label=1)

fpr, tpr, thresholds = roc_curve(y_test, Predicted_gb, pos_label=1)

test_auc = auc(fpr, tpr)

acc.append([model, test_score,test_recall, test_auc, fpr, tpr, thresholds])



model='ExtraTrees'

test_score = cross_val_score(model_et, X_train, y_train, cv=FOLDS, scoring='accuracy').mean() # Get recall for each parameter setting

test_recall = recall_score(y_test, Predicted_et, pos_label=1)

fpr, tpr, thresholds = roc_curve(y_test, Predicted_et, pos_label=1)

test_auc = auc(fpr, tpr)

acc.append([model, test_score, test_recall, test_auc, fpr, tpr, thresholds])



model='SVM'

test_score = cross_val_score(model_svc, X_train, y_train, cv=FOLDS, scoring='accuracy').mean() # Get recall for each parameter setting

test_recall = recall_score(y_test, Predicted_svm, pos_label=1)

fpr, tpr, thresholds = roc_curve(y_test, Predicted_svm, pos_label=1)

test_auc = auc(fpr, tpr)

acc.append([model, test_score, test_recall, test_auc, fpr, tpr, thresholds])



model='Xgboost'

test_score = cross_val_score(model_xgb, X_train, y_train, cv=FOLDS, scoring='accuracy').mean() # Get recall for each parameter setting

test_recall = recall_score(y_test, Predicted_xgb, pos_label=1)

fpr, tpr, thresholds = roc_curve(y_test, Predicted_xgb, pos_label=1)

test_auc = auc(fpr, tpr)

acc.append([model,test_score, test_recall, test_auc, fpr, tpr, thresholds])

report_performance(model_et)
result = pd.DataFrame(acc, columns=['Model', 'Accuracy', 'Recall', 'AUC', 'FPR', 'TPR', 'TH'])

result[['Model', 'Accuracy', 'Recall', 'AUC']]