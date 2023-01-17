# 1. Prepare Problem

# 1.a) Load libraries

import sys

import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import display

%matplotlib inline

import plotly.offline as py

import plotly.graph_objs as go

import plotly.tools as tls

import plotly.io as pio

pio.renderers.default = "svg"

py.init_notebook_mode(connected=True)

import warnings

warnings.filterwarnings('ignore')

from scipy.stats import skew

from scipy.stats import kurtosis



# Load libraries for evaluating algorithms

from pandas import set_option

from pandas.plotting import scatter_matrix

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler



from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.ensemble import VotingClassifier



# 1.b) Load dataset

df = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')

# Getting dataframe columns names

df_name=df.columns
# 2. Summarize Data

# 2.a) Descriptive statistics

print('dimension of data',df.shape)

df.info()

df.head()
df.describe()
# check missing values

sns.heatmap(df.isnull(),cbar=False,yticklabels=False,cmap = 'viridis')

plt.show()
# 2.b) Data visualizations:

# histogram of individual attributes

df.hist(bins=20,figsize=(18,12))

plt.show()
# density, skewness and kurtosis of each attribute

for i in range(len(df.columns)):

    sns.kdeplot(df[df_name[i]], shade=True);

    plt.show()

    print("%s: mean (%f), variance (%f), skewness (%f), kurtosis (%f)" % (df_name[i], np.mean(df[df_name[i]]), np.var(df[df_name[i]]), skew(df[df_name[i]]), kurtosis(df[df_name[i]])))
# box and whisker plots

df.plot(kind= 'box', subplots=True, layout=(4,4), sharex=False, sharey=False,fontsize=8,figsize=(18,12))

plt.show()
#  checking the target distribution

print(df.groupby('Outcome').size())

OutLabels = [str(df['Outcome'].unique()[i]) for i in range(df['Outcome'].nunique())]

OutValues = [df['Outcome'].value_counts()[i] for i in range(df['Outcome'].nunique())]

pie=go.Pie(labels=OutLabels,values=OutValues)

go.Figure([pie])
# visualizations of the interactions between variables :

sns.pairplot(df, hue="Outcome", palette='husl')

plt.show()
fig, ax = plt.subplots(figsize=(18, 12))

ax = sns.heatmap(df.corr(),vmin=-1, vmax=1, annot=True, fmt='.2f', cmap='coolwarm')

plt.show()
# 4. Evaluate Algorithms

# 4.a) Split-out validation dataset 

X =  df[df_name[0:8]]

Y = df[df_name[8]]

X_train, X_test, y_train, y_test =train_test_split(X,Y,test_size=0.2,random_state=1,stratify=df['Outcome']) 

# stratify is used to keep the same distribution of 'Outcome' in the train and test dataset
# 4.b) Data Transforms & Spot-Check Algorithms 

def GetScaledModel(nameOfScaler):

    

    if nameOfScaler == 'standard':

        scaler = StandardScaler()

    elif nameOfScaler =='minmax':

        scaler = MinMaxScaler()



    pipelines = []

    pipelines.append((nameOfScaler+'LR'  , Pipeline([('Scaler', scaler),('LR'  , LogisticRegression())])))

    pipelines.append((nameOfScaler+'LDA' , Pipeline([('Scaler', scaler),('LDA' , LinearDiscriminantAnalysis())])))

    pipelines.append((nameOfScaler+'KNN' , Pipeline([('Scaler', scaler),('KNN' , KNeighborsClassifier())])))

    pipelines.append((nameOfScaler+'CART', Pipeline([('Scaler', scaler),('CART', DecisionTreeClassifier(random_state=2))])))

    pipelines.append((nameOfScaler+'NB'  , Pipeline([('Scaler', scaler),('NB'  , GaussianNB())])))

    pipelines.append((nameOfScaler+'SVM' , Pipeline([('Scaler', scaler),('SVM' , SVC())])))

    pipelines.append((nameOfScaler+'AB'  , Pipeline([('Scaler', scaler),('AB'  , AdaBoostClassifier())])  ))

    pipelines.append((nameOfScaler+'GBM' , Pipeline([('Scaler', scaler),('GMB' , GradientBoostingClassifier(random_state=2))])  ))

    pipelines.append((nameOfScaler+'RF'  , Pipeline([('Scaler', scaler),('RF'  , RandomForestClassifier(random_state=2))])  ))

    pipelines.append((nameOfScaler+'ET'  , Pipeline([('Scaler', scaler),('ET'  , ExtraTreesClassifier(random_state=2))])  ))



    return pipelines 



# Test options and evaluation metric

num_folds = 10

seed = 7

scoring = 'accuracy'



# evaluate each model in turn

def EvaluateAlg(X,y,nameOfScaler):

    results = []

    names = []

    models = GetScaledModel(nameOfScaler)

    for name, model in models:

        kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

        cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)

        results.append(cv_results)

        names.append(name)    

    return results, names



standard_results, standard_names = EvaluateAlg(X_train,y_train,'standard')

minmax_results, minmax_names = EvaluateAlg(X_train,y_train,'minmax')

score = pd.DataFrame({'Model':standard_names, 'Score-mean':[np.mean(i) for i in standard_results], 

                   'Model_2':minmax_names, 'Score-mean_2':[np.mean(i) for i in minmax_results] })

print(score)



# Compare Algorithms

def CompAlg(results, names): 

    fig = plt.figure()

    ax = fig.add_axes([0,0,2,2])

    ax.boxplot(results, labels=names, showmeans=True, meanline=True, meanprops = dict(linestyle='--', linewidth=2.5, color='green'))

    ax.yaxis.grid(True)

    ax.set_title('Algorithm Comparison')

    fig.text(1.8, 1.9, 'mean : ---', color='green', weight='roman',size=14)

    plt.show()

    

CompAlg(standard_results, standard_names)

CompAlg(minmax_results, minmax_names)
# 4.d) Data Cleaning : outliers investigation

df_copy = df.copy()

# using box and whisker plots to visualize outliers

def OutliersBox(df,nameOfFeature):

    

    trace0 = go.Box(

        y = df[nameOfFeature],

        name = "All Points",

        jitter = 0.3,

        pointpos = -1.8,

        boxpoints = 'all',

        marker = dict(

            color = 'rgb(7,40,89)'),

        line = dict(

            color = 'rgb(7,40,89)')

    )



    trace1 = go.Box(

        y = df[nameOfFeature],

        name = "Only Whiskers",

        boxpoints = False,

        marker = dict(

            color = 'rgb(9,56,125)'),

        line = dict(

            color = 'rgb(9,56,125)')

    )



    trace2 = go.Box(

        y = df[nameOfFeature],

        name = "Suspected Outliers",

        boxpoints = 'suspectedoutliers',

        marker = dict(

            color = 'rgb(8,81,156)',

            outliercolor = 'rgba(219, 64, 82, 0.6)',

            line = dict(

                outliercolor = 'rgba(219, 64, 82, 0.6)',

                outlierwidth = 2)),

        line = dict(

            color = 'rgb(8,81,156)')

    )



    data = [trace0,trace1,trace2]



    layout = go.Layout(title = "{} Outliers".format(nameOfFeature))

                       

    go.Figure(data=data,layout=layout).show()



# function to remove outliers

def DropOutliers(df_copy,nameOfFeature):



    valueOfFeature = df_copy[nameOfFeature]

    

    # Calculate Q1 (25th percentile of the data) for the given feature

    Q1 = np.percentile(valueOfFeature, 25.)



    # Calculate Q3 (75th percentile of the data) for the given feature

    Q3 = np.percentile(valueOfFeature, 75.)



    # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)

    step = (Q3-Q1)*1.5

    

    # Index of outliers

    outliers_idx = valueOfFeature[~((valueOfFeature >= Q1 - step) & (valueOfFeature <= Q3 + step))].index.tolist()

    

    # Values of outliers

    outliers_val = valueOfFeature[~((valueOfFeature >= Q1 - step) & (valueOfFeature <= Q3 + step))].values



    # Remove the outliers

    print ("Number of outliers (inc duplicates): {} and outliers: {}".format(len(outliers_idx), outliers_val))

    good_data = df_copy.drop(df_copy.index[outliers_idx]).reset_index(drop = True)

    print ("New dataset with removed outliers has {} samples with {} features each.".format(*good_data.shape))

    return good_data
outliers_clmn=['BloodPressure', 'DiabetesPedigreeFunction', 'Insulin', 'BMI', 'Glucose', 'SkinThickness']

df_clean=df_copy

for i in range(len(outliers_clmn)):

    OutliersBox(df,outliers_clmn[i])

    df_clean = DropOutliers(df_clean,outliers_clmn[i])

    OutliersBox(df_clean,outliers_clmn[i])
# Comparing the accuracy of models after removing outliers

df_clean_name = df_clean.columns

X_c =  df_clean[df_clean_name[0:8]]

Y_c = df_clean[df_clean_name[8]]

X_train_c, X_test_c, y_train_c, y_test_c =train_test_split(X_c,Y_c,test_size=0.2, random_state=0, stratify=df_clean['Outcome'])  



standard_results_c, standard_names_c = EvaluateAlg(X_train_c,y_train_c,'standard')

minmax_results_c, minmax_names_c = EvaluateAlg(X_train_c,y_train_c,'minmax')



score_c = pd.DataFrame({'Model-s_c':standard_names_c, 'Score-s_c':[np.mean(i) for i in standard_results_c],

                        'Model-m_c':minmax_names_c, 'Score-m_c':[np.mean(i) for i in minmax_results_c]})

score=pd.concat([score,score_c],axis=1)
score
CompAlg(standard_results_c, standard_names_c)

CompAlg(minmax_results_c, minmax_names_c)
# 4.e) Feature selection:



clf = ExtraTreesClassifier(n_estimators=250,random_state=2)

clf.fit(X_train_c, y_train_c)



# feature importance

feature_importance = clf.feature_importances_

# make importances relative to max importance

feature_importance = 100.0 * (feature_importance / feature_importance.max())

# indexes from min to max value

sorted_idx = np.argsort(feature_importance)



Variable_importance = pd.DataFrame({'feature':df_clean_name[sorted_idx],'Relative Importance':feature_importance[sorted_idx]})

print(Variable_importance)
df_feature_imp=df_clean.drop(['Insulin','SkinThickness'], axis=1)

df_feature_imp_name = df_feature_imp.columns



X =  df_feature_imp[df_feature_imp_name[0:6]]

Y = df_feature_imp[df_feature_imp_name[6]]

X_train_c_imp, X_test_c_imp, y_train_c_imp, y_test_c_imp =train_test_split(X,Y,test_size=0.2,random_state=0,stratify=df_feature_imp['Outcome'])



minmax_results_c_imp, minmax_names_c_imp = EvaluateAlg(X_train_c_imp,y_train_c_imp,'minmax')



score_c_imp = pd.DataFrame({'Model-m_c_imp':minmax_names_c_imp, 'Score-m_c_imp':[np.mean(i) for i in minmax_results_c_imp]})

print(score_c_imp)



CompAlg(minmax_results_c_imp, minmax_names_c_imp)

score=pd.concat([score,score_c_imp],axis=1)                                     
score
# 5. Improve Accuracy

# 5.a) Algorithm Tuning



def GridSearch(X_train,y_train,model,hyperparameters):

    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

    clf = GridSearchCV(estimator=model,param_grid=hyperparameters,scoring=scoring,cv=kfold)

    # Fit grid search

    best_model = clf.fit(X_train, y_train)

    message = (best_model.best_score_, best_model.best_params_)

    print("Best: %f using %s" % (message))



    return best_model,best_model.best_params_
# model

model = Pipeline([('MinMaxScaler', MinMaxScaler()),('LR', LogisticRegression())])

# create regularization penalty space

penalty = ['l1', 'l2']

# create regularization hyperparameter distribution using uniform distribution

C = [0.01,0.1,0.5,1,1.2,1.4,1.6]

# Create hyperparameter options

hyperparameters = dict(LR__C=C,LR__penalty=penalty)



LR = GridSearch(X_train_c_imp,y_train_c_imp,model,hyperparameters)

#LR_best_model,LR_best_params = GridSearch.GridSearch()

model = Pipeline([('MinMaxScaler', MinMaxScaler()),('KNN', KNeighborsClassifier())])



neighbors = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

hyperparameters = dict(KNN__n_neighbors=neighbors)



KNN = GridSearch(X_train_c_imp,y_train_c_imp,model,hyperparameters)
model = Pipeline([('MinMaxScaler', MinMaxScaler()),('SVM', SVC())])



c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]

kernel_values = [ 'linear' , 'poly' , 'rbf' , 'sigmoid' ]

hyperparameters = dict(SVM__C=c_values, SVM__kernel=kernel_values)



SVC = GridSearch(X_train_c_imp,y_train_c_imp,model,hyperparameters)
model = Pipeline([('MinMaxScaler', MinMaxScaler()),('CART', DecisionTreeClassifier(random_state=2))])



max_depth_value = [2,3,4, None]

max_features_value =  [1,2,3,4]

min_samples_leaf_value = [1,2,3,4]

criterion_value = ["gini", "entropy"]



hyperparameters = dict(CART__max_depth = max_depth_value,

                  CART__max_features = max_features_value,

                  CART__min_samples_leaf = min_samples_leaf_value,

                  CART__criterion = criterion_value)



CART = GridSearch(X_train_c_imp,y_train_c_imp,model,hyperparameters)
model = Pipeline([('MinMaxScaler', MinMaxScaler()),('AB', AdaBoostClassifier())])



learning_rate_value = [.01,.05,.1,.5,1]

n_estimators_value = [50,100,150,200,250,300]



hyperparameters = dict(AB__learning_rate=learning_rate_value, AB__n_estimators=n_estimators_value)



AB = GridSearch(X_train_c_imp,y_train_c_imp,model,hyperparameters)
model = Pipeline([('MinMaxScaler', MinMaxScaler()),('GMB', GradientBoostingClassifier(random_state=2))])



learning_rate_value = [.01,.05,.1,.5,1]

n_estimators_value = [50,100,150,200,250,300]



hyperparameters = dict(GMB__learning_rate=learning_rate_value, GMB__n_estimators=n_estimators_value)



GMB = GridSearch(X_train_c_imp,y_train_c_imp,model,hyperparameters)
model = Pipeline([('Scaler', StandardScaler()),('ET', ExtraTreesClassifier(random_state=2))])



n_estimators_value = [50,60,80,100]

max_features_value =  [2,3]

min_samples_leaf_value = [2,4]

criterion_value = ["gini", "entropy"]



hyperparameters = dict(ET__n_estimators = n_estimators_value,

                  ET__max_features = max_features_value,

                  ET__min_samples_leaf = min_samples_leaf_value,

                  ET__criterion = criterion_value)



ET = GridSearch(X_train_c_imp,y_train_c_imp,model,hyperparameters)
from sklearn.svm import SVC

# 5.b) Ensembles

# Voting ensemble

model1 = LogisticRegression(C=1.2, penalty='l2')



model2 = KNeighborsClassifier(n_neighbors= 7)



model3 = SVC(C =0.1, kernel='poly')



model4 = DecisionTreeClassifier(criterion='entropy', max_depth= 2, max_features=3, min_samples_leaf= 1,random_state=2)



model5 = AdaBoostClassifier(learning_rate= 0.05, n_estimators=200)



model6 = GradientBoostingClassifier(learning_rate=0.5, n_estimators=50,random_state= 2)



model7 = GaussianNB()



model8 = RandomForestClassifier(random_state = 2)



model9 = ExtraTreesClassifier(criterion='entropy', n_estimators= 60, max_features=3, min_samples_leaf= 4,random_state=2)



# create the sub models

estimators = []

estimators.append(('MinMax '+'LR'  , Pipeline([('Scaler', MinMaxScaler()),('LR'  , model1)])))

estimators.append(('MinMax '+'KNN' , Pipeline([('Scaler', MinMaxScaler()),('KNN' , model2)])))

estimators.append(('MinMax '+'SVM' , Pipeline([('Scaler', MinMaxScaler()),('SVM' , model3)])))

estimators.append(('MinMax '+'CART', Pipeline([('Scaler', MinMaxScaler()),('CART', model4)])))

estimators.append(('MinMax '+'AB'  , Pipeline([('Scaler', MinMaxScaler()),('AB'  , model5)])  ))

estimators.append(('MinMax '+'GBM' , Pipeline([('Scaler', MinMaxScaler()),('GMB' , model6)])  ))

estimators.append(('MinMax '+'NB' , Pipeline([('Scaler', MinMaxScaler()),('NB' , model7)])  ))

estimators.append(('MinMax '+'RF'  , Pipeline([('Scaler', MinMaxScaler()),('RF'  , model8)])  ))

estimators.append(('MinMax '+'ET'  , Pipeline([('Scaler', MinMaxScaler()),('ET'  , model9)])  ))





kfold = StratifiedKFold(n_splits=num_folds, random_state=seed, shuffle=True)

ensemble = VotingClassifier(estimators)

results = cross_val_score(ensemble, X_train_c_imp, y_train_c_imp, cv=kfold)

print('Accuracy on train: ',results.mean())
# 6. Finalize Model : Predictions on test dataset



# prepare the model : training the model on the entire training dataset

sc = MinMaxScaler()

rescaledX = sc.fit_transform(X_train_c_imp)

model = VotingClassifier(estimators)

model.fit(rescaledX, y_train_c_imp)



# transform the test dataset

rescaledTestX = sc.transform(X_test_c_imp)

predictions = model.predict(rescaledTestX)



# Evaluate predictions

print(accuracy_score(y_test_c_imp, predictions))

print(confusion_matrix(y_test_c_imp, predictions)) 

print(classification_report(y_test_c_imp, predictions))