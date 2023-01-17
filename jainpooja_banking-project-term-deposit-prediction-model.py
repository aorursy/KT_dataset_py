import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier ,RandomForestClassifier ,GradientBoostingClassifier

from xgboost import XGBClassifier 

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.linear_model import Ridge,Lasso

from sklearn.metrics import roc_auc_score ,mean_squared_error,accuracy_score,classification_report,roc_curve,confusion_matrix

import warnings

warnings.filterwarnings('ignore')

from scipy.stats.mstats import winsorize

from sklearn.feature_selection import RFE

from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns',None)

import six

import sys

sys.modules['sklearn.externals.six'] = six
# accessing to the folder where the file is stored

path = '../input/banking-project-term-deposit/preprocessed_data.csv'



# Load the dataframe

dataframe = pd.read_csv(path)



print('Shape of the data is: ',dataframe.shape)



dataframe.head()



# Predictors

X = dataframe.iloc[:,:-1]



# Target

y = dataframe.iloc[:,-1]



# Dividing the data into train and test subsets

x_train,x_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=42)

# run Logistic Regression model

model = LogisticRegression()

# fitting the model

model.fit(x_train, y_train)

# predicting the values

y_scores = model.predict(x_val)





# getting the auc roc curve

auc = roc_auc_score(y_val, y_scores)

print('Classification Report:')

print(classification_report(y_val,y_scores))

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, y_scores)

print('ROC_AUC_SCORE is',roc_auc_score(y_val, y_scores))

    

#fpr, tpr, _ = roc_curve(y_test, predictions[:,1])

    

plt.plot(false_positive_rate, true_positive_rate)

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.title('ROC curve')

plt.show()
# Run Decision Tree Classifier

model = DecisionTreeClassifier()



model.fit(x_train, y_train)

y_scores = model.predict(x_val)

auc = roc_auc_score(y_val, y_scores)

print('Classification Report:')

print(classification_report(y_val,y_scores))

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, y_scores)

print('ROC_AUC_SCORE is',roc_auc_score(y_val, y_scores))

    

#fpr, tpr, _ = roc_curve(y_test, predictions[:,1])

    

plt.plot(false_positive_rate, true_positive_rate)

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.title('ROC curve')

plt.show()
from sklearn import tree

from sklearn.tree import export_graphviz # display the tree within a Jupyter notebook

from IPython.display import SVG

from graphviz import Source

from IPython.display import display

from ipywidgets import interactive, IntSlider, FloatSlider, interact

import ipywidgets

from IPython.display import Image

from subprocess import call

import matplotlib.image as mpimg
@interact

def plot_tree(crit=["gini", "entropy"],

              split=["best", "random"],

              depth=IntSlider(min=1,max=30,value=2, continuous_update=False),

              min_split=IntSlider(min=2,max=5,value=2, continuous_update=False),

              min_leaf=IntSlider(min=1,max=5,value=1, continuous_update=False)):

    

    estimator = DecisionTreeClassifier(random_state=0,

                                       criterion=crit,

                                       splitter = split,

                                       max_depth = depth,

                                       min_samples_split=min_split,

                                       min_samples_leaf=min_leaf)

    estimator.fit(x_train, y_train)

    print('Decision Tree Training Accuracy: {:.3f}'.format(accuracy_score(y_train, estimator.predict(x_train))))

    print('Decision Tree Test Accuracy: {:.3f}'.format(accuracy_score(y_val, estimator.predict(x_val))))



    graph = Source(tree.export_graphviz(estimator,

                                        out_file=None,

                                        feature_names=x_train.columns,

                                        class_names=['0', '1'],

                                        filled = True))

    

    display(Image(data=graph.pipe(format='png')))

    

    return estimator

# run Random Forrest Classifier

model = RandomForestClassifier()



model.fit(x_train, y_train)

y_scores = model.predict(x_val)

auc = roc_auc_score(y_val, y_scores)

print('Classification Report:')

print(classification_report(y_val,y_scores))

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, y_scores)

print('ROC_AUC_SCORE is',roc_auc_score(y_val, y_scores))

    

#fpr, tpr, _ = roc_curve(y_test, predictions[:,1])

    

plt.plot(false_positive_rate, true_positive_rate)

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.title('ROC curve')

plt.show()
# Selecting 8 number of features

#   selecting models

models = LogisticRegression()

#   using  rfe and selecting 8 features

rfe = RFE(models,8)

#   fitting the model

rfe = rfe.fit(X,y)

#   ranking features

feature_ranking = pd.Series(rfe.ranking_, index=X.columns)

plt.show()

print('Features  to be selected for Logistic Regression model are:')

print(feature_ranking[feature_ranking.values==1].index.tolist())

print('===='*30)



# Selecting 8 number of features

# Random Forrest classifier model

models = RandomForestClassifier()

#   using  rfe and selecting 8 features

rfe = RFE(models,8)

#   fitting the model

rfe = rfe.fit(X,y)

#   ranking features

feature_ranking = pd.Series(rfe.ranking_, index=X.columns)

plt.show()

print('Features  to be selected for Random Forrest Classifier are:')

print(feature_ranking[feature_ranking.values==1].index.tolist())

print('===='*30)



# splitting the data into train and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# selecting the data

rfc = RandomForestClassifier(random_state=42)

# fitting the data

rfc.fit(X_train, y_train)

# predicting the data

y_pred = rfc.predict(X_test)

# feature importances

rfc_importances = pd.Series(rfc.feature_importances_, index=X.columns).sort_values().tail(10)

# plotting bar chart according to feature importance

rfc_importances.plot(kind='bar')

plt.show()
# splitting the data

x_train,x_val,y_train,y_val = train_test_split(X,y, test_size=0.3, random_state=42, stratify=y)

# selecting the classifier

rfc = RandomForestClassifier()

# selecting the parameter

param_grid = { 

'max_features': ['auto', 'sqrt', 'log2'],

'max_depth' : [4,5,6,7,8],

'criterion' :['gini', 'entropy']

             }

# using grid search with respective parameters

grid_search_model = GridSearchCV(rfc, param_grid=param_grid)

# fitting the model

grid_search_model.fit(x_train, y_train)

# printing the best parameters

print('Best Parameters are:',grid_search_model.best_params_)
from sklearn.metrics import roc_auc_score,roc_curve,classification_report

from sklearn.model_selection import cross_val_score

from imblearn.over_sampling import SMOTE

from yellowbrick.classifier import roc_auc





# A function to use smote

def grid_search_random_forrest_best(dataframe,target):

    

    # splitting the data

    x_train,x_val,y_train,y_val = train_test_split(dataframe,target, test_size=0.3, random_state=42)

    

    # Applying Smote on train data for dealing with class imbalance

    smote = SMOTE()

    

    X_sm, y_sm =  smote.fit_sample(x_train, y_train)

    

    rfc = RandomForestClassifier(n_estimators=11, max_features='auto', max_depth=8, criterion='entropy',random_state=42)

    

    rfc.fit(X_sm, y_sm)

    y_pred = rfc.predict(x_val)

    print(classification_report(y_val, y_pred))

    print(confusion_matrix(y_val, y_pred))

    visualizer = roc_auc(rfc,X_sm,y_sm,x_val,y_val)





grid_search_random_forrest_best(X,y)
grid_search_random_forrest_best(X[['age', 'job', 'education', 'month', 'day_of_week', 'duration', 'campaign', 'poutcome']],y)
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import VotingClassifier





# splitting the data  

x_train,x_val,y_train,y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# using smote

smote = SMOTE()

X_sm, y_sm =  smote.fit_sample(x_train, y_train)

# models to use for ensembling  

model1 = RandomForestClassifier()

model3 = GradientBoostingClassifier()

model2 = LogisticRegression()

# fitting the model

model = VotingClassifier(estimators=[('rf', model1), ('lr', model2), ('xgb',model3)], voting='soft')

model.fit(X_sm,y_sm)

# predicting balues and getting the metrics

y_pred = model.predict(x_val)

print(classification_report(y_val, y_pred))

print(confusion_matrix(y_val, y_pred))

visualizer = roc_auc(model,X_sm,y_sm,x_val,y_val)
# Preprocessed Test File

test = pd.read_csv('../input/banking-project-term-deposit/new_train.csv')

test.head()

smote = SMOTE()



X_sm, y_sm =  smote.fit_sample(x_train, y_train)





rfc = RandomForestClassifier()

# selecting the parameter

param_grid = { 

'max_features': ['auto', 'sqrt', 'log2'],

'max_depth' : [4,5,6,7,8],

'criterion' :['gini', 'entropy']

             }

# using grid search with respective parameters

grid_search_model = GridSearchCV(rfc, param_grid=param_grid)



# fitting the model

grid_search_model.fit(X_sm, y_sm)

    

# Predict on the preprocessed test file

y_pred = grid_search.predict(test)

    

#prediction = pd.DataFrame(y_pred,columns=['y'])

#submission = pd.concat([Id,prediction['y']],1)



#submission.to_csv('submission.csv',index=False)