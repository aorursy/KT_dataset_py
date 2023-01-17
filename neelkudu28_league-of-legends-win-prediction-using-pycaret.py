import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import scipy.stats as ss

from scipy.stats import chi2_contingency

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

from sklearn.ensemble import RandomForestClassifier

!pip install pycaret

from pycaret.classification import *
data=pd.read_csv("../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv")

data.head()
data.shape
#Checking for missing values

data.isnull().sum()
#Checking datatypes of individual feature

data.dtypes
#Dropping 'gameId' feature as it's not required in model building and prediction

data.drop(["gameId"],1,inplace=True)
blue_features=[]

red_features=[]

for col in list(data):

    if(col[0]=='r'):

        red_features.append(col)

    if(col[0]=='b'):

        blue_features.append(col)
blues=data[blue_features]

red_features.append("blueWins")

reds=data[red_features]
#Dividing features into numerical and categorical features

categorical_reds=[]

categorical_blues=[]

numerical_reds=[]

numerical_blues=[]

for col in list(reds):

    if(len(reds[col].unique())<=30):

        categorical_reds.append(col)

    else:

        numerical_reds.append(col)



for col in list(blues):

    if(len(blues[col].unique())<=30):

        categorical_blues.append(col)

    else:

        numerical_blues.append(col)
print("Number of Categorical Features for Blue Team",len(categorical_blues))

print("Number of Categorical Features for Red Team",len(categorical_reds))

print("Number of Numerical Features for Blue Team",len(numerical_blues))

print("Number of Numerical Features for Red Team",len(numerical_reds))
def Chi_square(col_1,col_2):

    X=reds[col_1].astype('str')

    Y=reds[col_2].astype('str')

    observed_values=pd.crosstab(Y,X)

    chi2, p, dof, expected = ss.chi2_contingency(observed_values)

    if(p>0.05):

        print(col_1," is not required")

    else:

        print(col_1," is required")

        

for col in categorical_reds:

    Chi_square(col,"blueWins")
def Chi_square(col_1,col_2):

    X=blues[col_1].astype('str')

    Y=blues[col_2].astype('str')

    observed_values=pd.crosstab(Y,X)

    chi2, p, dof, expected = ss.chi2_contingency(observed_values)

    if(p>0.05):

        print(col_1," is not required")

    else:

        print(col_1," is required")

        

for col in categorical_blues:

    Chi_square(col,"blueWins")
X=reds[numerical_reds]

y=le.fit_transform(reds["blueWins"])



import statsmodels.api as sm

cols_red = list(X.columns)

pmax = 1

while (pmax>0.05):

    p=[]

    X_1 = X[cols_red]

    X_1 = sm.add_constant(X_1)

    model = sm.OLS(y,X_1).fit()

    p = pd.Series(model.pvalues.values[1:],index = cols_red)      

    pmax = max(p)

    feature_with_p_max = p.idxmax()

    if(pmax>0.05):

        cols_red.remove(feature_with_p_max)

    else:

        breakselected_features_BE = cols_red

print("Best features using Backward Elimination: ",cols_red)
X=blues[numerical_blues]

y=le.fit_transform(blues["blueWins"])



import statsmodels.api as sm

cols_blue = list(X.columns)

pmax = 1

while (pmax>0.05):

    p=[]

    X_1 = X[cols_blue]

    X_1 = sm.add_constant(X_1)

    model = sm.OLS(y,X_1).fit()

    p = pd.Series(model.pvalues.values[1:],index = cols_blue)      

    pmax = max(p)

    feature_with_p_max = p.idxmax()

    if(pmax>0.05):

        cols_blue.remove(feature_with_p_max)

    else:

        breakselected_features_BE = cols_blue

print("Best features using Backward Elimination: ",cols_blue)
Xr_rfc=reds.drop(["blueWins"],1)

yr_rfc=reds["blueWins"]
rfc_r=RandomForestClassifier(random_state=0)

rfc_r.fit(Xr_rfc,yr_rfc)
plt.figure(figsize=(10,10))

plt.barh(list(Xr_rfc),rfc_r.feature_importances_)

plt.title("Feature Imporatance using Random Forest Classifier")

plt.ylabel("Features")

plt.xlabel('Feature Importance Value')
Xb_rfc=blues.drop(["blueWins"],1)

yb_rfc=blues["blueWins"]
rfc_b=RandomForestClassifier(random_state=0)

rfc_b.fit(Xb_rfc,yb_rfc)
plt.figure(figsize=(10,10))

plt.barh(list(Xb_rfc),rfc_b.feature_importances_)
models=setup(data=blues,

             categorical_features=categorical_blues.remove('blueWins'),

             ignore_features=list(set(numerical_blues)-set(cols_blue)),

             target='blueWins',

             silent=True,

             session_id=269)
model_results=compare_models()

model_results
logreg_model=create_model('lr')
tunned_logreg_model=tune_model('lr')
plot_model(estimator=tunned_logreg_model,plot='parameter')
plot_model(estimator=tunned_logreg_model,plot='feature')
plot_model(estimator=tunned_logreg_model,plot='pr')
plot_model(estimator=tunned_logreg_model,plot='confusion_matrix')
plot_model(estimator=tunned_logreg_model,plot='class_report')
plot_model(tunned_logreg_model)
model_red=setup(data=reds,

               categorical_features=categorical_reds.remove('blueWins'),

               ignore_features=list(set(numerical_reds)-set(cols_red)),

               target='blueWins',

               silent=True,

               session_id=299)
compare_models()
logreg_model=create_model('lr')
tunned_lr_model=tune_model('lr')
plot_model(estimator=tunned_lr_model,plot='parameter')
plot_model(estimator=tunned_lr_model,plot='feature')
plot_model(estimator=tunned_lr_model,plot='confusion_matrix')
plot_model(estimator=tunned_lr_model,plot='pr')
plot_model(estimator=tunned_lr_model,plot='class_report')
plot_model(tunned_lr_model)