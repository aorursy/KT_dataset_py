import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sb

from sklearn.preprocessing import RobustScaler, Imputer

from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import train_test_split, KFold, GridSearchCV

from sklearn.metrics import f1_score, make_scorer

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import VarianceThreshold

from sklearn.manifold import TSNE

pd.set_option('display.max_columns', 200) # This allow us to see every column
data = pd.read_csv("../input/equipfails/equip_failures_training_set.csv")  # Read Data into a Pandas DataFrame for Analysis
data.describe(include = 'all')
def clean_data(df):

    df.replace('na', np.nan, inplace = True) #Replacing "na" with Numpy's Nan for Computation

    df = df.astype(float)

    r_c = []

    for i in range(df.shape[1]): #Removes features with more than 50% of the samples missing 

        if df[df.columns[i]].count() > (0.5*df.shape[0]):

            r_c.append(df.columns[i])

    imp_data = Imputer(strategy = 'median').fit_transform(df[r_c]) #Imputs missing data with median value of corresponding feature

    ret_data = pd.DataFrame(imp_data, columns = r_c)

    return ret_data, r_c #The Index kept are stored for use when readying the deployment dataset (i.e the test dataset)
data_fresh, index_in = clean_data(data)
#The Total Data is split into features (X) and a target set, the feature set correponds to a matrix of every eligible sensor reading

#y corresponds to a failure (1) or normal conditions(0)

X = data_fresh.iloc[:,2:] 

y = np.array(data_fresh.iloc[:,1])
def scale(df,method): #This function scales a dataset by the method set by the user

    scale_d = method.fit_transform(df)

    return scale_d
robust = RobustScaler(quantile_range=(2.5,97.5)) #Defining the Robust Scaler
X_scale = scale(X,robust) #Scaling Feature Set
var = VarianceThreshold()
var.fit(X_scale) 
var_feat = var.get_support() #This gets all index with non-zero variance, we stored it for use for the readying the deployment dataset
X_new_feat = X_scale[:,var_feat] #Extract all features with non-zero variance
X_train, X_test, y_train, y_test = train_test_split(X_new_feat,y, test_size = 0.3, random_state = 0, stratify = y)
def train_model(X_train,X_test,y_train,y_test, model): # This function trains the model

    model.fit(X_train,y_train)

    y_hat = model.predict(X_test)

    return ('The accuracy score is: ' + str(model.score(X_test, y_test)) +

            ' The F1 score is: ' + str(f1_score(y_test, y_hat)))
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression
#Initiate our models, all hyperparameters used were based on extensive GridSearch and trial and error

lr = LogisticRegression()

rfc = RandomForestClassifier(n_estimators=200, n_jobs = -1, max_features = 15)

xgb = XGBClassifier(n_jobs=-1)
train_model(X_train, X_test, y_train, y_test, lr)
train_model(X_train, X_test, y_train, y_test, rfc)
train_model(X_train, X_test, y_train, y_test, xgb)
test_data = pd.read_csv("../input/equipfails/equip_failures_test_set.csv")
test_d = test_data.iloc[:,1:] #Select all features except id
test_in = test_d[index_in[2:]] #Filter based on features
test_in.replace('na',np.nan, inplace = True) #Replace na with Nan as was done with TRAIN data
test_imp = Imputer(strategy = 'median').fit_transform(test_in) #Replace missing values with with median of feature
test_final = test_imp[:,var_feat]
test_final = robust.fit_transform(test_final) #Scale data using Robust scaler
rfc.fit(X_new_feat, y)
y_pred = rfc.predict(test_final) # We chose the Random Forest Classifier as we believe its less likely to overfit compared to Random Forest
export = test_data[['id']]
export['target'] = y_pred
print(export)