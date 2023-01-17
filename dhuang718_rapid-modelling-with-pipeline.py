import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
#import dataset and specify X (independent variables) and y (dependent variable)
df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
df = df.sample(frac=1).reset_index(drop=True)
df.head()
y = df['target']
X = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
       'oldpeak', 'slope', 'ca', 'thal']]
#import the pipeline
from sklearn.pipeline import Pipeline

#select preprocessing methods
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

#numerical pipeline
num_trans = Pipeline(steps = [('imputer', KNNImputer(n_neighbors = 2, weights = 'uniform')),
                              ('scaler', StandardScaler(copy = True))])

#categorical pipeline
cat_trans = Pipeline(steps = [('imputer', KNNImputer(n_neighbors = 2, weights = 'uniform')),
                              ('onehot', OneHotEncoder(drop='first'))])
numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_trans, numeric_features),
        ('cat', cat_trans, categorical_features)])
#select models 
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

#select validation method
from sklearn.model_selection import cross_val_score
#list all the models to be fitted
classifiers = [SVC(),
               LinearSVC(),
               LogisticRegression(),
               GaussianNB(),
               RandomForestClassifier(),
               GradientBoostingClassifier(),
               MLPClassifier(max_iter=2000),
               KNeighborsClassifier(n_neighbors = 6)]

#string values of the models
classifiers_names = ['SVC',
                     'LinearSVC',
                     'LogisticRegression', 
                     'GaussianNB', 
                     'RandomForestClassifier', 
                     'GradientBoostingClassifier',
                     'MLPClassifier',
                     'KNeighborsClassifier']
#empty array to hold peformance of all model
val_storage = []
#loop through all models and run each one according to the pipeline steps
for models in classifiers:
    #list steps
    steps = [('preprocess', preprocessor),
             ('model', models)]
    
    #run the pipeline
    pipeline = Pipeline(steps)
    
    #performance metrics
    scores = cross_val_score(pipeline, X, y, cv=4)
    avg_score = scores.mean()
    performance = str(round((avg_score)*100,2)) + ' %' + ' +/- ' + str(round((scores.max()-avg_score)*100,2)) + ' %'
    val_storage.append(performance)
#display performance 
df_metric = pd.DataFrame(data = {'Models:' : classifiers_names, 
                                 'Accuracy:' : val_storage})
df_metric