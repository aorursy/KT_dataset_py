# Install Yellowbrick Library

!pip install --upgrade pip -q

!pip install yellowbrick -q
# Generic

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, gc, warnings

warnings.filterwarnings("ignore")



# SKlearn

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold,StratifiedKFold



from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, SGDClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB



'''Yellowbrick Libraries'''



# Model Selection

from yellowbrick.classifier import ClassificationReport

from yellowbrick.model_selection import ValidationCurve,LearningCurve, CVScores, FeatureImportances, RFECV



# Feature Analysis Visualizers

from yellowbrick.features import JointPlotVisualizer, jointplot

from yellowbrick.features import rank1d, rank2d



# Target Visualizers Imports

from yellowbrick.target import BalancedBinningReference

from yellowbrick.target import ClassBalance

from yellowbrick.target import FeatureCorrelation, feature_correlation



# Confusion Matrix 

from yellowbrick.classifier import ConfusionMatrix



# ROC AUC

from yellowbrick.classifier import ROCAUC



# Precision - Recall Curves

from yellowbrick.classifier import PrecisionRecallCurve, prcurve



# Discrimination Threshold

from yellowbrick.classifier import DiscriminationThreshold



# Tabular Data

from tabulate import tabulate



# Matplotlib

import matplotlib.pyplot as plt
# Abalone Data Load

url = '../input/all-datasets-for-practicing-ml/Class/Class_Abalone.csv'

df = pd.read_csv(url, header='infer')



# Selecting only Male & Female

df = df[df['Sex'].isin(['M','F'])]



# Total Records

print("Total Records: ", df.shape[0])



# Records per Sex

print("Records per Sex:\n",df.Sex.value_counts())



# Feature Extraction - Label Encoding

encoder = LabelEncoder()

columns = df.columns

df['Sex']= encoder.fit_transform(df['Sex']) 



# Feature & Target Selection

target = ['Sex']   

features = columns [1:]



X = df[features]

y = df[target]



# Dataset Split

''' Training = 90% & Validation = 10%  '''

test_size = 0.1

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=0, shuffle=True) 
# Define SKLearn Models



models = [

           DecisionTreeClassifier(random_state=42),

           KNeighborsClassifier(),

           LinearDiscriminantAnalysis(),

           GaussianNB(),

           SVC(gamma='auto',verbose=0),

           RandomForestClassifier(n_estimators=300, verbose=0)

         ]
# Function to Calculate Scores



def calc_scores(X, y, estimator):

    

    kfold = KFold(n_splits=10, random_state=None, shuffle=False)

    cross_val_results = cross_val_score(estimator, X, y, cv=kfold, scoring='f1', verbose=0)

    

    f1_mean = cross_val_results.mean()

    model_name = estimator.__class__.__name__

    print(model_name, ":--", '{:.3}'.format(f1_mean))

    



# Calculate Scores

print("*** Mean F1 Scores ***")

for model in models:

    calc_scores(X_train, y_train, model)
tab_data = []

def calc_scores_viz(X, y, estimator):

    

    kfold = KFold(n_splits=10, random_state=None, shuffle=False)

    cross_val_results = cross_val_score(estimator, X, y, cv=kfold, scoring='f1', verbose=0)

    

    f1_mean = cross_val_results.mean()

    model_name = estimator.__class__.__name__

    #print(model_name, ":--", '{:.3}'.format(f1_mean))

    tab_data.append([model_name, '{:.3}'.format(f1_mean)])

    

    # Instantiate the classification model and visualizer

    visualizer = ClassificationReport(estimator, classes=['Female', 'Male'],

        cmap="PuBu", size=(600, 360))

    visualizer.fit(X, y)

    visualizer.score(X, y)

    visualizer.show()

    
for model in models:

    calc_scores_viz(X_train, y_train, model)



print(tabulate(tab_data, headers=['Classifiers','Mean F1 Score'], tablefmt='pretty'))
# Garbage Collect

gc.collect()
_, axes = plt.subplots(ncols=2, figsize=(18,10))



rank1d(X, ax=axes[0], show=False, orient='h', color='g')

rank2d(X, ax=axes[1], show=False, colormap='BuPu')

plt.show() 
# Garbage Collect

gc.collect()
# Length vs Whole_Weight

vis1 = JointPlotVisualizer(columns=["Length", "Whole_Weight"])



vis1.fit_transform(X, y)        # Fit and transform the data

vis1.show()    
# Diameter vs Rings

vis2 = JointPlotVisualizer(columns=["Diameter", "Rings"])



vis2.fit_transform(X, y)        # Fit and transform the data

vis2.show()    
# Garbage Collect

gc.collect()
# Balanced Binning Reference

visualizer = BalancedBinningReference()



visualizer.fit(y.T.squeeze())        # Fit the data to the visualizer

visualizer.show()        # Finalize and render the figure
# Class Balance

visualizer = ClassBalance(labels=["Male", "Female"])



visualizer.fit(y.T.squeeze())        # Fit the data to the visualizer

visualizer.show() 
'''Feature Correlation'''



visualizer = FeatureCorrelation(method='mutual_info-classification', feature_names=list(features), sort=True)



visualizer.fit(X, np.array(y.T.squeeze()))        # Fit the data to the visualizer

visualizer.show()              # Finalize and render the figure
# Garbage Collect

gc.collect()
# Classification Report - SVC



model = SVC(gamma='auto',verbose=0)

visualizer = ClassificationReport(model, classes=['Male','Female'], support=True, cmap='PuBu')



visualizer.fit(X_train, y_train)        # Fit the visualizer and the model

visualizer.score(X_val, y_val)        # Evaluate the model on the test data

visualizer.show() 
# Confusion Matrix - Abalone with Model = SVC



ab_cm = ConfusionMatrix(model, classes=['Male','Female'],label_encoder={1: 'Male', 2: 'Female'}, cmap='PuBu')



ab_cm.fit(X_train, y_train)

ab_cm.score(X_val, y_val)



ab_cm.show()
# ROC - AUC Curve for Abalone Dataset with model = GaussianNB



visualizer = ROCAUC(GaussianNB(), classes=['Male','Female'])



visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer

visualizer.score(X_val, y_val)        # Evaluate the model on the test data

visualizer.show()  
# Precision - Recall Curves for Abalone Dataset with model = SVC



viz = PrecisionRecallCurve(model)

viz.fit(X_train, y_train)

viz.score(X_val, y_val)

viz.show()
# Discrimination Threshold for Abalone Dataset with model = SVC



visualizer = DiscriminationThreshold(model)



visualizer.fit(X, y)        # Fit the data to the visualizer

visualizer.show()           # Finalize and render the figure
# Garbage Collection

gc.collect()
# Validation Curve for Abalone Dataset with Model = SVC

cv = StratifiedKFold(12)

param_range = np.logspace(-6, -1, 12)



viz = ValidationCurve(SVC(), param_name="gamma", param_range=param_range,logx=True, cv=cv, scoring="f1_weighted", n_jobs=8,)



viz.fit(X, y)

viz.show()
# Learning Curve for Abalone Dataset with GaussianNB



cv = StratifiedKFold(n_splits=12)

sizes = np.linspace(0.3, 1.0, 10)



# Instantiate the classification model and visualizer

model = GaussianNB()

visualizer = LearningCurve(model, cv=cv, scoring='f1_weighted', train_sizes=sizes, n_jobs=4)



visualizer.fit(X, y)        # Fit the data to the visualizer

visualizer.show()  
# Cross Validation Scores for Abalone Dataset with Model = SVC

cv = StratifiedKFold(n_splits=12, random_state=42)



# Instantiate the classification model and visualizer

model = SVC(gamma='auto',verbose=0)

visualizer = CVScores(model, cv=cv, scoring='f1_weighted', color='g')



visualizer.fit(X, y)        # Fit the data to the visualizer

visualizer.show()           # Finalize and render the figure
# Feature Importance for Abalone Dataset with Model = RandomForest



viz = FeatureImportances(RandomForestClassifier(n_estimators=300, verbose=0), labels=list(features), relative=False,colors='g')



# Fit and show the feature importances

viz.fit(X, y)

viz.show()
# Recursive Feature Elimination for Abalone Dataset with Model = SVC



visualizer = RFECV(SVC(kernel='linear', C=1))



visualizer.fit(X, y)        # Fit the data to the visualizer

visualizer.show()    
# Garbage Collect

gc.collect()