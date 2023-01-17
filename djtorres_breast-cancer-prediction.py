import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# keeps the plots in one place. calls image as static pngs

%matplotlib inline 

import matplotlib.pyplot as plt # side-stepping mpl backend

import matplotlib.gridspec as gridspec # subplots

import mpld3 as mpl



#Import models from scikit learn module:

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import KFold   #For K-fold cross validation

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn import metrics
df = pd.read_csv("../input/data.csv",header = 0)

df.head()
df.drop('id',axis=1,inplace=True)

df.drop('Unnamed: 32',axis=1,inplace=True)

# size of the dataframe

len(df)
df.diagnosis.unique()
df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})

df.head()
df.describe()
df.describe()

plt.hist(df['diagnosis'])

plt.title('Diagnosis (M=1 , B=0)')

plt.show()
features_mean=list(df.columns[1:11])

# split dataframe into two based on diagnosis

dfM=df[df['diagnosis'] ==1]

dfB=df[df['diagnosis'] ==0]
#Stack the data

plt.rcParams.update({'font.size': 8})

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8,10))

axes = axes.ravel()

for idx,ax in enumerate(axes):

    ax.figure

    binwidth= (max(df[features_mean[idx]]) - min(df[features_mean[idx]]))/50

    ax.hist([dfM[features_mean[idx]],dfB[features_mean[idx]]], bins=np.arange(min(df[features_mean[idx]]), max(df[features_mean[idx]]) + binwidth, binwidth) , alpha=0.5,stacked=True, normed = True, label=['M','B'],color=['r','g'])

    ax.legend(loc='upper right')

    ax.set_title(features_mean[idx])

plt.tight_layout()

plt.show()
traindf, testdf = train_test_split(df, test_size = 0.3)
#Generic function for making a classification model and accessing the performance. 

# From AnalyticsVidhya tutorial

def classification_model(model, data, predictors, outcome):

  #Fit the model:

  model.fit(data[predictors],data[outcome])

  

  #Make predictions on training set:

  predictions = model.predict(data[predictors])

  

  #Print accuracy

  accuracy = metrics.accuracy_score(predictions,data[outcome])

  print("Accuracy : %s" % "{0:.3%}".format(accuracy))



  #Perform k-fold cross-validation with 5 folds

  kf = KFold(data.shape[0], n_folds=5)

  error = []

  for train, test in kf:

    # Filter training data

    train_predictors = (data[predictors].iloc[train,:])

    

    # The target we're using to train the algorithm.

    train_target = data[outcome].iloc[train]

    

    # Training the algorithm using the predictors and target.

    model.fit(train_predictors, train_target)

    

    #Record error from each cross-validation run

    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))

    

    print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

    

  #Fit the model again so that it can be refered outside the function:

  model.fit(data[predictors],data[outcome]) 
predictor_var = ['radius_mean','perimeter_mean','area_mean','compactness_mean','concave points_mean']

outcome_var='diagnosis'

model=LogisticRegression()

classification_model(model,traindf,predictor_var,outcome_var)
predictor_var = ['radius_mean']

model=LogisticRegression()

classification_model(model,traindf,predictor_var,outcome_var)
predictor_var = ['radius_mean','perimeter_mean','area_mean','compactness_mean','concave points_mean']

model = DecisionTreeClassifier()

classification_model(model,traindf,predictor_var,outcome_var)
predictor_var = ['radius_mean']

model = DecisionTreeClassifier()

classification_model(model,traindf,predictor_var,outcome_var)
# Use all the features of the nucleus

predictor_var = features_mean

model = RandomForestClassifier(n_estimators=100,min_samples_split=25, max_depth=7, max_features=2)

classification_model(model, traindf,predictor_var,outcome_var)
#Create a series with feature importances:

featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)

print(featimp)
# Using top 5 features

predictor_var = ['concave points_mean','area_mean','radius_mean','perimeter_mean','concavity_mean',]

model = RandomForestClassifier(n_estimators=100, min_samples_split=25, max_depth=7, max_features=2)

classification_model(model,traindf,predictor_var,outcome_var)
predictor_var =  ['radius_mean']

model = RandomForestClassifier(n_estimators=100)

classification_model(model, traindf,predictor_var,outcome_var)
# Use all the features of the nucleus

predictor_var = features_mean

model = RandomForestClassifier(n_estimators=100,min_samples_split=25, max_depth=7, max_features=2)

classification_model(model, testdf,predictor_var,outcome_var)