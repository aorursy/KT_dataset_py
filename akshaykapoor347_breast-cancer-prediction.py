#Importing the libraries

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
%matplotlib inline

#scikit learn libraries
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


df = pd.read_csv("../input/data.csv")
df.head()
df.info()
#We can see Unnamed:32 has all null values hence we cannot use this column for our analysis and id will also be of no use for analysis
df.drop('Unnamed: 32', axis  = 1, inplace=True)
df.drop('id', axis = 1, inplace= True)

#Let us convert 'Malign' and 'Benign' to 1 and 0 respectively so it will be easier for analysis

df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})
df.describe()
sns.countplot(df['diagnosis'])
df.columns
#The mean, standard error, and "worst" or largest (mean of the three largest values) of these features were computed for each image,resulting in 30 features.
#For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.
#more info at https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names

first = list(df.columns[1:10])
second = list(df.columns[11:21])
third =  list(df.columns[21:30])

#Let us find the correlation between different attributes
corr1 = df[first].corr()
#Let us visualize with a heatmap
plt.figure(figsize=(14,10))
sns.heatmap(corr1, cmap='coolwarm', xticklabels = first,  yticklabels = first, annot=True)
#Let us perform analysis on the mean features

melign = df[df['diagnosis'] == 1][first]
bening = df[df['diagnosis'] == 0][first]

melign.columns
for columns in melign.columns:
    plt.figure()
    sns.distplot(melign[columns], kde=False, rug= True)
    sns.distplot(bening[columns], kde=False, rug= True)
    sns.distplot
plt.tight_layout()

color_function = {0: "green", 1: "red"}
colors = df["diagnosis"].map(lambda x: color_function.get(x))

pd.plotting.scatter_matrix(df[first], c=colors, alpha = 0.4, figsize = (15, 15));

#We divide the data into Training and test set 
train, test = train_test_split(df, test_size = 0.25)
# I have created a function to perform k folds cross validation which helps in obtaining a better insight to test the accuracy of the model
# More info at https://www.analyticsvidhya.com/blog/2018/05/improve-model-performance-cross-validation-in-python-r/

def classification_model(model, data, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])
  
  predictions = model.predict(data[predictors])
  
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print("Accuracy : %s" % "{0:.3%}".format(accuracy))

  #Perform k-fold cross-validation with 5 folds
  kf = KFold(data.shape[0],n_folds= 5)
  error = []
  for train, test in kf:
    # Filter the training data
    train_predictors = (data[predictors].iloc[train,:])
    train_target = data[outcome].iloc[train]
    model.fit(train_predictors, train_target)
    
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
    
    print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))
    
  model.fit(data[predictors],data[outcome]) 
#Using Logistic regression on the top five features
#more info at https://en.wikipedia.org/wiki/Logistic_regression

predictor_var = ['radius_mean','perimeter_mean','area_mean','compactness_mean','concave points_mean']
outcome_var='diagnosis'
model=LogisticRegression()
classification_model(model,train,predictor_var,outcome_var)
#Let us check the accuracy on test data
classification_model(model, test,predictor_var,outcome_var)
#Let us try to classify using a decision tree classifier 
predictor_var = ['radius_mean','perimeter_mean','area_mean','compactness_mean','concave points_mean']
model = DecisionTreeClassifier()
classification_model(model,train,predictor_var,outcome_var)
classification_model(model, test,predictor_var,outcome_var)
predictor_var = first
model = RandomForestClassifier()
classification_model(model, train,predictor_var,outcome_var)
#Let us find the most important features used for classification model

featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print(featimp)
predictor_var = ['concave points_mean','area_mean','radius_mean','perimeter_mean','concavity_mean']
model = RandomForestClassifier()
classification_model(model,train,predictor_var,outcome_var)
# I think we get a better prediction with all the features now let us try it on test data!
predictor_var = first
model = RandomForestClassifier()
classification_model(model, test,predictor_var,outcome_var)