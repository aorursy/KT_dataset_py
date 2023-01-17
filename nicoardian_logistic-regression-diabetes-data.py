# LINK : https://www.kaggle.com/nicoardian/lr-diabetes

# BINUS PPTI 6

# Alexander Agung Sunaringtyas 2201828045

# Nico Ardian Nugroho 2201827780



#Machine Learning Logistic Regression Diabetes



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 

from statistics import mean 

from sklearn.linear_model import LinearRegression, Ridge, Lasso 

from sklearn.model_selection import train_test_split, cross_val_score 

from sklearn.cross_validation import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/lrd.csv')

df.head(10)
df.isnull().sum()

#shows dataset is clean
# Compute the correlation matrix

corr = df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .8})
#defining our features and target variable

X = df.iloc[:, [2, 6]].values

y = df.iloc[:, 4].values
# Splitting the dataset into the Training set and Test set - 80-20 split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.9, random_state = 0)
#Feature scaling as range of estimated salary and age is different

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
#Classifying and prediction

classifier = LogisticRegression(random_state = 0)  #Logistic classifier

classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)  #predicting test results
#Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
#using K-Fold cross validation to get the mean Accuracy

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print('Mean Accuracy: {0:.2f}, Std of Accuracy: {1:.2f}'.format(accuracies.mean(),accuracies.std()))

# Bulding and fitting the Linear Regression model 

linearModel = LinearRegression() 

linearModel.fit(X_train, y_train) 

  

# Evaluating the Linear Regression model 

print(linearModel.score(X_test, y_test)) 
# List to maintain the different cross-validation scores 

cross_val_scores_ridge = [] 

  

# List to maintain the different values of alpha 

alpha = [] 

  

# Loop to compute the different values of cross-validation scores 

for i in range(1, 9): 

    ridgeModel = Ridge(alpha = i * 0.25) 

    ridgeModel.fit(X_train, y_train) 

    scores = cross_val_score(ridgeModel, X, y, cv = 10) 

    avg_cross_val_score = mean(scores)*100

    cross_val_scores_ridge.append(avg_cross_val_score) 

    alpha.append(i * 0.25) 

  

# Loop to print the different values of cross-validation scores 

for i in range(0, len(alpha)): 

    print(str(alpha[i])+' : '+str(cross_val_scores_ridge[i])) 
# Building and fitting the Ridge Regression model 

ridgeModelChosen = Ridge(alpha = 2) 

ridgeModelChosen.fit(X_train, y_train) 

  

# Evaluating the Ridge Regression model 

print(ridgeModelChosen.score(X_test, y_test)) 
# List to maintain the cross-validation scores 

cross_val_scores_lasso = [] 

  

# List to maintain the different values of Lambda 

Lambda = [] 

  

# Loop to compute the cross-validation scores 

for i in range(1, 9): 

    lassoModel = Lasso(alpha = i * 0.25, tol = 0.0925) 

    lassoModel.fit(X_train, y_train) 

    scores = cross_val_score(lassoModel, X, y, cv = 10) 

    avg_cross_val_score = mean(scores)*100

    cross_val_scores_lasso.append(avg_cross_val_score) 

    Lambda.append(i * 0.25) 

  

# Loop to print the different values of cross-validation scores 

for i in range(0, len(alpha)): 

    print(str(alpha[i])+' : '+str(cross_val_scores_lasso[i])) 
# Building and fitting the Lasso Regression Model 

lassoModelChosen = Lasso(alpha = 2, tol = 0.0925) 

lassoModelChosen.fit(X_train, y_train) 

  

# Evaluating the Lasso Regression model 

print(lassoModelChosen.score(X_test, y_test)) 
# Building the two lists for visualization 

models = ['Linear Regression', 'Ridge Regression', 'Lasso Regression'] 

scores = [linearModel.score(X_test, y_test), 

         ridgeModelChosen.score(X_test, y_test), 

         lassoModelChosen.score(X_test, y_test)] 

  

# Building the dictionary to compare the scores 

mapping = {} 

mapping['Linear Regreesion'] = linearModel.score(X_test, y_test) 

mapping['Ridge Regreesion'] = ridgeModelChosen.score(X_test, y_test) 

mapping['Lasso Regression'] = lassoModelChosen.score(X_test, y_test) 

  

# Printing the scores for different models 

for key, val in mapping.items(): 

    print(str(key)+' : '+str(val)) 
# Plotting the scores 

plt.bar(models, scores) 

plt.xlabel('Regression Models') 

plt.ylabel('Score') 

plt.show() 