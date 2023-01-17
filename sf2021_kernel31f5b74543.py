



# seed value for random number generators to obtain reproducible results

RANDOM_SEED = 1



# although we standardize X and y variables on input,

# we will fit the intercept term in the models

# Expect fitted values to be close to zero

SET_FIT_INTERCEPT = True



# import base packages into the namespace for this program

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt



# modeling routines from Scikit Learn packages

import sklearn.linear_model 

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #Random Forest Package

from sklearn.ensemble import GradientBoostingRegressor #Gradient Boosted Trees

import time

from sklearn.datasets import make_classification

from collections import OrderedDict

from sklearn.metrics import classification_report



from plotly import tools

from sklearn.datasets import fetch_openml

from sklearn.decomposition import PCA

from sklearn.metrics import f1_score

from sklearn import metrics

from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix

from math import sqrt  # for root mean-squared error calculation





import seaborn as sns  # pretty plotting, including correlation map

# Starter Code provided at the Start of each Kaggle Notebook

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



# Input data files are available in the "../input/" directory.

print('Directory Path where files are located')

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Validate working directory

os.getcwd() 

print(os.getcwd())

#Validate Current Path and create Path to data

from pathlib import Path

INPUT = Path("../input/digit-recognizer")

os.listdir(INPUT)
#Import CSV into Pandas dataframe and test shape of file 

train_df = pd.read_csv(INPUT/"train.csv")

train_df.head(3)
train_df.tail(3)
train_df.shape
#Split into train and validation prior to cross validation

from sklearn.model_selection import train_test_split





X_mtrain, X_valid, y_mtrain, y_valid = train_test_split(train_df.drop(['label'], axis=1), train_df["label"], shuffle=True,

                                                    train_size=.85, random_state=1)

# Check the shape of the trainig data set array

print('Shape of X_mtrain_data:', X_mtrain.shape)

print('Shape of y_mtrain_data:', y_mtrain.shape)

print('Shape of X_validation_data:', X_valid.shape)

print('Shape of y_validation_data:', y_valid.shape)
#S5 Split Train and Test

X_train, X_test, y_train, y_test = train_test_split(X_mtrain, y_mtrain, train_size = 0.7,

                                                    test_size =0.3, random_state=1)

print('Shape of X_train_data:', X_train.shape)

print('Shape of y_train_data:', X_test.shape)

print('Shape of X_test_data:',y_train.shape)

print('Shape of y_test_data:',y_test.shape)
#Check out what some of the data looks like



for digit_num in range(0,64):

    subplot(8,8,digit_num+1)

    grid_data = X_train.iloc[digit_num].as_matrix().reshape(28,28)  # reshape from 1d to 2d pixel array

    plt.imshow(grid_data, interpolation = "none", cmap = "bone_r")

    xticks([])

    yticks([])
#Try GridSearch for the Trees



from sklearn.model_selection import  GridSearchCV



tree_names = ['RandomForestClassifier']

tree_clfs = [RandomForestClassifier()]  

            

    

tree_param ={tree_names[0]: {'n_estimators': [10,100, 200], 'criterion': ['gini'],'max_features': ['sqrt'], 'n_jobs':[-1],

                      'max_depth':[4],  'random_state': [RANDOM_SEED], 

                       'bootstrap': [True], 'oob_score':[True]}

            }



            



for names, estimator in zip(tree_names,tree_clfs):

    print(names)

    print(tree_names)

    print(estimator)

    clf = GridSearchCV(estimator, tree_param[names] , return_train_score=True, cv=10) 

    clf.fit(X_train, y_train)



    print("best params: " + str(clf.best_params_))

    print("best scores: " + str(clf.best_score_))

    

    rmse = np.sqrt(mean_squared_error(y_train, clf.predict(X_train)))

    rmse_tst = np.sqrt(mean_squared_error(y_test, clf.predict(X_test)))



    print("rmse: {:}".format(rmse))

    print("rmse_tst: {:}".format(rmse_tst))

clf.best_params_
start_time = time.process_time() 

#Using the best parameters from the for the Extra Tree Regression, use it to show the Feature importance on the entire data set

myfit = RandomForestClassifier(n_estimators= 200, criterion='gini',max_features='sqrt', n_jobs=-1,

            max_depth=4, oob_score=True, random_state = 1, bootstrap=True)
myfit.fit(X_train,y_train)
print("Accuracy on training set: {:.3f}".format(myfit.score(X_train, y_train)))

print("Accuracy on test set: {:.3f}".format(myfit.score(X_test, y_test)))

f1 = f1_score(y_train, myfit.predict(X_train),average='weighted')

f1_tst = f1_score(y_test, myfit.predict(X_test),average='weighted')

f1_vld = f1_score(y_valid, myfit.predict(X_valid),average='weighted')



print("f1: {:}".format(f1))

print("f1_tst: {:}".format(f1_tst))

print("f1_vld: {:}".format(f1_vld))

    

# Extract single tree

print(metrics.classification_report(myfit.predict(X_train), y_train))

print(metrics.classification_report(myfit.predict(X_test), y_test))

print(metrics.classification_report(myfit.predict(X_valid), y_valid))

end_time = time.process_time() 

runtime = end_time - start_time  # seconds of wall-clock time

print(runtime)  # report in milliseconds
#check for the feature importance 

importances = myfit.feature_importances_

indices = np.argsort(importances)[::-1]



print("Feature ranking:")

for f in range(0,10):

    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))



# Plot the feature importances



figure(figsize(7,3))

plot(indices[:],importances[indices[:]],'k.')

yscale("log")

xlabel("feature",size=20)

ylabel("importance",size=20)

#Confusion Matrix iRFC

cm_trn = confusion_matrix(y_train, myfit.predict(X_train))

cm_trn_plt=sns.heatmap(cm_trn.T, square=True, annot=True, fmt='d', cbar=False, cmap="Reds")

plt.xlabel('Actual label')

plt.ylabel('Predicted label')

plt.title("Training");
#Confusion Matrix

cm_trn = confusion_matrix(y_test, myfit.predict(X_test))

cm_trn_plt=sns.heatmap(cm_trn.T, square=True, annot=True, fmt='d', cbar=False, cmap="Reds")

plt.xlabel('Actual label')

plt.ylabel('Predicted label')

plt.title("Testing");
#Confusion Matrix 

cm_trn = confusion_matrix(y_valid, myfit.predict(X_valid))

cm_trn_plt=sns.heatmap(cm_trn.T, square=True, annot=True, fmt='d', cbar=False, cmap="Reds")

plt.xlabel('Actual label')

plt.ylabel('Predicted label')

plt.title("Validation");
#Try out the PCA model with 95% of the components





start_time2 = time.process_time() 



pca = PCA(.95)

pca.fit(X_train)

transform = pca.transform(X_train)



plt.scatter(transform[:,0],transform[:,1], s=20, c = y_train)

plt.colorbar()

clim(0,9)



xlabel("PC1")

ylabel("PC2")

pca.n_components
end_time2 = time.process_time() 

runtime2 = end_time2 - start_time2  # seconds of wall-clock time

print(runtime2)  # report in milliseconds
#increase the number of components in PCA to see how many components are needed to capture most of the variance in the data. 



n_components_array=([1,2,3,4,5,10,20,50,100,200,500])

vr = np.zeros(len(n_components_array))

i=0;

for n_components in n_components_array:

    pca2 = PCA(n_components=n_components)

    pca2.fit(X_train)

    vr[i] = sum(pca2.explained_variance_ratio_)  # use the pca.explained_variance_ratio function to explain variance

    i=i+1    

    

    

#plot the PCA components to see how the variance is explained

plot(n_components_array,vr,'k.-')

xscale("log")

ylim(9e-2,1.1)

yticks(linspace(0.2,1.0,9))

xlim(0.9)

grid(which="both")

xlabel("number of PCA components",size=20)

ylabel("variance ratio",size=20)


# fit PCA model

pca.fit(train_df)



# transform data onto the first two principal components

X_pca = pca.transform(train_df)

print("Original shape: {}".format(str(train_df.shape)))

print("Reduced shape: {}".format(str(X_pca.shape)))

#RF PCA Model



X_pca_train, X_pca_test, y_pca_train, y_pca_test= train_test_split(X_pca, train_df["label"], train_size=.7, 

                                                             test_size=.3, random_state=1)

print(X_pca_train.shape)

print(X_pca_test.shape)

print(y_pca_train.shape)

print(y_pca_test.shape)

start_time = time.process_time() 



rfc_pca = RandomForestClassifier(n_estimators=200, n_jobs=-1, max_depth=5, criterion='gini',

                                max_features='sqrt', oob_score=True,  bootstrap = True, random_state=1)

# Train

rfc_pca= rfc_pca.fit(X_pca_train, y_pca_train)

print("Accuracy on training set: {:.3f}".format(rfc_pca.score(X_pca_train, y_pca_train)))

print("Accuracy on test set: {:.3f}".format(rfc_pca.score(X_pca_test, y_pca_test)))

f1 = f1_score(y_pca_train, rfc_pca.predict(X_pca_train),average='weighted')

f1_tst = f1_score(y_pca_test, rfc_pca.predict(X_pca_test),average='weighted')



print("f1: {:}".format(f1))

print("f1_tst: {:}".format(f1_tst))





print(metrics.classification_report(rfc_pca.predict(X_pca_train), y_pca_train))

print(metrics.classification_report(rfc_pca.predict(X_pca_test), y_pca_test))

end_time = time.process_time() 

runtime = end_time - start_time  # seconds of wall-clock time

print(runtime)  # report in milliseconds
cm_trn_pca = confusion_matrix(y_pca_train, rfc_pca.predict(X_pca_train))

cm_trn_pca_plt=sns.heatmap(cm_trn_pca.T, square=True, annot=True, fmt='d', cbar=False, cmap="Reds")

plt.xlabel('Actual label')

plt.ylabel('Predicted label')

plt.title("Training");
cm_trn_pca = confusion_matrix(y_pca_test, rfc_pca.predict(X_pca_test))

cm_trn_pca_plt=sns.heatmap(cm_trn_pca.T, square=True, annot=True, fmt='d', cbar=False, cmap="Reds")

plt.xlabel('Actual label')

plt.ylabel('Predicted label')

plt.title("Testing");
#Score test dataset

scr=myfit.predict(X_train)

#Conver array to Pandas dataframe with submission titles

pd_scr=pd.DataFrame(scr)

pd_scr.index.name = 'ImageId'

pd_scr.columns = ['label']

print(pd_scr)

#Export to Excel

pd_scr.to_excel("pd_scr4.xlsx")  