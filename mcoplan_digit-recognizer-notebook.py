# ------------------------------------------------------------

# Northwestern University 

# Predict 422

# W5 Principal Components Analysis



# -------------------------------------------------------------

# S1 Run SetUp Script to Install Packages 

import pandas as pd  # data frame operations  

import numpy as np  # arrays and math functions

import matplotlib.pyplot as plt  # static plotting

import matplotlib.cbook as cbook

import seaborn as sns  # pretty plotting, including heat map

import re # regular expressions

import scipy

import os # Operation System

#import mglearn

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import train_test_split

from matplotlib.backends.backend_pdf import PdfPages

from sklearn.metrics import roc_auc_score, confusion_matrix, mean_squared_error, r2_score  

from sklearn.naive_bayes import BernoulliNB

from sklearn import linear_model

from sklearn.model_selection import KFold, GridSearchCV, cross_validate, cross_val_score, cross_val_predict

from sklearn.datasets import make_blobs

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.decomposition import PCA

from math import sqrt  # for root mean-squared error calculation

from sklearn.ensemble import RandomForestClassifier,  BaggingClassifier, RandomForestRegressor

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import sklearn

import plotly

import plotly.graph_objs as go

import time

from sklearn.datasets import make_classification

from collections import OrderedDict

from plotly import tools

from sklearn.datasets import fetch_openml

from sklearn.decomposition import PCA

from sklearn.metrics import f1_score

from sklearn import metrics

import cProfile
#S3 Load/Import data

train_df = pd.read_csv("../input/digit-recognizer/train.csv")

train_df_X = np.array(train_df.drop(['label'], axis=1))

train_df_y = np.array(train_df["label"])

scoring_data = pd.read_csv("../input/digit-recognizer/test.csv")

train_df.head()

train_df.shape

#S4 Split into train and validation prior to cross validation

#I will never let the algo see the validation dataset

from sklearn.model_selection import train_test_split

X_trn, X_valid, y_trn, y_valid = train_test_split(train_df.drop(['label'], axis=1), train_df["label"], shuffle=True,

                                                    train_size=.85, random_state=1)

# Check the shape of the trainig data set array

print('Shape of X_trn_data:', X_trn.shape)

print('Shape of y_trn_data:', y_trn.shape)

print('Shape of X_valid:', X_valid.shape)

print('Shape of y_valid:', y_valid.shape)





#S5 Split Train and Test

X_train, X_test, y_train, y_test = train_test_split(X_trn, y_trn, train_size = 0.7,

                                                    test_size =0.3, random_state=1)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
#S6 Visualize Distributions

mn_plt =sns.countplot(train_df["label"], palette="Blues").set_title('Total Digit')

fig1 = mn_plt.get_figure()

#fig1.show()

fig1.savefig('TotalDistMNIST.png', 

        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 

        orientation='portrait', papertype=None, format=None, 

        #transparent=True, pad_inches=0.25, frameon=None)

         transparent=True, pad_inches=0.25)
mn_plt_trn =sns.countplot(y_train, palette="Blues").set_title('Train Digit')

fig2 = mn_plt_trn.get_figure()

#fig2.show()

fig2.savefig('TrainDistMNIST.png', 

        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 

        orientation='portrait', papertype=None, format=None, 

        transparent=True, pad_inches=0.25)
mn_plt_vld =sns.countplot(y_valid, palette="Blues").set_title('Valid Digit')

fig3 = mn_plt_vld.get_figure()

#fig3.show()

fig3.savefig('ValidDistMNIST.png', 

        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 

        orientation='portrait', papertype=None, format=None, 

        transparent=True, pad_inches=0.25)
mn_plt_tst =sns.countplot(y_test, palette="Blues").set_title('Test Digit')

fig4 = mn_plt_tst.get_figure()

#fig4.show()

fig4.savefig('TestDistMNIST.png', 

        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 

        orientation='portrait', papertype=None, format=None, 

        transparent=True, pad_inches=0.25)
#S7 RF Model

start_time = time.process_time() 

x = []

rfc = RandomForestClassifier(n_estimators=30, n_jobs=-1, max_depth=5, criterion='gini',

                             max_features='sqrt', oob_score=True,  bootstrap = True, random_state=1)

# Train

rfc= rfc.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(rfc.score(X_train, y_train)))

print("Accuracy on test set: {:.3f}".format(rfc.score(X_test, y_test)))

f1 = f1_score(y_train, rfc.predict(X_train),average='weighted')

f1_tst = f1_score(y_test, rfc.predict(X_test),average='weighted')

f1_vld = f1_score(y_valid, rfc.predict(X_valid),average='weighted')



print("f1: {:}".format(f1))

print("f1_tst: {:}".format(f1_tst))

print("f1_vld: {:}".format(f1_vld))

    

# Extract single tree

print(metrics.classification_report(rfc.predict(X_train), y_train))

print(metrics.classification_report(rfc.predict(X_test), y_test))

print(metrics.classification_report(rfc.predict(X_valid), y_valid))

end_time = time.process_time() 

runtime = end_time - start_time  # seconds of wall-clock time

print(runtime)  # report in milliseconds
#Confusion Matrix iRFC

cm_trn = confusion_matrix(y_train, rfc.predict(X_train))

cm_trn_plt=sns.heatmap(cm_trn.T, square=True, annot=True, fmt='d', cbar=False, cmap="Blues")

plt.xlabel('Actual label')

plt.ylabel('Predicted label')

plt.title("Training");

fig1 = cm_trn_plt.get_figure()

#fig1.show()

fig1.savefig('TrainCM.png', 

        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 

        orientation='portrait', papertype=None, format=None, 

        #transparent=True, pad_inches=0.25, frameon=None)

        transparent=True, pad_inches=0.25)
c_mat = confusion_matrix(y_test, rfc.predict(X_test))

cm_plt=sns.heatmap(c_mat.T, square=True, annot=True, fmt='d', cbar=False, cmap="Blues")

plt.xlabel('Actual label')

plt.ylabel('Predicted label')

plt.title("Test");

fig2 = cm_plt.get_figure()

#fig2.show()

fig2.savefig('TestCM.png', 

        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 

        orientation='portrait', papertype=None, format=None, 

        transparent=True, pad_inches=0.25)
cm_vld = confusion_matrix(y_valid, rfc.predict(X_valid))

cm_vld_plt=sns.heatmap(cm_vld.T, square=True, annot=True, fmt='d', cbar=False, cmap="Blues")

plt.xlabel('Actual label')

plt.ylabel('Predicted label')

plt.title("Valid");

fig3 = cm_vld_plt.get_figure()

#fig3.show()

fig3.savefig('ValidCM.png', 

        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 

        orientation='portrait', papertype=None, format=None, 

        transparent=True, pad_inches=0.25)
#S8 Estimate number PCA components

start_time = time.process_time() 

pca = PCA().fit(train_df_X)

pca_plt=plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance');

plt.savefig('PCAEstimate.png', 

        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 

        orientation='portrait', papertype=None, format=None, 

        transparent=True, pad_inches=0.25)
#S9 Keep the principal components that explain 95% of variation in the data



pca = PCA(n_components=0.95) #per instructions: represent 95 percent of the variability in the explanatory variables

# fit PCA model

pca.fit(train_df_X)



# transform data onto the first two principal components

X_pca = pca.transform(train_df_X)

print("Original shape: {}".format(str(train_df_X.shape)))

print("Reduced shape: {}".format(str(X_pca.shape)))

end_time = time.process_time() 

runtime = end_time - start_time  # seconds of wall-clock time

print(runtime)  # report in milliseconds
#S10 Split into train and validation prior to cross validation

X_pca_train, X_pca_vld, y_pca_train, y_pca_vld= train_test_split(X_pca, train_df_y, train_size=.85, 

                                                             test_size=.15, random_state=1)

print(X_pca_train.shape)

print(X_pca_vld.shape)

print(y_pca_train.shape)

print(y_pca_vld.shape)



#S11 Split Train and Test

X_pca_trn, X_pca_tst, y_pca_trn, y_pca_tst = train_test_split(X_pca_train, y_pca_train, train_size = 0.7,

                                                    test_size =0.3, random_state=1)

print(X_pca_trn.shape)

print(X_pca_tst.shape)

print(y_pca_trn.shape)

print(y_pca_tst.shape)
# #This cell took a long time to run. I wanted to makes sure I am using the correct parameters for the RFC



# n_estimators = [10, 20, 35, 50, 100]

# criterion = ['entropy','gini']

# max_features= ['sqrt']

# n_jobs= [-1]

# max_depth=[4,5,6,7,8]

# min_samples_leaf=[2,3,5,10,20]

# min_samples_split=[5,10,20]

# oob_score=[True]

# bootstrap = [True]

# random_state=[1]



# forest = RandomForestClassifier(random_state = 1)



# hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  

#              min_samples_split = min_samples_split, 

#              min_samples_leaf = min_samples_leaf,

#              criterion = criterion,

#              max_features = max_features,

#              oob_score = oob_score,

#              bootstrap = bootstrap,

#              random_state =random_state)



# gridF = GridSearchCV(forest, hyperF, cv = 3, verbose = 1, 

#                       n_jobs = -1)



# bestF = gridF.fit(X_pca_trn, y_pca_trn)
# bestF #These are the parameters I will use for the next model
# print("Accuracy on training set: {:.3f}".format(bestF.score(X_pca_trn, y_pca_trn)))

# print("Accuracy on test set: {:.3f}".format(bestF.score(X_pca_tst, y_pca_tst)))

# f1 = f1_score(y_pca_trn, bestF.predict(X_pca_trn),average='weighted')

# f1_tst = f1_score(y_pca_tst, bestF.predict(X_pca_tst),average='weighted')

# f1_vld = f1_score(y_pca_vld, bestF.predict(X_pca_vld),average='weighted')



# # Compare

# print(metrics.classification_report(rfc_pca.predict(X_pca_trn), y_pca_trn))

# print(metrics.classification_report(rfc_pca.predict(X_pca_tst), y_pca_tst))

# print(metrics.classification_report(rfc_pca.predict(X_pca_vld), y_pca_vld))
#S12 RF PCA Model

start_time = time.process_time() 

rfc_pca = RandomForestClassifier(n_estimators=30, n_jobs=-1, max_depth=5, criterion='gini',

                                max_features='sqrt', oob_score=True,  bootstrap = True, random_state=1)

# Train

rfc_pca= rfc_pca.fit(X_pca_trn, y_pca_trn)

print("Accuracy on training set: {:.3f}".format(rfc_pca.score(X_pca_trn, y_pca_trn)))

print("Accuracy on test set: {:.3f}".format(rfc_pca.score(X_pca_tst, y_pca_tst)))

f1 = f1_score(y_pca_trn, rfc_pca.predict(X_pca_trn),average='weighted')

f1_tst = f1_score(y_pca_tst, rfc_pca.predict(X_pca_tst),average='weighted')

f1_vld = f1_score(y_pca_vld, rfc_pca.predict(X_pca_vld),average='weighted')



print("f1: {:}".format(f1))

print("f1_tst: {:}".format(f1_tst))

print("f1_vld: {:}".format(f1_vld))

    

# Compare

print(metrics.classification_report(rfc_pca.predict(X_pca_trn), y_pca_trn))

print(metrics.classification_report(rfc_pca.predict(X_pca_tst), y_pca_tst))

print(metrics.classification_report(rfc_pca.predict(X_pca_vld), y_pca_vld))

end_time = time.process_time() 

runtime = end_time - start_time  # seconds of wall-clock time

print(runtime)  # report in milliseconds
#Confusion Matrix RFC-PCA

cm_trn_pca = confusion_matrix(y_pca_trn, rfc_pca.predict(X_pca_trn))

cm_trn_pca_plt=sns.heatmap(cm_trn_pca.T, square=True, annot=True, fmt='d', cbar=False, cmap="Blues")

plt.xlabel('Actual label')

plt.ylabel('Predicted label')

plt.title("Training");

fig4 = cm_trn_pca_plt.get_figure()

#fig4.show()

fig4.savefig('TrainPCACM.png', 

        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 

        orientation='portrait', papertype=None, format=None, 

        transparent=True, pad_inches=0.25)
c_tst_pca = confusion_matrix(y_pca_tst, rfc_pca.predict(X_pca_tst))

c_tst_pca_plt=sns.heatmap(c_tst_pca.T, square=True, annot=True, fmt='d', cbar=False, cmap="Blues")

plt.xlabel('Actual label')

plt.ylabel('Predicted label')

plt.title("Test");

fig5 = c_tst_pca_plt.get_figure()

#fig5.show()

fig5.savefig('TestPCACM.png', 

        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 

        orientation='portrait', papertype=None, format=None, 

        transparent=True, pad_inches=0.25)
cm_vld_pca = confusion_matrix(y_pca_vld, rfc_pca.predict(X_pca_vld))

cm_vld_pca_plt=sns.heatmap(cm_vld_pca.T, square=True, annot=True, fmt='d', cbar=False, cmap="Blues")

plt.xlabel('Actual label')

plt.ylabel('Predicted label')

plt.title("Valid");

fig6 = cm_vld_pca_plt.get_figure()

#fig6.show()

fig6.savefig('ValidPCACM.png', 

        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 

        orientation='portrait', papertype=None, format=None, 

        transparent=True, pad_inches=0.25)
#re-run but keep the test and validation data entirely seperate from PCA
#S8 Estimate number PCA components

start_time = time.process_time() 

pca = PCA().fit(X_train)

pca_plt=plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance');

plt.savefig('PCAEstimate_train.png', 

        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 

        orientation='portrait', papertype=None, format=None, 

        transparent=True, pad_inches=0.25)
#S9 Keep the principal components that explain 95% of variation in the data



pca = PCA(n_components=0.95) #per instructions: represent 95 percent of the variability in the explanatory variables

# fit PCA model

pca.fit(X_train)



# transform data onto the first two principal components

X_pca = pca.transform(X_train)

print("Original shape: {}".format(str(X_train.shape)))

print("Reduced shape: {}".format(str(X_pca.shape)))

end_time = time.process_time() 

runtime = end_time - start_time  # seconds of wall-clock time

print(runtime)  # report in milliseconds
#Transform test and validation data to pca

X_test_pca = pca.transform(X_test)

X_valid_pca = pca.transform(X_valid)



#RF PCA Model

start_time = time.process_time() 

rfc_pca = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

                       max_depth=None, max_features='auto', max_leaf_nodes=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=1, min_samples_split=2,

                       min_weight_fraction_leaf=0.0, n_estimators='warn',

                       n_jobs=None, oob_score=False, random_state=1, verbose=0,

                       warm_start=False)



# Train

rfc_pca= rfc_pca.fit(X_pca, y_train)

print("Accuracy on training set: {:.3f}".format(rfc_pca.score(X_pca, y_train)))

print("Accuracy on test set: {:.3f}".format(rfc_pca.score(X_test_pca, y_test)))

f1 = f1_score(y_train, rfc_pca.predict(X_pca),average='weighted')

f1_tst = f1_score(y_test, rfc_pca.predict(X_test_pca),average='weighted')

f1_vld = f1_score(y_valid, rfc_pca.predict(X_valid_pca),average='weighted')



print("f1: {:}".format(f1))

print("f1_tst: {:}".format(f1_tst))

print("f1_vld: {:}".format(f1_vld))

    

# Compare

print(metrics.classification_report(rfc_pca.predict(X_pca), y_train))

print(metrics.classification_report(rfc_pca.predict(X_test_pca), y_test))

print(metrics.classification_report(rfc_pca.predict(X_valid_pca), y_valid))

end_time = time.process_time() 

runtime = end_time - start_time  # seconds of wall-clock time

print(runtime)  # report in milliseconds
#Confusion Matrix RFC-PCA

cm_trn_pca = confusion_matrix(y_train, rfc_pca.predict(X_pca))

cm_trn_pca_plt=sns.heatmap(cm_trn_pca.T, square=True, annot=True, fmt='d', cbar=False, cmap="Blues")

plt.xlabel('Actual label')

plt.ylabel('Predicted label')

plt.title("Training");

fig4 = cm_trn_pca_plt.get_figure()

#fig4.show()

fig4.savefig('TrainPCACM.png', 

        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 

        orientation='portrait', papertype=None, format=None, 

        transparent=True, pad_inches=0.25)
c_tst_pca = confusion_matrix(y_test, rfc_pca.predict(X_test_pca))

c_tst_pca_plt=sns.heatmap(c_tst_pca.T, square=True, annot=True, fmt='d', cbar=False, cmap="Blues")

plt.xlabel('Actual label')

plt.ylabel('Predicted label')

plt.title("Test");

fig5 = c_tst_pca_plt.get_figure()

#fig5.show()

fig5.savefig('TestPCACM.png', 

        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 

        orientation='portrait', papertype=None, format=None, 

        transparent=True, pad_inches=0.25)
cm_vld_pca = confusion_matrix(y_valid, rfc_pca.predict(X_valid_pca))

cm_vld_pca_plt=sns.heatmap(cm_vld_pca.T, square=True, annot=True, fmt='d', cbar=False, cmap="Blues")

plt.xlabel('Actual label')

plt.ylabel('Predicted label')

plt.title("Valid");

fig6 = cm_vld_pca_plt.get_figure()

#fig6.show()

fig6.savefig('ValidPCACM.png', 

        bbox_inches = 'tight', dpi=None, facecolor='w', edgecolor='b', 

        orientation='portrait', papertype=None, format=None, 

        transparent=True, pad_inches=0.25)
#Score test dataset- FRC- full feature list



# scored_rfc=rfc.predict(scoring_data)

# #Conver array to Pandas dataframe with submission titles

# pd_scored_rfc=pd.DataFrame(scored_rfc)

# pd_scored_rfc.index += 1 

# pd_scored_rfc.index.name = 'ImageId'

# pd_scored_rfc.columns = ['Label']

# print(pd_scored_rfc)

# #Export to csv

# pd_scored_rfc.to_csv("rfc_pca_scrored.csv")  



scoring_data_pca = pca.transform(scoring_data)

scored_rfc=rfc_pca.predict(scoring_data_pca)

#Conver array to Pandas dataframe with submission titles

pd_scored_rfc=pd.DataFrame(scored_rfc)

pd_scored_rfc.index += 1 

pd_scored_rfc.index.name = 'ImageId'

pd_scored_rfc.columns = ['Label']

print(pd_scored_rfc)

#Export to csv

pd_scored_rfc.to_csv("rfc_pca_scrored.csv")  





# scoring_data_pca = pca.transform(scoring_data)

# scored_rfc_pca_bestF=bestF.predict(scoring_data_pca)

# #Conver array to Pandas dataframe with submission titles

# pd_scored_rfc_pca_bestF=pd.DataFrame(scored_rfc_pca_bestF)

# pd_scored_rfc_pca_bestF.index += 1 

# pd_scored_rfc_pca_bestF.index.name = 'ImageId'

# pd_scored_rfc_pca_bestF.columns = ['Label']

# print(pd_scored_rfc_pca_bestF)

# #Export to csv

# pd_scored_rfc_pca_bestF.to_csv("rfc_pca_bestF_scrored.csv")  





# scoring_data_pca = pca.transform(scoring_data)

# scored_rfc_pca_trained=rfc_pca.predict(scoring_data_pca)

# #Conver array to Pandas dataframe with submission titles

# pd_scored_rfc_pca_trained=pd.DataFrame(scored_rfc_pca_trained)

# pd_scored_rfc_pca_trained.index += 1 

# pd_scored_rfc_pca_trained.index.name = 'ImageId'

# pd_scored_rfc_pca_trained.columns = ['Label']

# print(pd_scored_rfc_pca_trained)

# #Export to csv

# pd_scored_rfc_pca_trained.to_csv("rfc_pca_trained_scrored.csv")  #this was my best score in Kaggle