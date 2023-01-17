# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Objectives:

#        i)   Using pandas and sklearn for modeling

#        ii)  Feature engineering

#                  a) Using statistical measures

#                  b) Using Random Projections

#                  c) Using clustering

#                  d) USing interaction variables

#       iii)  Feature selection

#                  a) Using derived feature importance from modeling

#                  b) Using sklearn FeatureSelection Classes

#        iv)  One hot encoding of categorical variables

#         v)  Classifciation using Decision Tree and RandomForest

# 1.0 Clear memory

%reset -f
# 1.1 Call data manipulation libraries

import pandas as pd

import numpy as np
# 1.2 Feature creation libraries

from sklearn.random_projection import SparseRandomProjection as sr  # Projection features

from sklearn.cluster import KMeans                    # Cluster features

from sklearn.preprocessing import PolynomialFeatures  # Interaction features
# 1.3 For feature selection

# Ref: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import mutual_info_classif  # Selection criteria
# 1.4 Data processing

# 1.4.1 Scaling data in various manner

from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale

# 1.4.2 Transform categorical (integer) to dummy

from sklearn.preprocessing import OneHotEncoder
# 1.5 Splitting data

from sklearn.model_selection import train_test_split
# 1.6 Decision tree modeling

# http://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree

# http://scikit-learn.org/stable/modules/tree.html#tree

from sklearn.tree import  DecisionTreeClassifier as dt

from sklearn.metrics import roc_curve, auc #for model evaluation

from sklearn.metrics import classification_report #for model evaluation

from sklearn.metrics import confusion_matrix #for model evaluation
# 1.7 RandomForest modeling

from sklearn.ensemble import RandomForestClassifier as rf
# 1.8 Plotting libraries to plot feature importance

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.tree import export_graphviz #plot tree
# 1.9 Misc

import os, time, gc

import eli5 #for purmutation importance

from eli5.sklearn import PermutationImportance

import shap #for SHAP values

from pdpbox import pdp, info_plots #for partial plots

np.random.seed(123) #ensure reproducibility
################## AA. Reading data from files and exploring ####################



# 2.0 Set working directory and read file

print(os.listdir("../input"))
# 2.1 Read heart.csv files

data = pd.read_csv("../input/heart.csv") #Loading of Data
# 2.2 Look at data

data.head(2)
data.shape    #(303,14)
# 2.3 Data types

data.dtypes.value_counts()   # All afeatures are integers except target i.e.float64
# 3 Check if there are Missing values? None

# 3 Check if there are Missing values? None

data.isnull().sum().sum()  # 0
#3.1 Splitting the data in Train and Test

X_train, X_test, y_train, y_test = train_test_split(data.drop('target', 1), data['target'], test_size = 0.3, random_state=10) 

#3.2 Check the splits

X_train.shape       # 212 X 13

X_test.shape        #  91 X 13

y_train.shape       # (212,)

y_test.shape        # ( 91,)





# Check if there are Missing values? None

X_train.isnull().sum().sum()  # 0

X_test.isnull().sum().sum()   # 0
############################ Using Statistical Numbers #####################





#  4. Feature 1: Row sums of features 1:93. More successful

#                when data is binary.



X_train['sum'] = X_train.sum(numeric_only = True, axis=1)  # numeric_only= None is default

X_test['sum'] = X_test.sum(numeric_only = True,axis=1)
# 4.1 Assume that value of '0' in a cell implies missing feature

#     Transform train and test dataframes

#     replacing '0' with NaN

#     Use pd.replace()

tmp_train = X_train.replace(0, np.nan)

tmp_test = X_test.replace(0,np.nan)
# 4.2 Check if tmp_train is same as train or is a view

#     of train? That is check if tmp_train is a deep-copy



tmp_train is X_train                # False

tmp_train._is_view                # False
# 4.3 Check if 0 has been replaced by NaN

tmp_train.head(2)

tmp_test.head(2)
# 5. Feature 2 : For every row, how many features exist

#                that is are non-zero/not NaN.

#                Use pd.notna()

tmp_train.notna().head(1)

X_train["count_not0"] = tmp_train.notna().sum(axis = 1)

X_test['count_not0'] = tmp_test.notna().sum(axis = 1)
# 6. Similary create other statistical features

#    Feature 3

#    Pandas has a number of statistical functions

#    Ref: https://pandas.pydata.org/pandas-docs/stable/reference/frame.html#computations-descriptive-stats



feat = [ "var", "median", "mean", "std", "max", "min"]

for i in feat:

    X_train[i] = tmp_train.aggregate(i,  axis =1)

    X_test[i]  = tmp_test.aggregate(i,axis = 1)
# 7 Delete not needed variables and release memory

del(tmp_train)

del(tmp_test)

gc.collect()
# 7.1 So what do we have finally

X_train.shape                # 212 X (1+ 13 + 7) ; 13th Index is target

X_train.head(1)

X_test.shape                 # 91 X (13 + 8)

X_test.head(2)
# 8  Store column names of our data somewhere

#     We will need these later (at the end of this code)

colNames = X_train.columns.values

colNames
################ Feature creation Using Random Projections ##################

# 9. Generate features using random projections

#     First stack train and test data, one upon another

tmp = pd.concat([X_train,X_test],

                axis = 0,            # Stack one upon another (rbind)

                ignore_index = True

                )
# 9.1

tmp.shape     # 303 X 21
# 9.2 Transform tmp to numpy array

tmp = tmp.values

tmp.shape    #(303, 21)
# 10. Let us create 4 random projections/columns

NUM_OF_COM = 4

# 10.1 Create an instance of class

rp_instance = sr(n_components = NUM_OF_COM)
# 10.2 fit and transform the (original) dataset

#      Random Projections with desired number

#      of components are returned

rp = rp_instance.fit_transform(tmp[:, :13])
# 10.3 Look at some features

rp[: 5, :  3]
# 10.4 Create some column names for these columns

#      We will use them at the end of this code

rp_col_names = ["r" + str(i) for i in range(5)]

rp_col_names
############################ Feature creation using kmeans ####################

# 11.1 Create a StandardScaler instance

se = StandardScaler()
# 11.2 fit() and transform() in one step

tmp = se.fit_transform(tmp)
# 11.3

tmp.shape               # 303 X 21 (an ndarray)
# 12. Perform kmeans using 13 features.

#     No of centroids is no of classes in the 'target'

centers = y_train.nunique()  

centers       # 2
# 13 Begin clustering

#13.1 First create object to perform clustering

kmeans = KMeans(n_clusters=centers, # How many

                n_jobs = 5)         # Parallel jobs for n_init
# 13.2 Next train the model on the original data only

kmeans.fit(tmp[:, : 13])
# 14 Get clusterlabel for each row (data-point)

kmeans.labels_

kmeans.labels_.size 
# 15. Cluster labels are categorical. So convert them to dummy



# 15.1 Create an instance of OneHotEncoder class

ohe = OneHotEncoder(sparse = False)
# 15.2 Use ohe to learn data

#      ohe.fit(kmeans.labels_)

ohe.fit(kmeans.labels_.reshape(-1,1))     # reshape(-1,1) recommended by fit()
# 15.3 Transform data now

dummy_clusterlabels = ohe.transform(kmeans.labels_.reshape(-1,1))

dummy_clusterlabels

dummy_clusterlabels.shape    (303,2)
# 15.4 We will use the following as names of new 2 columns

#      We need them at the end of this code



k_means_names = ["k" + str(i) for i in range(2)]

k_means_names
# 16. Will require lots of memory if we take large number of features

#     Best strategy is to consider only impt features



degree = 2

poly = PolynomialFeatures(degree,                 # Degree 2

                          interaction_only=True,  # Avoid e.g. square(a)

                          include_bias = False    # No constant term

                          )

# 16.1 Consider only first 5 features

#      fit and transform

df =  poly.fit_transform(tmp[:, : 5])
df.shape     # 303 X 15
# 16.2 Generate some names for these 15 columns

poly_names = [ "poly" + str(i)  for i in range(15)]

poly_names
################# concatenate all features now ##############################



# 17 Append now all generated features together

# 17 Append random projections, kmeans and polynomial features to tmp array



tmp.shape          # 303 X 21

#  17.1 If variable, 'dummy_clusterlabels', exists, stack kmeans generated

#       columns also else not. 'vars()'' is an inbuilt function in python.

#       All python variables are contained in vars().



if ('dummy_clusterlabels' in vars()):               #

    tmp = np.hstack([tmp,rp,dummy_clusterlabels, df])

else:

    tmp = np.hstack([tmp,rp, df])       # No kmeans      <==
tmp.shape     #(303,63)
# 18.1 Separate train and test

X = tmp[: X_train.shape[0], : ]
X.shape   #(212,63)
# 18.2

test = tmp[X_train.shape[0] :, : ]
test.shape   # (91,63)
# 18.3 Delete tmp

del tmp

gc.collect()
################## Model building #####################





# 19. Split train into training and validation dataset

X_train, X_test, y_train, y_test = train_test_split(

                                                    X,

                                                    y_train,

                                                    test_size = 0.3)
X_train.shape 
X_test.shape
# 24 Decision tree classification

# 24.1 Create an instance of class

clf = dt(min_samples_split = 5,

         min_samples_leaf= 5

        )

# 24.2 Fit/train the object on training data

#      Build model

clf = clf.fit(X_train, y_train)
# 24.3 Use model to make predictions

classes = clf.predict(X_test)
# 24.4 Check accuracy

(classes == y_test).sum()/y_test.size 
# 25. Instantiate RandomForest classifier

clf = rf(n_estimators=50)
# 25.1 Fit/train the object on training data

#      Build model



clf = clf.fit(X_train, y_train)
# 25.2 Use model to make predictions

classes = clf.predict(X_test)
# 25.3 Check accuracy

(classes == y_test).sum()/y_test.size      # 82%
################## Feature selection #####################



##****************************************

## Using feature importance given by model

##****************************************



# 26. Get feature importance

clf.feature_importances_        # Column-wise feature importance

clf.feature_importances_.size   # 63

# 26.1 To our list of column names, append all other col names

#      generated by random projection, kmeans (onehotencoding)

#      and polynomial features

#      But first check if kmeans was used to generate features



if ('dummy_clusterlabels' in vars()):       # If dummy_clusterlabels labels are defined

    colNames = list(colNames) + rp_col_names+ k_means_names + poly_names

else:

    colNames = colNames = list(colNames) + rp_col_names +  poly_names      # No kmeans      <==
# 26.1.1 So how many columns?

len(colNames)           # 65
#27 The Model

#The next part fits a random forest model to the data,

#split the data

X_train, X_test, y_train, y_test = train_test_split(data.drop('target', 1), data['target'], test_size = .2, random_state=10) 
model = rf(max_depth=5)
model.fit(X_train, y_train)
rf(bootstrap=True, class_weight=None, criterion='gini',

            max_depth=5, max_features='auto', max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=2,

            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,

            oob_score=False, random_state=None, verbose=0,

            warm_start=False)
#We plot the consequent decision tree, to see what it's doing
estimator = model.estimators_[1]
feature_names = [i for i in X_train.columns]
y_train_str = y_train.astype('str')
y_train_str[y_train_str == '0'] = 'no disease'
y_train_str[y_train_str == '1'] = 'disease'
y_train_str = y_train_str.values
#code from https://towardsdatascience.com/how-to-visualize-a-decision-tree-from-a-random-forest-in-python-using-scikit-learn-38ad2d75f21c



export_graphviz(estimator, out_file='tree.dot', 

                feature_names = feature_names,

                class_names = y_train_str,

                rounded = True, proportion = True, 

                label='root',

                precision = 2, filled = True)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
from IPython.display import Image
#This gives us on explainability tool. However, we can't glance at this and get a quick sense of the most important features.



Image(filename = 'tree.png')
y_predict = model.predict(X_test)

y_pred_quant = model.predict_proba(X_test)[:, 1]

y_pred_bin = model.predict(X_test)
#Assess the fit with a confusion matrix,

confusion_matrix = confusion_matrix(y_test, y_pred_bin)

confusion_matrix  # array([[26,  9],[ 3, 23]])
#Diagnostic tests are often sold, marketed, cited and used with sensitivity and specificity 

#as the headline metrics. 

#Sensitivity and specificity are defined as,

#Sensitivity=TruePositivesTruePositives+FalseNegatives

#Specificity=TrueNegativesTrueNegatives+FalsePositives

#Let's see what this model is giving,

total=sum(sum(confusion_matrix))

sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])
print('Sensitivity : ', sensitivity ) # 0.896551724137931
specificity = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])
print('Specificity : ', specificity) # 0.71875
#Now also check with a Receiver Operator Curve (ROC),

fpr, tpr, thresholds = roc_curve(y_test, y_pred_quant)
fig, ax = plt.subplots()

ax.plot(fpr, tpr)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC curve for diabetes classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)
#Another common metric is the Area Under the Curve, or AUC. 

#This is a convenient way to capture the performance of a model in a single number, although it's not without certain issues. 

#As a rule of thumb, an AUC can be classed as follows,

#0.90 - 1.00 = excellent

#0.80 - 0.90 = good

#0.70 - 0.80 = fair

#0.60 - 0.70 = poor

#0.50 - 0.60 = fail

#Checking what the above ROC gives us,



auc(fpr, tpr) #0.8901098901098901

#Now let's see what the model gives us from the ML explainability tools.

#Permutation importance is the first tool for understanding a machine-learning model, 

#and involves shuffling individual variables 

#in the validation data (after a model has been fit), 

#and seeing the effect on accuracy. Learn more here.



perm = PermutationImportance(model, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
base_features = data.columns.values.tolist()

base_features.remove('target')
feat_name = 'ca'

pdp_dist = pdp.pdp_isolate(model=model, dataset=X_test, model_features=base_features, feature=feat_name)



pdp.pdp_plot(pdp_dist, feat_name)

plt.show()