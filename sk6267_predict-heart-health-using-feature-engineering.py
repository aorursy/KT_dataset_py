#Objectives:

#   i)   Using pandas and sklearn for modeling

#  ii)  Feature engineering

#                  a) Using statistical measures

#                 b) Using Random Projections

#                 c) Using clustering

#                 d) USing interaction variables

# iii)  Feature selection

#                  a) Using derived feature importance from modeling

#                  b) Using sklearn FeatureSelection Classes

#  iv)  One hot encoding of categorical variables

#   v)  Classifciation using Decision Tree and RandomForest

#   vi) Visualizing Random Forest Tree
# Clear memory -  https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-reset

# Resets the namespace by removing all names defined by the user

# -f : force reset without asking for confirmation.

%reset -f
# Call data manipulation libraries

import pandas as pd

import numpy as np
# Feature creation libraries



# Ref: https://scikit-learn.org/stable/modules/random_projection.html#sparse-random-projection

# It reduces the dimensionality by projecting the original input space using a sparse random matrix

# Projection features

from sklearn.random_projection import SparseRandomProjection as sr



# Ref: https://scikit-learn.org/stable/modules/clustering.html#k-means

# Cluster features

from sklearn.cluster import KMeans



# Ref: https://scikit-learn.org/stable/modules/preprocessing.html#generating-polynomial-features

# Interaction features

from sklearn.preprocessing import PolynomialFeatures  
# For feature selection

# It  implements feature selection algorithms

# Ref: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection

from sklearn.feature_selection import SelectKBest   # Select features according to the k highest scores.

from sklearn.feature_selection import mutual_info_classif  # Estimate mutual information for a discrete target variable.
# Data processing



# Scaling data in various manner

from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale



# Transform categorical (integer) to dummy

from sklearn.preprocessing import OneHotEncoder
# Splitting data

from sklearn.model_selection import train_test_split
# Decision tree modeling

# http://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree

# http://scikit-learn.org/stable/modules/tree.html#tree

from sklearn.tree import  DecisionTreeClassifier
# RandomForest modeling

from sklearn.ensemble import RandomForestClassifier as rf
# Plotting libraries to plot feature importance

import matplotlib.pyplot as plt

import seaborn as sns
#Visualizing Decision Trees

# pip install graphviz

#pip install pydotplus

#export_graphviz function converts decision tree classifier into dot file 

#pydotplus convert this dot file to png or displayable form on Jupyter



from sklearn.tree import export_graphviz

from sklearn.externals.six import StringIO  

from IPython.display import Image  
# Read the Data File

dt = pd.read_csv("../input/heart.csv")
# Look at data

# Age : Age in years

# Sex (1 = male; 0 = female)

# cp chest pain type (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)

# trestbps : Resting blood pressure (in mm Hg on admission to the hospital)

# chol : Cholestoral in mg/dl

# fbs : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)

# restecg : resting electrocardiographic results

# thalach : maximum heart rate achieved

# exang : exercise induced angina (1 = yes; 0 = no)

# oldpeak : ST depression induced by exercise relative to rest

# slope : the slope of the peak exercise ST segment

# ca : number of major vessels (0-3) colored by flourosopy

# thal : A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)

# target : 1 - Yes or 0 - No 

dt.head(3)
#Look at data

dt.shape
# Data types

dt.dtypes.value_counts()
# Target classes are almost balanced

dt.target.value_counts()
########################################### Visualization ######3#####3################3#####################



############ Visualize - Heart Disease (1 =Yes, 0 = No ) for Sex (1 = male; 0 = female)%#####################



sns.countplot(x="target",hue = "sex", data=dt)

plt.legend(["Female", "Male"])

plt.xlabel("Heart Disease (0 = No Disease, 1= Disease)")

plt.show()
# HeartDisease Numbers for Male & Female 

Ct_Female = len(dt[dt.sex == 0])

Ct_Female_Disease = len(dt[(dt.sex == 0) & (dt.target == 1)])

Ct_Female_No_Disease = Ct_Female - Ct_Female_Disease

Ct_Male = len(dt[dt.sex == 1])

Ct_Male_Disease = len(dt[(dt.sex == 1) & (dt.target == 1)])

Ct_Male_No_Disease = Ct_Male - Ct_Male_Disease

Ct_Female,Ct_Female_Disease,Ct_Female_No_Disease,Ct_Male,Ct_Male_Disease,Ct_Male_No_Disease
############ Visualize - Pie chart for Female Disease % & Male Disease %#####################



labels = ['Female_No_Disease', 'Female_Disease',]

sizes = [Ct_Female_No_Disease,Ct_Female_Disease]



labels1 = ['Male_No_Disease', 'Male_Disease',]

sizes1 = [Ct_Male_No_Disease,Ct_Male_Disease]



#colors

colors = ['#ff9999','#66b3ff']



fig = plt.figure(figsize = (10,10))



# Add Pie Chart for Female Disease %

ax1 = fig.add_subplot(121)

ax1.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90)

#draw circle

centre_circle = plt.Circle((0,0),0.70,fc='white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)



ax1.axis('equal') 

plt.tight_layout()





# # Add Pie Chart for Female Disease %

ax2 = fig.add_subplot(122)

ax2.pie(sizes1, colors = colors, labels=labels1, autopct='%1.1f%%', startangle=90)

#draw circle

centre_circle = plt.Circle((0,0),0.70,fc='white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)



# Equal aspect ratio ensures that pie is drawn as a circle

ax2.axis('equal') 

plt.tight_layout()



plt.show()
############ Visualize - Heart Disease for age Decade#####################



# Create a new column listing whether a person

#     he in his twenties or thirtees

#     Note:  33.10//10 outputs 3, and 47//10 outputs 4. Try it yourself

#     As to what is lambda function, see:

#         https://www.w3schools.com/python/python_lambda.asp

dt['age_dec'] = dt.age.map(lambda age: 10 * (age // 10))

sns.countplot(x="age_dec",hue = "target", data=dt)

plt.legend(["No Disease", "Disease"])

plt.xlabel("Heart Disease for age Decade")

plt.show()
############ Visualize - Chest Pain for Age Decade %#####################



sns.countplot(x="age_dec",hue = "cp", data=dt)

plt.legend(["typical angina","atypical angina","Non-anginal pain","Asymptomatic"])

plt.xlabel("Chest Pain type for age Decade")

plt.show()

dt.drop(columns = ['age_dec'], inplace = True)
############ Visualize - Chest Pain type triggering to Heart Disease/No Disease %#####################



sns.countplot(x="cp",hue = "target", data=dt)

plt.legend(["No Disease","Disease"])

plt.xlabel("Chest Pain type to Heart Disease : (1: Aypical Angina, 2: Atypical Angina,3: Non-Anginal pain, 4: Asymptomatic")

plt.show()
############ Visualize - Scatter Plot for Heart Rate to Heart Disease/No Disease %#####################



plt.scatter(x=dt.age[dt.target==1], y=dt.thalach[(dt.target==1)], c="red")

plt.scatter(x=dt.age[dt.target==0], y=dt.thalach[(dt.target==0)], c="green")

plt.legend(["Disease", "Not Disease"])

plt.xlabel("Age")

plt.ylabel("Maximum Heart Rate")

plt.show()
# Checking if any Row having all zero value and if need to be dropped. None

x = np.sum(dt, axis = 1)

v = x.index[x == 0]

v
# Checking if there are Missing values? None

dt.isnull().sum().sum()
##Feature Engineering



#Feature 1: Row sums of features

dt['sum'] = dt.sum(numeric_only = True, axis=1)
# Feature 2 : Statistical Features ("var", "median", "mean", "std", "max", "min")



feat = [ "var", "median", "mean", "std", "max", "min"]

for i in feat:

    dt[i] = dt.aggregate(i,  axis =1)
# Data after Adding 7 Additional Features

dt.shape
dt.head(2)
# keep target feature separately

target = dt['target']
# drop 'target' column from dt

dt.drop(columns = ['target'], inplace = True)

dt.shape  
# Col Names

colNames = dt.columns.values

colNames
############################Feature creation Using Random Projections ####################



# transform dt to tmp - numpy array

tmp = dt.values

tmp.shape
# Let us create 10 random projections/columns

# This decision, at present, is arbitrary

NUM_OF_COM = 10
# Create an instance of class

rp_instance = sr(n_components = NUM_OF_COM)



# fit and transform the (original) dataset

# Random Projections with desired number of components are returned

rp = rp_instance.fit_transform(tmp[:, :13])
# Look at some features

rp[: 5, :  3]
# Create some column names for these columns

rp_col_names = ["r" + str(i) for i in range(10)]

rp_col_names
############################ Feature creation using kmeans ####################
# Creating a StandardScaler instance

se = StandardScaler()
# fit() and transform() in one step

tmp = se.fit_transform(tmp)
tmp.shape
# Perform kmeans using original 13 features.

#     No of centroids is no of classes in the 'target'

centers = target.nunique()

centers 
# Begin clustering :  First create object to perform clustering

kmeans = KMeans(n_clusters=centers, # How many

                n_jobs = 2)         # Parallel jobs for n_init
# Train the model on the original data only

kmeans.fit(tmp[:, : 13])
# Get clusterlabel for each row (data-point)

kmeans.labels_

kmeans.labels_.size
# As Cluster labels are categorical, So convert them to dummy

# Create an instance of OneHotEncoder class

ohe = OneHotEncoder(sparse = False)
# Use ohe to learn data - ohe.fit(kmeans.labels_)

ohe.fit(kmeans.labels_.reshape(-1,1))     # reshape(-1,1) recommended by fit()

                                          # '-1' is a placeholder for actual
# Transforming data

dummy_clusterlabels = ohe.transform(kmeans.labels_.reshape(-1,1))

dummy_clusterlabels

dummy_clusterlabels.shape 
# New Columns Name

k_means_names = ["k" + str(i) for i in range(2)]

k_means_names
############################ Interaction features #######################



# Will require lots of memory if we take large number of features

#     Best strategy is to consider only impt features



degree = 2

poly = PolynomialFeatures(degree,                 # Degree 2

                          interaction_only=True,  # Avoid e.g. square(a)

                          include_bias = False    # No constant term

                          )
# 21.1 Consider only first 5 features

#      fit and transform

df =  poly.fit_transform(tmp[:, : 5])

df.shape
# Generate some names for these 15 columns

poly_names = [ "poly" + str(i)  for i in range(15)]

poly_names
################# concatenate all features now ##############################



# Append now all generated features together

# Append random projections, kmeans and polynomial features to tmp array



tmp.shape       
#  If variable, 'dummy_clusterlabels', exists, stack kmeans generated

#       columns also else not. 'vars()'' is an inbuilt function in python.

#       All python variables are contained in vars().



if ('dummy_clusterlabels' in vars()):               #

    tmp = np.hstack([tmp,rp,dummy_clusterlabels, df])

else:

    tmp = np.hstack([tmp,rp, df])       # No kmeans      <==



tmp.shape 
# Split Data in train & test

X_train, X_test, Y_train, Y_test = train_test_split(

                                                    tmp,

                                                    target,

                                                    test_size = 0.2)
# Further Split Data in train & validation

X_train, X_val, Y_train, Y_val = train_test_split(

                                                    X_train,

                                                    Y_train,

                                                    test_size = 0.2)
X_train.shape
X_test.shape
X_val.shape
Y_train.shape
Y_test.shape
Y_val.shape
# Create Decision Tree classifer object

clf = DecisionTreeClassifier()
# Train Decision Tree Classifer

clf = clf.fit(X_train,Y_train)
#Predict the response for test dataset

Y_pred = clf.predict(X_test)
# Check accuracy

(Y_pred == Y_test).sum()/Y_test.size
# Instantiate RandomForest classifier

clf = rf(n_estimators=50)
#Build model

clf = clf.fit(X_train, Y_train)
#Predict the response for test dataset

Y_pred = clf.predict(X_test)
# Check accuracy

(Y_pred == Y_test).sum()/Y_test.size
#Visualizing Random Forest Trees

fit = clf.estimators_[1]

colNames_tmp = list(colNames) + rp_col_names+ k_means_names + poly_names

export_graphviz(fit, out_file='rftree.dot',  

                filled=True, rounded=True,

                special_characters=True,feature_names = colNames_tmp,class_names=['0','1'])

from subprocess import call

call(['dot', '-Tpng', 'rftree.dot', '-o', 'rftree.png', '-Gdpi=600'])

Image(filename = 'rftree.png')
################## Feature selection #####################



##****************************************

## Using feature importance given by model

##****************************************



clf.feature_importances_        # Column-wise feature importance

clf.feature_importances_.size
# list of column names

colNames = list(colNames) + rp_col_names+ k_means_names + poly_names

len(colNames)
# Create a dataframe of feature importance and corresponding

#      column names. Sort dataframe by importance of feature

feat_imp = pd.DataFrame({

                   "importance": clf.feature_importances_ ,

                   "featureNames" : colNames

                  }

                 ).sort_values(by = "importance", ascending=False)
feat_imp.shape
feat_imp.head(20)
#Plot feature importance for first 20 features

g = sns.barplot(x = feat_imp.iloc[  : 20 ,  1] , y = feat_imp.iloc[ : 20, 0])

g.set_xticklabels(g.get_xticklabels(),rotation=90)