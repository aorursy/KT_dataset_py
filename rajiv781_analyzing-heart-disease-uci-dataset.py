# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import sklearn.model_selection

import sklearn.linear_model

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
heart_data = pd.read_csv("../input/heart.csv")

heart_data.head()

#Show summary of numerical attributes

heart_data.describe()
#Plot histogram from each attribute/feature

heart_data.hist(bins=50, figsize=(20,15))

plt.show()

#Notice the features are on very different scales. Need feature scaling later. Some are tail heavy.

#some of the historgrams are bell shaped while others have no uniform shape at all.
#Create a traing and test set using RANDOM sampling from the data for calculating generalization error later.

train_set, test_set = train_test_split(heart_data, test_size=0.2, random_state=42)

print(len(train_set), "+", len(test_set) )
#Instead of random sampling, create training and test set that are representative of all age groups/other

#feature. Since we know age plays one of the key roles in heart disease also, let's create the set

#so there are equal no. of people from each age group because we don't want to excluded

#or create bias in our data.



from sklearn.model_selection import StratifiedShuffleSplit

#create age category attribute (20-30, 31-40, 41-50...)

heart_data_1 = heart_data

heart_data_1["age_cat"] = np.floor(heart_data_1["age"] / 10)

len(heart_data_1) #303

heart_data_1["age_cat"].value_counts() #cat 2.0 has only one example so remove it

#because StratifiedShuffleSplit method will throw error and needs atleast 2 examples for

#each category.



#drop row using age_cat = 2.0, then total count will be 302 and not 303

heart_data_1 = heart_data_1[heart_data_1.age_cat != 2.0]

len(heart_data_1)
heart_data_1["age_cat"].hist()
#now lets do STRATIFIED sampling based on the new age category

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) 

for train_index, test_index in split.split(heart_data_1, heart_data_1["age_cat"]):

        strat_train_set = heart_data_1.loc[train_index]

        strat_test_set = heart_data_1.loc[test_index]



print(len(strat_train_set), len(strat_test_set))

#Let's see if this worked as expected by looking at age_cat proportions in FULL/OVERALL heart set.

heart_data_1["age_cat"].value_counts() / len(heart_data_1) * 100

#5.0 is 41.39%

#6.0 is 26.49%

#.....
#Now lets measure the age_cat proportions in STRATIFIED TRAIN set.

strat_train_set["age_cat"].value_counts() / len(strat_train_set) * 100

#Now lets measure the age_cat proportions in STRATIFIED TEST set.

strat_test_set["age_cat"].value_counts() / len(strat_test_set) * 100
#Let's compare all 3 - OVERALL, RANDOM, STRATIFIED



def age_cat_proportions(data):

    return ((data["age_cat"].value_counts() / len(data)) * 100)



def age_cat_counts(data):

    return (data["age_cat"].value_counts())



#Get RANDOM proportions 

rand_train_set, rand_test_set = train_test_split(heart_data_1, test_size=0.2, random_state=42)



pd.DataFrame({

    "Overall": age_cat_proportions(heart_data_1),

    "Overall Count": age_cat_counts(heart_data_1),

    "Stratified": age_cat_proportions(strat_train_set),

    "Stratified Count": age_cat_counts(strat_train_set),

    "Random": age_cat_proportions(rand_train_set),

    "Random Count": age_cat_counts(rand_train_set),

}).sort_index()

#compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100

#compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
len(strat_train_set)
pd.DataFrame({

    "Overall": age_cat_proportions(heart_data_1),

    "Overall Count": age_cat_counts(heart_data_1),

    "Stratified Test Set": age_cat_proportions(strat_test_set),

    "Stratified Test Set Count": age_cat_counts(strat_test_set),

    "Random Test Set": age_cat_proportions(rand_test_set),

    "Random Test Set Count": age_cat_counts(rand_test_set),

}).sort_index()
strat_train_set.head()
strat_test_set.head()
#After completing analysis on stratified vs. Random sampling, remove age_cat attribute from stratified sets.

for set in (strat_train_set, strat_test_set):

    set.drop(["age_cat"], axis=1, inplace=True)
strat_train_set.head()
strat_test_set.head()
#Gather more insights from data

#Let’s create a copy so you can play with it without harming the training set:

strat_heart = strat_train_set.copy()

len(strat_heart)
strat_heart.head()
heart_test = strat_test_set.copy()

len(heart_test)
#Let's check whether heart disease increases with age

pd.crosstab(strat_heart.age,strat_heart.target).plot(kind="bar",figsize=(20,6))

#Let's look for correlations between every pair of attributes using corr() method:



corr_matrix = strat_heart.corr()



#let' see how much each attribute correlates to the age value:

corr_matrix["age"].sort_values(ascending=False)
#let' see how much each attribute correlates to the target value:

corr_matrix["target"].sort_values(ascending=False)
#Another way to check correlation between attributes is to use Pandas scatter_matrix

#function, which plots every numerical attribute against every other one. This will

#give us 14 * 14 = 196 plots. We'll plot only promising ones.

from pandas.plotting import scatter_matrix

attributes = ["target", "age", "cp", "thalach", "slope", "oldpeak"]

scatter_matrix(strat_heart[attributes], figsize=(20, 12))
#Clean the training set & let's separate the predictors and the labels since

#we don't necessarily want to apply the same transformations to the predictors

#and target values in dataframe.

heart = strat_heart.drop("target", axis=1)

heart_labels = strat_heart["target"].copy()
heart_test.head()
heart_test_labels = heart_test["target"].copy()

heart_test = heart_test.drop("target", axis=1)

heart_test.head()
#In case there is any missing values, we have option of either removing entire

#attribute or remove rows or set some default values (i.e. zero/mean/median).

#This can be accomplished using DataFrame's dropna(), drop(), and fillna().

#Use "sklearn.preprocessing import Imputer" class to replace each attribute

#missing values with the median of that attribute(but only used for numerical

#attributes).



#SciKit-Learn API has below main design principles:



#-Estimators - Any object (like Imputer) to estimate some paramters (say median) 

#based on a dataset is called estimators. The estimation is performed by fit()

#method & it takes dataset as parameter (training data & labels). Any other

#parameter needed to help in estimation process is considered hyperparameter.



#Transformers - Some estimators can also help in tranforming the dataset. 

#The transformation is 

#performed by transform() method & it returns transformed dataset. All 

#transformers have fit_transform() method that is equivalent of calling fit()

#and then transform() and its sometimes optimized to run faster.



#Predictors - Finally some estimators are capable of making predictions given 

#a dataset, they are called predictors. For e.g. the LinearRegression model is

#a predictor. A predictor has predict() method which takes dataset and returns

#a dataset of corressponding predictions.



#Inspection - All the estimator’s hyperparameters are accessible directly via 

#public instance variables (e.g., imputer.strategy), and all the estimator’s 

#learned parameters are also accessible via public instance variables with 

#an underscore suffix (e.g., imputer.statistics_).



#Nonproliferation of classes - Datasets are represented as NumPy arrays or 

#SciPy sparse matrices, instead of homemade classes. Hyperparameters are 

#just regular Python strings or numbers.



#Composition. Existing building blocks are reused as much as possible. 

#For example, it is easy to create a Pipeline estimator from an arbitrary 

#sequence of transformers followed by a final estimator, as we will see.



#You can convert text/categorical attributes to numbers using transformers

#such as LabelEncoder. You can apply transformations to convert from text

#categories to integer categories, then integer categories to one-hot

#vectors (which store 1 for 1 category attribute and 0 for remaining).



#Sometimes you will need to create custom transformers class to write your

#own tasks such as custom cleanup operations or combining specific attributes.



#One of the most important transformations you need to apply to your data is 

#feature scaling. With few exceptions, Machine Learning algorithms don’t perform 

#well when the input numerical attributes have very different scales.

#Note that scaling the target values is generally not required.



#age ranges from 29 to 77

#cp is 0 to 3

#thalach is 71 to 202

#.......



#There are two common ways to get all attributes to have the same scale: 

#min-max scaling (normalization) and standardization.



#min-max scaling (normalization) -> (x - xmin) / (xmax - xmin)

#Scikit-Learn provides a transformer called MinMaxScaler for this. It has a 

#feature_range hyperparameter that lets you change the range if you don’t 

#want 0–1 for some reason.



#Unlike min-max scaling, standardization does not bound values to a specific 

#range, which may be a problem for some algorithms (e.g., neural networks 

#often expect an input value ranging from 0 to 1). However, standardization 

#is much less affected by outliers.



heart.head()
#heart - training data

#heart_labels - training data labels

#heart_test - test data

#heart_test_labels - test data labels

x = (heart - np.min(heart)) / (np.max(heart) - np.min(heart)).values

x.head()
x_test = (heart_test - np.min(heart_test)) / (np.max(heart_test) - np.min(heart_test)).values

x_test.head()
#As you can see, there are many data transformation steps that need to be 

#executed in the right order. Fortunately, Scikit-Learn provides the Pipeline 

#class to help with such sequences of transformations. Here is a small pipeline 

#for the numerical attributes:

# from sklearn.pipeline import Pipeline

# from sklearn.preprocessing import StandardScaler

#     num_pipeline = Pipeline([

#             ('imputer', Imputer(strategy="median")),

#             ('attribs_adder', CombinedAttributesAdder()),

#             ('std_scaler', StandardScaler()),

#         ])

#     housing_num_tr = num_pipeline.fit_transform(housing_num)



#The Pipeline constructor takes a list of name/estimator pairs defining a 

#sequence of steps. All but the last estimator must be transformers (i.e., 

#they must have a fit_transform() method). The names can be anything you like.



#You now have a pipeline for numerical values, and you also need to apply the 

#LabelBi narizer on the categorical values: how can you join these transformations 

#into a sin‐ gle pipeline? Scikit-Learn provides a FeatureUnion class for this. 

#You give it a list of transformers (which can be entire transformer pipelines), 

#and when its transform() method is called it runs each transformer’s transform() 

#method in parallel, waits for their output, and then concatenates them and 

#returns the result (and of course calling its fit() method calls all each 

#transformer’s fit() method). 

x.to_csv('heart_training_data.csv', index=False)
len(x)
#There is an empty row in x. Drop empty row in the dataframe x

x.dropna(axis=0, inplace=True)
len(x)
len(heart_labels)
#Remove empty row from heart_labels dataframe as well.

heart_labels.dropna(axis=0, inplace=True)
len(heart_labels)
#Scikit-Learn's Stochastic Gradient Classifier - Training binary classifier

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)

sgd_clf.fit(x, heart_labels)

#Predict using Stochastic Gradient Classifier

sgd_predictions = sgd_clf.score(x_test, heart_test_labels)

print('Stochastic Gradient Classifier Accuracy score: ', sgd_predictions*100)
#Scikit-Learn's Logistic Regression

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()

log_reg.fit(x, heart_labels)

print("Logistic Regression Accuracy score: ", log_reg.score(x_test,heart_test_labels)*100)

#Scikit-Learn's KNN Model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)  # n_neighbors means k

knn.fit(x, heart_labels)

print("KNN Model Accuracy score: ", knn.score(x_test, heart_test_labels)*100)
#Scikit-Learn's Support Vector Machine (SVM)

from sklearn.svm import SVC

svm = SVC(random_state = 1)

svm.fit(x, heart_labels)

print("SVM Algorithm Accuracy score: ", svm.score(x_test,heart_test_labels)*100)

#Scikit-Learn's Naive Bayes Algorithm

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x, heart_labels)

print("Naive Bayes Algorithm Accuracy :", nb.score(x_test,heart_test_labels)*100)

#Scikit-Learn's Decision Tree Algorithm

from sklearn.tree import DecisionTreeClassifier

dt_class = DecisionTreeClassifier()

dt_class.fit(x, heart_labels)

print("Decision Tree Algorithm Accuracy :", dt_class.score(x_test, heart_test_labels)*100)

#Scikit-Learn's Random Forest Classification

from sklearn.ensemble import RandomForestClassifier

rf_class = RandomForestClassifier(n_estimators = 1000, random_state = 1)

rf_class.fit(x, heart_labels)

print("Random Forest Algorithm Accuracy :", rf_class.score(x_test,heart_test_labels)*100)
