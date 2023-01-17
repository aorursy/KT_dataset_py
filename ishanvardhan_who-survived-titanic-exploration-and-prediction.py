#import your libraries

import pandas as pd

import sklearn as sk

import numpy as np

import scipy

from scipy import stats

import plotly.express as px

import sqlite3

import re

import matplotlib.pyplot as plt

import seaborn as sns

from pandas.plotting import scatter_matrix

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn import metrics

from sklearn.metrics import roc_auc_score

from sklearn.metrics import accuracy_score


#Reading the training datafile and saving it as a dataframe

Titanic_Train_table = pd.read_csv('/kaggle/input/titanic/train.csv')



#Reading the test datafile and saving it as a dataframe

Titanic_Test_table = pd.read_csv('/kaggle/input/titanic/test.csv')

class Load:

    def __init__(self, df):

        """Initialising the constructor to save data in the different attributes"""

        self.data = df

        self._calculate(df)



    def _calculate(self, df):

        """calls internal functions to show headers, descriptive stats and nulls"""

        self._heading(df)

        self._stats(df)

        self._nulls(df)



    def _heading(self, df):

        """Using pandas library to get a glimpse of the dataset"""

        self.lines = df.head()

        self.inf = df.info()

        self.matrix = df.shape

        self.cols = df.columns

    

    def _stats(self, df):

        """Displays all the descriptive stats of the dataset"""

        self.description=df.describe()

        self.relation=df.corr()



        

    def _nulls(self, df):

        """Inspects the whole dataset and returns a list of columns and the respective number of Null values"""

        self.missing = df.isnull().sum()

        """Inspects the whole data frame and replaces any blanks with NaNs. The NaNs can then be preproessed on the basis of column."""

        self.cl = df.replace(r'^\s*$', np.nan, regex=True)
#Instantiating an object for the training dataset from the Load class

train = Load(Titanic_Train_table)



# Calling the functions of the Load Class

train._heading(Titanic_Train_table)

train._nulls(Titanic_Train_table)

train._stats(Titanic_Train_table)
# Having look at the data stored in the attributes and a Glimpse of the data

train.lines
train.description  

#The description shows a high number of blanks/NaNs for Age"""
train.missing       



#There are no nulls, apart from Age, Cabin and Embarked"""
train.cl            

#Returns a data frame with all the blank cells replaced with NaNs.
#Storing the result of the cleaned data in a new dataframe

train_cleaned = train.cl



#Converting the dtypes of the values in dataframe to numeric

train_cleaned[["Age", "Fare","Survived","SibSp","Parch","Pclass"]] = train_cleaned[["Age", "Fare","Survived","SibSp","Parch","Pclass"]].apply(pd.to_numeric)



#Performing quality checks: if there are values in age column above 120 (Approximate upper limit), convert them to NaN.

train_cleaned.loc[train_cleaned.Age > 120, 'Age'] = np.nan
#Check for the most frequently occuring value in a particular column of a dataframe. In Particular Embarked to replace NaNs.

train_cleaned['Embarked'].value_counts()

#Most Common value is S: Southanmpton

#Replacing the NaNs in Age column by the median Age Value

train_cleaned['Age'].fillna(train_cleaned['Age'].median(), inplace=True)





#Replacing the NaNs in Embarked column by the most common value, i.e. S

commonvalue_embarked="S"

train_cleaned["Embarked"].fillna(commonvalue_embarked, inplace = True) 

#Quality Check: Check if all the NANs have been replaced and only Cabin Column left. 

train_cleaned.isnull().sum()
class Visuals:

    def __init__(self, df):

        self.data = df

        self._calculate(df)



    def _calculate(self, df):

        """calls internal functions to show basic pandas visualisations"""

        self._visualisations(df)



    def _visualisations(self, df):

        """Display the outliers and the distribution of the data"""

        self.descriptions = df.boxplot(vert = False)

        self.histogram = df.hist(color='k')
#Instantiating an object of the Visualisation class and inspecting the outliers and data distribution

v_train=Visuals(train_cleaned)
#Visualisations using Seaborn





#lm plots using the Logistic model predict the passenger survival

figure1 = sns.lmplot(x="Age", y="Survived", logistic=True, col="Sex",

                         data=train_cleaned, aspect=1, x_jitter=.1, palette="Set1")

    

figure2 = sns.lmplot(x="Fare", y="Survived", logistic=True, col="Sex",

                         data=train_cleaned, aspect=1, x_jitter=.1, palette="Set1")







#Jointplot to understand the Pearson Corelation coefficient between the variables

     

figure3 = sns.jointplot("Age", "Fare", data=train_cleaned, color='b', kind='reg')

    
#Using Heatmap to visualise the Pearson corelation coefficient between the different , especially useful to understand 

#impact on the Survived column, due to other independent variables





fig, ax = plt.subplots(figsize=(14,14))         # Sample figsize in inches

sns.heatmap(train_cleaned.corr(), annot=True, linewidths=.5, ax=ax)
# Detailed histogram of the training data using plotly



figure5 = px.histogram(train_cleaned, x="Age", y="Fare", color="Survived",

                   marginal="box", # or violin, rug

                  hover_data=train_cleaned.columns)

                     

figure5.show()







#ViolinPlots to better understand distiribution of Fare data as a function of Sex and Survival



figure6 = px.violin(train_cleaned, y="Fare", x="Sex", color="Survived", box=True, points="all",

          hover_data=train_cleaned.columns)

figure6.show()
# scatter plots using Plotly to visualise the survival for each Parch leval as a function of Age and Fare



figure7 = px.scatter(train_cleaned, x="Fare", y="Age", color='Survived',facet_col="Parch")

figure7.show()





# scatter plots using Plotly to visualise the survival for each Parch leval as a function of Age and Fare



figure8 = px.scatter(train_cleaned, x='Fare', y='Age', color='Survived',

                facet_col='SibSp')

figure8.show()

# Scatter Plots Using Plotly to understand the survival rate for different classes and as a function of Age and Fare



figure9 = px.scatter(train_cleaned, x='Fare', y='Age', color='Survived',

                facet_col='Pclass')

figure9.show()





# Scatter Plots Using Plotly to understand the survival rate for different Boarding stations and as a function of Age and Fare

figure10 = px.scatter(train_cleaned, x='Fare', y='Age', color='Survived',

                facet_col='Embarked')

figure10.show()
class Engineering:

    def __init__(self, df, var1, var2):

        """Initialising the constructor to save the data"""

        self.data = df

        self.variable1 = var1

        self.variable2 = var2

        self._calculate(df, var1, var2)





    def _calculate(self, df, var1, var2):

        """Calculates all the internal functions defined, i.e. featureadded, featuredivision and featureSubtracted"""

        self._featureadded(var1,var2)

        self._featuredivision(var1,var2)

        self._featuremultiplied(var1,var2)

        self._featuresubtracted(var1,var2)

        

    

    def _featuredivision(self,var1,var2):

        """calculates ratio of given two variables"""

        self.dividedfeature=var1//var2

        

    

    def _featuremultiplied(self,var1,var2):

        """calculates product of given two variables"""

        self.multipliedfeature=var1*var2

        

        

    def _featureadded(self,var1,var2):

        """Calculates the addition of two variables"""

        self.addedfeature=var1+var2

        

        

    def _featuresubtracted(self,var1,var2):

        """calculates the subtraction of two variables"""

        self.subtractedfeature=var1-var2
#Instantiating two objects of the training data from Engineering class

eng_feature1 = Engineering(train_cleaned,train_cleaned['SibSp'], train_cleaned['Parch'])

eng_feature2 = Engineering(train_cleaned,train_cleaned['Age'], train_cleaned['Pclass'])





#Using addition method from the Engineering class on the first object

eng_feature1._featureadded(train_cleaned['SibSp'], train_cleaned['Parch'])





#Using multiplication method from the Engineering class on the second object

eng_feature2._featuremultiplied(train_cleaned['Age'], train_cleaned['Pclass'])





#Saving the data of above two methods into new columns of the cleaned Train dataset. 

train_cleaned['Family'] = eng_feature1.addedfeature

train_cleaned['AgeandClass'] = eng_feature2.multipliedfeature

# Inspecting the training dataset after Feature Engineering

train_cleaned
# Getting Dummy Variables for the categorical variables and inspecting the data

train_cleaned = pd.get_dummies(train_cleaned, columns=['Sex','Embarked'])

train_cleaned.head()

# Segregating the cleaned training dataset into Features and Target variables



train_target=train_cleaned.iloc[:,1]

train_features=train_cleaned.iloc[:,lambda train_cleaned:[2,4,8,10,11,12,13,14,15,16]]



#Inspecting the features dataset

train_features
class Outliers:



    def __init__(self,df):

        self.data=df

        self._calculate(df)

    

    def _calculate(self, df):

        """calls internal functions to calculate outliers using the zscore method and the IQR Method"""

        self._zscore(df)

        self._iqr(df)

        self._showiqr(df)

        

    def _zscore(self, df):

        self.z = np.abs(stats.zscore(df))

        print(self.z)

        

    def _iqr(self, df):

        self.Q1 = df.quantile(0.25)

        self.Q3 = df.quantile(0.75)

        self.IQR = self.Q3 - self.Q1

        print(self.IQR)

    

    def _showiqr(self, df):

        print(df < (self.Q1 - 1.5 * self.IQR)) |(df > (self.Q3 + 1.5 * self.IQR))

        

#https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba
#Instantiating an object for the test dataset from the Load class

test = Load(Titanic_Test_table)



# Calling the functions of the Load Class and storing all the values

test._heading(Titanic_Test_table)

test._nulls(Titanic_Test_table)

test._stats(Titanic_Test_table)
# Having look at the data stored in the attributes

test.description  

#The description shows a high number of blanks for Age, Cabin and i value for Fare"""
test.missing       

#There are no nulls, but there is presence of blank cells"""
test.cl            

#Returned a data frame with all the blank cells replaced with NaNs.
#Storing the result of the cleaned data in a new dataframe

test_cleaned = test.cl



#Converting the dtypes of the values in dataframe to numeric

test_cleaned[["Age", "Fare","SibSp","Parch","Pclass"]] = test_cleaned[["Age", "Fare","SibSp","Parch","Pclass"]].apply(pd.to_numeric)



#Performing quality checks: if there are values in age column above 120, convert them to NaN.

test_cleaned.loc[test_cleaned.Age > 120, 'Age'] = np.nan
#Quality Check

test_cleaned.isnull().sum()
#Replacing the NaNs in Age column by the median Age Value"""



test_cleaned['Age'].fillna(test_cleaned['Age'].median(), inplace=True)





#Replacing the NaNs in Fare column by the median Fare Value"""

test_cleaned['Fare'].fillna(test_cleaned['Fare'].median(), inplace=True)
#Instantiating two objects of test dataset from the Engineering class

eng_feature1 = Engineering(test_cleaned,test_cleaned['SibSp'],test_cleaned['Parch'])

eng_feature2 = Engineering(test_cleaned,test_cleaned['Age'],test_cleaned['Pclass'])



#Using addition method from the Engineering class on the first object

eng_feature1._featureadded(test_cleaned['SibSp'],test_cleaned['Parch'])



#Using multiplication method from the Engineering class on the second object

eng_feature2._featuremultiplied(test_cleaned['Age'],test_cleaned['Pclass'])



#Saving the data of above two methods into new columns of the original dataset. 

test_cleaned['Family'] = eng_feature1.addedfeature

test_cleaned['AgeandClass'] = eng_feature2.multipliedfeature
# Getting Dummy Variables for the categorical variables in the test dataset



test_cleaned = pd.get_dummies(test_cleaned, columns=['Sex','Embarked'])

test_cleaned.head()
# Removing the un-important columns from the test dataset and making it exactly similar to the train dataset.



test_features=test_cleaned.iloc[:,lambda test_cleaned:[1,3,7,9,10,11,12,13,14,15]]



# Inspecting the cleaned test dataset features

test_features
class Modelling:

    def __init__(self,var1,var2):

        """Initialising the constructor method to save the data"""

        self.X=var1

        self.Y=var2

        self._calculate(var1, var2)

        

    def _calculate(self, var1, var2):

        """Calculates all the internal functions that contain the different models"""

        self._baselinemodel(var1,var2)

        self._forest(var1,var2)

        self._boosting(var1,var2)

        self._neighbours(var1,var2)

        self._vectors(var1,var2)

        

    def _baselinemodel(self,var1,var2):

        """We start by building a simple logistic regression and then improve upon the results"""

        self.model=LogisticRegression(random_state=42)

        self.model_1=self.model.fit(var1,var2)

        

    def _forest(self,var1,var2):

        """Use the ensemble methods to improve on the baseline model"""

        self.model=RandomForestClassifier(n_estimators=25, max_depth=7, random_state=42)

        self.model_2=self.model.fit(var1,var2)

        

    def _boosting(self,var1,var2):

        """Implemetation of Boosting algorithm to understand differences from the ensemble ones"""

        self.model=GradientBoostingClassifier(criterion='friedman_mse', init=None,

              learning_rate=0.05, loss='deviance', max_depth=7,

              max_features=1.0, max_leaf_nodes=None,

              min_impurity_decrease=0.0, min_impurity_split=None,

              min_samples_leaf=9, min_samples_split=2,

              min_weight_fraction_leaf=0.0, n_estimators=50,

              presort='auto', random_state=42, subsample=1.0, verbose=0,

              warm_start=False)

        self.model_3=self.model.fit(var1,var2)

        

    def _neighbours(self,var1,var2):

        """A simple model based on nearest neighbours methodology"""

        self.model=KNeighborsClassifier(n_neighbors = 7)

        self.model_4=self.model.fit(var1,var2)

        

    def _vectors(self,var1,var2):

        """implementing support vector machines"""

        self.model=SVC(kernel = 'linear', C = 1)

        self.model_5=self.model.fit(var1,var2)
#Instantiating an object of the Modelling class

mod=Modelling(train_features, train_target)



# Using the calculate function to Implement all the models

mod._calculate(train_features, train_target)





#Saving the results of all the models in different variables

lr=mod.model_1

rf=mod.model_2

gb=mod.model_3

kn=mod.model_4

sv=mod.model_5
class Eval:

    def __init__(self, v1, v2, v3, v4):

        """Initialising the constructor and saving all the variable data"""

        self.X_test=v1

        self.X_train=v2

        self.model=v3

        self.Y_train=v4

        self._calculate(v1, v2, v3, v4)

        

    def _calculate(self, v1, v2, v3, v4):

        self._pred(v1,v3)

        self._cv(v3, v2, v4)

        

    def _pred(self, v1, v3):

        """Predicting the target variable"""

        self.predicted_value=v3.predict(v1)

        

    def _cv(self, v3, v2, v4):

        """Cross validation"""

        self.scores=cross_val_score(v3, v2, v4, cv=4)

#Instantiating objects for the five different models

eval1=Eval(test_features, train_features, lr, train_target)

eval2=Eval(test_features, train_features, rf, train_target)

eval3=Eval(test_features, train_features, gb, train_target)

eval4=Eval(test_features, train_features, kn, train_target)

eval5=Eval(test_features, train_features, sv, train_target)





# Calculating the cross-validation scores as well as prediction values for all the models

eval1._calculate(test_features, train_features, lr, train_target)

eval2._calculate(test_features, train_features, rf, train_target)

eval3._calculate(test_features, train_features, gb, train_target)

eval4._calculate(test_features, train_features, kn, train_target)

eval5._calculate(test_features, train_features, sv, train_target)



#Storing the cross validation scores of all the models in different variables

scores_lr = eval1.scores

scores_rf = eval2.scores

scores_gb = eval3.scores

scores_kn = eval4.scores

scores_sv = eval5.scores





# Creating a list of all the Cross validation scores and plotting them

cv_scores=[scores_lr, scores_rf, scores_gb, scores_kn, scores_sv]



for x in cv_scores:

    print(x.mean())

    plt.plot(x)

    



# Inspecting the feature importance from the random Forest Model (Prescriptive Statistics)

rf.feature_importances_





#Plottng the Feature Importances

feature_importances = pd.Series(rf.feature_importances_, index=train_features.columns)

print(feature_importances)

feature_importances.sort_values(inplace=True)

feature_importances.plot(kind='barh', figsize=(7,6))
# Hyper parameter Tuning Using Grid Search

# Define Parameters for Gradient Boosting Model as it shows the best cv score



param_grid = {"max_depth": [2,3,7],

              "max_features" : [1.0,0.3,0.1],

              "min_samples_leaf" : [3,5,9],

              "n_estimators": [8,10,25,50],

              "learning_rate": [0.05,0.1,0.02,0.2]}





# Perform Grid Search CV

from sklearn.model_selection import GridSearchCV

gb_cv = GridSearchCV(gb, param_grid=param_grid, cv = 4, verbose=10, n_jobs=-1 ).fit(train_features, train_target)







# Best hyperparmeter setting

gb_cv.best_estimator_
# Selecting the best model results out of all the models, (Gradient Boosting)



Best_prediction_result=pd.DataFrame(eval3.predicted_value)

PassID=pd.DataFrame(test_cleaned['PassengerId'])

Submission_file=pd.concat([PassID, Best_prediction_result], axis = 1)

Submission_file.columns=['PassengerId', 'Survived']


