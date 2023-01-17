# Loading pandas
import pandas as pd
# Loading numpy
import numpy as np

# calling predefined magic function
%matplotlib inline

# Loading the seaborn class for plotting the various bar graphs
import seaborn as sns

# Setting random seed
np.random.seed(0)
# Loading the training data into dataframe
pd_train = pd.read_csv("E:/train.csv")
# Viewing the training dataset
pd_train.head()
# Loading the test data into dataframe
pd_test = pd.read_csv("E:/test.csv")
# Viewing the test dataset
pd_test.head()
# Getting the details of each of the columns of the training data set
pd_train.info()
# Getting data analysis done on the various numeric fields in the training set
pd_train.describe()
# Plotting the count of total count of persons who survived versus who did not
sns.countplot(x='Survived', data=pd_train);
# Factorizing sex attribute of the training dataset into male or female as 0 or 1
pd_train["sex"] = pd.factorize(pd_train["Sex"])[0]
# Factorizing sex attribute of the test dataset into male or female as 0 or 1
pd_test["sex"] = pd.factorize(pd_test["Sex"])[0]

# Calculating the mode of the Age column of the training dataset
freq_age=pd_train.Age.dropna().mode()[0]
# Printing the modal age
freq_age
# Replacing all the NaN values in the Age column with the modal age in the training dataset
pd_train['Age']=pd_train['Age'].fillna(freq_age)
# Replacing all the NaN values in the Age column with the modal age in the test dataset
pd_test['Age']=pd_test['Age'].fillna(freq_age)
# Printing the modified test dataset
pd_test.head()
# Loading the Gaussian Naive Bayes function from the naive bayes library
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing

# Declaring a Gaussian Naive Bayes classifier object
clf =GaussianNB()
# Creating a new train dataset by selecting Pclass Sex and Age of the person
# to predict the chance of survival
X = pd.concat([pd_train.iloc[:,2],pd_train.iloc[:,5],pd_train.iloc[:,12]], axis=1)

# Creating an object Y to store the taining target data
Y = pd_train["Survived"]

# Creating a new dataset to store the test features
test_df = pd.concat([pd_test.iloc[:,1],pd_test.iloc[:,4],pd_test.iloc[:,11]], axis=1)
# Viewing X
X.head()
# Viewing Y
Y.head()
# Fitting our X and Y datasets in the Gaussian classifier
clf.fit(X,Y)
# Predicting the Survival chance 
clf.predict(test_df)
# Calculating the accuracy of our predictor
clf.score(X,Y)
# Loading the KNN Classifier from sklearn class
from sklearn import neighbors
# Declaring the KNN Classifier object
clf2 = neighbors.KNeighborsClassifier(3,weights="uniform")
# Creating a trained model using KNN Classifier
trained_model = clf2.fit(X,Y)
# Viewing the trained model
print("trained_model: ",trained_model)
# Calculating the accuracy of our predictor
print(trained_model.score(X,Y))
# Calculating the prediction probability of our training model
print(trained_model.predict_proba(test_df))
# Prediction of survival chance based on our testing model
print(trained_model.predict(test_df))
# Plotting the heatmap of the given data
# A heatmap is a overall view of the total dataset
# It is vital in the case where we need to have a look at the various data points in a dataset
sns.heatmap(pd_train[["Pclass","SibSp","Parch","Fare","Age","Survived"]].corr(),annot=True, fmt = ".2f", cmap = "Blues");
# Explore Fare distribution 
g = sns.distplot(pd_train["Age"], color="m", label="Skewness : %.2f"%(pd_train["Fare"].skew()))
g = g.legend(loc="best")
g
# Loading pyplot class of matplot library
import matplotlib.pyplot as plt

# Calculating the total count of persons died and survived
sizes = pd_train.groupby('Survived')["sex"].count()
# Viewing the total count
sizes
# Calculating the total percentage of people died
p1 = (sizes[0]*100)/(sizes[0]+sizes[1])

# Calculating the total percentage of people survived
p2= (sizes[1]*100)/(sizes[0]+sizes[1])

# Storing the percentages in a list
size = [p1,p2]

# Declaring objects of subplots
figureObject, axesObject = plt.subplots()

# Creating a pie chart with percentages of people died or survived
axesObject.pie(size,labels=["Died","Survived"],autopct='%.2f%%',startangle=90)
axesObject.axis("equal")
plt.show()
# Plotting the swarm graph to show the correlation between the Fare and survival chance
sns.swarmplot(x='Survived', y='Fare', data=pd_train);
# Show the correlation between the Fare and survival chance
pd_train.groupby('Survived').Fare.describe()
#import library
#from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
#Building Decision Tree-CART Algorithm (gini criteria)
#max_depth=5 signifies the depth of tree
dt_train_gini=DecisionTreeClassifier(criterion="gini",random_state=100,max_depth=5,min_samples_leaf=5)
#Train
dt_train_gini.fit(X,Y)
#To see the decision tree plot we'll use graphviz
# "gini" option helps in leveraging CART - Classification and Regression Tree -algorithm for fitting the decision tree.
from sklearn import tree
with open("dt_train_gini.txt","w")as f:
    f=tree.export_graphviz(dt_train_gini,rounded=True,filled=True,out_file=f)
    
#creating an object or instance of DecisionTree Classifier
clf3 = tree.DecisionTreeClassifier(max_depth=5)
#Fitting in the datamodels
clf3.fit(X, Y)
Y_pred = clf3.predict(test_df)
pd_test['Survived'] = Y_pred

# displaying the pd_test
pd_test
# analyzing the score of model i.e accuracy 
clf3.score(X,Y)
