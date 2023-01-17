# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt # plotting and exploration
%matplotlib inline

from sklearn.linear_model import LogisticRegression #Our Linear Regression Model
from sklearn.tree import DecisionTreeClassifier #Add a tree for testing
from sklearn.ensemble import ExtraTreesClassifier #Add extra tree classifier for comparison
from sklearn.ensemble import GradientBoostingClassifier #Add a gradient boosted classifier for more comparison
from sklearn.ensemble import VotingClassifier #Add a voting classifier to see if it helps

from sklearn.preprocessing import StandardScaler #Scale the model
from sklearn.metrics import classification_report #Report on the classification

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("../input/us-census-data/adult-training.csv") #Load the training data
train.head() #Take a peek at the training data
test = pd.read_csv("../input/us-census-data/adult-test.csv") #Load the testing data
test.head() #Take a peek at the testing data
test.reset_index(inplace = True) #Reset the indecies of test to pull the information into proper columns
test.head() #Take a peek at the test set
#Add all the name conversions based on the original data source
names = {"level_0" : "Age", "level_1" : "WorkClass", "level_2" : "fnlwgt", "level_3" : "Education", "level_4" : "EducationIndex",
        "level_5" : "MaritalStatus", "level_6" : "Occupation", "level_7" : "Relationship", "level_8" : "Race", "level_9" : "Gender",
        "level_10" : "CapitalGain", "level_11" : "CapitalLoss", "level_12" : "HoursWorked", "level_13" : "NativeCountry", 
        "|1x3 Cross validator" : "IncomeBracket"}
test = test.rename(columns = names) #Rename the columns
test.head() #Take a peek at the spiffy test set
train.head() #Take a peek at the train set in order to compare and reassign column names
#Add all the name conversions based on the original data source
namesTrain = {"39" : "Age", " State-gov" : "WorkClass", " 77516" : "fnlwgt", " Bachelors" : "Education", " 13" : "EducationIndex",
              " Never-married" : "MaritalStatus", " Adm-clerical" : "Occupation", " Not-in-family" : "Relationship", " White" : "Race", 
              " Male" : "Gender", " 2174" : "CapitalGain", " 0" : "CapitalLoss", " 40" : "HoursWorked", " United-States" : "NativeCountry", 
              " <=50K" : "IncomeBracket"}
train = train.rename(columns = namesTrain) #Rename the train columns
train.head() #Take a peek at the spiffy train data
train.replace(' ?', np.nan, inplace=True) #Change training ? into null
test.replace(' ?', np.nan, inplace=True) #Change testing ? into null
test.head() #Check the test set for if it worked, as the ? appeared early in test
print("Training Set Nulls:\n", train.isnull().any()) #Check the train set for nulls
print("\nTesting Set Nulls:\n", test.isnull().any()) #Check the test set for nulls
train["NativeCountry"] = train["NativeCountry"].fillna("Unknown") #Fill NativeCountry with "Unknown"
train = train.fillna("None") #Fill the other nulls with "None"
test["NativeCountry"] = test["NativeCountry"].fillna("Unknown") #Fill NativeCountry with "Unknown"
test = test.fillna("None") #Fill the other nulls with "None"
test.head() #Take a peek at the test data to make sure it applied
print("Training Set Nulls:\n", train.isnull().any()) #Check the train set for nulls
print("\nTesting Set Nulls:\n", test.isnull().any()) #Check the test set for nulls
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,6)) #Set the figures
train.hist(column = "Age", bins = 25, ax = axes[0], edgecolor = "white") #Load a histogram for the train ages
test.hist(column = "Age", bins = 25, ax = axes[1], edgecolor = "white") #Load a histogram for the test ages
axes[1].set_title("Age Test") #Set the Test title to identify it
plt.show() #Show the plot
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,6)) #Set the figures
train["WorkClass"].value_counts().plot.bar(ax = axes[0]) #Load a bar graph for the train WorkClass
test["WorkClass"].value_counts().plot.bar(ax = axes[1]) #Load a bar graph for the test WorkClass
axes[0].set_title("WorkClass") #Set the Train title to identify it
axes[1].set_title("WorkClass Test") #Set the Test title to identify it
plt.show() #Show the plot
#Replace Never-worked and Without-pay with None
train["WorkClass"].replace(" Without-pay", "None", inplace=True)
test["WorkClass"].replace(" Without-pay", "None" , inplace=True)
train["WorkClass"].replace(" Never-worked", "None", inplace=True)
test["WorkClass"].replace(" Never-worked", "None" , inplace=True)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,6)) #Set the figures
train["WorkClass"].value_counts().plot.bar(ax = axes[0]) #Load a bar graph for the train WorkClass
test["WorkClass"].value_counts().plot.bar(ax = axes[1]) #Load a bar graph for the test WorkClass
axes[0].set_title("WorkClass") #Set the Train title to identify it
axes[1].set_title("WorkClass Test") #Set the Test title to identify it
plt.show() #Show the plot
#Drop the fnlwgt of each dataset
train = train.drop(columns = ["fnlwgt"])
test = test.drop(columns = ["fnlwgt"])
test.head() #Take a peek at one to show it is gone
print(train[["Education", "EducationIndex"]].value_counts())
#Drop the Education Index of each dataset
train = train.drop(columns = ["EducationIndex"])
test = test.drop(columns = ["EducationIndex"])
test.head() #Take a peek at one to show it is gone
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,6)) #Set the figures
train["Education"].value_counts().plot.bar(ax = axes[0]) #Load a bar graph for the train Education
test["Education"].value_counts().plot.bar(ax = axes[1]) #Load a bar graph for the test Education
axes[0].set_title("Education") #Set the Train title to identify it
axes[1].set_title("Education Test") #Set the Test title to identify it
plt.show() #Show the plot
#ConvertEduc: Converts education levels into more congregated groups
#Input: the education level
#Output: the converted label
def convertEduc(level):
    #A list of labels that come together for the non-grad category
    levelsToChange = [" Preschool", " 1st-4th", " 5th-6th", " 7th-8th", " 9th", " 10th", " 11th", " 12th"]
    
    #If the person never graduated high school
    if level in levelsToChange:
        return " Non-Grad" #Return the collective non-grad category
    
    #If the person has an associate's degree, no matter where from
    if level == " Assoc-voc" or level == " Assoc-acdm":
        return " Associates" #Return that it is an associate's degree
    
    #If the level is prof-school, which I would consider essentially a doctorate
    if level == " Prof-school":
        return " Doctorate" #Return the doctorate label
    
    return level #Return the level label if there is nothing to change

train["Education"] = train["Education"].apply(convertEduc) #Convert the train education levels
test["Education"] = test["Education"].apply(convertEduc) #Convert the test education levels

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,6)) #Set the figures
train["Education"].value_counts().plot.bar(ax = axes[0]) #Load a bar graph for the train Education
test["Education"].value_counts().plot.bar(ax = axes[1]) #Load a bar graph for the test Education
axes[0].set_title("Education") #Set the Train title to identify it
axes[1].set_title("Education Test") #Set the Test title to identify it
plt.show() #Show the plot
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,6)) #Set the figures
train["MaritalStatus"].value_counts().plot.bar(ax = axes[0]) #Load a bar graph for the train Marital Status
test["MaritalStatus"].value_counts().plot.bar(ax = axes[1]) #Load a bar graph for the test Marital Status
axes[0].set_title("MaritalStatus") #Set the Train title to identify it
axes[1].set_title("MaritalStatus Test") #Set the Test title to identify it
plt.show() #Show the plot
#ConvertMarry: Converts marital status into more congregated groups
#Input: the marital status
#Output: the converted label
def convertMarry(label):
    labelsToChange = [" Married-AF-spouse", " Married-spouse-absent", " Married-civ-spouse"]
    
    #If the person is an AF/absent Spouse
    if label in labelsToChange:
        return " Married" #Return the collective married category
    
    return label #Return the label if there is nothing to change

train["MaritalStatus"] = train["MaritalStatus"].apply(convertMarry) #Convert the train education levels
test["MaritalStatus"] = test["MaritalStatus"].apply(convertMarry) #Convert the test education levels

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,6)) #Set the figures
train["MaritalStatus"].value_counts().plot.bar(ax = axes[0]) #Load a bar graph for the train Education
test["MaritalStatus"].value_counts().plot.bar(ax = axes[1]) #Load a bar graph for the test Education
axes[0].set_title("MaritalStatus") #Set the Train title to identify it
axes[1].set_title("MaritalStatus Test") #Set the Test title to identify it
plt.show() #Show the plot
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,6)) #Set the figures
train["Occupation"].value_counts().plot.bar(ax = axes[0]) #Load a bar graph for the train Occupation
test["Occupation"].value_counts().plot.bar(ax = axes[1]) #Load a bar graph for the test Occupation
axes[0].set_title("Occupation") #Set the Train title to identify it
axes[1].set_title("Occupation Test") #Set the Test title to identify it
plt.show() #Show the plot
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,6)) #Set the figures
train["Relationship"].value_counts().plot.bar(ax = axes[0]) #Load a bar graph for the train Relationship
test["Relationship"].value_counts().plot.bar(ax = axes[1]) #Load a bar graph for the test Relationship
axes[0].set_title("Relationship") #Set the Train title to identify it
axes[1].set_title("Relationship Test") #Set the Test title to identify it
plt.show() #Show the plot
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,6)) #Set the figures
train["Race"].value_counts().plot.bar(ax = axes[0]) #Load a bar graph for the train Race
test["Race"].value_counts().plot.bar(ax = axes[1]) #Load a bar graph for the test Race
axes[0].set_title("Race") #Set the Train title to identify it
axes[1].set_title("Race Test") #Set the Test title to identify it
plt.show() #Show the plot
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,6)) #Set the figures
train["Gender"].value_counts().plot.bar(ax = axes[0]) #Load a bar graph for the train Gender
test["Gender"].value_counts().plot.bar(ax = axes[1]) #Load a bar graph for the test Gender
axes[0].set_title("Gender") #Set the Train title to identify it
axes[1].set_title("Gender Test") #Set the Test title to identify it
plt.show() #Show the plot
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16,12)) #Set the figures
train.hist(column = ["CapitalGain", "CapitalLoss"], ax = axes[0]) #Load a histogram for the train capital
test.hist(column = ["CapitalGain", "CapitalLoss"], ax = axes[1]) #Load a histogram for the test capital
plt.show() #Show the plot
#Change Capital Gain/Loss into 0 and 1 for semi-boolean values
train["CapitalGain"] = train["CapitalGain"].apply(lambda x: 1 if x>0 else 0)
test["CapitalGain"] = test["CapitalGain"].apply(lambda x: 1 if x>0 else 0)
train["CapitalLoss"] = train["CapitalLoss"].apply(lambda x: 1 if x>0 else 0)
test["CapitalLoss"] = test["CapitalLoss"].apply(lambda x: 1 if x>0 else 0)

test.head() #Take a peek at the test dataframe, which has a 1 in the first five
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,12)) #Set the figures
train.hist(column = "HoursWorked", bins = 10, ax = axes[0], edgecolor = "white") #Load a histogram for the train capital
test.hist(column = "HoursWorked", bins = 10, ax = axes[1], edgecolor = "white") #Load a histogram for the test capital
plt.show() #Show the plot
print(train["HoursWorked"].value_counts())
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,6)) #Set the figures
train["NativeCountry"].value_counts().plot.bar(ax = axes[0]) #Load a bar graph for the train NativeCountry
test["NativeCountry"].value_counts().plot.bar(ax = axes[1]) #Load a bar graph for the test NativeCountry
axes[0].set_title("NativeCountry") #Set the Train title to identify it
axes[1].set_title("NativeCountry Test") #Set the Test title to identify it
plt.show() #Show the plot
#Country Groupings
Asia = [" Laos", " Thailand", " Taiwan", " Vietnam", " Japan", " Cambodia", " China", " India", " Philippines", " Hong", " Iran"]
Europe = [" Hungary", " Yugoslavia", " France", " Scotland", " Ireland", " Greece", " Poland", " Portugal", " Italy", " England",
         " Germany", " Holand-Netherlands"]
LatinAmerica = [" Honduras", " Trinadad&Tobago",  " Nicaragua", " Peru", " Ecuador", " Guatemala", " Jamaica", " Columbia", " Haiti",
               " Dominican-Republic", " Cuba", " El-Salvador"]
AmericanTerritories = [" Puerto-Rico", " Outlying-US(Guam-USVI-etc)"]
#ConvertCountry: Converts countries into more congregated groups
#Input: the country
#Output: the converted label
def convertCountry(country):
    #If the person is from an Asian country
    if country in Asia:
        return " Asia" #Return the collective Asian category
    
    #If the person is from a European country
    if country in Europe:
        return " Europe" #Return the collective European category
    
    #If the person is from a Latin American country
    if country in LatinAmerica:
        return " LatinAmerica" #Return the collective Latin America category
    
    #If the person is from a US Territory
    if country in AmericanTerritories:
        return " United-States" #Return United-States, as it is part of the US
    
    #If the country is just listed as south
    if country == " South":
        return "Unknown" #Return unknown because what do you mean "south"?
    
    return country #Return the country if there is nothing to change

train["NativeCountry"] = train["NativeCountry"].apply(convertCountry) #Convert the train countries
test["NativeCountry"] = test["NativeCountry"].apply(convertCountry) #Convert the test countries

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,6)) #Set the figures
train["NativeCountry"].value_counts().plot.bar(ax = axes[0]) #Load a bar graph for the train NativeCountry
test["NativeCountry"].value_counts().plot.bar(ax = axes[1]) #Load a bar graph for the test NativeCountry
axes[0].set_title("NativeCountry") #Set the Train title to identify it
axes[1].set_title("NativeCountry Test") #Set the Test title to identify it
plt.show() #Show the plot
#Changes <=50K into 0 and >50K into 1, indicating less than or more than 50K per year respectively
#Also, it checks if x is already 0, because if so, it will turn it into a 1 and everything will be 1 (happy day for everyone under 50K)
train["IncomeBracket"] = train["IncomeBracket"].apply(lambda x: 0 if x == " <=50K" or x == 0 else 1)
test["IncomeBracket"] = test["IncomeBracket"].apply(lambda x: 0 if x == " <=50K." or x == 0 else 1)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,6)) #Set the figures
train["IncomeBracket"].value_counts().plot.bar(ax = axes[0]) #Load a bar graph for the train IncomeBracket
test["IncomeBracket"].value_counts().plot.bar(ax = axes[1]) #Load a bar graph for the test IncomeBracket
axes[0].set_title("IncomeBracket") #Set the Train title to identify it
axes[1].set_title("IncomeBracket Test") #Set the Test title to identify it
plt.show() #Show the plot
#These are the y (value being tracked) for our train test split
incomeTrain = train["IncomeBracket"].copy()
incomeTest = test["IncomeBracket"].copy()


#These are the x (values being tested against) for our train test split
characterTrain = train.drop("IncomeBracket", axis = 1)
characterTest = test.drop("IncomeBracket", axis = 1)

#Encode our variables so they can be read by the regression
characterTrain = pd.get_dummies(characterTrain)
characterTest = pd.get_dummies(characterTest)

characterTest.head() #Print one of the characteristics segments to make sure structure is right.
stdScale = StandardScaler() #Bring in the standard scaler to scale the data
stdScale.fit(characterTrain) #Fit the scaler to our train set
characterTrain = stdScale.transform(characterTrain) #Scale the train set
characterTest = stdScale.transform(characterTest) #Scale the test set
regression = LogisticRegression(C = 0.9, class_weight = "Balanced") #Initialize the logistic regression
regression.fit(characterTrain, incomeTrain) #Fit the logistic regression

clf = ExtraTreesClassifier(n_estimators = 100, max_depth = 5, min_samples_split = 2, random_state = 0) #Initialize the extra trees classifier
clf.fit(characterTrain, incomeTrain) #Fit the lots of trees model

clf2 = DecisionTreeClassifier(max_depth = None, random_state = 0) #Classifies with just a tree
clf2.fit(characterTrain, incomeTrain) #Fit the single tree

clf3 = GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.5, max_depth = 1, random_state = 0) #Initialize the gradient boosting classifier
clf3.fit(characterTrain, incomeTrain) #Fit the gradient boost

ereg = VotingClassifier(estimators = [('lr', regression), ('et', clf), ('dt', clf2), ('gb', clf3)]) #Initialize the voting classifier
ereg = ereg.fit(characterTrain, incomeTrain) #Fit the voting
#Print the scores for all the different classifiers
print("Logistic Regression Score: ", regression.score(characterTest, incomeTest))
print("Extreme Random Forest Score: ", clf.score(characterTest, incomeTest))
print("Decision Tree Score: ", clf2.score(characterTest, incomeTest))
print("Gradient Boosting Score: ", clf3.score(characterTest, incomeTest))
print("Voting Classification Score: ", ereg.score(characterTest, incomeTest))