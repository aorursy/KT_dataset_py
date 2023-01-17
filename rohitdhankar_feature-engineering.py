



%matplotlib inline



#

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.cross_validation import KFold #cross_val_score         # DHANK - This import option may be changed 

from sklearn.model_selection import cross_val_score                 # DHANK - This import option may be changed 

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.feature_selection import SelectKBest, f_classif



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import re                                                          # Reg-ex for Family Names Mr etc 

import operator

from scipy import stats

#

# Basis info here -SPLITTER CLASSES - http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection

# looking at various SPLITTER Options 

# 

# My GitHubRepo - https://github.com/RohitDhankar

# Kaggle Profile - https://www.kaggle.com/rohitdhankar



# Python 3.5 Notebook to be uploaded to Kaggle - Python 3.5 virtual env activated - usually work with 2.7 

#

#conda create -n py35 python=3.5 ipykernel

#source activate py35

#

# conda create -n py27 python=2.7 ipykernel

# source activate py27



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import mixture

from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix

%matplotlib inline

#

#dfTrn = pd.read_csv("train.csv") # On Local Machine 

#dfTest = pd.read_csv("test.csv") # On Local Machine 

dfTrn = pd.read_csv("../input/train.csv") # On Kaggle 

dfTest = pd.read_csv("../input/test.csv") # On Kaggle 

print('Number of rows: {}, Number of columns: {}'.format(*dfTrn.shape))

print ("_"*90)

print('Number of rows: {}, Number of columns: {}'.format(*dfTest.shape))

#print ("_"*90)

#print(dfTrn.head(5))

#print ("_"*90)

#print(dfTest.head(5))



# If any Encoding or Mapping of String values in Features with Categorical or Numeric is required use MAP 

# MAP being own function it replaces in place and not creating another feature like One Hot Encoder etc 

feature_labels = []

missing_values = []



for col in dfTrn.columns:

    feature_labels.append(col)

    missing_values.append(dfTrn[col].isnull().values.ravel().sum()) # Append Feature wise Missing Values  

    print(col,"=", missing_values[-1]) # prints Feature Labels with Missing Values Count ...



#Source - http://stackoverflow.com/questions/28199524/best-way-to-count-the-number-of-rows-with-missing-values-in-a-pandas-dataframe    
feature_labels_tst = []

missing_values_tst = []



for col in dfTest.columns:

    feature_labels_tst.append(col)

    missing_values_tst.append(dfTest[col].isnull().values.ravel().sum()) # Append Feature wise Missing Values  

    print(col,"=", missing_values_tst[-1]) # prints Feature Labels with Missing Values Count ...



#Source - http://stackoverflow.com/questions/28199524/best-way-to-count-the-number-of-rows-with-missing-values-in-a-pandas-dataframe    
# Source - Kaggle - https://www.kaggle.com/michielkalkman/titanic/kaggle-titanic-001

# This source is similar in approach and code to DataQuest.io - Titanic Tute 

# All code below is standard DataQuest.io code -





# df_Any is any data_set TRAIN or TEST passed to this function

def harmonize_data(df_Any):

    

    df_Any["Age"] = df_Any["Age"].fillna(df_Any["Age"].median()) # Median values --- 17th JAN -- OK 

   # df_Any["Age"] = df_Any["Age"].fillna(df_Any["Age"].mean()) # Mean values

# Age - Multiple Imputation etc tried ....





    df_Any.loc[df_Any["Sex"] == "male", "Sex"] = 0     # SWAP this is and check performance

    df_Any.loc[df_Any["Sex"] == "female", "Sex"] = 1

#    

    df_Any["Embarked"] = df_Any["Embarked"].fillna("S") # Impute missing values

#

    df_Any.loc[df_Any["Embarked"] == "S", "Embarked"] = 0

    df_Any.loc[df_Any["Embarked"] == "C", "Embarked"] = 1

    df_Any.loc[df_Any["Embarked"] == "Q", "Embarked"] = 2



    df_Any["Fare"] = df_Any["Fare"].fillna(df_Any["Fare"].median())



    return df_Any



# Get Harmonized DF's from Train and Test Data 

# Cant be running this code cell or function more than once for a particular - df_Any # Test or Train 



dfHarTrn = harmonize_data(dfTrn)

#print(dfHarTrn.head(10))

names_Trn = dfHarTrn.columns.values

print ("_"*90)

print (names_Trn)





dfHarTst = harmonize_data(dfTest)

#print(dfHarTst.head(10))



names_Test = dfHarTst.columns.values

print ("_"*90)

print (names_Test)


## Feature Engineering - New features from old - Title , Family_Size , Family_ID, Family_Name , Deck  etc . 

## Imputation of Missing Values in both Train and Test  - Age , Cabin , Embarked , Fare 



# Getting values for Deck from the Cabin Variable 



dfHarTrn["Deck"]=dfHarTrn.Cabin.str[0]              # Add a Feature called DECK to Train DF 

dfHarTst["Deck"]=dfHarTst.Cabin.str[0]              # Add a Feature called DECK to Test DF 

dfHarTrn["Deck"].unique()                           # nan -- null values



# Histogram of Titanic Train Set 

#

dfHarTrn.hist(bins=10,figsize=(7,7),grid=True)



# Histogram of Titanic Test Set 

#

dfHarTst.hist(bins=10,figsize=(7,7),grid=True)

#

Fare_list = dfHarTrn["Fare"].tolist()



sns.distplot(Fare_list)
# Original Features - ['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch' 'Ticket' 'Fare' 'Cabin' 'Embarked']

# Drop Features -- 'PassengerId' 'Survived' 'Pclass' 'Name' , 'SibSp' 'Parch' 'Ticket''Cabin' and 'Embarked'

# Plot BoxPlots to see distribution of  Means and Variances 



  

dfTrn1 = dfHarTrn.drop(dfHarTrn.columns[[0,1,2,3,6,7,8,10,11]],axis=1,inplace=False) # 

print(dfTrn1.head(5))

print ("_"*90)

print('Number of rows: {}, Number of columns: {}'.format(*dfTrn1.shape))

names_DfTrn1 = dfTrn1.columns.values

print("names_DfTrn1")

print ("_"*90)



# Dataset boxplot for Means and Variances 



dfTrn1.plot(kind='box', vert=False)



# Seen below - Features have large Variance - data set needs std and scaling 


# Source -- http://seaborn.pydata.org/generated/seaborn.factorplot.html

# Seaborn Plots 



#sns.color_palette(palette=deep)

#sns.color_palette("RdBu", n_colors=7)

p1=sns.color_palette("bright")

p2=sns.color_palette("pastel")

p3=sns.color_palette("RdBu")



# deep, muted, bright, pastel, dark, colorblind





Plt1 = sns.factorplot("Survived", col="Deck", col_wrap=4,

                    data=dfHarTrn[dfHarTrn.Deck.notnull()],

                    kind="count", size=2.5, aspect=.8,palette=p1)





Plt2 = sns.factorplot("Survived", col="Sex", col_wrap=4,

                    data=dfHarTrn[dfHarTrn.Deck.notnull()],

                    kind="count", size=2.5, aspect=.8,palette=p3)





Plt3 = sns.factorplot("Survived", col="Pclass", col_wrap=4,

                    data=dfHarTrn[dfHarTrn.Deck.notnull()],

                    kind="count", size=2.5, aspect=.8,palette=p2)



#Source - https://www.kaggle.com/poonaml/titanic/titanic-survival-prediction-end-to-end-ml-pipeline#

#



g = sns.FacetGrid(dfHarTrn, hue="Survived", col="Pclass", margin_titles=True,

                  palette={1:"seagreen", 0:"gray"})

g=g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend()



# As seen below - for 

# Pclass-1 -  Fares are Higher and equally distributed between Ages - 20 to 60 

# Pclass-2 -  Fares are comparitively Lower and equally distributed between Ages - 0 to 60 

# Pclass-3 -  Fares are comparitively Lower and equally distributed between Ages - 0 to 40 

#Source - https://www.kaggle.com/poonaml/titanic/titanic-survival-prediction-end-to-end-ml-pipeline#

#



#PClass vise Viz of Age to be used to compare "Imputed AGE" with Original Data Age 



dfHarTrn.Age[dfHarTrn.Pclass == 1].plot(kind='kde')    

dfHarTrn.Age[dfHarTrn.Pclass == 2].plot(kind='kde')

dfHarTrn.Age[dfHarTrn.Pclass == 3].plot(kind='kde')

 

# plots an axis lable

plt.xlabel("Age")    

plt.title("Age Distribution within classes")

# sets our legend for our graph.

plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best') 
# Correlation amongst Features 



corr_df = dfHarTrn.corr(method='pearson')



print(corr_df.head(len(dfHarTrn))) # Not required as we are plottng the Correlation above 



# We can look at Column 1 of the Print out below - see what all Features have a 

# greater than 0.1 Corr value - Negative or Positive both considered . 
#Source -- http://www.tradinggeeks.net/2015/08/calculating-correlation-in-python/



print("--------------- CREATE A HEATMAP ---------------")

# Create a mask to display only the lower triangle of the matrix (since it's mirrored around its 

# top-left to bottom-right diagonal).

mask = np.zeros_like(corr_df)

mask[np.triu_indices_from(mask)] = True

# Create the heatmap using seaborn library. 

# List if colormaps (parameter 'cmap') is available here: http://matplotlib.org/examples/color/colormaps_reference.html

sns.heatmap(corr_df, cmap='RdYlGn_r', vmax=1.0, vmin=-1.0 , mask = mask, linewidths=2.5)

 

# Show the plot we reorient the labels for each column and row to make them easier to read.

plt.yticks(rotation=0) 

plt.xticks(rotation=90) 

plt.show()



# High Positive Corr :- 

# Survived and Fare = [0.257307]

#

# High Negative Corr :- 

# PClass and Survived = [-0.338481] -- PClass -1 Survived the Most and Pclass 3 the Least 

# 
# Imputing in Age 

# Source - https://www.kaggle.com/poonaml/titanic/titanic-survival-prediction-end-to-end-ml-pipeline#

# Imputation Method -1 - RF 



with sns.plotting_context("notebook",font_scale=1.5):

    sns.set_style("whitegrid")

    sns.distplot(dfHarTrn["Age"].dropna(),

                 bins=80,

                 kde=False,

                 color="red")

    sns.plt.title("Age Distribution")

    plt.ylabel("Count")
from sklearn.ensemble import RandomForestRegressor

#predicting missing values in age using Random Forest

def fill_missing_age(df):

    

    #Feature set

    #age_df = df[['Age','Embarked','Fare', 'Parch', 'SibSp',

    #             'TicketNumber', 'Title','Pclass','FamilySize',

    #             'FsizeD','NameLength',"NlengthD",'Deck']]

    

    

  #  age_df = df[['Fare', 'Parch', 'SibSp',

  #               'Title','Pclass','FamilySize']]

  





    # Also add the Test Set dfHarTest --------------------------- TBD 



    age_df = df[['Fare', 'Parch', 'SibSp','Pclass']]

  

    

    

    # Split sets into train and test

    train  = age_df.loc[ (df.Age.notnull()) ]# known Age values

    test = age_df.loc[ (df.Age.isnull()) ]# null Ages

    

    # All age values are stored in a target array

    y = train.values[:, 0]

    

    # All the other values are stored in the feature array

    X = train.values[:, 1::]

    

    # Create and fit a model

    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)

    rtr.fit(X, y)

    

    # Use the fitted model to predict the missing values

    predictedAges = rtr.predict(test.values[:, 1::])

    

    # Assign those predictions to the full data set

    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 

    

    return df
# Copy the DF's to Impute Age into 

# Print DF's - to see existing NaN values



dfHarTrn1 = dfHarTrn

print(dfHarTrn1.head(10))



print("_"*90)



dfHarTst1 = dfHarTst

print(dfHarTst1.tail(10))
#



dfHarTrn1=fill_missing_age(dfHarTrn1)

dfHarTst1=fill_missing_age(dfHarTst1)



# Print DF's - to see Filled in / Imputed -  NaN values



print(dfHarTrn1.head(10))

print("_"*90)

print(dfHarTst1.tail(10))

# Feature Engineering --- FAMILY SIZE 



# within the df_harmonized_train - which is our TRAIN DF of INDEPENDENT FEATURES 

# Add a FEATURE called - FamilySize



#dfHarTrn1["FamilySize"] = dfHarTrn1["SibSp"] + dfHarTrn1["Parch"]

dfHarTrn["FamilySize"] = dfHarTrn["SibSp"] + dfHarTrn["Parch"]

#df_harmonized_train["NameLength"] = df_harmonized_train["Name"].apply(lambda x: len(x))



dfHarTrn.head(15)
# Feature Engineering --- TITLE 

#DHANK-- Create a FUNC "get_title" - use Reg-Ex to get the Titles 



def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    if title_search:

        return title_search.group(1)

    return ""



#DHANK-- Apply .apply - above function to df_TRAIN --- same done to df_TEST below in another code cell 



titles = dfHarTrn["Name"].apply(get_title)

print(pd.value_counts(titles))



# Prints out the Titles available in df_TRAIN these are different for df_TEST 

# We create - title_mapping , below basis this above print out - print(pd.value_counts(titles))



title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, 

                 "Major": 7, "Col": 7, "Capt": 7,

                 "Mlle": 8, "Mme": 8, 

                 "Don": 9, 

                 "Lady": 10, "Countess": 10, "Jonkheer": 10, 

                 "Sir": 9, 

                 "Ms": 2}

for k,v in title_mapping.items():

    titles[titles == k] = v



print(pd.value_counts(titles))



dfHarTrn["Title"] = titles   # Add a Feature called - "Title"

dfHarTrn.head(5)

family_id_mapping = {} # Instantiate empty Dict 



def get_family_id(row):

    last_name = row["Name"].split(",")[0]   # In values of feature "Name" (",") separates LASTNAME from other Names

    family_id = "{0}{1}".format(last_name, row["FamilySize"])

    if family_id not in family_id_mapping:

        if len(family_id_mapping) == 0:

            current_id = 1

        else:

            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)

        family_id_mapping[family_id] = current_id

    return family_id_mapping[family_id]



family_ids = df_harmonized_train.apply(get_family_id, axis=1)  #Create PD_Core_Series traverse Axis=1-ROWS_Top_down 



family_ids[df_harmonized_train["FamilySize"] < 3] = -1             # All FamilySize's < 3 give default ID(-1)



print(pd.value_counts(family_ids))





df_harmonized_train["FamilyId"] = family_ids                       # Add a Feature - "FamilyId"

df_harmonized_train.head(5)

#print(type(family_ids))
from sklearn.linear_model import LogisticRegression

#from sklearn import cross_validation    # Depreciation Warning 

from sklearn.model_selection import cross_val_score # in place of "cross_validation" try "model_selection" 

# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html





#predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked","FamilySize"]

#predictors = ["Pclass", "Sex", "Fare", "Embarked","FamilySize"] # "SibSp", "Parch", "Age"-- REMOVED -- 79.46%

predictors = ["Pclass", "Sex", "Fare", "FamilySize"] # "Embarked" , "SibSp", "Parch", "Age"-- REMOVED -- 79.24%





alg    = LogisticRegression(random_state=1)



scores = cross_val_score(

    alg,

    dfHarTrn1[predictors],

    dfHarTrn1["Survived"],

    cv=3

)



print("{0:.2f}%".format(scores.mean() * 100))



# Check how "predictors" can be "names" from above - not to to be typed manually 

# 79.35% , 79.46%



from sklearn.ensemble import RandomForestClassifier

#from sklearn import cross_validation

from sklearn.model_selection import cross_val_score    # in place of "cross_validation" try "model_selection" 

# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html





#predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"] #> 0.835056689342

#predictors = ["Pclass", "Sex", "Age", "Fare","SibSp"] #> 0.0.835034013605

#predictors = ["Pclass", "Sex", "Age", "Fare" ,"FamilySize"] #> 0.842857142857 = 84.29% 

predictors = ["Pclass", "Sex", "Age", "Fare" ,"FamilySize","Title"] #> 0.842857142857 = 84.29% 







alg = RandomForestClassifier(       # Various parameter values passed :- 

    random_state=124,               # 1 , 123 , 124

    n_estimators=250,               # 150 , 250 , 

    min_samples_split=6,            # 4 , 6  

    min_samples_leaf=2              # 2 , 2

)

                              



scores = cross_val_score(

    alg,                           # ALGORITHM in Cross_val_score Documentation is called - ESTIMATOR

    dfHarTrn[predictors],         # X Array like INDEPENDENT VARIABLES 

    dfHarTrn["Survived"],         # y Array like TARGET or DEPENDENT VARIABLE

    cv=18                          # Default = 3 Fold Cross Validation , we used - 5 , 15, 18 

)



print("{0:.2f}%".format(scores.mean() * 100))



# 17th JAN --- predictors = ["Pclass", "Sex", "Age", "Fare"] 

#

# For Age imputation Random Forest Predictors used SET_1 == [['Fare', 'Parch', 'SibSp','Pclass']]

# 83.28% --- with Age imputed with Random Forest Values SET_1 

# 84.29%  --- with Age imputed with Median values of Age Feature 

# 84.29%  --- with Age imputed with Mean values of Age Feature







# 17th JAN --- predictors = ["Pclass", "Sex", "Age", "Fare" ,"FamilySize"] 

# 82.73% --- with Age imputed with Random Forest Values SET_1 #######################################

# 84.18%  --- with Age imputed with Median values of Age Feature 







# 17th JAN ---predictors = ["Pclass", "Sex", "Age", "Fare" ,"FamilySize","Title"] 

# 84.07% --- Age imputed with Median values of Age Feature 

# 83.73% --- Age imputed with MEAN values of Age Feature --- MEAN worst than MEDIAN

# Create a FUNC "get_title" - use Reg-Ex to get the Titles 



def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    if title_search:

        return title_search.group(1)

    return ""



# Apply .apply - above function to df_TRAIN --- same done to df_TEST below in another code cell 



titles_testDF = dfHarTst["Name"].apply(get_title)

print(pd.value_counts(titles_testDF))

# Above Code Cell -  Prints out Titles available in df_Test these are different for df_TRAIN - DONA  

# We create - title_mapping , below basis this above print out - print(pd.value_counts(titles))



title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6,"Col": 7,"Dona":8,"Ms": 9}



for k,v in title_mapping.items():

    titles_testDF[titles_testDF == k] = v



print(pd.value_counts(titles_testDF))



dfHarTst["Title"] = titles_testDF   # Add a Feature called - "Title"

dfHarTst.tail(5)

from sklearn.ensemble import RandomForestClassifier

#from sklearn import cross_validation

from sklearn.model_selection import cross_val_score    # in place of "cross_validation" try "model_selection" 

# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html





#predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"] #> 0.835056689342

#predictors = ["Pclass", "Sex", "Age", "Fare","SibSp"] #> 0.0.835034013605

#predictors = ["Pclass", "Sex", "Title", "Fare"] #> 83.61% # As suggested by - SelectKBest - Score Dropped ?? 

#predictors = ["Pclass", "Sex", "Title", "Age","Fare","SibSp"] #> 83.96%

#predictors = ["Pclass", "Sex", "Age", "Fare"] #> 0.842857142857 = 84.29% 

#predictors = ["Pclass", "Sex", "Title", "Age","Fare","SibSp"] #> 83.85%  ---- 7 JAN 17 



predictors = ["Pclass", "Sex", "Title", "Age","Fare"] #> 84.75%  ---- 7 JAN 17 



alg = RandomForestClassifier(       # Various parameter values passed :- 

    random_state=124,               # 1 , 123 , 124 

    n_estimators=350,               # 350 #> 84.74% , 650 #>84.63%    

    max_features=2,                 # 2 Dont Change 2 is IDEAL , 

    min_samples_split=6,            # 4 , 6  # The Deafult Val = 2

    min_samples_leaf=2,             # 2 , if 4 #> 83.95%

    verbose=0                       # Make it 1 for print in console 

)



                              

#kFold = KFold(df_harmonized_train.shape[0], random_state=124, n_folds=10)

#print(kFold)

#print("_"*90)





scores = cross_val_score(

    alg,                           # ALGORITHM in Cross_val_score Documentation is called - ESTIMATOR

    dfHarTrn[predictors],          # X Array like INDEPENDENT VARIABLES 

    dfHarTrn["Survived"],            # y Array like TARGET or DEPENDENT VARIABLE

    cv=20                            # cv=Default = 3 Fold Cross Validation , we used -18 , 20 #>84.75% 

)                                    # kFold = for n_folds= 22 #>84.40% ,  



print("{0:.2f}%".format(scores.mean() * 100))


# FINAL SUBMISSION - 17th JAN 





from sklearn import cross_validation, metrics   #Additional scklearn functions

from sklearn.grid_search import GridSearchCV 



predictors_final = ["Pclass", "Sex", "Title", "Age","Fare"] #> 84.75%  ---- 7 JAN 17 



alg = RandomForestClassifier(       # Various parameter values passed :- 

    random_state=124,               # 1 , 123 , 124 

    n_estimators=350,               # 350 #> 84.74% , 650 #>84.63%    

    max_features=2,                 # 2 Dont Change 2 is IDEAL , 

    min_samples_split=6,            # 4 , 6  # The Deafult Val = 2

    min_samples_leaf=2,             # 2 , if 4 #> 83.95%

    verbose=0                       # Make it 1 for print in console 

)





#full_predictions = []

#for alg, predictors in algorithms:

alg.fit(dfHarTrn[predictors], dfHarTrn["Survived"])

predictions = alg.predict_proba(dfHarTst[predictors].astype(float))[:,1]

#full_predictions.append(predictions)



#predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4



predictions[predictions > 0.5] = 1

predictions[predictions <= 0.5] = 0

predictions = predictions.astype(int)



submission = pd.DataFrame({

        "PassengerId": dfHarTst["PassengerId"],

                               "Survived": predictions

    })

submission.to_csv('Submission_17JAN.csv',index=False)




