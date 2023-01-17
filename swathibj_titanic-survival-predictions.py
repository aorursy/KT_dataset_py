# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
##Importing all packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



#preprocessing

from sklearn import preprocessing



##Modelling



from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn import tree
##Load both Train and test dataset

df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_test = pd.read_csv("/kaggle/input/titanic/test.csv")

df_train.shape , df_test.shape
##Check first 10 rows

df_train.head(10)
fig,((ax1,ax2, ax3),(ax4, ax5,ax6)) = plt.subplots(nrows=2,

                                           ncols=3, 

                                           figsize=(12,8))



sns.barplot(x='Pclass',y='Survived',data=df_train, ax=ax1)

sns.barplot(x='Sex',y='Survived',data=df_train, ax=ax2)

sns.barplot(x='SibSp',y='Survived',data=df_train, ax=ax3)

sns.barplot(x='Parch',y='Survived',data=df_train, ax=ax4)

sns.barplot(x='Embarked',y='Survived',data=df_train, ax=ax5);
df_train.info() ## Quick check of data type, number of columns and row
df_train.describe()
## Let's check the missing values in all columns

df_train.isna().sum() ## Can see that Age, Cabin and Embarked has missing values
##Filling missing values

def fill_missing_values(df):

    ## 1.Fill Age column missing value

    ## Let's fill Age missing columns with Median value

    df["Age"].fillna(df["Age"].median() , inplace=True)

    ## Filling Cabin value with Unknown as its unknown

    df["Cabin"].fillna("Unknown", inplace=True)

    ##Adding new column to capture Cabin unique values

    df["Cabin_Unique"]= df["Cabin"].str[0]

    ##3. Will fill 2 missing records with most commanly available value i.e with S

    df["Embarked"].fillna("S",inplace=True)

    

    return df
fill_missing_values(df_train)
##Lets check again to see if still any missing values exists

df_train.isna().sum() ## No more missing values
##Let's do few extraction/modification on Columns



def feature_eng(df):

    ##Extracting only the Title from Name column

    df["Title"] = df.Name.str.extract('([A-Za-z]+)\. ',expand=False).str.strip()

    ##Replacing most of the unique title except Miss, MRs and Mr to Others

    df["Title"].replace([ 'Master', 'Don', 'Rev', 'Dr', 'Mme', 'Ms',

       'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'Countess',

       'Jonkheer'], "Others", inplace=True)

    

    ## Since we dont require Names and it is unique will drop this column from dataset

    df.drop("Name", axis=1, inplace=True)

    ##Will drop Ticket as they dont contribute much to survival rate/model

    df.drop("Ticket", axis=1, inplace=True)

    ##Will drop Cabin column has this column value has been already retrived in Cabin_Num 

    df.drop("Cabin", axis=1 , inplace=True)

    

    return df
feature_eng(df_train)
df_train.columns[3][:]
## Since model does not take String/object datatype values, we need to conver these to numbers.

## Will do this using label encoding



def label_encode(df):

    label_encoder=preprocessing.LabelEncoder()

    df["Cabin_Unique"]=label_encoder.fit_transform(df["Cabin_Unique"])

    df["Title"]=label_encoder.fit_transform(df["Title"])

    df["Sex"]=label_encoder.fit_transform(df["Sex"])

    df["Embarked"]=label_encoder.fit_transform(df["Embarked"])

    return df
label_encode(df_train)
## Lets check corr of each column to target variable



plt.figure(figsize=(10,7))

heatmap = sns.heatmap(df_train.corr(),

                      annot=True,

                     fmt=".2f",

                     cbar=False,

                     cmap="YlGnBu");
#fig , ax = plt.subplots()

#ax.hist(df_train["Age"] , df_train["Survived"]);

df_train["decade"]= df_train['Age'].apply(lambda x: int(x/10))

df_train[["decade" , "Survived"]].groupby("decade").mean().plot();

# Youngsters have more chnace of survival and between 70-80 have high chance
df_train[["Cabin_Unique","Survived" ]].groupby("Cabin_Unique").mean().plot();

## We can see that Higher cabin number means lower its ditribution and have less chance of survival.

## Since we have filled missing Cabin has N which is in last cabin have very less survival rate
##Lets drop Decade which we used for Anapysis as this is not helpful for Predictions

df_train.drop("decade", axis=1, inplace=True)
## Now will proprocess test data using functions created for train dataset

df_test.head(20)
##Data cleaning to test data set

fill_missing_values(df_test)

feature_eng(df_test)

label_encode(df_test)
##We have one record with Fare as 'NULL' values and will fill this with Mean value

df_test["Fare"].fillna(df_test["Fare"].mean() , inplace=True)

df_test.isna().sum()
## Splitting the data to X_train and Y_train

X_train = df_train.drop("Survived", axis=1)

Y_train = df_train["Survived"]
models={"RandomForest": RandomForestClassifier(),

       "GradientBoost": GradientBoostingClassifier(),

       "Logistic": LogisticRegression(),

       "DecisionTree":tree.DecisionTreeClassifier()}



def model_scores(models, X_train, Y_train):

    ##set random seed

    np.random.seed(45)

    model_scores={}

    

    for name,model in models.items():

        #Fit the model

        model.fit(X_train,Y_train)

       

        

        #Evaluate the model

        model_scores[name]=model.score(X_train,Y_train)

      

        

    return model_scores  
Model_scores = model_scores(models ,X_train, Y_train)

Model_scores
Accuracy_Table = pd.DataFrame(Model_scores.values(), index=Model_scores.keys(),columns=["Accuracy"])

Accuracy_Table.sort_values(by="Accuracy", ascending=False)
##Will predict score for test data using RandomForestClassifier



rf_model=RandomForestClassifier()

rf_model.fit(X_train,Y_train)



Pred_score_Random=rf_model.predict(df_test)
## Will save the predicted output in Gender_Submission csv file



dict_pred={"PassengerId":df_test.PassengerId,

           "Predicted_Score":Pred_score_Random        

          }



Gender_Submission = pd.DataFrame(dict_pred)
Gender_Submission.to_csv("Gender_Submission_Predicted_Outputs.csv", index=False)