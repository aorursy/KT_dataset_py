# General Essential Libraries for data analysis:
import numpy as np 
import pandas as pd 

import seaborn as sns 
sns.set(style = "whitegrid")
import matplotlib.pyplot as plt 

import warnings
warnings.filterwarnings("ignore")

# Libraries for data visualisation: 
import plotly.plotly as py
import plotly.tools as tls
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1)
%matplotlib inline
import seaborn as sns
# Libraries for Machine Learning Algroithyms:
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder 

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# This creates a pandas dataframe and assigns it to the titanic variable.
titanic = pd.read_csv("../input/train.csv")
titanic_test = pd.read_csv("../input/test.csv")

# Print the first 5 rows of the dataframe.
titanic.head()
#shape command will give number of rows/samples and number of columns/features/predictors in dataset
#(rows,columns)
#titanic.shape
print ("The shape of the train data is (row, column):"+ str(titanic.shape))
#Describe gives statistical information about numerical columns in the dataset
titanic.describe()
#Describe gives statistical information about obj/Categorial columns in the dataset
titanic.describe(include =['O','category'])
#info method provides information about dataset like 
#total values in each column, null/not null, datatype, memory occupied etc
titanic.info()
print("-"*70)
titanic_test.info()
# Missing values in Train Datasets

total = titanic.isnull().sum().sort_values(ascending = False)
percent = round(titanic.isnull().sum().sort_values(ascending = False)/len(titanic)*100, 2)
pd.concat([total, percent], axis = 1,keys= ['Total', 'Percent'])
#Lets find distinct percentage of Embarked Category in Traing Dataset.
percent = pd.DataFrame(round(titanic.Embarked.value_counts(dropna=False, normalize=True)*100,2))
total = pd.DataFrame(titanic.Embarked.value_counts(dropna=False))
total.columns = ["Total"]
percent.columns = ['Percent']
pd.concat([total, percent], axis = 1)
titanic[titanic.Embarked.isnull()]
fig, ax = plt.subplots(figsize=(16,12),ncols=3)
ax1 = sns.boxplot(x="Embarked", y="Fare", hue="Survived", data=titanic, ax = ax[0] , palette="colorblind");
ax2 = sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=titanic, ax = ax[1] , palette="colorblind");
ax3 = sns.boxplot(x="Embarked", y="Fare", hue="Sex", data=titanic, ax = ax[2] , palette="colorblind");

ax1.set_title("Embarked~Fare~Survived", fontsize = 18)
ax2.set_title("Embarked~Fare~Pclass",  fontsize = 18)
ax3.set_title("Embarked~Fare~Sex",  fontsize = 18)
fig.show()
# Fill the NAN with C for Embarked column
titanic["Embarked"] = titanic["Embarked"].fillna('C')
[i[0] for i in titanic.Cabin]
titanic[titanic.Age.isnull()].head(5)
def MissingAge_RandomForest(df):
    #Feature set
    age_df = df[['Age','Embarked','Fare', 'Parch', 'SibSp','Ticket', 'Pclass','Cabin']]
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
with sns.plotting_context("notebook",font_scale=1.5):
    sns.set_style("whitegrid")
    sns.distplot(titanic["Age"].dropna(),
                 bins=80,
                 kde=False,
                 color="blue")
    plt.title("Age Distribution")
    plt.ylabel("Count");
MissingAge_RandomForest(titanic)
#age = titanic.loc[:,"Age":] 
#temp = age.loc[age.Age.notnull()] ## df with age values
#temp_test = age_df.loc[age_df.Age.isnull()] ## df without age values
#age
#y = temp_train.Age.values ## setting target variables(age) in y 
#x = temp_train.loc[:, "Sex":].values