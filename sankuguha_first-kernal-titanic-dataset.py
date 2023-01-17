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
#importing pandas,numpy,matplotlib etc



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
#importing data set

train_df = pd.read_csv(r"../input/titanic/train.csv")

test_df = pd.read_csv(r"../input/titanic/test.csv")
#shape of train data set

train_df.shape
#shape of test data set

test_df.shape
#we can see there some null values for Age,Cabin,Embarked in training data set

train_df.info()
#we can see there some null values for Age,Fare,Cabin in test data set

test_df.info()
#count of null values in train_df

pd.isnull(train_df).sum()
#count of null values in train_df

pd.isnull(test_df).sum()
#selecting rows which has any of the columns as null values from train data set

train_df[train_df.isnull().any(axis ="columns")]
#selecting sample 5 rows from train_df

train_df.sample(n=5)
train_df["Survived"].value_counts()
#Percentage & Values together of Survived/Not Survived based on train data frame using matplotlib using style as "ggplot"

survivalrates = train_df["Survived"].value_counts().tolist()

plt.figure(figsize=(6,6))

sns.set()

plt.pie(survivalrates,

        labels = ['Not Survived','Srvived'],

        autopct=lambda p : '{:.2f}%  ({:,.0f})'.format(p,p * sum(survivalrates)/100))
#Survival Distribution by Pclass using different ploting technique

train_df_grouped = train_df.groupby(by=["Pclass","Survived"])

train_df_grouped_by_size = train_df_grouped.size()

train_df_grouped_by_size
#Now lets visualize the above data set using matplotlib pie chart

sns.set()

train_df_grouped_by_size.plot(kind="pie",

                              autopct='%.2f',

                              figsize=(6,6),

                              title="Survival Distribution by Pclass")
#installing pandas bokeh. Its supported on python 3.5 and above

!pip install pandas-bokeh
#importing pandas bokeh

import pandas_bokeh

pandas_bokeh.output_notebook()
#Plotting the same data using plot_bokeh

train_df_grouped_by_size.plot_bokeh(kind="pie",

                                    title="Survival Distribution by Pclass")
#You can see survival rate in "Pclass = 3" is very low, So "Pclass" is important feature to identify the survival rate

#Survival rate is highest in Pclass = 1 i.e Upper Class 

pd.crosstab(train_df["Pclass"],train_df["Survived"]).plot.bar(figsize=(12,12),

                                                              title="Survival Rate by Pclass",

                                                              color='gy')
#Survival rate based on "SibSp".

pd.crosstab(train_df["SibSp"],train_df["Survived"]).plot.bar(figsize=(12,12),

                                                             title="Survival Rate by SibSp",

                                                             color = ["gold","firebrick"])
#Survival rate based on "Parch".We can see survival rate is lower in case of Parch = 0

pd.crosstab(train_df["Parch"],train_df["Survived"]).plot.bar(figsize=(12,12),

                                                             title="Survival Rate by Parch",

                                                             color='mc')
train_df["Title"] = train_df["Name"].str.split(",").str.get(1).str.split(".").str.get(0).str.strip()

train_df["Title"] = train_df["Title"].replace(["Dr","Rev","Col","Mlle","Major","Ms","the Countess","Sir","Jonkheer","Capt","Lady","Mme","Don"],

                         "Other")

test_df["Title"] = test_df["Name"].str.split(",").str.get(1).str.split(".").str.get(0).str.strip()

test_df["Title"] = test_df["Title"].replace(["Dr","Rev","Col","Mlle","Major","Ms","the Countess","Sir","Jonkheer","Capt","Lady","Mme","Don","Dona"],

                         "Other")
#Survival rate based on Title column i.e we created in previous step

pd.crosstab(train_df["Title"],train_df["Survived"]).plot.bar(figsize=(12,12),

                                                             title="Survival Rate by Title",

                                                             color = 'gb')
#You can see survival rate is higher in case of Southampton port

pd.crosstab(train_df["Embarked"],train_df["Survived"]).plot.bar(figsize=(12,12),

                                                                title="Survival Rate by Embarked",

                                                                color=["green","yellowgreen"])
#Lets check the data where Embarked is null from training dataset.We can see survived=1 in both the rows,so we will update Embarked as 'S' since survival rate is higher

train_df[train_df["Embarked"].isnull()]
#Filing Null Values for Embarked in train_df data set since survival rate is higher in case Embarked = 'S'

train_df["Embarked"].fillna("S",inplace = True)

train_df["Embarked"].unique()
#Now Lets fill the Cabin value as first letter(str[0]) from Cabin value and fill NULL values with "U" for Cabin feature

train_df["Cabin"] = train_df["Cabin"].str[0]

test_df["Cabin"] = test_df["Cabin"].str[0]

train_df["Cabin"].fillna("U",inplace=True)

test_df["Cabin"].fillna("U",inplace=True)
#Lets visualize the survival rate by Cabin fetaure from tarin data set

pd.crosstab(train_df["Cabin"],train_df["Survived"]).plot.bar(figsize=(12,12),

                                                             title="Survival Rate by Cabin")
#Mean age by each class

train_df.groupby(["Pclass"]).mean()["Age"].plot.bar(figsize=(6,6),title="Mean Age by Pclass",color=["lightblue"])
#age distribution by Survival rate

sns.kdeplot(train_df[train_df["Survived"] == 1]["Age"],label='Survived',shade = True);

sns.kdeplot(train_df[train_df["Survived"] == 0]["Age"],label='Not Survived',shade = True);

plt.show()
#Lets see mean of 'Fare' by Survived & Sex

train_df.groupby(["Survived","Sex"]).mean()["Fare"].plot.bar(figsize=(6,6),

                                                       title="Mean Fare by Survived & Sex",color = ["seagreen"])
#You can see sum of fare is higher PClass = 1

plt.figure(figsize=(12,12))

sns.barplot(x='Pclass',y='Fare',data=train_df,estimator=sum,palette="husl",capsize=.2)
#Fare distribution by Pclass & Embarked

plt.figure(figsize=(12,12))

sns.violinplot(x='Pclass',

               y='Fare',

               data=train_df,

               hue='Embarked')
#Fare distribution by Pclass & Sex

plt.figure(figsize=(12,12))

sns.boxplot(x='Pclass',

            y='Fare',

            data=train_df,

            hue='Sex')
train_df.plot_bokeh.scatter(x="Fare",

                            y="Pclass",

                            category="Sex",

                            title="Fare Distribution by Sex using Scatter Plot")
#Lets create a grid plot using scatter plot and plot-bokeh table from Dataframe

#Create Bokeh-Table with DataFrame

from bokeh.models.widgets import DataTable, TableColumn

from bokeh.models import ColumnDataSource



train_df_sample_table = DataTable(

    columns=[TableColumn(field=Ci, title=Ci) for Ci in train_df.columns],

    source=ColumnDataSource(train_df),

    height=300,

)



#create the scatter plot

train_df_scatter = train_df.plot_bokeh.scatter(x="Fare",

                                               y="Pclass",

                                               category="Sex",

                                               title="Titanic DataSet Visualization",

                                               show_figure=False)



# Combine Table and Scatterplot via grid layout:

pandas_bokeh.plot_grid([[train_df_sample_table, train_df_scatter]], plot_width=400, plot_height=350)





#Lets replace Sex fetaue as below for train and test data set

train_df["Sex"].replace({"male" : 1, "female" :0},inplace=True)

test_df["Sex"].replace({"male" : 1, "female" :0},inplace=True)
#Lets create family column by combining "SibSp" & "Parch" & assign family column as 4 as below

train_df["Family"] = train_df["SibSp"] + train_df["Parch"]

train_df.loc[train_df["Family"] > 3,"Family"] = 4

test_df["Family"] = test_df["SibSp"] + test_df["Parch"]

test_df.loc[test_df["Family"] > 3,"Family"] = 4
#Creating category columns for 'Embarked','Cabin','Title','Family','Pclass' for training data set

train_df_fam_Pclass = train_df.loc[:,['Embarked','Title','Family','Pclass']]

train_df_fam_Pclass_dummy = pd.get_dummies(train_df_fam_Pclass.astype('str'))

train_df_age_fare_sex = train_df.loc[:,['Sex','Age','Fare']]
#Creating train and test data set from training data set by splitting the input and output

x_train = pd.concat([train_df_age_fare_sex,train_df_fam_Pclass_dummy],axis="columns")

y_train = train_df.loc[:,"Survived"]
x_train.info()
#Lets update null values with mean value for male and female respectively

train_male_average_age = x_train[x_train["Sex"] == 1]["Age"].mean()

train_female_average_age = x_train[x_train["Sex"] == 0]["Age"].mean()

x_train.loc[x_train["Age"].isnull() & x_train["Sex"] == 1,"Age"] = train_male_average_age

x_train.loc[x_train["Age"].isnull(),"Age"] = train_female_average_age
#Now lets create co-relation matrix for train data set

x_train_corr = x_train.corr(method='pearson')

plt.figure(figsize = (15,15))

sns.heatmap(x_train_corr, xticklabels=x_train_corr.columns,yticklabels=x_train_corr.columns,cmap = "coolwarm_r",annot=True,annot_kws = {'size': 6})

plt.title("Correlation")

plt.show()
#lets see the skewness of the data

x_train.skew()
#Lets fill the missing values for test data set for Age column

test_male_average_age = test_df[test_df["Sex"] == 1]["Age"].mean()

test_female_average_age = test_df[test_df["Sex"] == 0]["Age"].mean()

test_df.loc[test_df["Age"].isnull() & test_df["Sex"] == 1,"Age"] = test_male_average_age

test_df.loc[test_df["Age"].isnull(),"Age"] = test_female_average_age
#Lets fill null value for Fare column by taking mean value of Fare for Pclass = 3

test_df[test_df["Fare"].isnull()]
#Lets take the mean fare for pclass=3 and apply for null value for fare

test_fare_pclass3_mean = test_df.loc[test_df["Pclass"] == 3,"Fare"].mean()

test_df.loc[test_df["Fare"].isnull(),"Fare"] = test_fare_pclass3_mean
#Lets create family column by combining "SibSp" & "Parch" for test data set

test_df["Family"] = test_df["SibSp"] + test_df["Parch"]

test_df.loc[test_df["Family"] > 3,"Family"] = 4

test_df_fam_Pclass = test_df.loc[:,['Embarked','Title','Family','Pclass']]

test_df_fam_Pclass_dummy = pd.get_dummies(test_df_fam_Pclass.astype('str'))

test_df_age_fare_sex = test_df.loc[:,['PassengerId','Sex','Age','Fare']]
x_test = pd.concat([test_df_age_fare_sex,test_df_fam_Pclass_dummy],axis="columns")
x_test.info()
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
# prepare models

models = []

models.append(('LR', LogisticRegression(solver='liblinear')))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC(gamma='auto')))

# load dataset into variables

X = x_train.values

Y = y_train.values

# evaluate each model in turn

results = []

names = []

scoring = 'accuracy'

for name, model in models:

    kfold = KFold(n_splits=7)

    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)

# boxplot algorithm comparison

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()    
#Lets put this data set into XGBClassifier

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

xgbc = XGBClassifier(max_depth = 4)

test_size = 0.33

seed = 123

X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size = test_size, random_state = seed)

xgbc.fit(X_train, Y_train)

Y_pred = xgbc.predict(X_test)



xgbc_train_acc = round(xgbc.score(X_train, Y_train) * 100, 2)

print('Training Accuracy: ', xgbc_train_acc)

xgbc_test_acc = round(xgbc.score(X_test, Y_test) * 100, 2)

print('Testing Accuracy: ', xgbc_test_acc)
x_test['Survived'] = xgbc.predict(x_test.drop(['PassengerId'], axis = 1))
x_test[['PassengerId', 'Survived']].to_csv('submission.csv', index = False)