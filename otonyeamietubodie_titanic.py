import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly_express as px

import matplotlib.image as mpimg

from tabulate import tabulate

import missingno as msno 

from IPython.display import display_html

from PIL import Image

import gc

import cv2

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
train.head(5)
train.info()
test.info()
def missing_values(df):

    total_mv = df.isnull().sum().sort_values(ascending=False)

    percentage = round(df.isnull().sum().sort_values(ascending=False) / len(df) * 100, 2)

    return pd.concat([total_mv, percentage], axis=1, keys=['Total', 'Percentage'])
missing_values(train)
missing_values(test)
def percentage_value_count(df, feature):

    percent = pd.DataFrame(round(df.loc[:, feature].value_counts(dropna=False, normalize=True)*100, 2))

    

    total = pd.DataFrame(df.loc[:, feature].value_counts(dropna=False))

    total.columns = ['Total']

    percent.columns = ['Percent']

    return pd.concat([total, percent], axis=1)
percentage_value_count(train, 'Embarked')
train[train['Embarked'].isnull()]
fig, ax = plt.subplots(figsize=(16,12),ncols=2)

ax1 = sns.barplot(x="Embarked", y="Fare", hue="Pclass", data=train, ax = ax[0]);

ax2 = sns.barplot(x="Embarked", y="Fare", hue="Pclass", data=test, ax = ax[1]);

ax1.set_title("Training Set", fontsize = 18)

ax2.set_title('Test Set',  fontsize = 18)





fig.show()
test[test.Fare.isnull()]
test['Fare'] = test['Fare'].fillna(0)
print('Train age missing value' + str((train.Age.isnull().sum() / len(train))* 100)+str('%'))

print('Test age missing value' + str((test.Age.isnull().sum() / len(test))* 100) + str('%'))
colors = ["#0101DF", "#DF0101"]

f, ax = plt.subplots(figsize=(12, 5))

sns.countplot('Sex', data=train, palette=colors)
colors = ["#0101DF", "#DF0101"]

f, ax = plt.subplots(figsize=(12, 5))

sns.countplot('Survived', data=train, palette=colors)
pal = {'male':"white", 'female':"Pink"}

sns.set(style="darkgrid")

plt.subplots(figsize = (15,8))

ax = sns.barplot(x = "Sex", 

                 y = "Survived", 

                 data=train, 

                 palette = pal,

                 linewidth=5,

                 order = ['female','male'],

                 capsize = .05,



                )



plt.title("Survived/Non-Survived Passenger Gender Distribution", fontsize = 25,loc = 'center', pad = 40)

plt.ylabel("% of passenger survived", fontsize = 15, )

plt.xlabel("Sex",fontsize = 15);
from plotly.offline import init_notebook_mode,iplot
train['Pclass']
upper_class = train[train.Pclass== 3]

middle_class = train[train.Pclass== 2]

lower_class = train[train.Pclass== 1]
plt.subplots(figsize = (15,10))

sns.barplot(x = "Pclass", 

            y = "Survived", 

            data=train, 

            linewidth=6,

            capsize = .05,

            errcolor='blue',

            errwidth = 3

            



           )

plt.title("Passenger Class Distribution - Survived vs Non-Survived", fontsize = 25, pad=40)

plt.xlabel("Socio-Economic class", fontsize = 15);

plt.ylabel("% of Passenger Survived", fontsize = 15);

names = ['Upper', 'Middle', 'Lower']

#val = sorted(train.Pclass.unique())

val = [0,1,2] ## this is just a temporary trick to get the label right. 

plt.xticks(val, names);
# Kernel Density Plot

fig = plt.figure(figsize=(15,8),)

ax=sns.kdeplot(train.loc[(train['Survived'] == 0),'Fare'] , color='red',shade=True,label='not survived')

ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Fare'] , color='green',shade=True, label='survived')

plt.title('Fare Distribution Survived vs Non Survived', fontsize = 25, pad = 40)

plt.ylabel("Frequency of Passenger Survived", fontsize = 15, labelpad = 20)

plt.xlabel("Fare", fontsize = 15, labelpad = 20);
train[train.Fare > 280]
# Kernel Density Plot

fig = plt.figure(figsize=(15,8),)

ax=sns.kdeplot(train.loc[(train['Survived'] == 0),'Age'] , color='red',shade=True,label='not survived')

ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Age'] , color='green',shade=True, label='survived')

plt.title('Age Distribution Survived vs Non Survived', fontsize = 25, pad = 40)

plt.ylabel("Frequency of Passenger Survived", fontsize = 15, labelpad = 20)

plt.xlabel("Age", fontsize = 15, labelpad = 20);
pal = {'Q':"white", 'S':"Pink", 'C':"Green"}

sns.set(style="darkgrid")

plt.subplots(figsize = (15,8))

ax = sns.barplot(x = "Embarked", 

                 y = "Survived", 

                 data=train, 

                 palette = pal,

                 linewidth=5,

                 order = ['Q','S', 'C'],

                 capsize = .05,



                )



plt.ylabel("% of passenger survived", fontsize = 15, )

plt.xlabel("Embarked",fontsize = 15);
pal = {1:"Green", 0:"Red"}

g = sns.FacetGrid(train,size=5, col="Sex", row="Embarked", margin_titles=True, hue = "Survived",

                  palette = pal

                  )

g = g.map(plt.hist, "Age", edgecolor = 'white').add_legend();

g.fig.suptitle("Survived by Sex and Age", size = 25)

plt.subplots_adjust(top=0.90)
g = sns.FacetGrid(train, size=5,hue="Survived", col ="Sex", margin_titles=True,

                palette=pal,)

g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend()

g.fig.suptitle("Survived by Sex, Fare and Age", size = 25)

plt.subplots_adjust(top=0.85)
train.describe(include='all')
#Below is a heatmap of the correlation of the normal data:

correlation_matrix = train.corr()

fig = plt.figure(figsize=(20,8))

sns.heatmap(correlation_matrix, vmax=0.8, square=True)
correlation_matrix['Survived'].sort_values(ascending=False)
pd.DataFrame(abs(train.corr()['Survived']).sort_values(ascending = False))
train['Sex'] = train['Sex'].map({'male':1, 'female':0})

test['Sex'] = test['Sex'].map({'male':1, 'female':0})

male_mean = train[train['Sex'] == 1].Survived.mean()

female_mean = train[train['Sex'] == 0].Survived.mean()



print('Male survival mean' + str(male_mean))

print('Female survival mean' + str(female_mean))



print ("The mean difference between male and female survival rate: " + str(female_mean - male_mean))
import random

male = train[train['Sex'] == 1]

female = train[train['Sex'] == 0]



## empty list for storing mean sample

m_mean_samples = []

f_mean_samples = []



for i in range(50):

    m_mean_samples.append(np.mean(random.sample(list(male['Survived']),50,)))

    f_mean_samples.append(np.mean(random.sample(list(female['Survived']),50,)))

    



# Print them out

print (f"Male mean sample mean: {round(np.mean(m_mean_samples),2)}")

print (f"Male mean sample mean: {round(np.mean(f_mean_samples),2)}")

print (f"Difference between male and female mean sample mean: {round(np.mean(f_mean_samples) - np.mean(m_mean_samples),2)}")
train = train.drop(['PassengerId', 'Cabin'], axis=1)
test = test.drop(['PassengerId', 'Cabin'], axis=1)
train = train.drop(['Name', 'Ticket'], axis=1)

test = test.drop(['Name', 'Ticket'], axis=1)
train['family_size'] = train.SibSp + train.Parch+1

test['family_size'] = train.SibSp + train.Parch+1
def family_group(size):

    a = ''

    if (size <= 1):

        a = 'loner'

    elif (size <= 4):

        a = 'small'

    else:

        a = 'large'

    return a
train['family_group'] = train['family_size'].map(family_group)

test['family_group'] = test['family_size'].map(family_group)
train['is_alone'] = [1 if i<2 else 0 for i in train.family_size]

test['is_alone'] = [1 if i<2 else 0 for i in test.family_size]
## Calculating fare based on family size.

train['calculated_fare'] = train.Fare / train.family_size

test['calculated_fare'] = test.Fare / test.family_size
def fare_group(fare):

    a = ''

    if fare <= 4:

        a = 'Very Low'

        

    elif fare <= 10:

        a = 'Low'

        

    elif fare <= 20:

        a = 'mid'

        

    elif fare <= 45:

        a = 'high'

        

    else:

        a = 'very high'

    return a



train['fare_group'] = train['calculated_fare'].map(fare_group)

test['fare_group'] = test['calculated_fare'].map(fare_group)
train.drop(['family_size','Fare'], axis=1, inplace=True)

test.drop(['family_size',"Fare"], axis=1, inplace=True)

train = pd.get_dummies(train, columns=["Pclass",'Embarked', 'family_group', 'fare_group'], drop_first=False)

test = pd.get_dummies(test, columns=["Pclass",'Embarked', 'family_group', 'fare_group'], drop_first=False)
train.head(5)
train = pd.concat([train[['Survived', 'Age', 'Sex', 'SibSp', 'Parch']], train.loc[:, 'is_alone':]], axis=1)
test = pd.concat([test[['Age', 'Sex']], test.loc[:, 'SibSp':]], axis=1)
from sklearn.ensemble import RandomForestRegressor



def completing_age(df):

    ## gettting all the features except Survived

    age_df = df.loc[:, 'Age':]

    

    temp_train = age_df.loc[age_df.Age.notnull()]## df with age values

    temp_test = age_df.loc[age_df.Age.isnull()]

    

    y = temp_train.Age.values

    x = temp_train.loc[:, 'Sex':].values

    

    rfr = RandomForestRegressor(n_estimators=1500, n_jobs=-1)

    rfr.fit(x, y)

    predicted_age = rfr.predict(temp_test.loc[:, 'Sex':])

    

    df.loc[df.Age.isnull(), "Age"] = predicted_age

    return df



## Implementing the completing_age function in both train and test dataset. 

completing_age(train)

completing_age(test);
## Let's look at the his

plt.subplots(figsize = (22,10),)

sns.distplot(train.Age, bins = 100, kde = True, rug = False, norm_hist=False);
##we can create a new column by grouping the age group
def age_group_fun(age):

    a = ''

    if age <= 1:

        a = 'infant'

    elif age <= 4:

        a = 'toddler'

    elif age <= 13:

        a = 'teenager'

    elif age <= 18:

        a = 'teenager'

    elif age <= 35:

        a = 'young_adult'

    elif age <= 45:

        a = 'adult'

    elif age <= 55:

        a = 'middle_aged'

    elif age <= 65:

        a = 'senior_citizen'

    else:

        a = 'old'

    return a



train['age_group'] = train['Age'].map(age_group_fun)

test['age_group'] = test['Age'].map(age_group_fun)
train = pd.get_dummies(train, columns=['age_group'], drop_first=True)

test = pd.get_dummies(test, columns=['age_group'], drop_first=True);
# It is important to separate dependent and independent variables.

#Our dependent variable or target variable is something that we are trying to find,

#and our independent variable is the features we use to find the dependent variable.
y = train['Survived']

X = train.drop(['Survived'], axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=0)
len(X_train)
len(X_test)
train.sample(5)
#Here Age and Calculated_fare is much higher in magnitude compared to others machine learning features.

#This can create problems as many machine learning models will get confused thinking Age and Calculated_fare

#have more weight than other features. Therefore, we need to do feature scaling to get a better result.

#There are multiple ways to do feature scaling.
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()

X_train = std_scaler.fit_transform(X_train)

X_test = std_scaler.transform(X_test)
# import LogisticRegression model in python. 

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_absolute_error, accuracy_score
reg = LogisticRegression(solver='liblinear', penalty='l1', random_state=42)

reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
y_pred
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_pred, y_test)

cm
#Accuracy is the measure of how often the model is correct.
accuracy_score(y_test, y_pred)
from sklearn.metrics import classification_report, balanced_accuracy_score

print(classification_report(y_test, y_pred))
y_pred_test = reg.predict(test)
y_pred_test
submission = pd.DataFrame({

        "PassengerId": submission.PassengerId,

        "Survived": y_pred_test

    })



submission.PassengerId = submission.PassengerId.astype(int)

submission.Survived = submission.Survived.astype(int)



submission.to_csv("titanic1_submission.csv", index=False)