# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Read the data

df = pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv")

df.head()
df.tail()
#Show the shape of dataset

df.shape
#Check for missing values

df.isnull().sum()
#Remove columns

df.drop(["Province/State"], inplace = True, axis=1)

df.head()
#Remove columns

df.drop(["Lat"], inplace = True, axis=1)

df.head()
#Remove columns

df.drop(["Long"], inplace = True, axis=1)

df.head()
#Describe of dataset

df.describe()
df.hist(figsize = (15,10))
#Import libraries for visualization features

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")
#Correlation between features

gr = df.corr()

sns.heatmap(gr, annot = True)
#Check for values in each country

df["Country/Region"].value_counts()
#Check for values of confirmed cases

df["ConfirmedCases"].value_counts()
#Check for values of fatalities

df["Fatalities"].value_counts()
#Convert object "Date" to datetime

df["Date"] = pd.to_datetime(df["Date"])

df.head()
#Visualization fatalities by date

df["Fatalities"] = df["Fatalities"].astype("int64")

fatalities_by_date = df.groupby("Date")["Fatalities"].mean()

sns.barplot(fatalities_by_date.index, fatalities_by_date.get_values())

plt.title("Fatalities by date")

plt.xlabel("Date")

plt.ylabel("Fatalities")
#Visualization fatalities by confirmed cases

df["Fatalities"] = df["Fatalities"].astype("int64")

fatalities_by_cases = df.groupby("ConfirmedCases")["Fatalities"].mean()

sns.barplot(fatalities_by_cases.index, fatalities_by_cases.get_values())

plt.title("Fatalities by confirmed cases")

plt.xlabel("Confirmed cases")

plt.ylabel("Fatalities")
#Remove "Id" column

df.drop(["Id"], inplace = True, axis=1)

df.head()
#Import LabelEncoder and apply label encoder

from sklearn.preprocessing import LabelEncoder

country = ["Country/Region"]

encoder = LabelEncoder()

encoded = df[country].apply(encoder.fit_transform)

encoded.head(20)
#Join columns

data = df[["ConfirmedCases", "Fatalities"]].join(encoded)

data.head()
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

import random as rd

from sklearn.preprocessing import StandardScaler
#Define X and y

features = ["ConfirmedCases","Country/Region","Fatalities"]

scaler = StandardScaler()

X = scaler.fit_transform(data[features].values)

y = data["Fatalities"].values
#Split dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state =1)
#Import necessary libraries

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

#Define and fit model

lda = LinearDiscriminantAnalysis()

lda.fit(X,y)

predicted = lda.predict(X_test)

accuracy_score(y_test, predicted)
#Calculate mean absolute error

test_predictions = lda.predict(X_test)

test_mae = mean_absolute_error (test_predictions, y_test)

print(test_mae)
sub = pd.read_csv("../input/covid19-global-forecasting-week-1/submission.csv")

sub.to_csv("covsubmission.csv", index = False)