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
# I am still a very beginner of Data Science field (just started few months ago)

# So, I will try my best to tackle this dataset.

# Feel free to comment anything if I have done anything wrong or any improvement needed.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



weather_data = pd.read_csv("/kaggle/input/mount-rainier-weather-and-climbing-data/Rainier_Weather.csv")

climb_data = pd.read_csv("/kaggle/input/mount-rainier-weather-and-climbing-data/climbing_statistics.csv")
weather_data.isnull().sum()

# no missing values
climb_data.isnull().sum()

# no missing values
weather_data.describe()
climb_data.describe()

# seems like there is an abnormal value (14.2) on column "Success Percentage"

# "Success Percentage" shouldn't have value more than 1
# we can combine those two datasets 

df = weather_data.merge(climb_data, on="Date")



df.info()

# we can change that "Date" from object format into datetime format



df.drop(["Attempted","Succeeded"], axis=1, inplace=True)
df["Date"] = pd.to_datetime(df["Date"])



df.info()

# "Date" has changed from object format into datetime format



# we can crete column "Month" from "Date"

df["Month"] = df["Date"].dt.month
############################### Exploratory Data Analysis (EDA) #################################

plt.figure(figsize=(16,8))

sns.countplot(x=df["Route"])

plt.xticks(rotation=90)

plt.show()

# "Disapointment Cleaver" is the highest route taken
plt.figure(figsize=(16,8))

sns.lineplot(x=df["Date"], y=df["Success Percentage"])

plt.show()

# the graph shows that there are high number of "Success Percentage" around June to August 2015
plt.figure(figsize=(16,8))

sns.lineplot(x=df["Month"], y=df["Success Percentage"])

plt.show()

# the graph show that month for the highest number of "Success Percentage" is June
plt.figure(figsize=(16,8))

sns.lineplot(x=df["Month"], y=df["Temperature AVG"], label="Temperature")

sns.lineplot(x=df["Month"], y=df["Relative Humidity AVG"], label="Relative Humidity")

sns.lineplot(x=df["Month"], y=df["Wind Speed Daily AVG"], label="Wind Speed")

plt.show()

# Judging from the graph, "Relative Humidity AVG" and "Wind Speed Daily AVG" starts to decrease while 

# "Temperature AVG" starts to increase when near month of "June".

# Perhaps, lower "Relative Humidity AVG" and "Wind Speed Daily AVG", and higher "Temperature AVG" will increase the chance of success.
#################################### Feature Engineer our data #############################

# check for outliers



sns.boxplot(df["Success Percentage"])

# there is an extreme outlier at around 14
sns.boxplot(df["Battery Voltage AVG"])

# there are outliers at lower and upper ends.
sns.boxplot(df["Temperature AVG"])

# there are outliers at lower end
sns.boxplot(df["Relative Humidity AVG"])

# no outlier
sns.boxplot(df["Wind Speed Daily AVG"])

# there are outliers at upper end
sns.boxplot(df["Wind Direction AVG"])

# no outlier
sns.boxplot(df["Solare Radiation AVG"])

# there are outliers at lower end
# we can drop the rows containing "Success Percentage" higher than 1

index = df[df["Success Percentage"] > 1].index

df.drop(index, inplace=True)



# we can change the value of "Wind Speed Daily AVG" being 0 into 0.753458 (the second lowest value after 0).

# it is hard to imagine where there will be totally no wind.

df["Wind Speed Daily AVG"] = np.where(df["Wind Speed Daily AVG"] == 0, 0.753458, df["Wind Speed Daily AVG"])

df["Wind Speed Daily AVG"] = np.where(df["Wind Speed Daily AVG"] >= 30.02925, 30.02925, df["Wind Speed Daily AVG"])



# we can change the value of "Solare Radiation AVG" being 0 into 0.0330833 (the second lowest value after 0).

# it is hard to imagine where there will be totally no radiation.

df["Solare Radiation AVG"] = np.where(df["Solare Radiation AVG"] == 0, 0.0330833, df["Solare Radiation AVG"])

df["Solare Radiation AVG"] = np.where(df["Solare Radiation AVG"] <= 22.502249, 22.502249, df["Solare Radiation AVG"])



# we can apply "top-coding" and "bottom-coding" on those outliers.

# we can limit cap those outliers with interquantile proximity rule. 

df["Battery Voltage AVG"] = np.where(df["Battery Voltage AVG"] >= 13.685626, 13.685626, df["Battery Voltage AVG"])

df["Battery Voltage AVG"] = np.where(df["Battery Voltage AVG"] <= 13.313955, 13.313955, df["Battery Voltage AVG"])



# we can limit cap those outliers as well.

df["Temperature AVG"] = np.where(df["Temperature AVG"] <= 10.4985425, 10.4985425, df["Temperature AVG"])

df["Route"].nunique()

# it has 22 unique values which are quite a lot.

# it is a high cardinality variable.
df["Route"].value_counts()
# I decided to keep only top 9 highest counts of unique values.

# As for the rest, I will group all of them under a new value "Others".



Route = pd.Series(df["Route"].value_counts())

Route_to_keep = Route.index[0:9]

Route_to_remove = [i for i in df["Route"].unique() if i not in Route_to_keep]



for i in Route_to_remove:

    df["Route"] = np.where(df["Route"] == i, "Others", df["Route"])



df["Route"].value_counts()
# apply one hot encoding on "Route" variable

dummies = pd.get_dummies(df["Route"], drop_first=True)

df = pd.concat([dummies, df], axis=1)

df.drop("Route", axis=1, inplace=True)
df.drop(["Date", "Month"], axis=1, inplace=True)

# I am not going to put these two variables into my model building



# check for correlation

corrmat = df.corr()

plt.figure(figsize=(16,8))

sns.heatmap(df.corr(), annot=True)
# I am going to check what is the correlation between "Success Percentage" and other variables

corrmat = df.corr()

corrmat["Success Percentage"].abs().sort_values(ascending=False)



# from the result, we can observe that "Success Percentage" does not have strong correlation with any other variables.

# I do not think our linear regression model is going to work very well on this data.

# Anyway, we still go ahead and try it out.
# normalize the data



def norm_func(i):

    return ((i-i.min()) / (i.max()-i.min()))



df = df.apply(norm_func)



df.describe()

# all variables are being kept in range between 0 and 1
# split the data into predictors and output

X = df.drop("Success Percentage", axis=1)

Y = df["Success Percentage"]
# split the data into train and test datasets.

from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)
####################################### Model Building #####################################

from statsmodels.regression.linear_model import OLS



model = OLS(Y_train, X_train).fit()

model.summary()

# seems like there are many variables with p_value more than 0.05

# we can drop those variables with p-value more than 0.05
X_train = X_train.drop(["Emmons-Winthrop", "Gibralter Ledges", "Ingraham Direct", "Kautz Glacier", "Liberty RIngraham Directge", "Little Tahoma", "Others", "Wind Direction AVG", "Relative Humidity AVG", "Wind Speed Daily AVG"], axis=1)

X_test = X_test.drop(["Emmons-Winthrop", "Gibralter Ledges", "Ingraham Direct", "Kautz Glacier", "Liberty RIngraham Directge", "Little Tahoma", "Others", "Wind Direction AVG", "Relative Humidity AVG", "Wind Speed Daily AVG"], axis=1)
# rebuild the same model with new training data



model = OLS(Y_train, X_train).fit()

model.summary()

# seems like "Temperature AVG" now has p-value more than 0.05

# so we can drop that variable as well.
X_train = X_train.drop("Temperature AVG", axis=1)

X_test = X_test.drop("Temperature AVG", axis=1)
model = OLS(Y_train, X_train).fit()

model.summary()



# we can observe that adjusted R-squared is only 0.527.

# perhaps due to our dependent variable ("Success Percentage") does not has any strong correlation with any predictors.
# evaluate the model with mean-squared-error

from sklearn.metrics import mean_squared_error



pred = model.predict(X_test)

mean_squared_error(Y_test, pred)
# As a conclusion, I am not going to say that this model is very good since mean_squared error is quite large.

# Maybe we still lack of other significant information such as how many mountains have those climbers climbed before (veteran or rookie), years of experience being climbers, etc