import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv("../input/gender_submission.csv")

data_train = pd.read_csv("../input/train.csv")

data_test = pd.read_csv("../input/test.csv")
data.head()

data_train.head()

data_test.head()
data_train.count()
list(data_train.columns)
survived_column = data_train["Survived"]

survived_column.head()
data_train.groupby("Survived").count()
np.mean(survived_column)
survived_column.values[:5]
numerical_features = data_train[["Fare", "Pclass", "Age"]]

numerical_features.head()
numerical_features.count()
median_values = numerical_features.dropna().median()

median_values
inputed_features = numerical_features.fillna(median_values)

inputed_features.head()
inputed_features.count()
features_array = inputed_features.values

features_array
features_array.shape

features_array.dtype
from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(features_array,survived_column, test_size = .2, random_state = 0)

x_train.shape
x_test.shape
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(C = 1)

logreg.fit(x_train, y_train)
target_predicted = logreg.predict(x_test)
from sklearn.metrics import accuracy_score

accuracy_score(target_predicted, y_test)
logreg.score(x_test, y_test)
features_names = numerical_features.columns

features_names
features_names.values
logreg.coef_
numeric_features = ["Age", "Fare"]

ordinal_features = ["Pclass", "SibSp", "Parch"]

nominal_features = ["Sex", "Embarked"]
# Adding new column by changing the binary notation to actual phrases

data_train["target_name"] = data_train["Survived"].map({0: "Not Survived", 1: "Survived"})
# Target variable explorations

import seaborn as sns

import matplotlib.pyplot as plt

sns.countplot(data_train["target_name"])

plt.xlabel("Survived ?")

plt.ylabel("Number of occurences")

plt.show()
## Correlation matrix heatmap

cor_mat = data_train[numeric_features + ordinal_features].corr().round(2)

#plottin heatmap

fig = plt.figure(figsize = (12, 12))

sns.heatmap(cor_mat, annot = True, center = 0, cmap = sns.diverging_palette(250, 10, as_cmap = True),

           ax = plt.subplot(111))

plt.show()
for column in  numeric_features:

    #give figure details

    fig = plt.figure(figsize = (18, 12))

    

    #Histogram plot that shows the relation of column(age or fare) and their density in the whole passengers

    sns.distplot(data_train[column].dropna(), ax = plt.subplot(221))

    plt.xlabel(column, fontsize = 14)

    plt.ylabel("Density", fontsize = 14)

    plt.suptitle("Plots for " + column, fontsize = 18)

    

    #Distribution of the survived and non-survived per age

    sns.distplot(data_train.loc[data_train.Survived == 0, column].dropna(),

                color = "red", label  = "Not Survived", ax = plt.subplot(222))

    sns.distplot(data_train.loc[data_train.Survived == 1, column].dropna(),

                color = "blue", label = "Survived", ax = plt.subplot(222))

    #These are the axes and the legend

    plt.legend(loc = "best")

    plt.xlabel(column, fontsize = 14)

    plt.ylabel("Density by Survived and Non Survived Passengers")

    

    #This  barplot demostrates the average of the non-survived and survived passenegers

    sns.barplot(x = "target_name", y = column, data = data_train, ax = plt.subplot(223))

    plt.xlabel("Survived or not survived", fontsize= 14)

    plt.ylabel("Average" + column, fontsize = 14)

    

    #Boxplot of column per Survived / Not Survived Value

    sns.boxplot(x = "target_name", y = column, data = data_train, ax = plt.subplot(224))

    plt.xlabel("Survived or Not Survived ?", fontsize = 14)

    plt.ylabel(column, fontsize  = 14)

    plt.show()

    
#Plotting Categorical Features

#Looping before like we did above



for column in ordinal_features:

    fig = plt.figure(figsize= (18,12))

    

    sns.barplot(x = "target_name", y = column, data = data_train, ax = plt.subplot(321))

    plt.xlabel("Survived or not survived", fontsize = 14)

    plt.ylabel("Average " + column , fontsize = 14)

    

    plt.suptitle("Plots for " + column, fontsize = 14)

    

    #this block will execute a boxplot of survived - not survived values

    sns.boxplot(x = "target_name", y = column, data = data_train, ax = plt.subplot(322))

    plt.xlabel("Survived or not  survived", fontsize = 14)

    plt.ylabel("Average " + column, fontsize = 14)

    

    #this block will execute 

    ax = sns.countplot(x = column, hue = "target_name", data = data_train, ax = plt.subplot(312))

    plt.xlabel(column, fontsize = 14)

    plt.ylabel("Number of occurences", fontsize = 14)

    plt.legend(loc = "best")

    

    #Adding percents over bars

    #Getting heights of our bars

    height = [p.get_height() if p.get_height() == p.get_height() else 0 for p in ax.patches]

    #Counting number of bar groups 

    ncol = int(len(height)/2)

    #Counting total height of groups

    total = [height[i] + height[i + ncol] for i in range(ncol)] *2

    #Looping through bars

    for i, p in enumerate(ax.patches):

        #Adding percentages

        ax.text(p.get_x() + p.get_width()/2, height[i]*1.01 + 10,

               "{:1.0%}".format(height[i] / total[i]), ha = "center", size = 14)

        

    ##Survived percentage for every value of feature

    sns.pointplot(x = column, y = "Survived", data = data_train, ax = plt.subplot(313))

    plt.xlabel(column, fontsize = 14)

    plt.ylabel("Survived Percentage", fontsize = 14)

    

    plt.plot()
##PLotting categorical Features

# Looping through and plotting the categorical features



for column  in nominal_features:

    

    fig = plt.figure(figsize = (18, 12))

    

    #number of occurences per category - target pair

    ax = sns.countplot(x = column, hue = "target_name", data = data_train, ax = plt.subplot(211))

    plt.xlabel(column, fontsize = 14)

    plt.ylabel("Number of Occurences",  fontsize = 14)

    plt.legend(loc = "best")

    plt.suptitle("Plots for " + column, fontsize = 18)   

    

    # Adding percents over bars

    

    height = [p.get_height() for p in ax.patches]

    ncol = int(len(height) / 2)

    total = [height[i] + height[i + ncol]  for i in range(ncol)] *2

    #loooping through bars

    for i, p in enumerate(ax.patches):

        #adding percentages

        ax.text(p.get_x() + p.get_width()/2, height[i]*1.01 + 10,

               "{:1.0%}".format(height[i]/total[i]), ha = "center", size = 14)

        

    #Survived percentage for every value of feature

    sns.pointplot(x = column, y = "Survived", data = data_train, ax = plt.subplot(212))

    plt.xlabel(column, fontsize = 14)

    plt.ylabel("Survived percentage", fontsize = 14)

    plt.show()