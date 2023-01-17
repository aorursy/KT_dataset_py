# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt #Visualization

import seaborn as sns





from collections import Counter 

import warnings

warnings.filterwarnings('ignore')





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Firstly, i want to combine train and test data sets.

df1 = pd.read_csv("/kaggle/input/titanic/train.csv")
df2 = pd.read_csv("/kaggle/input/titanic/test.csv")
df = pd.concat([df1 , df2]) # concatenate two dataframes
df.head()
df.info()
df.Sex.value_counts()
df.Sex.value_counts().plot(kind = "pie", autopct='%1.1f%%'); # autopct provides that the plot can show percentile.
df.Survived.value_counts()
df.Survived.value_counts().plot(kind = "bar");
df.Fare.plot.hist(bins = 20); # "bins" means Number of histogram bins to be used.
df.Age.plot.kde();
import missingno as msno

msno.matrix(df);
msno.bar(df)

plt.show()
fig, ax = plt.subplots()

fig, ax = plt.subplots(2,2)
fig, ax = plt.subplots()

ax.hist(df.Age, bins = 20)

ax.set_xlabel("Age")

ax.set_ylabel("Number of observations")

plt.show()
sns.catplot(x="Sex", y="Age", data=df);
g = sns.factorplot(x = "SibSp", y = "Survived", data=df, kind = "bar", size = 5)

g.set_ylabels("Survived Probability")

plt.show()
g = sns.factorplot(x = "Parch", y = "Survived", data=df, kind = "bar", size = 5)

g.set_ylabels("Survived Probability")

plt.show()
g = sns.factorplot(x = "Pclass", y = "Survived", data=df, kind = "bar", size = 5)

g.set_ylabels("Survived Probability")

plt.show()
sns.distplot(df.Fare);
g = sns.FacetGrid(df, col = "Survived") # col argument provides we can see data acording to the Survived value.

g.map(sns.distplot, "Age", bins= 25)

plt.show()
g = sns.FacetGrid(df, col="Embarked") # We see as many graphs as the value of the "Embarked" argument.

g.map(plt.hist, "Age", bins = 25)

plt.show()
sns.lmplot(x = "Age", y = "Fare", data=df)

plt.show()
sns.scatterplot(x = "Age", y = "Fare",  data=df);
sns.countplot(df.Sex)

plt.title("Gender Comparison",  color = "blue", fontsize = 20) #plt.title is a feature which Matplotlib has. 

                                                               #this way we can edit the title.

plt.show()


### sns.set_palette("RdBu") ## if you want to change color of palette, you can use this code.

sns.countplot(x = "Embarked", data = df)

plt.show()
# We can add new variable(y) so we have more information about each group's mean.



sns.catplot(x = "Embarked", y = "Pclass", data = df, kind = "bar")

plt.show()
# I changed just "kind" type and i earned new information about the distribution of groups.



sns.catplot(x  = "Embarked", y = "Pclass", data = df, kind="box")

plt.show()
# For check



df[df.Embarked == "Q"].Pclass.mean()



# when we analysed Q port's mean with this code, we see that People who embarked on Q port are usally prefer 3. class

list1 = ["SibSp", "Parch", "Age", "Fare", "Survived"]

sns.heatmap(df[list1].corr(), annot = True)

plt.show()
sns.factorplot("Pclass", "Survived", hue = "Sex", data = df)

plt.show()
sns.catplot(x="Embarked", y = "Age", kind = "violin", data=df)

plt.show()
pal = sns.cubehelix_palette(2, rot=-.5, dark = .3) # If we want to change palette

sns.violinplot(data = df.iloc[:,6:8], palette = pal, inner = "points") # inner shows dots.

plt.show()
sns.catplot(x="Embarked", y = "Age", hue = "Sex", kind = "violin", data=df)

plt.show()
sns.boxplot(x = "Sex", y = "Age", hue = "Embarked", data = df )

plt.show()
sns.boxplot(x = "SibSp", y= "Parch", data = df);
sns.swarmplot(x = "Sex", y = "Age", hue = "Pclass", data = df )

plt.show()
sns.scatterplot(x = "Age", y = "Fare", data = df);

sns.scatterplot(x = "Age", y = "Fare", hue="Embarked", data = df);

sns.scatterplot(x = "Age", y = "Fare", hue="Embarked", style = "Pclass",data = df);
sns.scatterplot(x = "Age", y = "Fare", hue = "SibSp",size = "SibSp",data = df);
sns.lmplot(x = "Age", y = "Fare", hue = "Pclass", data = df)

plt.show()
sns.lmplot(x = "Age", y = "Fare", hue = "Pclass", col  = "Embarked", data = df)

plt.show()
sns.lineplot(x = "Age", y = "Survived", data = df);
sns.lineplot(x = "Age", y = "Survived",hue = "Sex", data = df);
sns.lineplot(x = "SibSp", y = "Survived",hue = "Sex", data = df);
def Cabin_type(df):

    df.loc[df["Cabin"].notnull(), "Cabin"] = "Known"

    df.loc[df["Cabin"].isnull(), "Cabin"] = "Unknown"

    return df



train = Cabin_type(df)

sns.countplot("Cabin", hue = "Survived", data = df)

plt.show()
def detect_outliers(df, features):

    outlier_indeces = []

    

    for c in features:

        # 1st quantile

        Q1 = np.percentile(df[c],25)

        

        # 3rd quantile

        Q3 = np.percentile(df[c],75)

        

        # IQR

        IQR = Q3 - Q1

        

        # Outlier step

        outlier_step = IQR * 1.5

        

        # detect outlier and their indeces

        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step  )].index

        

        # store indeces

        outlier_indeces.extend(outlier_list_col)

        

    outlier_indeces = Counter(outlier_indeces)

    multiple_outliers = list(i for i, v in outlier_indeces.items() if v > 0)

    

    return multiple_outliers
df.loc[detect_outliers(df,["Age","SibSp","Parch","Fare"])]
# drop outliers

df = df.drop(detect_outliers(df,["Age","SibSp","Parch","Fare"]), axis = 0).reset_index(drop = True)
# we want to observe missing values 

df.isnull().sum()
# Missing values of Embarked Feature;

df[df["Embarked"].isnull()]
df.boxplot(column="Fare",by="Embarked")

plt.show()
# We fill the missing values, as we have decided above

df.Embarked = df.Embarked.fillna("C")