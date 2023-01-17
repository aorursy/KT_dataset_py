import warnings

warnings.filterwarnings('ignore')



import pandas as pd

import numpy as np
train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")
print("Train data dimensions: ", train_data.shape)

print("Test data dimensions: ", test_data.shape)
train_data.head()
print("Number of missing values",train_data.isnull().sum().sum())
train_data.describe()
#create continous column names list

contFeatureslist = []

for colName,x in train_data.iloc[1,:].iteritems():

    #print(x)

    if(not str(x).isalpha()):

        contFeatureslist.append(colName)
print(contFeatureslist)
contFeatureslist.remove("id")

contFeatureslist.remove("loss")
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
plt.figure(figsize=(13,9))

sns.boxplot(train_data[contFeatureslist])
# Include  target variable also to find correlation between features and target feature as well

contFeatureslist.append("loss")
correlationMatrix = train_data[contFeatureslist].corr().abs()



plt.subplots(figsize=(13, 9))

sns.heatmap(correlationMatrix,annot=True)



# Mask unimportant features

sns.heatmap(correlationMatrix, mask=correlationMatrix < 1, cbar=False)

plt.show()
plt.figure(figsize=(13,9))

sns.distplot(train_data["loss"])

sns.boxplot(train_data["loss"])
plt.figure(figsize=(13,9))

sns.distplot(train_data["loss"])
plt.figure(figsize=(13,9))

sns.distplot(np.log1p(train_data["loss"]))


catCount = sum(str(x).isalpha() for x in train_data.iloc[1,:])

print("Number of categories: ",catCount)
catFeatureslist = []

for colName,x in train_data.iloc[1,:].iteritems():

    if(str(x).isalpha()):

        catFeatureslist.append(colName)
print(train_data[catFeatureslist].apply(pd.Series.nunique))
from sklearn.preprocessing import LabelEncoder
for cf1 in catFeatureslist:

    le = LabelEncoder()

    le.fit(train_data[cf1].unique())

    train_data[cf1] = le.transform(train_data[cf1])
train_data.head(5)
sum(train_data[catFeatureslist].apply(pd.Series.nunique) > 2)
filterG5 = list((train_data[catFeatureslist].apply(pd.Series.nunique) > 5))
catFeaturesG5List = [i for (i, v) in zip(catFeatureslist, filterG5) if v]
len(catFeaturesG5List)
catFeaturesG5List
#lets plot for cats >5

for x in catFeaturesG5List:

    plt.figure(figsize=(8,4))

    sns.distplot(train_data[x])
filterG2 = list((train_data[catFeatureslist].apply(pd.Series.nunique) == 2))

catFeaturesG2List = [i for (i, v) in zip(catFeatureslist, filterG2) if v]

catFeaturesG2List.append("loss")
corrCatMatrix = train_data[catFeaturesG2List].corr().abs()

s = corrCatMatrix.unstack()

sortedSeries= s.sort_values(ascending=False)

print(sortedSeries[sortedSeries != 1.0][0:10])