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
filterG5_10 = list((train_data[catFeatureslist].apply(pd.Series.nunique) > 5) & 

                (train_data[catFeatureslist].apply(pd.Series.nunique) < 10))
catFeaturesG5_10List = [i for (i, v) in zip(catFeatureslist, filterG5_10) if v]
len(catFeaturesG5_10List)
ncol = 2

nrow = 4

try:

    for rowIndex in range(nrow):

        f,axList = plt.subplots(nrows=1,ncols=ncol,sharey=True,figsize=(13, 9))

        features = catFeaturesG5_10List[rowIndex*ncol:ncol*(rowIndex+1)]

        

        for axIndex in range(len(axList)):

            sns.boxplot(x=features[axIndex], y="loss", data=train_data, ax=axList[axIndex])

                        

            # With original scale it is hard to visualize because of outliers

            axList[axIndex].set(yscale="log")

            axList[axIndex].set(xlabel=features[axIndex], ylabel='log loss')

except IndexError:

    print("")
filterG2 = list((train_data[catFeatureslist].apply(pd.Series.nunique) == 2))

catFeaturesG2List = [i for (i, v) in zip(catFeatureslist, filterG2) if v]

catFeaturesG2List.append("loss")
corrCatMatrix = train_data[catFeaturesG2List].corr().abs()



s = corrCatMatrix.unstack()

sortedSeries= s.order(kind="quicksort",ascending=False)



print("Top 5 most correlated categorical feature pairs: \n")

print(sortedSeries[sortedSeries != 1.0][0:9])