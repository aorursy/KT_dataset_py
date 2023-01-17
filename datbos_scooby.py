%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np 

import pandas as pd 

import seaborn as sns

sns.set_style("darkgrid")

train = pd.read_csv("../input/train/train.csv")

# train.info()

train.head(20)
column_list1 = train.columns.values.tolist()

cmap = sns.diverging_palette(220,10, as_cmap=True)

correlations1 = train[column_list1].corr()

print ("Correlation matrix")

print (sns.heatmap(correlations1, cmap=cmap))
# histagram of Adoption Speed Training Data

train['AdoptionSpeed'].hist(bins=5)

train['Type'].hist(bins=2)
### Cat vs Dog

#### Most of the youngest pets are kittens rather than puppies. It may be istructive to segregate the Type (dog/cat) populations in doing the analysis. 
sns.violinplot(x="Type", y="Age", data=train)


sns.jointplot(x="AdoptionSpeed", y="Age", data=train);
%matplotlib inline



sns.pairplot(train[['Type',  'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize', 'FurLength','Vaccinated']], hue= 'Type')