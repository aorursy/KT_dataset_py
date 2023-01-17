# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/train.csv")
data.head()
data.shape
data.info()
data.describe()
data.describe(include=['object'])
# Univariate visualization

# Univariate analysis looks at one feature at a time.

features = ['Age','Fare']

data[features].hist()
# Histograms are useful to find distribution of numerical features like(Gaussian,skewed)

# Similar to histogram we can use kernal density plots which are smooth version of histogram 

# and advantage its not dependent on bin size.

data[features].plot(kind='density',subplots=True)
sns.distplot(data['Fare'])
#Box plot

sns.boxplot('Fare',data=data)
sns.violinplot('Fare',data=data)
# Visualization for Categorical and binary features

# Binary : feature has exactly 2 values

# Categorical : ordinal or non-ordinal

data['Survived'].value_counts()
# Count plot is used to display frequency table in visual form.

sns.countplot('Survived',data=data)
sns.countplot('Embarked',data=data)
# Histogram vs Barplot(count plot)

# Histograms are best suited for looking at the distribution of numerical variables while bar plots are used for categorical features.
# Quantitative vs. Quantitative

# Correlation Matrix

# correlations among the numerical variables in our dataset. This information is important to know as there are Machine Learning algorithms 

# (for example, linear and logistic regression) that do not handle highly correlated input variables well.

# data.corr()

# Correlation matrix is not useful for this dataset as we don't have numerical features.
# Countplot

sns.countplot('Sex',data=data,hue='Survived')
# Countplot

sns.countplot('Embarked',data=data,hue='Survived')
# Scatterplot

sns.scatterplot('Age','Fare',data=data,hue='Survived')