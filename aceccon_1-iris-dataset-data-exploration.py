# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib as plt

%pylab inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#First examples from dataset

df = pd.read_csv("../input/Iris.csv")

print(df.head())
#Size of dataset

print(df.shape)
#Checking for missing value

missing_df = df.isnull().sum()

print(missing_df)
#Let's see how many examples we have of each target variable (each specie)

df['Species'].value_counts().plot(kind='bar')



#Add count over the bar

ax=df['Species'].value_counts().plot.bar(width=.8)

for i, v in df['Species'].value_counts().reset_index().iterrows():

    ax.text(i, v.Species + 0.3, v.Species, color='blue')

    

plt.xticks(rotation='horizontal') #Rotate xticks

plt.xlabel('Types of Iris')

plt.ylabel('Count')

plt.title('Count of types of iris on dataset')

plt.show()
#Pairwise correlation between attributes

sns.heatmap(df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm' ]].corr(), vmin = -1, vmax=1, annot=True)

plt.show()
#Plotting the distributions of examples over each class

plt.subplot(221)

sns.violinplot(x = "Species", y = "PetalLengthCm", data=df, size =6)

plt.subplot(222)

sns.violinplot(x = "Species", y = "SepalWidthCm", data=df, size = 6)

plt.subplot(223)

sns.violinplot(x = "Species", y = "PetalLengthCm", data=df, size = 6)

plt.subplot(224)

sns.violinplot(x = "Species", y = "PetalWidthCm", data=df, size = 6)

plt.show()
#Plot relationship between pairwise

sns.pairplot(df.drop("Id", axis = 1), hue="Species", size=3)

plt.show()
#Boxplot of variables

df.drop("Id", axis=1).boxplot(by = "Species", figsize = (10, 10))

plt.show()
from pandas.plotting import parallel_coordinates

plt.figure()

parallel_coordinates(df.drop("Id", axis=1), "Species")
