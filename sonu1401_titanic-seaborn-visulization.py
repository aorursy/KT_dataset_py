# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
df=pd.read_csv("../input/test.csv")
df.head()
df1=pd.read_csv("../input/train.csv")
df1.head()
# We will take training data and practice all seaborn visulization

#Basically we can categorize seaborn plots into 5 more important types which is mostly used in 
#data science 
# 1. Continuos or Distribution Plots
#2. Categorical Plots
#3. Matrix Plots
#4. Regression Plots
#5. Grid Plots 
# 1. Continuos or Distribution Plots- as it is clear we will take continuos varibles to plot 
# this type of plots
# a distplots
import seaborn as sns
sns.distplot(df1['Age'].fillna(value=0))

#To remove the kde layer and just have the histogram use:
sns.distplot(df1['Age'].fillna(value=0),kde=False)
sns.distplot(df1['Age'].fillna(value=0),kde=False,bins=50)
#width of histogram varies by using bins variation 
sns.distplot(df1['Age'].fillna(value=0).value_counts())
##now putting kernel density equation as True (y - axis gives number of ages count )
#jointplot() allows you to basically match up two distplots for bivariate data. With your choice of what kind parameter to compare with: 
#“scatter” 
#“reg” 
#“resid” 
#“kde” 
#“hex”
df1.head()
df1.fillna(value=0,inplace= True)
sns.jointplot(x="Age",y="Fare",data=df1,kind='scatter')
#here x and y both should be continuos 
# on top side age is distributed in distribution and right side fare is distributed as in right side and 
#how fare and age are related can be seen in scatter plot in middle
sns.jointplot(x="Age",y="Fare",data=df1,kind='reg')
# on top side age is distributed in distribution and right side fare is distributed as in right side and 
#how fare and age are related can be seen in reg plot in middle
sns.jointplot(x='Age',y='Fare',data=df1,kind='hex',color='r')
sns.jointplot(x='Age',y='Fare',data=df1,kind='resid',color='r')
sns.jointplot(x='Age',y='Fare',data=df1,kind='kde',color='b')
# pairplot : pairplot will plot pairwise relationships across an entire dataframe (for the numerical columns) and 
# supports a color hue argument (for categorical columns). 
sns.pairplot(df1)
# concepts:
# If two varaibles are same then histogram will be plotted.
# If two variables are continuos then scatter plot will be plotted.
# But if two variables are categorical then || or == graph will be plotted
# If data has null values hen no graph will be plotted.
sns.pairplot(df1,hue='Sex',palette='coolwarm')
# rugplot : rugplots are actually very simple concept , they just draw a dash mark for every point on univaraite distribution. They are building
# block of KDE.
#sns.rugplot(df1['Age'])
sns.rugplot(df1['Fare'])
#kdeplots : kdeplots are "Kernal Density Estimation Plots". These KDE plots replace every single observation with a Guassian(Normal) Distribution
# centered around that value

sns.kdeplot(df1['Age'])
#2. Categorical Plots (It allows you to get the aggregate(Y) data off a categorical(X) feature in your data.)
#a. barplot
#b. countplot
#c. boxplot
#d. violinplot
#e. stripplot
#f. swarmplot
#g. factorplot
#h. lvplot

df1.head()
#a. barplot : barplot is a general plot that allows you to aggregate the categorical data based off some function, by default the mean:

sns.barplot(x='Sex',y='Fare',data=df1)
# You can change the estimator object to your own function, that converts a vector to a scalar:
import numpy as np
sns.barplot(x='Sex',y='Fare',data=df1,estimator=np.std)

sns.barplot(x='Sex',y='Fare',data=df1,estimator=np.sum)
#b. countplot : This is essentially the same as barplot except the estimator is explicitly counting the number of occurrences. 
# Which is why we only pass the x value:
sns.countplot(x='Pclass',data=df1)

#c. boxplot() : boxplots and violinplots are used to shown the distribution of categorical data. 
#  A box plot (or box-and-whisker plot) shows the distribution of quantitative data in a way that facilitates comparisons between variables or across levels 
# of a categorical variable. The box shows the quartiles of the dataset while the whiskers extend to show the rest of the distribution, except for points 
# that are determined to be “outliers” using a method that is a function of the inter-quartile range.

df1.head()
sns.boxplot(x='Pclass',y='Fare',data=df1)
sns.boxplot(x='Sex',y='Fare',data=df1,palette='rainbow')
# we Can do entire dataframe with orient='h'
sns.boxplot(data=df1,palette='rainbow',orient='h')
sns.boxplot(x='Sex',y='Fare',data=df1,hue='Pclass',palette='coolwarm')
#d. violinplot¶
#A violin plot plays a similar role as a box and whisker plot. It shows the distribution of quantitative data across several levels of one (or more)
#categorical variables such that those distributions can be compared. Unlike a box plot, in which all of the plot components correspond to actual 
#datapoints, the violin plot features a kernel density estimation of the underlying distribution.

df1.head()
sns.violinplot(x='Pclass',y='Age',data=df1,palette='rainbow')
sns.violinplot(data=df1,palette='rainbow',orient='h')
sns.violinplot(x='Pclass',y='Age',data=df1,palette='Set1',hue='Sex')
sns.violinplot(x='Pclass',y='Age',data=df1,palette='Set1',hue='Sex',split=True)
#e. stripplot  :
#The stripplot will draw a scatterplot where one variable is categorical. A strip plot can be drawn on its own, but it is also a good complement 
#to a box or violin plot in cases where you want to show all observations along with some representation of the underlying distribution.
df1.head()
sns.stripplot(x='Pclass',y='Fare',data=df1)
sns.stripplot(x='Pclass',y='Age',data=df1,jitter=True)
sns.stripplot(x='Pclass',y='Age',hue='Sex',data=df1,jitter=True)
sns.stripplot(x='Pclass',y='Age',hue='Sex',data=df1,jitter=True,palette='Set1')
sns.stripplot(x='Pclass',y='Age',hue='Sex',data=df1,jitter=True,palette='Set1',split=True)
sns.swarmplot(x='Pclass',y='Age',data=df1)
sns.swarmplot(x='Pclass',y='Age',data=df1,hue='Sex')
sns.swarmplot(x='Pclass',y='Age',data=df1,hue='Sex',palette='Set1',split=True)
### Combining Categorical Plots
sns.violinplot(x='Pclass',y='Age',data=df1,hue='Sex',palette='rainbow',split=True)
sns.swarmplot(x='Pclass',y='Age',data=df1,hue='Sex',palette='Set1',split=True)
##factorplot: factorplot is the most general form of a categorical plot. It can take in a kind parameter to adjust the plot type
sns.factorplot(x='Pclass',y='Fare',data=df1,kind='bar')
##  3. MATRIX PLOTS (Matrix plots allow you to plot data as color-encoded matrices and can also be used to indicate clusters within the data )
# TYPES: HEATMAP AND CLUSTERMAP
# HEATMAP In order for a heatmap to work properly, your data should already be in a matrix form, the sns.heatmap function basically just colors
# it in for you.
df1.head()
sns.heatmap(df1.corr())
sns.heatmap(df1.corr(),cmap='coolwarm',annot=True)
# making pivot of data and then using heatmap
df1.pivot_table(values='Fare',index='Pclass',columns='Sex')
df1_pv=df1.pivot_table(values='Fare',index='Pclass',columns='Sex')
sns.heatmap(df1_pv,cmap='coolwarm',annot=True)
#clustermap: The clustermap uses hierarchal clustering to produce a clustered version of the heatmap. For example:
sns.clustermap(df1_pv)
sns.clustermap(df1.corr(),cmap='coolwarm',annot=True)
# Regression Plots
# lmplot allows you to display linear models, but it also conveniently allows you to split up those plots based off of features, 
# as well as coloring the hue based off of features.( x and y both should be continuous)
df1.head()
sns.lmplot(x='Age',y='Fare',data=df1)
sns.lmplot(x='Pclass',y='Fare',data=df1)
sns.lmplot(x='Pclass',y='Fare',data=df1,hue='Sex')
sns.lmplot(x='Pclass',y='Fare',data=df1,hue='Sex',palette='coolwarm')
sns.lmplot(x='Pclass',y='Fare',data=df1,col='Sex')
df1.head()
sns.lmplot(x='Pclass',y='Fare',data=df1,col='Sex',row='Embarked')
# 5. GRID PLOTS  : Grids are general types of plots that allow you to map plot types to rows and columns of a grid, 
#               this helps you create similar plots separated by features.
#Types :
#PairGrid : Pairgrid is a subplot grid for plotting pairwise relationships in a dataset.
#pairplot
#FacetGrid
#JointGrid
#PairGrid : Pairgrid is a subplot grid for plotting pairwise relationships in a dataset.
df1.head()
import matplotlib.pyplot as plt
%matplotlib inline
sns.PairGrid(df1)

# then map the graph on the grid
pg=sns.PairGrid(df1)
pg.map(plt.scatter)
# Map to upper,lower, and diagonal
pg = sns.PairGrid(df1)
pg.map_diag(plt.hist)
pg.map_upper(plt.scatter)
pg.map_lower(sns.kdeplot)
#Pairplot : pairplot is a simpler version of PairGrid (you'll use quite often)
sns.pairplot(df1)
sns.pairplot(df1,hue='Sex',palette='rainbow')
#FacetGrid: FacetGrid is the general way to create grids of plots based off of a feature:
#Create just the Grid 
fg=sns.FacetGrid(df1,col='Pclass',row='Sex') 
#now map the plot on the same
fg=sns.FacetGrid(df1,col='Pclass',row='Sex') 
fg.map(plt.hist,"Fare")
fg=sns.FacetGrid(df1,row='Survived',col='Pclass',hue='Sex') 
fg.map(plt.scatter,"Fare","Age").add_legend()

#JointGrid: JointGrid is the general version for jointplot() type grids, for a quick example:
jg=sns.JointGrid(x='Age',y='Fare',data=df1)
jg.plot(sns.regplot,sns.distplot)