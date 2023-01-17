#Set Work Directory

# import os

# os.chdir('C:\\Users\\Akshay\\Documents\\iPython\\Personal\\Practice')

# os.getcwd()



#Call Libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
#Load data

# df = pd.read_excel('titanic.xls')

df = pd.read_csv('../input/titanic_data.csv')

df.head(1)
#Example of blank canvas (step 1 only)

## Sex has 2 values, while pclass has 3 values. Thus you get a 2 X 3 matrix of charts

g = sns.FacetGrid(df, row='Sex', col='Pclass',hue='Pclass',size=3)
#First example is a single 1X1 grid chart, without any rows, columns or depth.

#This is why i dont mention any row or column or hue in the FacetGrid statement.

g = sns.FacetGrid(df,size=5)



#Next is the drawing itself. I need a frequency distribution of the Pclass. Opacity = 70%

g.map(sns.countplot,'Pclass', alpha=0.7)



#General hygiene of adding legends.

g.add_legend()
#In second example I make a 2X3 grid with Gender levels as Rows and Embarking levels as columns. 

#Hue (Depth, color) is gender as well. Chart screen size is 3 

g = sns.FacetGrid(df,row='Sex',col='Embarked',hue='Sex',size=3)



#Frequency distribution is that of number of passengers in each passenger class. Opacity is 70%

g.map(sns.countplot,'Pclass', alpha=0.7)



#General hygiene of adding legends

g.add_legend()
#Grid is 1X3 with only pclass as columns. Hue(depth) is same as columns for color only.

g = sns.FacetGrid(df,col='Pclass',hue='Pclass',size=4)

g.map(plt.hist,'Age',alpha=0.7)

g.add_legend()
#Grid is 1X3 with only pclass as columns. Hue(depth) is same as columns for color only.

g = sns.FacetGrid(df,col='Pclass',hue='Pclass',size=4)

g.map(sns.barplot,'Sex','Age',alpha=0.7)

g.add_legend()
#Grid is 2X3 with Embarked levels as columns and Gender levels as rows. Hue(depth) is same as rows for color only.

g = sns.FacetGrid(df, row='Sex',col='Embarked',hue='Sex',size=3)

g.map(plt.scatter,'Age','Fare', alpha=0.7)

g.add_legend()
#Grid is 1X3 with embarked levels as columns. Hue is gender levels for color.

g = sns.FacetGrid(df,col='Embarked',hue='Sex', size=4)

g.map(plt.scatter, 'Age', 'Fare')

g.add_legend()
#Grid is 1X1 with hue as pclass

g = sns.FacetGrid(df,hue='Pclass',size=4)

g.map(plt.scatter,'Age','Fare', alpha=0.7)

g.add_legend()
#Non Grid method for scatter plot with bonus correlation and freq distributions

sns.jointplot(x='Age', y='Fare', data=df, size=6)
#Grid is 1X3 with only pclass as columns.

g = sns.FacetGrid(df,col='Pclass',size=4)

g.map(sns.boxplot, 'Sex', 'Age')

g.add_legend()
#The non grid method of making boxplots is shown below, which is much more colorful by default!

ax = sns.boxplot(x="Pclass", y="Age", data=df)
#Grid is 3X3 with no hue variable

g = sns.FacetGrid(df,row='Pclass',col='Embarked',size=3)

g.map(sns.violinplot, 'Sex', 'Age')

g.add_legend()
#Simple 1X3 grid of line charts without a Hue variable

g = sns.FacetGrid(df,col='Pclass',size=4)

g.map(sns.kdeplot, 'Age')

g.add_legend()
#Grid is 1X2 with only gender levels on columns. Hue is pclass which creates a really cool visualization.

g = sns.FacetGrid(df,col='Sex',hue='Pclass',size=5)

g.map(sns.kdeplot, 'Age')

g.add_legend()
# Hue needs to be a categorical variables else it displays it as a variable on the diagram

sns.pairplot(df.fillna(0), vars=['Age','Fare'] , hue="Pclass", size=3)
#Another minor variation over the pairplot

sns.pairplot(df.fillna(0), vars=['Age','Fare'] , hue="Pclass", size=3, diag_kind="kde")