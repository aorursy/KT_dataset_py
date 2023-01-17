# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

import numpy as np

import csv

import re



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Convert text file into csv

#with open('../input/auto-mpg.txt') as input_file:

#    lines = input_file.readlines()

#    newLines = []

#    for line in lines:

#        newLine = line.strip().split()

#        newLines.append(newLine)



#with open('../input/auto-mpg.csv', 'wb') as test_file:

#    file_writer = csv.writer(test_file)

#    file_writer.writerows(newLines)    
df = pd.read_csv("../input/auto-mpg.csv", header=None)

df.head()
# Renaming the columns

df.columns = ['mpg','cylinders','displacement','horsepower','weight','acceleration',

              'model year', 'origin', 'car name']

df.head()
# Lets split variable 'car name' as 'brand name' and 'model name'

df['brand name'], df['model name'] = df['car name'].str.split(' ',1).str
# Lets group and view the data

df.groupby(['brand name']).sum().head()
# what is 'hi'? Is it a wrong split and where is it located in the data file. 

## lets find out....Seriously this is how i learn python .i'm crazy!

df['brand name'].str.contains('hi').head(30)
# Hmm no, the split is correct! lets continue...

df.iloc[28,:]
# Correct brand name 

df['brand name'] = df['brand name'].str.replace('chevroelt|chevrolet|chevy','chevrolet')

df['brand name'] = df['brand name'].str.replace('maxda|mazda','mazda')

df['brand name'] = df['brand name'].str.replace('mercedes|mercedes-benz|mercedes benz','mercedes')

df['brand name'] = df['brand name'].str.replace('toyota|toyouta','toyota')

df['brand name'] = df['brand name'].str.replace('vokswagen|volkswagen|vw','volkswagen')



df.groupby(['brand name']).sum().head()
df.dtypes
# Convert horsepower from Object to numeric

df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')

df.head()
# check horsepower

df.dtypes
df.describe().round(2)
# Fill the 6 missing values of horsepower by mean value

meanhp = df['horsepower'].mean()

df['horsepower'] = df['horsepower'].fillna(meanhp)

df.describe().round(2)
# plot distribution plot to view the distribution of target variable

sns.distplot(df['mpg'])
# Skewness and kurtosis

print("Skewness: %f" %df['mpg'].skew())

print("Kurtosis: %f" %df['mpg'].kurt())
# Counts of each brands

plt.figure(figsize=(12,6))

sns.countplot(x = "brand name", data=df)

t = plt.xticks(rotation=45)
# Car Counts Manufactured by countries

fig, ax = plt.subplots(figsize = (12, 6))

sns.countplot(x = df.origin.values, data=df)

labels = [item.get_text() for item in ax.get_xticklabels()]

labels[0] = 'USA'

labels[1] = 'Europe'

labels[2] = 'Japan'

ax.set_xticklabels(labels)

ax.set_title("Cars manufactured by Countries")

plt.show()
# Exploring the range and distribution of numerical Variables 



fig, ax = plt.subplots(6, 2, figsize = (15, 13))

sns.boxplot(x= df["mpg"], ax = ax[0,0])

sns.distplot(df['mpg'], ax = ax[0,1])



sns.boxplot(x= df["cylinders"], ax = ax[1,0])

sns.distplot(df['cylinders'], ax = ax[1,1])



sns.boxplot(x= df["displacement"], ax = ax[2,0])

sns.distplot(df['displacement'], ax = ax[2,1])



sns.boxplot(x= df["horsepower"], ax = ax[3,0])

sns.distplot(df['horsepower'], ax = ax[3,1])



sns.boxplot(x= df["weight"], ax = ax[4,0])

sns.distplot(df['weight'], ax = ax[4,1])



sns.boxplot(x= df["acceleration"], ax = ax[5,0])

sns.distplot(df['acceleration'], ax = ax[5,1])



plt.tight_layout()
# Plot Numerical Variables 

plt.figure(1)

f,axarr = plt.subplots(4,2, figsize=(12,11))

mpgval = df.mpg.values



axarr[0,0].scatter(df.cylinders.values, mpgval)

axarr[0,0].set_title('Cylinders')



axarr[0,1].scatter(df.displacement.values, mpgval)

axarr[0,1].set_title('Displacement')



axarr[1,0].scatter(df.horsepower.values, mpgval)

axarr[1,0].set_title('Horsepower')



axarr[1,1].scatter(df.weight.values, mpgval)

axarr[1,1].set_title('Weight')



axarr[2,0].scatter(df.acceleration.values, mpgval)

axarr[2,0].set_title('Acceleration')



axarr[2,1].scatter(df["model year"].values, mpgval)

axarr[2,1].set_title('Model Year')



axarr[3,0].scatter(df.origin.values, mpgval)

axarr[3,0].set_title('Country Mpg')

# Rename x axis label as USA, Europe and Japan

axarr[3,0].set_xticks([1,2,3])

axarr[3,0].set_xticklabels(["USA","Europe","Japan"])



# Remove the blank plot from the subplots

axarr[3,1].axis("off")



f.text(-0.01, 0.5, 'Mpg', va='center', rotation='vertical', fontsize = 12)

plt.tight_layout()

plt.show()
# Correlation between Numerical Features

corr = df.select_dtypes(include=['float64','int64']).iloc[:,0:].corr()

plt.figure(figsize=(12,12))

sns.heatmap(corr,vmax=1, square=True, annot=True)
# Car Manufactured by Countries (USA,Europe,Japan) and multivariate analysis

valtoreplace = {1:'USA', 2:'Europe', 3:'Japan'}

df['norigin'] = df['origin'].map(valtoreplace)

sns.pairplot(df, hue="norigin")