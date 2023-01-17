#importing important libraries

from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#I renamed the file as train and test files to keep it simple.
# Distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow # To fit the graphs with in the area of notebook properly.

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()

# Correlation matrix

def plotCorrelationMatrix(df, graphWidth):

    filename = df.dataframeName

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title(f'Correlation Matrix for {filename}', fontsize=15)

    plt.show()

# Scatter and density plots

def plotScatterMatrix(df, plotSize, textSize):

    df = df.select_dtypes(include =[np.number]) # keep only numerical columns

    # Remove rows and columns that would lead to df being singular

    df = df.dropna('columns')

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    columnNames = list(df)

    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots

        columnNames = columnNames[:10]

    df = df[columnNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = df.corr().values

    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')

    plt.show()

import pandas as pd

nRowsRead = 1000 # specify 'None' if want to read whole file

# test.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('/kaggle/input/test.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'test.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)#printing the first five rows of the test dataset
plotPerColumnDistribution(df1, 10, 5) 

#calling the plotting function to plot different graphs to visualize relations between different features
#plotting the correlation matrix colors changing from dark to light show an increase in correlation 

#the darker colors suggest negative correlation

plotCorrelationMatrix(df1, 8)
#plotting correlation by making pairs between different features to figure out which pairs are more correlated.

plotScatterMatrix(df1, 20, 10)
nRowsRead = 1000 # specify 'None' if want to read whole file

# train.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df2 = pd.read_csv('/kaggle/input/train.csv', delimiter=',', nrows = nRowsRead)

df2.dataframeName = 'train.csv'

nRow, nCol = df2.shape

print(f'There are {nRow} rows and {nCol} columns')
df2.head(5) #printing first five values of train dataset.
#plotting the number of counts of each region which vary from 1-4

plotPerColumnDistribution(df2, 10, 5)
#plotting the correlation matrix of test dataset

#both test and train correlation matrix show similarities in between these pairs (physicians,hospital_beds),(hospital_beds,income) as must 

#be clear from the fact that more physicians are required if no. of hospitals increase.

plotCorrelationMatrix(df2, 8)
plotScatterMatrix(df2, 20, 10)
import seaborn as sns

ax = sns.scatterplot(x=df2['physicians'] ,y= df2['income'], data=df2)
ax = sns.scatterplot(x=df2['percent_senior'],y=df2['percent_city'],data=df2)

#Negative correlation can be clearly seen in the graph
ax = sns.scatterplot(x=df2['physicians'] ,y= df2['labor'], data=df2)
ax = sns.scatterplot(x=df2['graduates'] ,y= df2['percent_senior'], data=df2)

#There is a clear correlation between graduates and percent senior
X_train = df2.drop('crime_rate',axis=1)

Y_train = df2[['crime_rate']]

X_test = df1.dropna(axis=1)

X_test.head()
#Lets split the data into train and test 1:3 split

from sklearn.model_selection import train_test_split

X = df2.drop('crime_rate', axis=1)

Y = df2[['crime_rate']]

train_x, test_x, train_y, test_y = train_test_split(X,Y

            ,test_size=0.33,random_state=0)

train_x.shape[0], test_x.shape[0]
#using Linear Model for Predictions

from sklearn import linear_model

from sklearn.metrics import mean_squared_error 

lin   = linear_model.LinearRegression()

lin.fit(train_x,train_y)

preds = lin.predict(test_x)

mean_squared_error(test_y,preds)
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state= 0)

regressor.fit(train_x, train_y)

preds = regressor.predict(test_x)

mean_squared_error(test_y,preds)

from sklearn import svm

model = svm.SVR()

model.fit(train_x,train_y)

preds = model.predict(test_x)

mean_squared_error(test_y,preds)

import statsmodels.api as ssm



train_x = ssm.add_constant(train_x)

model = ssm.OLS(train_y,train_x).fit()

preds = model.summary()

preds
#Linear Model from sklearn library for predictions

from sklearn import linear_model

from sklearn import metrics

lin   = linear_model.LinearRegression()

lin.fit(X_train, Y_train)

preds = lin.predict(X_test)

indexes = len(list(preds)) #generating index





crime_rate_predictions = pd.DataFrame(data = preds, index=range(1,indexes+1),columns=['crime_rate'])

print(crime_rate_predictions)
crime_rate_predictions.to_csv('CrimeRatePredictions.csv', index=True)