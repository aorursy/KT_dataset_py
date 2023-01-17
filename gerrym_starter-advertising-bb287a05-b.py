from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

print(os.listdir('../input'))
# Distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

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

nRowsRead = 1000 # specify 'None' if want to read whole file

df1 = pd.read_csv('../input/advertising.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'advertising.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
df1.head()# check all file colums feature
sns.pairplot(df1,hue='Clicked on Ad',palette='bwr')
print("Age range")

sns.set_style('whitegrid')

df1['Age'].hist(bins=30)

plt.xlabel('Age')
print("Age vs Area Income")

sns.jointplot(x='Age',y='Area Income',data=df1)
sns.jointplot(x='Age',y='Daily Time Spent on Site',data=df1)
df1.describe() 
df1.isnull().sum() # Check if there is missing values on each column
X_Catagoric = df1.select_dtypes(include = ['object'])

X_Catagoric.head()
X = df1[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]

y = df1['Clicked on Ad'].values
# X_Catagoric.head()

# X.head()

# y.shape
pd.crosstab(df1['Country'], df1['Clicked on Ad']).sort_values(1,0, ascending = False).head(10)
df1.groupby(['Ad Topic Line','Male'])['Clicked on Ad'].count().head() #what type of advertise are man and woman interested
df1['Timestamp'] = pd.to_datetime(df1['Timestamp']) 

# Converting timestamp column into datatime object in order to extract new features

df1['Month'] = df1['Timestamp'].dt.month 

# Creates a new column called Month

df1['Day'] = df1['Timestamp'].dt.day     

# Creates a new column called Day

df1['Hour'] = df1['Timestamp'].dt.hour   

# Creates a new column called Hour

df1["Weekday"] = df1['Timestamp'].dt.dayofweek 

# Creates a new column called Weekday with sunday as 6 and monday as 0

# Other way to create a weekday column

#df['weekday'] = df['Timestamp'].apply(lambda x: x.weekday()) # Monday 0 .. sunday 6

# Dropping timestamp column to avoid redundancy

df1 = df1.drop(['Timestamp'], axis=1) # deleting timestamp
df1.head()
df1.groupby(['Male'])['Clicked on Ad'].sum()
df1.groupby(['Male','Clicked on Ad'])['Clicked on Ad'].count().unstack()
sns.factorplot('Hour', 'Clicked on Ad', hue='Male', data = df1)

plt.show()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.23, random_state=0)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
from sklearn.metrics import r2_score, classification_report, confusion_matrix,accuracy_score
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))