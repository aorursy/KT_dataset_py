from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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
df1 = pd.read_csv('../input/Automobile.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'Automobile.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
plotPerColumnDistribution(df1, 10, 5)
plotCorrelationMatrix(df1, 8)
plotScatterMatrix(df1, 20, 10)
#Define category columns
cat_cols = ['symboling', 'fuel-type', 'aspiration', 'num-of-doors', 
                'body-style', 'drive-wheels', 'engine-location', 'fuel-system',
                'engine-type', 'num-of-cylinders']
#Converting type of categorical columns to category
for col in cat_cols:
    df1[col] = df1[col].astype('category')
#Converting  numeric to categorical variables
dummied = pd.get_dummies(df1[cat_cols], drop_first = True)
plotCorrelationMatrix(dummied, 8)
#adding price column with categorical varaibles
df2=pd.concat([df1['price'],dummied],axis =1)
# Plotting graph between price and categorical variables
plotCorrelationMatrix(df2, 8)
# Coorelation amnong price and categorical variables
df2.corr(method='pearson', min_periods=1)
# create X and y
feature_cols = ['symboling']
X = df1[feature_cols]
y = df1.price
#Convertng categorical variable to numeric for make 
#dummy_make = pd.get_dummies(df1['make'], drop_first = True)
#df_make=pd.concat([df1['price'],dummy_make],axis =1)
df1.info()
# import, instantiate, fit
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X, y)
# print the coefficients
print (linreg.intercept_)
print (linreg.coef_)
# Plot the graph between symboling and price
df1.plot(kind='scatter', x='height', y='price', alpha=0.2)
# Seaborn scatter plot with regression line
import seaborn as sns
sns.lmplot(x='height', y='price', data=df1, aspect=1.5, scatter_kws={'alpha':0.2})

feature_cols = ['length', 'width', 'height']
import seaborn as sns
# multiple scatter plots in Seaborn
sns.pairplot(df1, x_vars=feature_cols, y_vars='price', kind='reg')

# multiple scatter plots in Pandas
fig, axs = plt.subplots(1, len(feature_cols), sharey=True)
for index, feature in enumerate(feature_cols):
    df1.plot(kind='scatter', x=feature, y='price', ax=axs[index], figsize=(16, 3))

#Line plot for price

df1.price.plot()
#boxplot fir price group by length
df1.boxplot(column='price', by='length')