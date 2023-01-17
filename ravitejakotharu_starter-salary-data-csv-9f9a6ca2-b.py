from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#importing libaries for linear regression classifier

import numpy as np
import pandas as pd
import seaborn as sns
import pandas.util.testing as tm
import matplotlib.pyplot as plt
from sklearn import preprocessing,svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

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
# salary_data.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df = pd.read_csv('/kaggle/input/salary_data.csv', delimiter=',', nrows = nRowsRead)
df.dataframeName = 'salary_data.csv'
nRow, nCol = df.shape
print(f'There are {nRow} rows and {nCol} columns')
#Read the dataset
df_set = df[['YearsExperience', 'Salary']]
# Taking only the selected two attributes from the dataset 
df_set.columns = ['YearsExperience', 'Salary']
df.head(5)
plotPerColumnDistribution(df, 10, 5)
plotCorrelationMatrix(df, 8)
plotScatterMatrix(df, 6, 15)
#models for 50% train and 50% test

X = np.array(df_set['YearsExperience']).reshape(-1, 1) 
y = np.array(df_set['Salary']).reshape(-1, 1) 
# Separating the data into independent and dependent variables 
# Converting each dataframe into a numpy array 
# since each dataframe contains only one column 
#df_set.dropna(inplace = True) 
# Dropping any rows with Nan values 
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = 0.5,test_size = 0.5,random_state=0) 
# Splitting the data into training and testing data 
regr = LinearRegression() 
regr.fit(X_train, y_train) 
X_train.shape
#predicting the test result and visualizing the test result
y_pred=regr.predict(X_test)
y_pred
plt.scatter(X_test,y_test,color='orange')
plt.plot(X_test,regr.predict(X_test),color='black')
plt.title('Salary vs YearsExperience (Test Data 50%)')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()
#predicting the train result and visualizing the results
X_pred = regr.predict(X_train)
X_pred
plt.scatter(X_train,y_train,color='brown')
plt.plot(X_train,regr.predict(X_train),color='black')
plt.title('Salary vs YearsExperience(Train data 50%)')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()
#Calculating the function for Mean squared error and Root mean squared error
#MSE
MSE = np.square(np.subtract(y_test,y_pred)).mean()
print("Mean squared error of dataset is ",MSE)

#RMSE
RMSE = np.sqrt(MSE)
print("Root Mean squared error of dataset is ",RMSE)

#crosschecking the MSE/RMSE results with sklearn metrics
from sklearn import metrics
print('Mean Squared Error of the Model: ',metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error of the Model: ',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
#models for 70% train and 30% test
X = np.array(df_set['YearsExperience']).reshape(-1, 1) 
y = np.array(df_set['Salary']).reshape(-1, 1) 
# Converting each dataframe into a numpy array 
# since each dataframe contains only one column 
#df_set.dropna(inplace = True) 
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = 0.7,test_size = 0.3,random_state=0) 
# Splitting the data into training and testing data 
regr = LinearRegression() 
regr.fit(X_train, y_train) 
X_train.shape
y_test.shape
#predicting the test result and visualizing the test result on the regression line 
y_pred=regr.predict(X_test)
y_pred
plt.scatter(X_test,y_test,color='orange')
plt.plot(X_test,regr.predict(X_test),color='black')
plt.title('Salary vs YearsExperience (Test Data 30%)')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()
#predicting the train result and visualizing the results on the regression line
X_pred = regr.predict(X_train)
X_pred
plt.scatter(X_train,y_train,color='brown')
plt.plot(X_train,regr.predict(X_train),color='black')
plt.title('Salary vs YearsExperience(Train data 70%)')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()
#Calculating the function for Mean squared error and Root mean squared error
#MSE
MSE = np.square(np.subtract(y_test,y_pred)).mean()
print("Mean squared error of dataset is ",MSE)

#RMSE
RMSE = np.sqrt(MSE)
print("Root Mean squared error of dataset is ",RMSE)

#crosschecking the MSE/RMSE results with sklearn metrics
from sklearn import metrics
print('Mean Squared Error of the Model: ',metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error of the Model: ',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
#models for 80% train and 20% test
X = np.array(df_set['YearsExperience']).reshape(-1, 1) 
y = np.array(df_set['Salary']).reshape(-1, 1) 
# Converting each dataframe into a numpy array 
# since each dataframe contains only one column 
#df_set.dropna(inplace = True) 
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = 0.8,test_size = 0.2,random_state=0) 
# Splitting the data into training and testing data 
regr = LinearRegression() 
regr.fit(X_train, y_train) 
X_train.shape 
y_test.shape
#predicting the test result and visualizing the test result on the regression line
y_pred=regr.predict(X_test)
y_pred
plt.scatter(X_test,y_test,color='orange')
plt.plot(X_test,regr.predict(X_test),color='black')
plt.title('Salary vs YearsExperience (Test Data 20%)')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()
#predicting the train result and visualizing the results on the regression line
X_pred = regr.predict(X_train)
X_pred
plt.scatter(X_train,y_train,color='brown')
plt.plot(X_train,regr.predict(X_train),color='black')
plt.title('Salary vs YearsExperience(Train data 80%)')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()
#Calculating the function for Mean squared error and Root mean squared error
#MSE
MSE = np.square(np.subtract(y_test,y_pred)).mean()
print("Mean squared error of dataset is ",MSE)

#RMSE
RMSE = np.sqrt(MSE)
print("Root Mean squared error of dataset is ",RMSE)

#crosschecking the MSE/RMSE results with sklearn metrics
from sklearn import metrics
print('Mean Squared Error of the Model: ',metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error of the Model: ',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))