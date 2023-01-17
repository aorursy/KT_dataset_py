from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

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

df1 = pd.read_csv('/kaggle/input/data.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'data.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
plotPerColumnDistribution(df1, 10, 5)
plotCorrelationMatrix(df1, 8)
plotScatterMatrix(df1, 20, 10)
sns.pairplot(df1, hue='diagnosis',vars = ['radius_mean','concave points_mean','area_mean','smoothness_mean'])
# The first order of business now is to create new variables which you will then feed into the model functions

# We will create training and test sets to both train our data on and then test our model as well.



X = df1.drop(['diagnosis'],axis=1)#drop the target class from train set

Y = df1['diagnosis']#seperating out the target class

print(X.columns)
## Extra step to deal with null data

X = X.dropna(axis=1) # axis-1 is running horizontal across columns. axis=0 is downwards across rows

print(X.columns)

#NOTE : I was getting a 'null' error while making my model. I tried removing 'Unnamed: 32' just as I had 

# removed the 'diagnosis' column, however the error still crept up, so I used the old tried and trusted

# .dropna() function.
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

#notice how the train_test_split function takes the test size and not the 

#size of the train size.Also 42 is the answer to everything.
#Now that our sets to train are ready, we may just choose to look at their

#new dimensions as well

print(X_train.shape)
print(Y_train.shape)
# I love using the sns heatmap to visualize the null values

sns.heatmap(X_train.isnull())
#Now to actually building the model

from sklearn.svm import SVC #from support vector machine get support vector classifier

from sklearn.metrics import classification_report, confusion_matrix #these will make sense later



svc_model = SVC() #initialize the code module

svc_model.fit(X_train,Y_train) #fit your data variables to the model
y_predict = svc_model.predict(X_test) # get 'y' by predict FOR the TEST values.

cm = confusion_matrix(Y_test,y_predict) # we use the confusion matrix to check

                                        #how our predicted y compares to the

                                        # real one.

sns.heatmap(cm,annot=True)
print(classification_report(Y_test,y_predict))
min_train = X_train.min()

min_train
range_train = (X_train-min_train).max()

range_train
X_train_scaled = (X_train-min_train)/range_train

X_train_scaled
#without scaling

sns.scatterplot(x=X_train['area_mean'],y=X_train['smoothness_mean'],hue=Y_train)
#with scaling

sns.scatterplot(x=X_train_scaled['area_mean'],y=X_train_scaled['smoothness_mean'],hue=Y_train)
#Now we need to do the same normalization for our TEST set

min_test = X_test.min()

range_test = (X_test-min_test).max()

X_test_scaled = (X_test-min_test)/range_test
#setting up our new SVC model

svc_model = SVC()

svc_model.fit(X_train_scaled,Y_train)
y_predict = svc_model.predict(X_test_scaled)

cm = confusion_matrix(Y_test,y_predict)

sns.heatmap(cm,annot=True,fmt='d')
print(classification_report(Y_test,y_predict))

# we get a 99% precision already.
param_grid = {'C':[0.1,1,10,100],'gamma':[0.1,1,0.01,0.001],'kernel':['rbf']}

from sklearn.model_selection import GridSearchCV #read GridSearchCrossValidation

grid = GridSearchCV(SVC(),param_grid,)
grid.fit(X_train_scaled,Y_train)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test_scaled)
cm = confusion_matrix(Y_test,grid_predictions)

sns.heatmap(cm,annot=True)
print(classification_report(Y_test,grid_predictions))