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

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import StandardScaler



class CustomScaler(BaseEstimator):

    def __init__(self, columns ):

        self.scaler = StandardScaler()

        self.columns = columns

        self.mean_ = None

        self.std_ = None

    

    def fit(self, X, y=None):

        self.scaler.fit(X[self.columns], y)

        self.mean_ = np.mean(X[self.columns])

        self.std_ = np.std(X[self.columns])

        return self

    

    def transform(self, X, y=None):

        init_col_order = X.columns

        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns,index=X.index)

        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]

        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]
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

#     if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots

#         columnNames = columnNames[:10]

    df = df[columnNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = df.corr().values

    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')

    plt.show()

import matplotlib.pyplot as plt # plotting

import seaborn as sns

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing

from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

from sklearn import metrics
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/Boston-house-price-data.csv', delimiter=',')

df.sample(5)
df.info()
print('total number of null values : {0}'.format(df.isna().sum().sum()))
df.describe()
plt.figure(figsize=(11,9))

corr = df.corr().round(2)



mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

# # Want diagonal elements as well

# mask[np.diag_indices_from(mask)] = False



sns.heatmap(data=corr, annot=True,cmap='coolwarm',mask=mask)

plt.xticks(rotation=90)

plt.show()
for col in df.columns:

    fig,ax = plt.subplots(1,2,figsize=(15,1.5))

    if len(np.unique(df[col]))<10:

        sns.countplot(df[col],ax=ax[0])

    else:

        sns.distplot(df[col],bins=50 if len(np.unique(df[col]))>50 else None,ax=ax[0])

        

    sns.boxplot(df[col],ax=ax[1])

    plt.suptitle(col,fontsize=20,y=1.2)

    plt.show()
columns = [col for col in df.columns if len(np.unique(df[col]))>50]

columns.remove('MEDV')

columns
for col in columns:

    fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(20,1.5))

    

    sns.distplot(df[col],bins=50,ax=ax[0])

    ax[0].set_title('original')

    

    quantile_transformer = preprocessing.QuantileTransformer(output_distribution='normal',n_quantiles=int(len(df)/20), random_state=0)

    X_trans = quantile_transformer.fit_transform(df[col].values.reshape((len(df),1)))

    sns.distplot(X_trans,bins=50,ax=ax[1])

    ax[1].set_title('normalized')

    

    plt.suptitle(col,fontsize=20,y=1.2)

    plt.show()
columns
c = columns.copy()

c.append('MEDV')

X = df[c]

X_train = X.copy()
for k, v in X_train.items():

        q1 = v.quantile(0.25)

        q3 = v.quantile(0.75)

        irq = q3 - q1

        v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]

        perc = np.shape(v_col)[0] * 100.0 / np.shape(X_train)[0]

        print("Column %s outliers = %.2f%%" % (k, perc))
len(X_train)
Q1 = X_train.quantile(0.25)

Q3 = X_train.quantile(0.75)

IQR = Q3 - Q1



X_train = X_train[~((X_train < (Q1 - 1.5 * IQR)) |(X_train > (Q3 + 1.5 * IQR))).any(axis=1)]
len(X_train)
sns.distplot(X_train['MEDV']);plt.show()
cols = 3

rows = int(len(X_train.drop('MEDV',axis=1).columns)/cols)



plt.figure(figsize=(15,10))

for i,col in enumerate(X_train.drop('MEDV',axis=1).columns):

    ax = plt.subplot(rows, cols, i+1)

    sns.distplot(X_train[col],ax=ax)
X.columns
X_train, y_train = X_train.drop('MEDV',axis=1), X_train['MEDV']
y_train = np.log(y_train)
quantile_transformer = preprocessing.QuantileTransformer(output_distribution='normal',n_quantiles=int(len(X_trans)/20), random_state=0)

X_train.loc[:,columns] = quantile_transformer.fit_transform(X_train[columns].values.reshape((len(X_train),len(columns))))
cols = 3

rows = int(len(X_train.columns)/cols)



plt.figure(figsize=(15,10))

for i,col in enumerate(X_train.columns):

    ax = plt.subplot(rows, cols, i+1)

    sns.distplot(X_train[col],ax=ax)
scaler = CustomScaler(columns)#check at the start of the book to find the CustomScaler

scaler.fit(X_train)

X_train = scaler.transform(X_train)
cols = 3

rows = int(len(X_train.columns)/cols)



plt.figure(figsize=(15,10))

for i,col in enumerate(X_train.columns):

    ax = plt.subplot(rows, cols, i+1)

    sns.distplot(X_train[col],ax=ax)
X = X_train.copy()

X.loc[:,'MEDV']=y_train

Q1 = X.quantile(0.25)

Q3 = X.quantile(0.75)

IQR = Q3 - Q1



X = X[~((X < (Q1 - 1.5 * IQR)) |(X > (Q3 + 1.5 * IQR))).any(axis=1)]

y_train = X['MEDV']

X_train = X.drop('MEDV',axis=1)
for k, v in X_train.items():

        q1 = v.quantile(0.25)

        q3 = v.quantile(0.75)

        irq = q3 - q1

        v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]

        perc = np.shape(v_col)[0] * 100.0 / np.shape(X_train)[0]

        print("Column %s outliers = %.2f%%" % (k, perc))
scores_map={}
from sklearn.linear_model import LinearRegression

LR_model = LinearRegression()

scores = cross_val_score(LR_model,X_train,y_train,cv=10,n_jobs=-1,scoring='neg_mean_squared_error')

scores_map['LR']=scores

print('Logistic Regression negative RMSE {:.3f} (+/- {:.3f})'.format(scores.mean(),scores.std()))
from sklearn.svm import SVR





svr_rbf = SVR(kernel='rbf')

grid = GridSearchCV(svr_rbf, cv=10, param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)}, scoring='neg_mean_squared_error')

grid.fit(X_train, y_train)

print("Best parameters :", grid.best_params_)

print("Best Score :{:.3f}".format(grid.best_score_))
svr_rbf = SVR(kernel='rbf',C=10,gamma=0.01)



scores = cross_val_score(svr_rbf,X_train,y_train,cv=10,n_jobs=-1,scoring='neg_mean_squared_error')

scores_map['SVR']=scores

print('SVR negative RMSE {:.3f} (+/- {:.3f})'.format(scores.mean(),scores.std()))
from sklearn.tree import DecisionTreeRegressor



tree = DecisionTreeRegressor(random_state=0)

grid = GridSearchCV(tree, cv=10, param_grid={"max_depth" : [1, 2, 3, 4, 5, 6, 7]}, scoring='neg_mean_squared_error')

grid.fit(X_train, y_train)

print("Best parameters : ", grid.best_params_)

print("Best Score :{:.3f}".format(grid.best_score_))
tree = DecisionTreeRegressor(max_depth=7)

scores = cross_val_score(tree, X_train, y_train, cv=10, scoring='neg_mean_squared_error')

scores_map['DTree'] = scores

print("D.Tree negative RMSE {:.3f} (+/- {:.3f})".format(scores.mean(),scores.std()))
from sklearn.neighbors import KNeighborsRegressor



knn = KNeighborsRegressor()



grid = GridSearchCV(knn, cv=10, param_grid={"n_neighbors" : [2, 3, 4, 5, 6, 7]}, scoring='neg_mean_squared_error')

grid.fit(X_train, y_train)

print("Best parameters :", grid.best_params_)

print("Best Score :{:.3f}".format(grid.best_score_))
knn = KNeighborsRegressor(n_neighbors=4)

scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='neg_mean_squared_error')

scores_map['KNN'] = scores

print("KNN negative RMSE {:.3f} (+/- {:.3f})".format(scores.mean(),scores.std()))
from sklearn.ensemble import GradientBoostingRegressor



gbr = GradientBoostingRegressor(random_state=0)

param_grid={'n_estimators':[50,100,150, 200], 'learning_rate': [0.5,0.1,0.05,0.02,0.001]

            , 'max_depth':[2, 3,4,5,6,7,8], 'min_samples_leaf':[3,5,9,11,14,16]

            ,'min_samples_split':[2,4,6,8,10], 'alpha':[0.05,0.1,0.3,0.5]}

# grid = GridSearchCV(gbr, cv=10, param_grid=param_grid, scoring='neg_mean_squared_error')

grid = RandomizedSearchCV(gbr, cv=10, param_distributions=param_grid, scoring='neg_mean_squared_error')

grid.fit(X_train, y_train)

print("Best params :", grid.best_params_)

print("Best Score :{:.3f}".format(grid.best_score_))
gbr = GradientBoostingRegressor(n_estimators=200,min_samples_split=2,min_samples_leaf=3,max_depth=8,learning_rate=0.02,alpha=0.05,   random_state=0)

scores = cross_val_score(gbr, X_train, y_train, cv=10, scoring='neg_mean_squared_error')

scores_map['GBR'] = scores

print("GBR negative RMSE {:.3f} (+/- {:.3f})".format(scores.mean(),scores.std()))
plt.figure(figsize=(15, 7))

scores_map = pd.DataFrame(scores_map)

sns.boxplot(data=scores_map)

plt.xticks(fontsize=30)

plt.show()