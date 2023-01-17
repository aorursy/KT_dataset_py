# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
nRowsRead = 1000 # specify 'None' if want to read whole file

# innercity.csv has 21613 rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('/kaggle/input/credit-score-prediction/CreditScore_train.csv', delimiter=',')

df1.dataframeName = 'CreditScore_train.csv'

nRow, nCol = df1.shape

print(f'TRAIN DATA : There are {nRow} rows and {nCol} columns')



df2 = pd.read_csv('/kaggle/input/credit-score-prediction/CreditScore_test.csv', delimiter=',')

df2.dataframeName = 'CreditScore_test.csv'

nRow, nCol = df2.shape

print(f'TEST DATA : There are {nRow} rows and {nCol} columns')



df1["source"] = "train"

df2["source"] = "test"



merged_df = pd.concat([df1,df2])

merged_df.dataframeName = 'Merged_DF'



nRow, nCol = merged_df.shape

print(f'MERGED DATA : There are {nRow} rows and {nCol} columns')
merged_df.head(5)
merged_df.info()
merged_df.columns
merged_df.dtypes
merged_df.isnull().any()
merged_df.isna().sum()
merged_df.duplicated().sum()
merged_df.describe(include='all').T
##missing data

total = merged_df.count()

sumcol=merged_df.isnull().sum()

countcol=merged_df.isnull().count()



percent = (merged_df.isnull().sum()/countcol*100).sort_values(ascending=False)

missing_data = pd.concat([total, percent,sumcol,countcol], axis=1, keys=['Total', 'Percent','Sumcol','countcol'])

missing_data.sort_values(['Percent'], axis=0, ascending=False)

#missing_data.head(20)



miss_perc=missing_data.sort_values(['Percent'], axis=0, ascending=False)

miss_perc




#missing data

total = merged_df.count()

sumcol=merged_df.isnull().sum()

countcol=merged_df.isnull().count()



percent = (merged_df.isnull().sum()/countcol*100).sort_values(ascending=False)

missing_data = pd.concat([total, percent,sumcol,countcol], axis=1, keys=['Total', 'Percent','Sumcol','countcol'])

#missing_data.head(20)

miss_perc=missing_data.sort_values(['Percent'], axis=0, ascending=False)

m_per = miss_perc[miss_perc.Percent > 60]

print(m_per)

drop_cols=m_per.index

print(drop_cols)

#[cols.append(i) for i in drop_cols if df[i].isnull().sum()/row*100 > 60 ]

#count=0

filtered_df=merged_df.drop(columns=drop_cols,axis=1)



#for i in drop_cols:

 #   print(i)

#    count=count+1

#filt_concat_df=df_concat.drop(columns=[i],axis=1)

print(filtered_df.shape)
filtered_df.head()
filtered_df['y']
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

filtered_df.corr()
#PEARSON CORRELATION



plt.figure(figsize = (15,10))

sns.heatmap(filtered_df.corr(method="pearson"))

plt.title('PEARSON CORRELATION', fontsize=15)
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

%matplotlib inline

plt.figure(figsize = (28,8))

sns.boxplot(data=filtered_df)
print(pd.isnull(filtered_df).any())
filtered_df.head()
##filtered_df = filtered_df.drop(columns=['source'])
filtered_df.isnull().any()
filtered_df.head()
#Correlation with output variable

cor_target = abs(filtered_df.corr()["y"])

#Selecting highly correlated features

relevant_features = cor_target[cor_target<0.3]

relevant_features
relevant_features.item
filtered_df.shape
lst_key=[]

null_key=[]

for i,j in relevant_features.items():

    lst_key.append(i)

#print(lst_key.count())



final_df=filtered_df.drop(columns=lst_key,axis=1)

print(final_df.shape)
a=final_df.isnull().any()==True
for i,j in a.items():

    if j==True:

        null_key.append(i)

print(null_key)

final_df=filtered_df.drop(columns=lst_key,axis=1)
type(a)
for i,j in a.items():

    if j==True:

        null_key.append(i)

print(null_key)
for i in null_key:

    final_df[i].fillna(final_df[i].mean(),inplace=True)

final_df.shape 
train_final = final_df[final_df.source=="train"]

test_final = final_df[final_df.source=="test"]



print(train_final.shape)

print(test_final.shape)



train_final.drop(columns="source",inplace=True)

test_final.drop(columns="source",inplace=True)
X = train_final.drop("y", axis=1)

Y = train_final["y"]

print(X.shape)

print(Y.shape)
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = MinMaxScaler().fit(X)

scaled_X = scaler.transform(X)
from sklearn.model_selection import train_test_split



seed      = 42

test_size = 0.20



X_train, X_test, Y_train, Y_test = train_test_split(scaled_X, Y, test_size = test_size, random_state = seed)



print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)

print(Y_test.shape)
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.linear_model import ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor



from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error



import time

import datetime



start = 0

end = 0

start = time.time()



# user variables to tune

folds   = 10

metric  = "neg_mean_absolute_error"



# hold different regression models in a single dictionary

models = {}

models["Linear"]        = LinearRegression()

models["Lasso"]         = Lasso()

models["Ridge"]         = Ridge()

models["ElasticNet"]    = ElasticNet()

models["DecisionTree"]  = DecisionTreeRegressor()

models["KNN"]           = KNeighborsRegressor()

models["RandomForest"]  = RandomForestRegressor()

models["AdaBoost"]      = AdaBoostRegressor()

models["GradientBoost"] = GradientBoostingRegressor()

models["XGBoost"] = XGBRegressor()



# 10-fold cross validation for each model

model_results = []

model_names   = []

for model_name in models:

	model   = models[model_name]

	k_fold  = KFold(n_splits=folds, random_state=seed)

	results = cross_val_score(model, X_train, Y_train, cv=k_fold, scoring=metric)

	

	model_results.append(results)

	model_names.append(model_name)

	print("{}: {}, {}".format(model_name, round(results.mean(), 3), round(results.std(), 3)))

	end = time.time()

	list_lapse = end - start

	print("Time taken for processing {}: {}".format(model_name, str(datetime.timedelta(seconds=list_lapse))))



# box-whisker plot to compare regression models

figure = plt.figure(figsize = (20,8))



figure.suptitle('Regression models comparison')

axis = figure.add_subplot(111)

plt.boxplot(model_results)

axis.set_xticklabels(model_names, rotation = 45, ha="right")

axis.set_ylabel("Mean Absolute Error (MAE)")

plt.margins(0.05, 0.1)
model = XGBRegressor(objective ='reg:squarederror')

model.fit(X_train,Y_train)



#Predicting TEST & TRAIN DATA

train_predict = model.predict(X_train)

test_predict = model.predict(X_test)



error_percent = np.mean(np.abs((Y_train - train_predict) / Y_train)) * 100

print("MAPE - Mean Absolute Percentage Error (TRAIN DATA): ",error_percent )

Y_train, train_predict = np.array(Y_train), np.array(train_predict)
model = XGBRegressor(objective ='reg:squarederror')

model.fit(X_test,Y_test)



#Predicting TEST & TRAIN DATA

train_predict = model.predict(X_train)

test_predict = model.predict(X_test)



error_percent = np.mean(np.abs((Y_train - train_predict) / Y_train)) * 100

print("MAPE - Mean Absolute Percentage Error (TEST DATA): ",error_percent )

Y_train, train_predict = np.array(Y_train), np.array(train_predict)
# plot between predictions and Y_test

x_axis = np.array(range(0, test_predict.shape[0]))

plt.figure(figsize=(20,10))

plt.plot(x_axis, test_predict, linestyle="--", marker="o", alpha=0.7, color='r', label="predictions")

plt.plot(x_axis, Y_test, linestyle="--", marker="o", alpha=0.7, color='g', label="Y_test")

plt.xlabel('Row number')

plt.ylabel('PRICE')

plt.title('Predictions vs Y_test')

plt.legend(loc='lower right')
feature_importance = model.feature_importances_

feature_importance = 100.0 * (feature_importance / feature_importance.max())



sorted_idx = np.argsort(feature_importance)

pos        = np.arange(sorted_idx.shape[0]) + .5



plt.figure(figsize = (15,18))



#Make a horizontal bar plot.

plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, df1.columns[sorted_idx])

plt.xlabel('Relative Importance')

plt.title('Variable Importance')