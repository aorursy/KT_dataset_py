

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats



from IPython.display import Image

from IPython.core.display import HTML 



import warnings

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
"""listing training data"""

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

df = train_data.copy()

df
"""identifying the datatypes of each column"""

df.info()
"""describing statistics of dataframe"""



train_data.describe()
"""Finding the missing values in dataset"""

df.isnull().sum()#.max()
"""grouping the total and percentage of missing values"""

total = df.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)



missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(25)
"""Removing the columns having missing values"""



step1 = missing_data[(missing_data['Percent'] > .19)]

step1
len(df), len(df.columns)
df = df.drop(step1.index, 1)
len(df), len(df.columns)
"""Dropping the rows with missing values """

step2 = missing_data[(missing_data['Percent'] <= .10) & (missing_data['Percent'] > 0.002)]

step2
df = df.dropna(subset=step2.index)
len(df), len(df.columns)
"""checking the existance of missing values"""

df.isnull().sum().max()
df2 = train_data.transpose() #data.T
df2
"""Again listing total and percent"""

total = df2.isnull().sum().sort_values(ascending=False)

percent = (df2.isnull().sum()/df2.isnull().count()).sort_values(ascending=False)



missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(25)
"""Listing the columns in dataset"""

df.columns
"""Finding outliers in the datas"""



"""Univariate Outliers"""



sns.boxplot(x=df['Fare'])
sns.boxplot(x=df['Pclass'])
"""Bivariate outliers"""



fig, ax = plt.subplots(figsize=(8,4))



ax.scatter(df['Fare'], df['Pclass'])



ax.set_xlabel('Fare')

ax.set_ylabel('Pclass')

plt.show()
fig, ax = plt.subplots(figsize=(8,4))



ax.scatter(df['Fare'], df['Sex'])



ax.set_xlabel('Fare')

ax.set_ylabel('Sex')

plt.show()
"""Splitting numerical and categorical datas"""



numerical   = df.select_dtypes(exclude=['object'])

categorical = df.select_dtypes(include=['object'])
numerical
categorical
"""Discovering outliers using mathematical function(z score)"""

z = np.abs(stats.zscore(numerical))
threshold = 3

np.where(z > 3)
z[1][5]
rows, cols = np.where(z > 3)
for i in range(len(rows)):

    print (z[rows[i]][cols[i]])
numerical = numerical[(z < 3).all(axis=1)]
numerical
categorical = categorical[(z < 3).all(axis=1)]
categorical
"""Finding Interquartile range"""



Q1 = numerical.quantile(0.25)

Q3 = numerical.quantile(0.75)

IQR = Q3 - Q1

# print(IQR)
threshold = 10

iqrdf = (numerical < (Q1 - threshold * IQR)) | (numerical > (Q3 + threshold * IQR))
numerical = numerical[~iqrdf.any(axis=1)]

categorical = categorical[~iqrdf.any(axis=1)]
categorical.shape, numerical.shape
"""Concatenate the categorical and numeric values obtained above"""

df = pd.concat([categorical, numerical], axis=1)

df
"""Make dummies"""

df = pd.concat([df, df], axis=0)

df
"""Remove dummies"""



df = df.drop_duplicates(keep='first')

df
"""Data transformation"""







from sklearn import preprocessing







le = preprocessing.LabelEncoder()
def preprocess(df, str_labels):

    for label in str_labels:

        le.fit(df[label])

        df[label] = le.transform(df[label])

    return df



df = preprocess(df, categorical.columns)
df
df['Survived'].unique()
"""Data normalisation"""

#histogram and normal probability plot

sns.distplot(df['Pclass'], fit=norm);

fig = plt.figure()

# normalize the data attributes

normalized = df.copy()

normalized.iloc[:,:-1] = preprocessing.normalize(df.iloc[:,:-1])
normalized
#histogram and normal probability plot

sns.distplot(normalized['Pclass'], fit=norm);

fig = plt.figure()
"""Maxabs scaler"""



from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()

scaler.fit_transform(normalized.Fare.values.reshape(-1, 1))
normalized.Fare
"""minmaxscaler"""



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-3,3))

scaler.fit_transform(normalized.Fare.values.reshape(-1, 1))
normalized.Fare
"""Standard scaler"""



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit_transform(normalized.Fare.values.reshape(-1, 1))
"""Data descretization"""

from sklearn.preprocessing import KBinsDiscretizer
# transform the dataset with KBinsDiscretizer

enc = KBinsDiscretizer(n_bins=50, encode='ordinal')
normalized['Pclass'] = enc.fit_transform(np.array(normalized['Fare']).reshape(-1, 1))
normalized
normalized['Pclass'][0]
"""Aggregation"""

normalized['Fare'].sum()
normalized['Fare'].max()
normalized['Fare'].min()
normalized['Fare'].mean()
normalized['Fare'].median()
normalized.mean()
normalized.mean(axis='columns')
normalized.groupby('Pclass').sum()
normalized.groupby('Pclass').aggregate(['min', np.median, max])
"""Feature selection"""

#using chi square statistics

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2
#apply SelectKBest class to extract top 10 best features

bestfeatures = SelectKBest(score_func=chi2, k=1)
X = normalized.iloc[:,0:-2]  #independent columns

yr = normalized.iloc[:,-2]    #target column i.e fare

yc = normalized.iloc[:,-1]    #target column i.e fare
fit = bestfeatures.fit(X,yr)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns

featureScores.nlargest(1,'Score')  #print 10 best features
"""Correlation"""

#get correlations of each features in dataset

corrmat = normalized.iloc[:,0:-1].corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

#plot heat map

g=sns.heatmap(train_data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
#Constructing decision treee
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split 

from sklearn import metrics
col_names = ['Id', 'Survived', 'Pclass', 'name', 'sex', 'age', 'Sibsp', 'Parch', 'ticket','Fare','Cabin','Embarked']

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

df
x = df[list(df.columns[:-1])]
y = df.Survived
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=1)

list(y)
clf = DecisionTreeClassifier()



clf = clf.fit(x_train, y_train)



y_pred = clf.predict(x_test)
metrics.accuracy_score(y_test, y_pred)