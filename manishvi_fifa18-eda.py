# Import Pandas
import pandas as pd

#Import Numpy for numerical computation
import numpy as np

# Import matplotlib & seaborn for visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt

%matplotlib inline
import seaborn as sns
import scipy.stats as st
from sklearn import ensemble, tree, linear_model
import missingno as msno
train = pd.read_csv("../input/Football_Data.csv")
test = pd.read_csv("../input/Test.csv")
ID = train['ID']
Sno= train['Sno']
#Drop unnecessary columns
train.drop('ID',axis=1,inplace=True)
train.drop('Sno',axis=1,inplace=True)

train.shape , test.shape
train.describe()
numeric_features = train.select_dtypes(include=[np.number])
numeric_features.columns
categorical_features = train.select_dtypes(include=[np.object])
categorical_features.columns
%matplotlib inline
msno.matrix(train.sample(250))
msno.heatmap(train)
msno.bar(train.sample(1000))
train.skew(), train.kurt()
y = train['Wage']
plt.figure(1); plt.title("Distribution")
sns.distplot(y, kde=False, fit=st.johnsonsu)
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=st.norm)
plt.figure(3); plt.title('Log Normal')
sns.distplot(y, kde=False, fit=st.lognorm)
sns.distplot(train.skew(),color='blue',axlabel ='Skewness')
plt.figure(figsize = (12,8))
sns.distplot(train.kurt(),color='r',axlabel ='Kurtosis',norm_hist= False, kde = True,rug = False)
#plt.hist(train.kurt(),orientation = 'vertical',histtype = 'bar',label ='Kurtosis', color ='blue')
plt.show()
plt.hist(train['Wage'],orientation = 'vertical',histtype = 'bar', color ='blue')
plt.show()
correlation = numeric_features.corr()
print(correlation['Wage'].sort_values(ascending = False),'\n')
f , ax = plt.subplots(figsize = (14,12))
plt.title('Correlation of Numeric Features with players wage',y=1,size=16)
sns.heatmap(correlation,square = True,  vmax=0.8)
k= 11
cols = correlation.nlargest(k,'Wage')['Wage'].index
print(cols)
cm = np.corrcoef(train[cols].values.T)
f , ax = plt.subplots(figsize = (14,12))
sns.heatmap(cm, vmax=.8, linewidths=0.01,square=True,annot=True,cmap='viridis',
            linecolor="white",xticklabels = cols.values ,annot_kws = {'size':12},yticklabels = cols.values)
sns.set()
columns = ['Wage','Potential','Overall']
sns.pairplot(train[columns],size = 2 ,kind ='scatter',diag_kind='kde')
plt.show()
total = numeric_features.isnull().sum().sort_values(ascending=False)
percent = (numeric_features.isnull().sum()/numeric_features.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1,join='outer', keys=['Total Missing Count', '% of Total Observations'])
missing_data.index.name =' Numeric Feature'

missing_data.head(20)
missing_values = numeric_features.isnull().sum(axis=0).reset_index()
missing_values.columns = ['column_name', 'missing_count']
missing_values = missing_values.loc[missing_values['missing_count']>0]
missing_values = missing_values.sort_values(by='missing_count')

ind = np.arange(missing_values.shape[0])
width = 0.1
fig, ax = plt.subplots(figsize=(50,20))
rects = ax.barh(ind, missing_values.missing_count.values, color='b')
ax.set_yticks(ind)
ax.set_yticklabels(missing_values.column_name.values, rotation='horizontal')
ax.set_xlabel("Missing Observations Count")
ax.set_title("Missing Observations Count - Numeric Features")
plt.show()
total = categorical_features.isnull().sum().sort_values(ascending=False)
percent = (categorical_features.isnull().sum()/categorical_features.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1,join='outer', keys=['Total Missing Count', ' % of Total Observations'])
missing_data.index.name ='Feature'
missing_data.head(20)
for column_name in train.columns:
    if train[column_name].dtypes == 'object':
        train[column_name] = train[column_name].fillna(train[column_name].mode().iloc[0])
        unique_category = len(train[column_name].unique())
        print("Feature '{column_name}' has '{unique_category}' unique categories".format(column_name = column_name,
                                                                                         unique_category=unique_category))
 
for column_name in test.columns:
    if test[column_name].dtypes == 'object':
        test[column_name] = test[column_name].fillna(test[column_name].mode().iloc[0])
        unique_category = len(test[column_name].unique())
        print("Features in test set '{column_name}' has '{unique_category}' unique categories".format(column_name = column_name, unique_category=unique_category))