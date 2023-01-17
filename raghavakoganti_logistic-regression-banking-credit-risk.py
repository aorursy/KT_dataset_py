import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import datasets, linear_model

from mpl_toolkits.mplot3d import axes3d

import seaborn as sns

from sklearn.preprocessing import scale

import sklearn.linear_model as skl_lm

from sklearn.metrics import mean_squared_error, r2_score



import statsmodels.api as sm



print("Packages LOADED")
df_base = pd.read_csv('/kaggle/input/Log-Reg-Case-Study.csv')

df_base.head()
from sklearn.linear_model import LogisticRegression



data = df_base

arr = df_base.values

arr



# Declare the independent variables by just taking 4 variables

X = data.ix[:,(1,10,11,13)]

X
# Declare the dependent variable - Default_On_Payment

y = data.ix[:,16]

y
# Run the regression

regr = LogisticRegression()

regr.fit(X,y)



# Predict the results

pred = regr.predict(X)

pred
y
# Build the Confusion Matrix



from sklearn.metrics import confusion_matrix,classification_report



cm_df = pd.DataFrame(confusion_matrix(y,pred).T, index=regr.classes_, columns=regr.classes_)

cm_df.index.name = 'Predicted'

cm_df.columns.name = 'True'

print(cm_df)
# Partition into 70% Train and 30% Test data



# Train data

model_data = df_base.sample(frac=0.7, random_state=1234)

model_data.head()
df_base.shape
print(model_data.shape)
# Test data

test_data = df_base.loc[~df_base.index.isin(model_data.index),:]

print(test_data.shape)
df = model_data

df.describe()
# Age



print(df.Age.describe())
# check 99.5 quantile

print(df.Age.quantile(q=0.995))
# check 99.7 quantile

print(df.Age.quantile(q=0.997))
# check 99.9 quantile

print(df.Age.quantile(q=0.999))
# Capping the Age values greater than 75 to 75

df.loc[df['Age']>75,'Age'] = 75
print(min(df.Age))
print(max(df.Age))
df.isnull().values.any()
df.isnull().sum()
df.isnull().sum().sum()
# Housing



pd.crosstab(data.Housing,data.Default_On_Payment)
sns.countplot(x='Housing', data=df, palette='hls')

plt.show()
print(df.Housing.isnull().sum())
pd.crosstab(data.Housing.isnull(), data.Default_On_Payment)
pd.crosstab(data.Housing, data.Default_On_Payment).plot(kind='bar')

plt.title('Frequency of Defaulters')

plt.xlabel('Housing')

plt.ylabel('Frequency of Defaults')

plt.show()
df.Housing.isnull().sum()
df['Housing'].mode()[0]
df.Housing[df.Housing=='A152'].count()
df['Housing'].fillna(df['Housing'].mode()[0], inplace=True)

df.Housing[df.Housing=='A152'].count()
df.Housing.isnull().sum()
print(df.Num_Dependents.value_counts())
sns.countplot(x='Num_Dependents', data=df, palette='hls')

plt.show()
pd.crosstab(data.Num_Dependents, data.Default_On_Payment).plot(kind='bar')

plt.title('Frequency of Defaulters')

plt.xlabel('Num_Dependents')

plt.ylabel('Frequency of Defaults')

plt.show()
def get_Percent(col,df):

    grps = df.groupby([col,'Default_On_Payment'])

    df2 = pd.DataFrame()

    for name,group in grps:

        df2.loc[name[0],name[1]] = len(group)

    df2['Percentage 0'] = df2[0]*100/(df2[0]+df2[1])

    df2['Percentage 1'] = df2[1]*100/(df2[0]+df2[1])

    print(df2.sort_values(by='Percentage 1'))

    

cols = ['Num_Dependents']



for col in cols:

    get_Percent(col,df_base)
df.shape
df = df.drop(['Customer_ID', 'Num_Dependents'], axis=1)

df.shape
df['Job_Status'].unique()
df['Job_Status'].describe()
df['Job_Status'].isnull().sum()
print(df.Job_Status.value_counts())
sns.countplot(x='Job_Status', data=df, palette='hls')

plt.show()
pd.crosstab(data.Job_Status, data.Default_On_Payment)
pd.crosstab(data.Job_Status, data.Default_On_Payment).plot(kind='bar')

plt.title('Frequency of Defaulters')

plt.xlabel('Job_Status')

plt.ylabel('Frequency of Defaults')

plt.show()
cols = ['Job_Status']



for col in cols:

    get_Percent(col,df_base)
df['Job_Status'].mode()
df['Job_Status'].fillna(df['Job_Status'].mode()[0], inplace=True)

df.Job_Status.isnull().sum()
df.Job_Status.unique()
print(model_data.shape)
f1 = model_data['Job_Status'] == 'A171'

f2 = model_data['Job_Status'] == 'A172'

f3 = model_data['Job_Status'] == 'A173'
model_data['Dummy_A171'] = np.where(f1,1,0)

model_data['Dummy_A172'] = np.where(f2,1,0)

model_data['Dummy_A173'] = np.where(f3,1,0)



print(model_data.shape)
model_data = model_data.drop(['Job_Status'], axis=1)



model_data.shape
cols = ['Purpose_Credit_Taken']



for col in cols:

    get_Percent(col, df_base)
# For Low Default Percentage



f1 = model_data['Purpose_Credit_Taken']=='P41'

f2 = model_data['Purpose_Credit_Taken']=='P43'

f3 = model_data['Purpose_Credit_Taken']=='P48'



print(model_data.shape)
model_data['Dummy_Purpose_Credit_Taken_Low'] = np.where(np.logical_or(f1,

                                                                     np.logical_or(f2,f3)),1,0)



print(model_data.shape)
# For High Default peecentage



f1 = model_data['Purpose_Credit_Taken']=='P49'

f2 = model_data['Purpose_Credit_Taken']=='P40'

f3 = model_data['Purpose_Credit_Taken']=='P45'

f4 = model_data['Purpose_Credit_Taken']=='P50'

f5 = model_data['Purpose_Credit_Taken']=='P46'



print(model_data.shape)
model_data['Dummy_Purpose_Credit_Taken_High'] = np.where(np.logical_or(f1,

                                                                      np.logical_or(f2,

                                                                                   np.logical_or(f3,

                                                                                                np.logical_or(f4,f5)))),1,0)



print(model_data.shape)
model_data = model_data.drop(['Purpose_Credit_Taken'],axis=1)



model_data.shape
pd.crosstab(data.Age,data.Default_On_Payment)
pd.crosstab(data.Age,data.Default_On_Payment).plot(kind='bar')

plt.title('Frequency of Defaulters')

plt.xlabel('Age')

plt.ylabel('Frequency of Defaults')

plt.show()
bins = [0,30,100]

ages = model_data.Age

lbls = [1,0]



model_data['Dummy_Age_Group'] = pd.cut(ages,labels=lbls,bins=bins)
model_data[['Age','Dummy_Age_Group']].head()
model_data.shape
# Drop the 'Age' column as we have created a Dummy Variable for it.

model_data = model_data.drop(['Age'],axis=1)

model_data.shape
# Drop non-numeric columns

df2 = model_data._get_numeric_data()

df2.head()
# Independent variables

X = df2.loc[:,df2.columns!='Default_On_Payment'].values

X
# Dependent variable

y = df2.iloc[:,5].values

y
logit = sm.Logit(y,sm.add_constant(X))

lg = logit.fit()

lg.summary()
# Find out the significant variables



def get_significant_vars(lm):

    var_p_vals_df = pd.DataFrame(lm.pvalues)

    var_p_vals_df['vars'] = var_p_vals_df.index

    var_p_vals_df.columns = ['pvals','vars']

    return list(var_p_vals_df[var_p_vals_df.pvals<=0.05]['vars'])



significant_vars = get_significant_vars(lg)

significant_vars
# Consider only the significant variables and fit function again

X = df2.ix[:,(0,2,4,9)].values

X
logit = sm.Logit(y,sm.add_constant(X))

lg = logit.fit()

lg.summary()
significant_vars = get_significant_vars(lg)

significant_vars
# Consider only the significant variables and fit function again

X = df2.ix[:,(0,1,4)].values

X
logit = sm.Logit(y,sm.add_constant(X))

lg = logit.fit()

lg.summary()
significant_vars = get_significant_vars(lg)

significant_vars
regr = LogisticRegression()

regr.fit(X,y)

print(regr.coef_)

print(regr.intercept_)
pred = regr.predict(X)

pred