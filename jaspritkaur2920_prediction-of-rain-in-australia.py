import numpy as np #linear algebra

import pandas as pd #data processing



#import libraries for plotting

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



# importing data

df = pd.read_csv("../input/weather-dataset-rattle-package/weatherAUS.csv")
#preview the dataset

df.head()
#dimentions of dataset

df.shape
#viewing column names

df.columns
#drop RISK_MM variable (it is given in the description to drop the feature)

df.drop(["RISK_MM"], axis=1, inplace=True)
#viewing the summary of dataset

df.info()
#view statistical properties of dataset

df.describe()
#Explore "RainTomorrow" target variable

#check for missing values

df["RainTomorrow"].isnull().sum()
#view number of unique values

df["RainTomorrow"].nunique()
#view the unique values

df["RainTomorrow"].unique()
#view the frequency distribution of values

df['RainTomorrow'].value_counts()
#view percentage of frequency distribution of values

df["RainTomorrow"].value_counts()/len(df)
#find categorical values



categorical = [var for var in df.columns if df[var].dtype=='O']

print("There are {} categorical values\n".format(len(categorical)))

print("The categorical variavles are : ", categorical)
#view categorical variables

df[categorical].head()
#check missing values in categorical variables

df[categorical].isnull().sum()
#view frequency count of categorical variables

for var in categorical:

    print(df[var].value_counts())
#check for cardinality in categorical variables

for var in categorical:

    print(var, " contains ",len(df[var].unique()), " labels")
df['Date'].dtypes
#parse the dates, currently coded as strings, into datetime format

df["Date"] = pd.to_datetime(df['Date'])
#extract year from date

df['Year'] = df['Date'].dt.year

df['Year'].head()
#extract month from date

df['Month'] = df['Date'].dt.month

df['Month'].head() 
#extract day from date

df['Day'] = df['Date'].dt.day

df['Day'].head()
#again viewing the summary of the dataset

df.info()
#As there are three additional columns from Date variable, I will drop the original Date variable.

df.drop('Date', axis = 1, inplace = True)
df.head()
#find categorical values

categorical = [var for var in df.columns if df[var].dtype=='O']

df[categorical].isnull().sum()
#print number of labels in Location variable

print('Location contains', len(df.Location.unique()), 'labels')
#check labels in location variable

df.Location.unique()
#check frequency distribution of values in Location variabe

df.Location.value_counts()
# let's do One Hot Encoding of Location variable

# get k-1 dummy variables after One Hot Encoding 

# preview the dataset with head() method



pd.get_dummies(df.Location, drop_first = True).head()
#print number of labels in WindGustDir variable

print('WindGustDir contains',len(df.WindGustDir.unique()),'labels')
#check labels in WindGustDir variable

df['WindGustDir'].unique()
#check frequency distribution of values in WindGustDir variable

df.WindGustDir.value_counts()
# let's do One Hot Encoding of WindGustDir variable

# get k-1 dummy variables after One Hot Encoding 

# also add an additional dummy variable to indicate there was missing data

# preview the dataset with head() method

pd.get_dummies(df.WindGustDir, drop_first = True, dummy_na = True).head()
# sum the number of 1s per boolean variable over the rows of the dataset

# it will tell us how many observations we have for each category

pd.get_dummies(df.WindGustDir, drop_first = True, dummy_na = True).sum(axis=0)
#check number of labels in WindDir9am variable

print('WindDir9am contains', len(df.WindDir9am.unique()),'labels')
#check lables in WindDir9am variable

df['WindDir9am'].unique()
#check frequency distribution of values in WindDir9am variable

df['WindDir9am'].value_counts()
# let's do One Hot Encoding of WindDir9am variable

# get k-1 dummy variables after One Hot Encoding 

# also add an additional dummy variable to indicate there was missing data

# preview the dataset with head() method



pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).head()
# sum the number of 1s per boolean variable over the rows of the dataset

# it will tell us how many observations we have for each category



pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).sum(axis=0)
#print number of lables in WindDir3pm variable

print('WindDir3pm contains',len(df.WindDir3pm.unique()),'labels')
#check labels in WindDir3pm variable

df['WindDir3pm'].unique()
#check for frequency distribution of values in WindDir3pm variable

df['WindDir3pm'].value_counts()
# let's do One Hot Encoding of WindDir3pm variable

# get k-1 dummy variables after One Hot Encoding 

# also add an additional dummy variable to indicate there was missing data

# preview the dataset with head() method



pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).head()
# sum the number of 1s per boolean variable over the rows of the dataset

# it will tell us how many observations we have for each category



pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).sum(axis=0)
# print number of labels in RainToday variable

print('RainToday contains', len(df['RainToday'].unique()), 'labels')
# check labels in WindGustDir variable

df['RainToday'].unique()
# check frequency distribution of values in WindGustDir variable

df.RainToday.value_counts()
# let's do One Hot Encoding of RainToday variable

# get k-1 dummy variables after One Hot Encoding 

# also add an additional dummy variable to indicate there was missing data

# preview the dataset with head() method



pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).head()
# sum the number of 1s per boolean variable over the rows of the dataset

# it will tell us how many observations we have for each category



pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).sum(axis=0)
# find numerical variables

numerical = [var for var in df.columns if df[var].dtype!='O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :', numerical)
#view the numberical variables

df[numerical].head()
#check missing values in numerical variable

df[numerical].isnull().sum()
#view summary statistics in numerical variables

print(round(df[numerical].describe()),2)
plt.figure(figsize=(15,10))



plt.subplot(2,2,1)

fig = df.boxplot(column='Rainfall')

fig.set_title('')

fig.set_label('Rainfall')



plt.subplot(2,2,2)

fig = df.boxplot(column='Evaporation')

fig.set_title('')

fig.set_label('Evaporation')



plt.subplot(2,2,3)

fig = df.boxplot(column='WindSpeed9am')

fig.set_title('')

fig.set_label('WindSpeed9am')



plt.subplot(2,2,4)

fig = df.boxplot(column='WindSpeed3pm')

fig.set_title('')

fig.set_label('WindSpeed3pm')
# plot historams to check distribution

plt.figure(figsize=(15,10))



plt.subplot(2,2,1)

fig = df.Rainfall.hist(bins=10)

fig.set_xlabel('Rainfall')

fig.set_ylabel('RainTomorrow')



plt.subplot(2,2,2)

fig = df.Evaporation.hist(bins=10)

fig.set_xlabel('Evaporation')

fig.set_ylabel('RainTomorrow')



plt.subplot(2,2,3)

fig = df.WindSpeed9am.hist(bins=10)

fig.set_xlabel('WindSpeed9am')

fig.set_ylabel('RainTomorrow')



plt.subplot(2,2,4)

fig = df.WindSpeed3pm.hist(bins=10)

fig.set_xlabel('WindSpeed3pm')

fig.set_ylabel('RainTomorrow')
# find outliers for Rainfall variable



IQR = df.Rainfall.quantile(0.75) - df.Rainfall.quantile(0.25)

lower_fence = df.Rainfall.quantile(0.25) - (IQR * 3)

upper_fence = df.Rainfall.quantile(0.75) + (IQR * 3)



print('Rainfall outlier values are < {} or > {}'.format(lower_fence, upper_fence))
# find outliers for Evaporation variable



IQR = df.Evaporation.quantile(0.75) - df.Evaporation.quantile(0.25)

lower_fence = df.Evaporation.quantile(0.25) - (IQR * 3)

upper_fence = df.Evaporation.quantile(0.75) + (IQR * 3)



print('Evaporation outliers values are < {} or > {}'.format(lower_fence, upper_fence))
# find outliers for WindSpeed9am variable



IQR = df.WindSpeed9am.quantile(0.75) - df.WindSpeed9am.quantile(0.25)

lower_fence = df.WindSpeed9am.quantile(0.25) - (IQR * 3)

upper_fence = df.WindSpeed9am.quantile(0.75) + (IQR * 3)

print('WindSpeed9am outlier values are < {} or > {}'.format(lower_fence, upper_fence))
# find outliers for WindSpeed3pm variable



IQR = df.WindSpeed3pm.quantile(0.75) - df.WindSpeed3pm.quantile(0.25)

lower_fence = df.WindSpeed3pm.quantile(0.25) - (IQR * 3)

upper_fence = df.WindSpeed3pm.quantile(0.75) + (IQR * 3)

print('WindSpeed3pm outliers values are < {} or > {}'.format(lower_fence, upper_fence))
correlation = df.corr()
plt.figure(figsize=(16, 12))

plt.title('Correlation Heatmap of Rain in Australia Dataset')

ax = sns.heatmap(correlation, square=True, annot=True, fmt='.2f',linecolor='white')

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

ax.set_yticklabels(ax.get_yticklabels(), rotation=30)



plt.show()
num_var = ['MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm', 'WindGustSpeed', 'WindSpeed3pm', 'Pressure9am', 'Pressure3pm']
#sns.pairplot(df[num_var], kind='scatter', diag_kind='hist', palette='Rainbow')

#plt.show()
X = df.drop(['RainTomorrow'], axis=1)

y = df['RainTomorrow']
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train.shape, X_test.shape
# check data types in x_train

X_train.dtypes
# display categorical variables

categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

categorical
# display numerical variables

numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

numerical
X_train[numerical].isnull().sum()
X_test[numerical].isnull().sum()
# print percentage of missing values in the numerical variables in train set

for col in numerical:

    if X_train[col].isnull().mean()>0:

        print(col, round(X_train[col].isnull().mean(), 4))
# inpute missing values in X_train and X_test with respective column meadian in X_train

for df1 in [X_train, X_test]:

    for col in numerical:

        col_median = X_train[col].median()

        df1[col].fillna(col_median, inplace = True)
X_train[numerical].isnull().sum()
X_test[numerical].isnull().sum()
# print percentage of missing values in the categorical variables in training set

X_train[categorical].isnull().mean()
#inpute missing categorical variables with most frequent value

for df2 in [X_train, X_test]:

    df2['WindGustDir'].fillna(X_train['WindGustDir'].mode()[0], inplace=True)

    df2['WindDir9am'].fillna(X_train['WindDir9am'].mode()[0], inplace=True)

    df2['WindDir3pm'].fillna(X_train['WindDir3pm'].mode()[0], inplace=True)

    df2['RainToday'].fillna(X_train['RainToday'].mode()[0], inplace=True)
X_train[categorical].isnull().sum()
X_test[categorical].isnull().sum()
def max_value(df3, variable, top):

    return np.where(df3[variable]>top, top, df3[variable])



for df3 in [X_train, X_test]:

    df3['Rainfall'] = max_value(df3, 'Rainfall', 3.2)

    df3['Evaporation'] = max_value(df3, 'Evaporation', 21.8)

    df3['WindSpeed9am'] = max_value(df3, 'WindSpeed9am', 55)

    df3['WindSpeed3pm'] = max_value(df3, 'WindSpeed3pm', 57)
X_train.Rainfall.max(), X_test.Rainfall.max()
X_train.Evaporation.max(), X_test.Evaporation.max()
X_train.WindSpeed9am.max(), X_test.WindSpeed9am.max()
X_train.WindSpeed3pm.max(), X_test.WindSpeed3pm.max()
X_train[numerical].describe()
# print categorical variables

categorical
X_train[categorical].head()
#encode RainToday variable

import category_encoders as ce

encoder = ce.BinaryEncoder(cols=['RainToday'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)
X_train.head()
X_train = pd.concat([X_train[numerical], X_train[['RainToday_0','RainToday_1']],pd.get_dummies(X_train.Location), pd.get_dummies(X_train.WindGustDir), pd.get_dummies(X_train.WindDir9am), pd.get_dummies(X_train.WindDir3pm)],axis=1)
X_train.head()
X_test = pd.concat([X_test[numerical], X_test[['RainToday_0','RainToday_1']],pd.get_dummies(X_test.Location), pd.get_dummies(X_test.WindGustDir), pd.get_dummies(X_test.WindDir9am), pd.get_dummies(X_test.WindDir3pm)],axis=1)
X_test.head()
X_train.describe()
cols = X_train.columns
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])
X_train.describe()