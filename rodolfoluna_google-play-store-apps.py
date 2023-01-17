import pandas as pd

import numpy as np

import seaborn as sns

from sklearn import metrics

import matplotlib.pyplot as plt

%matplotlib inline



from pylab import rcParams

from sklearn import preprocessing
df = pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')
df.info()
df.isnull().any()
df.isnull().sum()
df.head()
#Handling null values on Rating feature



df['Rating'] = df['Rating'].fillna(df['Rating'].median())
df['Type'].value_counts()
df["Type"].fillna("Free", inplace = True) 
df['Content Rating'].value_counts()
df["Content Rating"].fillna("Everyone", inplace = True) 
df['Current Ver'].value_counts()
df["Current Ver"].fillna("1.0", inplace = True)
df['Android Ver'].value_counts()
df["Android Ver"].fillna("4.1 and up", inplace = True)
df.isnull().sum()
df['Rating'].describe()
def type_class(types):

    if types == 'Free':

        return 0

    if types == 'Paid':

        return 1

    else:

        pass

    

df['Type'] = df['Type'].map(type_class)    
#Testing the changes:



df['Type'].value_counts()
df.Installs=df.Installs.apply(lambda x: x.strip('+'))

df.Installs=df.Installs.apply(lambda x: x.replace(',',''))

df.Installs=df.Installs.replace('Free',np.nan)

df.Installs.value_counts()
#Testing the numeric data.

df.Reviews.str.isnumeric().sum()
df[~df.Reviews.str.isnumeric()]
df.Reviews = df.Reviews.apply(lambda x:x.replace('3.0M','3000000'))
df['Reviews'] = pd.to_numeric(df['Reviews'])
rcParams['figure.figsize'] = 12,9

g = sns.kdeplot(df.Rating, color="Blue", shade = True)

g.set_xlabel("Rating")

g.set_ylabel("Frequency")

plt.title('Distribution of Rating',size = 23)
g = sns.countplot(x="Category",data=df, palette = "Set1")

g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right")

g 

plt.title('Apps per category',size = 22)
df['Installs'].unique()
df['Installs'].isnull().sum(axis = 0)
#Dropping null rows on Installs column

df.dropna(subset=['Installs'], how='all', inplace=True)
#Searching null values on Installs column

df['Installs'].isnull().sum(axis = 0)
#Converting Installs feature to numeric

df["Installs"] = pd.to_numeric(df["Installs"])
sorted_installs = sorted(list(df['Installs'].unique()))

df['Installs'].replace(sorted_installs, range(0, len(sorted_installs),1), inplace = True)
df['Installs'].head()
plt.figure(figsize = (12,12))

sns.regplot(x="Installs", y="Rating", color = 'teal',data=df);

plt.title('Rating VS Installs',size = 22)
#Converting values to numeric values.



le = preprocessing.LabelEncoder()

df['App'] = le.fit_transform(df['App'])
#Encoding category feature.



le = preprocessing.LabelEncoder()

df['Category'] = le.fit_transform(df['Category'])
#Encoding genre features.



le = preprocessing.LabelEncoder()

df['Genres'] = le.fit_transform(df['Genres'])
#Content Rating encoding.



le = preprocessing.LabelEncoder()

df['Content Rating'] = le.fit_transform(df['Content Rating'])
#Cleaning Price feature.



df['Price'] = df['Price'].apply(lambda x : x.strip('$'))
# Type encoding



df['Type'] = pd.get_dummies(df['Type'])
df = df[df['Last Updated'] != 0]
#Encoding Last Updated feature.



import time

import datetime



df['Last Updated'] = df['Last Updated'].apply(lambda x : time.mktime(datetime.datetime.strptime(x, '%B %d, %Y').timetuple()))
df.select_dtypes(exclude=['int', 'float']).columns
#Type encoding.



le = preprocessing.LabelEncoder()

df['Type'] = le.fit_transform(df['Type'])
# Convert kbytes to Mbytes 



k_indices = df['Size'].loc[df['Size'].str.contains('k')].index.tolist()

converter = pd.DataFrame(df.loc[k_indices, 'Size'].apply(lambda x: x.strip('k')).astype(float).apply(lambda x: x / 1024).apply(lambda x: round(x, 3)).astype(str))

df.loc[k_indices,'Size'] = converter
#Size encoding.



le = preprocessing.LabelEncoder()

df['Size'] = le.fit_transform(df['Size'])
#Cleaning Size feature

#df['Size'] = df['Size'].apply(lambda x: x.strip('+'))

#df['Size'] = df['Size'].apply(lambda x: x.strip('M'))



df[df['Size'] == 'Varies with device'] = 0

df['Size'] = df['Size'].astype(float)
#Seacrching for more non numerical data



df.info()
#Showing lines with non numerical data



df.loc[~df['Price'].astype(str).str.isdigit()]
#Replacing values on Current Ver feature



df['Current Ver'].replace('Varies with device', np.nan, inplace = True )

df['Current Ver'] = pd.to_numeric(df['Current Ver'], errors='coerce') 

df['Current Ver'] = df['Current Ver'].fillna(df['Current Ver'].mean())
df['Android Ver']
df['Android Ver'] = df['Android Ver'].str.replace(r'\D', '')
#Replace the blank values with the most frequent one



df['Android Ver'].replace('', 403)

df['Android Ver'].fillna(403, inplace=True)
df['Android Ver'] = pd.to_numeric(df['Android Ver'])
df[df.isna().any(axis=1)]
#Finding the mode



df.mode()['Android Ver'][0]
#We will replace de null values with the mode



df['Android Ver'] = df['Android Ver'].fillna(df['Android Ver'].mode()[0])
df.isnull().sum()
#Splitting data in to train and test.



from sklearn.model_selection import train_test_split



X = df.drop(labels = ['Rating'], axis=1)

Y = df.Rating

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
#K Nearest Neighbors



from sklearn.neighbors import KNeighborsRegressor



knear = KNeighborsRegressor(n_neighbors=50)

modelKnn = knear.fit(X_train, y_train)



#Show score

y_pred = modelKnn.predict(X_test)

metrics.mean_absolute_error(y_test, y_pred)
#Stochastic Gradient Descent



from sklearn import linear_model



SGD = linear_model.SGDRegressor()

modelSGD = SGD.fit(X_train, y_train)



#Score

y_pred = modelSGD.predict(X_test)

metrics.mean_absolute_error(y_test, y_pred)
#Support Vector Regressor



from sklearn.svm import SVR



svr = SVR()

modelSVR = svr.fit(X_train, y_train)



#Score

y_pred = modelSVR.predict(X_test)

metrics.mean_absolute_error(y_test, y_pred)
#Linear Regression



from sklearn.linear_model import LinearRegression



LR = LinearRegression()

modelLr = LR.fit(X_train, y_train)



#Score

y_pred = modelLr.predict(X_test)

metrics.mean_absolute_error(y_test, y_pred)