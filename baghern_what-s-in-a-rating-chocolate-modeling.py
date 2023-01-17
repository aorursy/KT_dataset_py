import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv('../input/flavors_of_cacao.csv')

#removing some of the unicode non printing characters for convenience
for c in df.columns.values:
    if c != 'Rating':
        df[c] = df[c].apply(lambda x: str(x).replace(u'\xa0', ''))

        
print("Number of records: " + str(df.shape[0]))
print("Baseline Average Rating: " + str(df['Rating'].mean()))
df.head()
sns.countplot(df['Rating']).set_title('Distribution Over Chocolate Ratings')
plt.show()
print("Least liked chocolates: ")
df[df['Rating'] == 1]
print("Most liked chocolates: ")
df[df['Rating'] == 5]
sns.countplot(df['Review\nDate']).set_title('Rating Volume Over Time')
plt.show()
sns.set_style("darkgrid")

#fitting a linear regression line to a scatterplot
sns.regplot(x=df['Review\nDate'].apply(lambda x: float(x)), 
            y=df['Rating'].apply(lambda x: float(x)))

plt.title('Rating Over Time')
plt.show()
print("Total unique Cocoa Percent: " + str(len(df['Cocoa\nPercent'].unique())))
df['Cocoa\nPercent'] = df['Cocoa\nPercent'].apply(lambda x: float(str(x).replace('%', '')))

sns.regplot(x=df['Cocoa\nPercent'], y=df['Rating'])

plt.title('Rating By Cocoa Percentage')
plt.show()
df.replace('', np.nan, ).isnull().sum()
df['Bean\nType'].fillna("Uknown", inplace = True)
#for one pesky row that has "nan"
df['Bean\nType'].replace("nan", "Uknown", inplace = True)

print("Total unique bean types: " + str(len(df['Bean\nType'].unique())))

f = {'Rating':['size','mean','std']}
df.groupby('Bean\nType').agg(f)
print("Total unique bean types: " + str(len(df['Company\nLocation'].unique())))

f = {'Rating':['size','mean','std']}
df.groupby('Company\nLocation').agg(f)
print("Total unique Specific Bean Origin or Bar Name: " + str(len(df['Specific Bean Origin\nor Bar Name'].unique())))
print("Total unique REF: " + str(len(df['REF'].unique())))
X = df.drop('Rating', axis = 1)
y = df['Rating']

X.head()
#using pd.get_dummies to create a one hot encoded matrix
dummies = pd.get_dummies(X['Company\xa0\n(Maker-if known)'])
#Adding the variable to the column names so I can keep track of which original variable it came from
dummies.columns = ['Company_' + k for k in dummies.columns.values]
X = pd.concat([X, dummies], axis=1)

#dropping the original column 
del X['Company\xa0\n(Maker-if known)']

X.head()
#using pd.get_dummies to create a one hot encoded matrix
dummies = pd.get_dummies(X['Company\nLocation'])
#Adding the variable to the column names so I can keep track of which original variable it came from
dummies.columns = ['Company_Loc_' + k for k in dummies.columns.values]

X = pd.concat([X, dummies], axis=1)

#dropping the original column 
del X['Company\nLocation']

X.head()
#using pd.get_dummies to create a one hot encoded matrix
dummies = pd.get_dummies(X['REF'])
#Adding the variable to the column names so I can keep track of which original variable it came from
dummies.columns = ['REF_' + k for k in dummies.columns.values]

X = pd.concat([X, dummies], axis=1)

#dropping the original column 
del X['REF']

X.head()
#using pd.get_dummies to create a one hot encoded matrix
dummies = pd.get_dummies(X['Review\nDate'])
#Adding the variable to the column names so I can keep track of which original variable it came from
dummies.columns = ['Date_' + k for k in dummies.columns.values]

X = pd.concat([X, dummies], axis=1)

#dropping the original column 
del X['Review\nDate']

X.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=7)

X_train.head()
from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer(ngram_range=(1,2))

#fitting to the training data
tf.fit(X_train['Specific Bean Origin\nor Bar Name'])

#transforming on both data set
train_transformed = tf.transform(X_train['Specific Bean Origin\nor Bar Name'])
test_transformed = tf.transform(X_test['Specific Bean Origin\nor Bar Name'])

#converting to a dataframe so we can see the column names easier later
train_transformed = pd.DataFrame(data = train_transformed.todense(), 
                              index = X_train.index.values, 
                              columns = ["Bar_Name_" + k for k in tf.vocabulary_] )
test_transformed = pd.DataFrame(data = test_transformed.todense(), 
                              index = X_test.index.values, 
                              columns = ["Bar_Name_" + k for k in tf.vocabulary_] )

#appending back to the original data
X_train = pd.concat([X_train, train_transformed], axis=1)
X_test = pd.concat([X_test, test_transformed], axis=1)

del X_train['Specific Bean Origin\nor Bar Name']
del X_test['Specific Bean Origin\nor Bar Name']

X_train.head()
tf = TfidfVectorizer(ngram_range=(1,2))

tf.fit(X_train['Bean\nType'])

train_bean_name = tf.transform(X_train['Bean\nType'])
test_bean_name = tf.transform(X_test['Bean\nType'])

#transforming on both data set
train_transformed = tf.transform(X_train['Bean\nType'])
test_transformed = tf.transform(X_test['Bean\nType'])

#converting to a dataframe so we can see the column names easier later
train_transformed = pd.DataFrame(data = train_transformed.todense(), 
                              index = X_train.index.values, 
                              columns = ["Bean_Type_" + k for k in tf.vocabulary_] )
test_transformed = pd.DataFrame(data = test_transformed.todense(), 
                              index = X_test.index.values, 
                              columns = ["Bean_Type_" + k for k in tf.vocabulary_] )

#appending back to the original data
X_train = pd.concat([X_train, train_transformed], axis=1)
X_test = pd.concat([X_test, test_transformed], axis=1)

del X_train['Bean\nType']
del X_test['Bean\nType']

X_train.head()
tf = TfidfVectorizer(
    ngram_range=(1,2))

tf.fit(X_train['Broad Bean\nOrigin'])

#transforming on both data set
train_transformed = tf.transform(X_train['Broad Bean\nOrigin'])
test_transformed = tf.transform(X_test['Broad Bean\nOrigin'])

#converting to a dataframe so we can see the column names easier later
train_transformed = pd.DataFrame(data = train_transformed.todense(), 
                              index = X_train.index.values, 
                              columns = ["Bean_Origin_" + k for k in tf.vocabulary_] )
test_transformed = pd.DataFrame(data = test_transformed.todense(), 
                              index = X_test.index.values, 
                              columns = ["Bean_Origin_" + k for k in tf.vocabulary_] )

#appending back to the original data
X_train = pd.concat([X_train, train_transformed], axis=1)
X_test = pd.concat([X_test, test_transformed], axis=1)

del X_train['Broad Bean\nOrigin']
del X_test['Broad Bean\nOrigin']

X_train.head()
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X_train['Cocoa\nPercent'].values.reshape(-1, 1))

X_train['Cocoa\nPercent'] = sc.transform(X_train['Cocoa\nPercent'].values.reshape(-1, 1))
X_test['Cocoa\nPercent'] = sc.transform(X_test['Cocoa\nPercent'].values.reshape(-1, 1))

X_train.head()
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

#parameter combinations to try
param_grid = {'n_estimators': [10, 30, 50, 90], 
              'max_depth': [5, 10, 20, None]
             }

regr = RandomForestRegressor()

#fitting the model to each combination in the grid
model = GridSearchCV(regr, param_grid)
#fining the best parameters based on the search grid
model.fit(np.matrix(X_train), y_train)

#pulling the fitted model on the best settings so we can see the variable importances
regr = model.best_estimator_

print(model.best_score_)
#finding the indices that would sort the array 
sorted_indices = np.argsort(regr.feature_importances_)

#finding the most important features and associated importance
variables = regr.feature_importances_[sorted_indices]
importance_rating = X_train.columns.values[sorted_indices]

importances = pd.DataFrame({'variable':variables, 'importance':importance_rating})
importances.tail(10)
mask = df['Specific Bean Origin\nor Bar Name'].apply(lambda x: 'honduras' in str(x).lower())

print("Number of bars for Honduras: " + str(df[mask].shape[0]))
df[mask]
print("Number of bars for Soma: " + str(df[df['Company\xa0\n(Maker-if known)'] == 'Soma'].shape[0]))
print("Average Soma bar rating: " + str(df[df['Company\xa0\n(Maker-if known)'] == 'Soma']['Rating'].mean()))
print("Number of bars for 887: " + str(df[df['REF'] == '887'].shape[0]))
print("Average bar rating: " + str(df[df['REF'] == '887']['Rating'].mean()))
mask = df['Specific Bean Origin\nor Bar Name'].apply(lambda x: 'del toro' in str(x).lower())

print("Number of bars for del toro: " + str(df[mask].shape[0]))
df[mask]
from sklearn.linear_model import Ridge

param_grid = {'alpha': [0.001, 0.01, 1, 3, 5, 10]
             }

regr = Ridge()

model = GridSearchCV(regr, param_grid)
#fining the best parameters based on the search grid
model.fit(np.matrix(X_train), y_train)

#pulling the fitted model on the best settings so we can see the variable importances
regr = model.best_estimator_

print(model.best_score_)
#finding the indices that would sort the array 
sorted_indices = np.argsort(regr.coef_)

#finding the most important features and associated importance
variables = regr.coef_[sorted_indices]
importance_rating = X_train.columns.values[sorted_indices]

importances = pd.DataFrame({'variable':variables, 'coefficient':importance_rating})
print("Total non zero coefficients: " + str(len(importances[importances['coefficient'] != 0.0])))
importances.head()
print("Number of bars for Callebaut: " + str(df[df['Company\xa0\n(Maker-if known)'] == 'Callebaut'].shape[0]))
df[df['Company\xa0\n(Maker-if known)'] == 'Callebaut']
importances.tail()
print("Number of bars for Amedei: " + str(df[df['Company\xa0\n(Maker-if known)'] == 'Amedei'].shape[0]))
df[df['Company\xa0\n(Maker-if known)'] == 'Amedei']
print("Number of bars for 111: " + str(df[df['REF'] == '111'].shape[0]))
df[df['REF'] == '111']
print("Number of bars for Patric: " + str(df[df['Company\xa0\n(Maker-if known)'] == 'Patric'].shape[0]))
df[df['Company\xa0\n(Maker-if known)'] == 'Patric']
print("Number of bars for Cacao Sampaka: " + str(df[df['Company\xa0\n(Maker-if known)'] == 'Cacao Sampaka'].shape[0]))
df[df['Company\xa0\n(Maker-if known)'] == 'Cacao Sampaka']