def getfeat(df):

    """

    Returns lists of numeric & categorical features

    """

    numfeat, catfeat = list(df.select_dtypes(include=np.number)), list(df.select_dtypes(exclude=np.number))

    return numfeat, catfeat
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



plt.rcParams["figure.figsize"] = (8,6) ## preferred size of plots

plt.style.use('fivethirtyeight')



import seaborn as sns

sns.set(style="darkgrid") ## beautiful plots



from scipy import stats



import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sample = pd.read_csv('../input/sample_submission.csv')
train.info()
## Dropping the redundant column of 'Loan_ID'

train.drop('Loan_ID', axis=1, inplace=True)

test.drop('Loan_ID', axis=1, inplace=True)
## Seperating features by type and target column

target = 'Interest_Rate'

numfeat, catfeat = getfeat(train)

numfeat.remove(target)



temp = 0    ## a temporary variable for random usage
train.shape, test.shape
## Mapping target variable to categories as integer type won't give best results. 

match = {1:'Cat_1', 2:'Cat_2', 3:'Cat_3'}  

unmatch = {'Cat_1': 1, 'Cat_2': 2, 'Cat_3':3} ## to be used during submission to unmap

train[target] = train[target].map(match)
train[target].value_counts().plot(kind='bar', figsize=(8,6))

plt.show()  
# drop_indices = np.random.choice(train[train[target]=='Cat_2'].index, 30000, replace=False)

# train = train.drop(drop_indices).reset_index()

# drop_indices = np.random.choice(train[train[target]=='Cat_3'].index, 25000, replace=False)

# train = train.drop(drop_indices).reset_index()

# train[target].value_counts()

# # train.head()
train_labels = train[target]

df = pd.concat([train, test], keys = ['train', 'test'])

df
# for temp in range(len(df['Loan_Amount_Requested'])):

#     df['Loan_Amount_Requested'][temp] = int(df['Loan_Amount_Requested'][temp].replace(',', ''))

# df['Loan_Amount_Requested'] = df['Loan_Amount_Requested'].astype('int64')



## Fixing 'Loan_Amount_Requested' 

df['Loan_Amount_Requested'] = df['Loan_Amount_Requested'].str.replace(",","").astype('int64')
## Creating new features before moving towards transformation

df['Loan_Amount_Requested_range'] = pd.cut(df['Loan_Amount_Requested'], 

                                             bins= range(0, df['Loan_Amount_Requested'].max(), 2000)).astype('object')

df['Loan_Amount_Requested_category'] = pd.cut(df['Loan_Amount_Requested'], 

                                                 bins=[int(x) for x in np.linspace(0,df['Loan_Amount_Requested'].max(), 8)], 

                                                 labels=['Very Low', 'Low', 'Medium-Low', 'Medium', 'Medium-High', 'High', 'Very High']).astype('object')
sns.countplot(x = 'Loan_Amount_Requested_category', data = df.xs('train'), hue = 'Interest_Rate')

plt.show()
sns.distplot(df['Loan_Amount_Requested'], fit = stats.norm)

plt.legend(['Normal fit', 'Skew: {:.4f}'.format(df['Loan_Amount_Requested'].skew())])

plt.show()
temp = 'Length_Employed'

df[temp].isnull().sum()/len(df[temp])*100
df[temp].unique()
sns.countplot(x=temp, data = df)

plt.xticks(rotation=45)

plt.show()
## My guesses for null values --> either unemployed i.e. a housewife or student in that case '0 year' 

##                            --> fill with mode 



df[temp].fillna('0 year', inplace=True)
temp = 'Home_Owner'

df[temp].isnull().sum()/len(df)*100
df[temp].value_counts()
sns.countplot(temp, data=df, hue='Length_Employed')

plt.show()
## My guesses for null values --> people let go of filling just because none of it applies

##                            --> fill with mode 



df[temp].fillna('None', inplace = True)
temp = 'Annual_Income'

df[temp].isnull().sum()/len(df)*100
df[temp].describe()
## My guesses for null values --> fill with median as data has an extreme outlier

##                            -->  

df[temp].fillna(df[temp].median(), inplace=True)
df['Annual_Income_range'] = pd.cut(df[temp], bins=[int(x) for x in np.linspace(0, df[temp].max(), 15)]).astype('object')

df['Annual_Income_cat'] = pd.cut(df[temp], bins=[int(x) for x in np.linspace(0, df[temp].max(), 8)],

                                 labels = ['Very Low', 'Low', 'Medium-Low', 'Medium', 'Medium-High', 'High', 'Very High']).astype('object')
f,a = plt.subplots(1,2,figsize=(16,6))



sns.distplot(df[temp], fit=stats.norm, ax=a[0])

a[0].set_title("Distribution with Skew: {:.4f}".format(df[temp].skew()))



tempdf = pd.Series(stats.boxcox(1+df[temp], lmbda=0))



sns.distplot(tempdf, fit=stats.norm, ax=a[1])

a[1].set_title("Transformed distribution with Skew: {:.4f}".format(tempdf.skew()))



df[temp] = stats.boxcox(1+df[temp], lmbda=0)

plt.show()
temp = 'Income_Verified'

df[temp].isnull().sum()
df[temp].unique()
temp = 'Purpose_Of_Loan'

df[temp].isnull().sum()
df[temp].unique()
sns.countplot(df[temp])

plt.xticks(rotation=90)

plt.show()
temp = 'Debt_To_Income'

df[temp].isnull().sum()
df[temp].describe(percentiles=[0.20,0.40,0.60,0.9])
df['Debt'] = (df[temp]*df['Annual_Income']).astype('float')
f,a = plt.subplots(1,2,figsize=(16,6))



sns.distplot(df[temp], fit=stats.norm, ax=a[0])

a[0].set_title("Distribution with Skew: {:.4f}".format(df[temp].skew()))



tempdf = pd.Series(stats.boxcox(1+df[temp], lmbda=0.817))  ## I prefer setting lambda values manually although 'boxcox_normmax' gives good results



sns.distplot(tempdf, fit=stats.norm, ax=a[1])

a[1].set_title("Transformed distribution with Skew: {:.4f}".format(tempdf.skew()))



df[temp] = stats.boxcox(1+df[temp], lmbda=0.817) ##    ONE TIME RUN

plt.show()
temp = 'Inquiries_Last_6Mo'

df[temp].isnull().sum()
sorted(df[temp].unique())
df['Requirement'] = pd.cut(df[temp], bins=[0,1,5,8], labels = ['least', 'wanted', 'highly wanted']).astype('object')
temp = 'Months_Since_Deliquency'

df[temp].isnull().sum()/len(df)*100
df.drop(temp, axis=1,inplace=True)
temp = 'Number_Open_Accounts'

print(df[temp].isnull().sum(), sorted(df[temp].unique()))
df['Number_Open_Accounts_cat'] = pd.cut(df[temp], bins = [0,9,20,45,85], labels=['few', 'ok_ok', 'many', 'far_too_many']).astype('object')
f,a = plt.subplots(1,2,figsize=(16,6))



sns.distplot(df[temp], fit=stats.norm, ax=a[0])

a[0].set_title("Distribution with Skew: {:.4f}".format(df[temp].skew()))



tempdf = pd.Series(stats.boxcox(1+df[temp], lmbda=0.15))  ## I prefer setting lambda values manually although 'boxcox_normmax' gives good results



sns.distplot(tempdf, fit=stats.norm, ax=a[1])

a[1].set_title("Transformed distribution with Skew: {:.4f}".format(tempdf.skew()))



df[temp] = stats.boxcox(1+df[temp], lmbda=0.15) ##    ONE TIME RUN

plt.show()
temp = 'Total_Accounts'

df[temp].isnull().sum()
df[temp].describe()
df['Total_Accounts_cat'] = pd.cut(df[temp], bins = [0,9,20,45,85,156], labels=['few', 'ok_ok', 'many', 'very_many','far_too_many']).astype('object')
f,a = plt.subplots(1,2,figsize=(16,6))



sns.distplot(df[temp], fit=stats.norm, ax=a[0])

a[0].set_title("Distribution with Skew: {:.4f}".format(df[temp].skew()))  ## I prefer setting lambda values manually although 'boxcox_normmax' gives good results



tempdf = pd.Series(stats.boxcox(1+df[temp], lmbda=0.348))



sns.distplot(tempdf, fit=stats.norm, ax=a[1])

a[1].set_title("Transformed distribution with Skew: {:.4f}".format(tempdf.skew()))



df[temp] = stats.boxcox(1+df[temp], lmbda=0.348) ##    ONE TIME RUN

plt.show()
temp = 'Gender'

df[temp].isnull().sum()
import lightgbm as lgb

from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, plot_confusion_matrix

from sklearn.model_selection import RandomizedSearchCV
## Rebuilding test and training set



numfeat, catfeat = getfeat(df)

catfeat.remove(target) 



X = df.xs('train').drop(target, axis=1)

y = df.xs('train')[target]



X_test = df.xs('test').drop(target, axis=1)
X.shape, y.shape, X_test.shape
## Encoding categorical variables



X = pd.get_dummies(X, columns=catfeat)

X_test = pd.get_dummies(X_test, columns=catfeat)



from sklearn import preprocessing as prep



## Scaling for better results on NN 



minmaxscalar = prep.MinMaxScaler()  

X = pd.DataFrame(minmaxscalar.fit_transform(X), columns = X.columns)

X_test = pd.DataFrame(minmaxscalar.fit_transform(X_test), columns = X_test.columns)



X.shape, y.shape
## Before submitting dataframe to lgbm, replacing all non-alphanumeric characters in the column names

## The problem is that few of the features contain a comma (,) which lgbm doesn't accept anymore.



X.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in X.columns]

X_test.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in X_test.columns]



X_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=100)
learning_rate = [float(x) for x in np.linspace(0.005, 0.05, 1000)]

n_estimators = [int(x) for x in range(500,5000,500)]

max_depth = [int(x) for x in range(2,30,2)]

num_leaves = [int(x) for x in range(1,5000,100)]

bagging_fraction = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]



early_stopping_rounds = 50

min_data_in_leaf = 100

lambda_l1 = 0.5



lgbgrid = {'num_leaves': num_leaves,

           'bagging_fraction': bagging_fraction,

           'early_stopping_rounds': early_stopping_rounds,

           'min_data_in_leaf': min_data_in_leaf,

           'lambda_l1': lambda_l1,

           'max_depth': max_depth,

           'n_estimators': n_estimators,

           'learning_rate': learning_rate

           }
# lgbm = lgb.LGBMClassifier(objective='multiclass', random_state=5, bagging_freq=1)

# clf = RandomizedSearchCV(lgbm, lgbgrid, cv=5, n_iter=300, n_jobs=1)

# search = clf.fit(X_train,y_train)

# search.best_params_
scores = {'last':0,'current':0}
lgbm = lgb.LGBMClassifier(boost = 'dart', objective='multiclass', num_class = 3,

                          learning_rate = 0.075, max_bin = 470,

                          max_depth = 38, min_data_in_leaf = 60, num_leaves = 3501,

                          lambda_l1 = 0.5                          

                          )



lgbm.fit(X_train, y_train,eval_set = (x_test,y_test), early_stopping_rounds=100, verbose=200)



y_pred = lgbm.predict(x_test)



scores['current'] = f1_score(y_test, y_pred, average='weighted')

print('previous score = ', scores['last'],

      '\ncurrent score = ', scores['current'])

scores['last'] = scores['current']
plot_confusion_matrix(lgbm, x_test, y_test, normalize='pred')

plt.show()
from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder



## encoding y variables for multiclass NN prediction

encoded_y = LabelEncoder().fit_transform(y_train)

dummy_y = np_utils.to_categorical(encoded_y)



encoded_y_test = LabelEncoder().fit_transform(y_test)

dummy_y_test = np_utils.to_categorical(encoded_y_test)
## Creating metrics to be used in epochs as the evaluation metric for the competition is F1-score



from keras import backend as K



def recall_m(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())

    return recall



def precision_m(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision



def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))

model = Sequential()

model.add(Dense(12, input_dim=92 ,activation='relu'))

model.add(Dense(24, activation='relu'))

model.add(Dense(3, activation='softmax'))

## using softmax as final layer activation layer as it is a multiclass prediction. Sigmoid would've been good had it been a binary classification.



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[f1_m])



model.fit(X_train,dummy_y,epochs=100, validation_split=0.1,batch_size=128, verbose=100)

f1_sc = model.evaluate(x_test, dummy_y_test, verbose=0, batch_size=32)

f1_sc[1]
# sub = pd.Series(lgbm.predict(X_test))

# sample.iloc[:,1] = sub.map(unmatch)

# sample.to_csv('../input/submission.csv', index=False)