import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import random



pd.set_option("display.max_rows", 30)

pd.set_option("display.max_columns", 100)

np.set_printoptions(threshold=np.nan)

sns.set(style="darkgrid")



from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from sklearn.model_selection import train_test_split



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout

from sklearn.metrics import confusion_matrix
df = pd.read_csv('../input/train.csv')
# use pd.cut to bin values into groups

bins = [0, 12, 17, 60, np.inf]

labels = ['child', 'teenager', 'adult', 'elder']

age_groups = pd.cut(df.Age, bins, labels=labels)

df['age_group'] = age_groups
# use unstack to check and reshape part of the data

groups = df.groupby(['age_group', 'Pclass'])

groups.size().unstack()
# extract features from Name

def extractName(s):

    b = s.strip().split(',')[1]

    return b.strip().split()[0]



df['title'] = df.Name.apply(extractName)
# extract cabin category

def extractCab(c):

    return str(c)[0] if type(c) is str else np.nan

df['cabin_class'] = df.Cabin.apply(extractCab)
# plot categorical features

ax = sns.countplot(x="Survived", data=df)

plt.show()

ax = sns.countplot(x="Pclass", hue = 'Survived', data=df)

plt.show()

ax = sns.countplot(x="Sex", hue = 'Survived', data=df)

plt.show()

ax = sns.countplot(x="age_group", hue = 'Survived', data=df)

plt.show()

ax = sns.countplot(x="SibSp", hue = 'Survived', data=df)

plt.show()

ax = sns.countplot(x="Parch", hue = 'Survived', data=df)

plt.show()

ax = sns.countplot(x="Cabin", data=df)

plt.show()

ax = sns.countplot(x="Embarked", hue = 'Survived', data=df)

plt.show()

ax = sns.countplot(x="title", data=df)

plt.show()

ax = sns.countplot(x="cabin_class", data=df)

plt.show()
# use sns.FacetGrid to check data

g = sns.FacetGrid(df, row='Survived', col='Pclass')

g.map(sns.distplot, "Age") # or Fare

plt.show()
# use sns.jointplot to check data

sns.jointplot(data=df, x='Age', y='Fare', kind='reg', color = 'r')

plt.show()
# plot continuous features

ax = sns.distplot(df.Age.dropna())

plt.show()

ax = sns.distplot(df.Fare.dropna())

plt.show()
# heatmap show correlation

sns.heatmap(df.corr(), annot=True, fmt=".2f")

plt.show()
# select data

features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','title']

ori = df.loc[:, features]

x = ori.copy()

y = df.loc[:, 'Survived'].values



# read in test data

testData = pd.read_csv('../input/test.csv')

testData['title'] = testData.Name.apply(extractName)

test = testData.loc[:, features]



def fill_missing_category_by_new_catetory(df, column, trainData = None, fill = 'MISSING'):

    """Fill NaN values for category columns with new category."""

    df.fillna(value = {column:fill},inplace = True)

    

def fill_missing_category_by_random_catetory(df, column, trainData = None):

    """Fill NaN values for category columns with random category from the column."""

    if trainData is None:

        trainData = df

    df[column] = np.where(df[column].isnull(), trainData[column].dropna().sample(len(trainData[column]), replace=True), df[column])     

    

def fill_missing_category_by_most_freq_catetory(df, column, trainData = None):

    """Fill NaN values for category columns with the most frequent category from the column."""

    if trainData is None:

        trainData = df

    df[column] = np.where(df[column].isnull(), trainData[column].dropna().value_counts().idxmax(), df[column])

    

def fill_missing_continous_by_mean(df, column, trainData = None):

    """Fill NaN values for continous columns with mean value."""

    if trainData is None:

        trainData = df

    df[column] = np.where(df[column].isnull(), trainData[column].dropna().mean(), df[column])



def fill_missing_continous_by_median(df, column, trainData = None):

    """Fill NaN values for continous columns with median value."""

    if trainData is None:

        trainData = df

    df[column] = np.where(df[column].isnull(), trainData[column].dropna().median(), df[column])

    



def fill_missing(df, columnNames, methods, td = None):

    """Fill all missing values in a dataframe. 

    Available methods: 

        for category: new, random, freq;

        for continuous: mean, median, freq

    """

    for column, method in zip(columnNames, methods):

        if column not in df.columns:

            raise ValueError("Invalid column name %s" % column)

        

        if method == 'new':

            f = fill_missing_category_by_new_catetory

        elif method == 'random':

            f = fill_missing_category_by_random_catetory

        elif method == 'freq':

            f = fill_missing_category_by_most_freq_catetory

        elif method == 'mean':

            f = fill_missing_continous_by_mean

        elif method == 'median':

            f = fill_missing_continous_by_median

        else:

            raise ValueError("Invalid method %s for column %d" % (method, column))

        f(df, column, trainData = td)

    

fill_missing(x, ['Age','Embarked'],['median','random'])

fill_missing(test, ['Age','Fare'],['median','median'], td = ori)
mer = pd.concat([x,test],keys = ['x','test'])
def labelEncode(df, columnIndex):

    if columnIndex is int:

        columnIndex = [columnIndex]

    oneHotList = []

    oneHotLength = []

    X = df.values

    

    for idx in columnIndex:

        labelencoder = LabelEncoder()

        X[:,idx] = labelencoder.fit_transform(X[:,idx])



        if len(np.unique(X[:,idx])) > 2:

            oneHotList.append(idx)

            oneHotLength.append(len(np.unique(X[:,idx])) + sum(oneHotLength))



    # perform one hot encoder if necessary

    if oneHotList:

        onehotencoder = OneHotEncoder(categorical_features=oneHotList)

        X = onehotencoder.fit_transform(X).toarray()

        oneHotLength = [x-1 for x in oneHotLength]

        X = np.delete(X, oneHotLength, axis=1)

    return X
mer_encode = labelEncode(mer, [1,6,7])

X = mer_encode[range(x.shape[0]),:]

test = mer_encode[x.shape[0]:,:]
# split data

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
# feature scaling

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

test = sc.transform(test)
# define sequence of layers

classifier = Sequential()

classifier.add(Dense(26, kernel_initializer='uniform', activation= 'relu', input_shape = (25,))) 

classifier.add(Dense(26, kernel_initializer='uniform', activation= 'relu'))

classifier.add(Dense(1, kernel_initializer='uniform', activation= 'sigmoid'))

classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# fit train dataset

classifier.fit(X_train, y_train, batch_size= 10, epochs= 100)
# predict on X_test

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)



cm = confusion_matrix(y_test, y_pred)

cm
# predict on test data

test_pred = classifier.predict(test)

test_pred = np.where(test_pred > 0.5,1,0)
id = testData.PassengerId

result = pd.DataFrame(test_pred).iloc[:,0]

finalSubmit = pd.DataFrame(dict(PassengerId = id, Survived = result))
finalSubmit.to_csv('submission2.tsv', index=False)