import matplotlib.pyplot as plt, pandas as pd, numpy as np

from sklearn.cross_validation import train_test_split

from sklearn.preprocessing import LabelEncoder

# import classification algorithms

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

%matplotlib inline
df = pd.read_csv('../input/tips.csv',index_col='UID',encoding = "ISO-8859-1")

print(df.head())
print(df.dtypes)
# collecting categorical varibale names 

cat_var = df.dtypes.loc[df.dtypes=='object'].index

print(cat_var)
# check for unique variables in cat_var

df[cat_var].apply(lambda x: len(x.unique()))
print(df['Bet Type'].value_counts())

print('~'*30)

print(df['Bet Type'].value_counts()/df['Bet Type'].shape[0]*100)
# Lose vs Win ratio

print(df['Result'].value_counts())

print('~'*30)

print(df['Result'].value_counts()/df.Result.shape[0]*100)
# Tipster active vs unactive

print(df.TipsterActive.value_counts()/df.TipsterActive.shape[0]*100)
print(df.Odds.describe())
# Tipster's performance

cross_tab = pd.crosstab(df['Tipster'], df['Result'], margins=True)

cross_tab.iloc[:-1,:-1].plot(kind='bar', stacked=True, color=['red', 'blue'], grid=False)
# percentage conversion

def perConvert(ser):

    return ser/float(ser[-1])

cross_tab1 = cross_tab.apply(perConvert, axis=1)

cross_tab1.iloc[:-1,:-1].plot(kind='bar', stacked=True, color=['red', 'blue'], grid=False)
# checking for outliers in continous data

df.plot.scatter('ID', 'Odds')
df.apply(lambda x: sum(x.isnull()))

# dataset looks clear of any null values, good here!
# since we cannot plug categorical values we need to convert it to numerical values 

# this is taken care by the LabelEncoder() function from sklearm.preprocessing

le = LabelEncoder()

for var in cat_var:

    df[var] = le.fit_transform(df[var])
print(cat_var)

print(df.head())
# creating a seperate prediction set for post cross validation

df_X_test = df.iloc[-20:,:-2] 

df_y_test = df.iloc[-20:,-2:-1]

df_X_test = df_X_test[['Tipster', 'Track', 'Horse', 'Bet Type', 'Odds']]

df_y_test = df_y_test['Result'] 



print(df_X_test.shape, df_y_test.shape)

# allocating data for cross validation train test split

df_train = df.iloc[:-20]

X = df_train[['Tipster', 'Track', 'Horse', 'Bet Type', 'Odds']]

# I choose to leave out column 'TipsterActive' (Bool variable)

y = df_train.Result.values

print(X.shape, y.shape)

print(X.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# list of classifiers

classifers = [GaussianNB(), LogisticRegression(n_jobs=-1), DecisionTreeClassifier(min_samples_leaf=5,min_samples_split=17,random_state=1), KNeighborsClassifier(n_neighbors=5, leaf_size=50, p=3)]

for cl in classifers:

    clf = cl

    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)*100

    print('Accuracy of %r Classifier = %2f' % (cl, accuracy) + ' %')

    print('\n')

           