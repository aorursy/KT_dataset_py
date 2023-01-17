# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Shubham Paliwal
import pandas as pd

df_titanic_test = pd.read_csv("../input/titanic-dataset-from-kaggle/test.csv")

df_titanic_train = pd.read_csv("../input/titanic-dataset-from-kaggle/train.csv")
#Setting options to display all rows and columns



pd.options.display.max_columns=None

pd.options.display.max_rows=None

pd.options.display.width=None
df_titanic_train.head()
df_titanic_train.shape
df_titanic_train.isnull().sum()
df_titanic_train['Pclass'].value_counts().plot.barh()
# Here survived counts shows 0 - dead, and 1 = Alive, after the titanic incedent.

df_titanic_train['Survived'].value_counts().plot.bar()
df_titanic_train['Sex'].value_counts()
pd.crosstab(df_titanic_train['Survived'],df_titanic_train['Sex']).plot.bar(figsize = (10,6))

print("Total male and female in titanic",df_titanic_train['Sex'].value_counts())

print("_________________________________")

print("Total Survived male and female in titanic")

print(pd.crosstab(df_titanic_train['Survived'],df_titanic_train['Sex']))
df_titanic_train['Embarked'].value_counts().plot.barh()
pd.crosstab(df_titanic_train['Survived'],df_titanic_train['Embarked']).plot.bar(figsize = (10,6))

print("Total Port of Embarkation in titanic")

print(df_titanic_train['Embarked'].value_counts())

print("_________________________________")

print("Total Survived as per the port of Embarkation in titanic")

print(pd.crosstab(df_titanic_train['Survived'],df_titanic_train['Embarked']))
print("Total Port of Embarkation in titanic")

print(df_titanic_train['Embarked'].value_counts())

print("_________________________________")

print("Total Survived as per the port of Embarkation in titanic (In percent)")

print(pd.crosstab(df_titanic_train['Survived'],df_titanic_train['Embarked'])/df_titanic_train['Embarked'].value_counts()*100)
df_titanic_train.dtypes
df_titanic_train.describe()
df_titanic_train['Embarked'].fillna('S', inplace=True)
df_titanic_train['Embarked'].isnull().sum()
import os 

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.boxplot(df_titanic_train['Age'])
data = [df_titanic_train, df_titanic_test]



for dataset in data:

    mean = df_titanic_train["Age"].mean()

    std = df_titanic_test["Age"].std()

    is_null = dataset["Age"].isnull().sum()

# compute random numbers between the mean, std and is_null

    rand_age = np.random.randint(mean - std, mean + std, size = is_null)

    # fill NaN values in Age column with random values generated

    age_slice = dataset["Age"].copy()

    age_slice[np.isnan(age_slice)] = rand_age

    dataset["Age"] = age_slice

    dataset["Age"] = df_titanic_train["Age"].astype(int)

df_titanic_train["Age"].isnull().sum()
df_titanic_train.isnull().sum()
df_titanic_train['Died'] = 1 - df_titanic_train['Survived']
df_titanic_train.groupby('Sex').agg('sum')[['Survived', 'Died']].plot(kind='bar', 

                                                                      figsize=(15,8),

                                                          stacked=True, colors=['g', 'r']);
figure = plt.figure(figsize=(20, 9))

plt.hist([df_titanic_train[df_titanic_train['Survived'] == 1]['Fare'], df_titanic_train[df_titanic_train['Survived'] == 0]['Fare']], 

         stacked=True, color = ['g','b'],

         bins = 50, label = ['Survived','Dead'])

plt.xlabel('Fare')

plt.ylabel('Number of passengers')

plt.legend();

survived = 'survived'

not_survived = 'not survived'

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(20, 8))

women = df_titanic_train[df_titanic_train['Sex']=='female']

men = df_titanic_train[df_titanic_train['Sex']=='male']

ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)

ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)

ax.legend()

ax.set_title('Female')

ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)

ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)

ax.legend()

_ = ax.set_title('Male')
import re

deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}

data = [df_titanic_train, df_titanic_test]



for dataset in data:

    dataset['Cabin'] = dataset['Cabin'].fillna("U0")

    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

    dataset['Deck'] = dataset['Deck'].map(deck)

    dataset['Deck'] = dataset['Deck'].fillna(0)

    dataset['Deck'] = dataset['Deck'].astype(int)
# we can now drop the cabin feature

df_titanic_train = df_titanic_train.drop(['Cabin'], axis=1)

df_titanic_test = df_titanic_test.drop(['Cabin'], axis=1)
df_titanic_train.head()
data = [df_titanic_train, df_titanic_test]

for dataset in data:

    dataset['Age'] = df_titanic_train["Age"].astype(int)
df_titanic_train = df_titanic_train.drop(['PassengerId'], axis=1)
df_titanic_train = df_titanic_train.drop(['Died'], axis=1)
data = [df_titanic_train, df_titanic_test]

for dataset in data:

    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']

    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0

    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1

    dataset['not_alone'] = dataset['not_alone'].astype(int)

df_titanic_train['not_alone'].value_counts()
data = [df_titanic_train, df_titanic_test]



for dataset in data:

    dataset['Fare'] = dataset['Fare'].fillna(0)

    dataset['Fare'] = dataset['Fare'].astype(int)
df_titanic_train.head()
df_titanic_train = df_titanic_train.drop(['Ticket'], axis=1)

df_titanic_test = df_titanic_test.drop(['Ticket'], axis=1)
Ports = {"S": 0, "C": 1, "Q": 2}

data = [df_titanic_train, df_titanic_test]



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].map(Ports)
df_titanic_train.head()
data = [df_titanic_train, df_titanic_test]

for dataset in data:

    dataset['Age'] = dataset['Age'].astype(int)

    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6

    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6
df_titanic_train['Age'].value_counts()
df_titanic_train = df_titanic_train.drop(['Name'], axis=1)
df_titanic_train.columns
df_titanic_train["Fare"].describe()
data = [df_titanic_train, df_titanic_test]



for dataset in data: 

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3

    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4

    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5

    dataset['Fare'] = dataset['Fare'].astype(int)
df_titanic_train.head()
genders = {"male": 0, "female": 1}

data = [df_titanic_train, df_titanic_test]



for dataset in data:

    dataset['Sex'] = dataset['Sex'].map(genders)
df_titanic_train.head()
## Age times class

# data = [df_titanic_train, df_titanic_train]

#for dataset in data:

#    dataset['Age_Class']= dataset['Age']* dataset['Pclass']
## Fare per person 

# for dataset in data:

#    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1)

 #   dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)

# Let's take a last look at the training set, before we start training the models.

# df_titanic_train.head(10)
#FUNCTIONS TAB

def set_plot_sizes(sml, med, big):

    plt.rc('font', size=sml)          # controls default text sizes

    plt.rc('axes', titlesize=sml)     # fontsize of the axes title

    plt.rc('axes', labelsize=med)    # fontsize of the x and y labels

    plt.rc('xtick', labelsize=sml)    # fontsize of the tick labels

    plt.rc('ytick', labelsize=sml)    # fontsize of the tick labels

    plt.rc('legend', fontsize=sml)    # legend fontsize

    plt.rc('figure', titlesize=big)  # fontsize of the figure title

def add_datepart(df, fldnames, drop=True, time=False, errors="raise"):	

    """add_datepart converts a column of df from a datetime64 to many columns containing

    the information from the date. This applies changes inplace.

    Parameters:

    -----------

    df: A pandas data frame. df gain several new columns.

    fldname: A string or list of strings that is the name of the date column you wish to expand.

        If it is not a datetime64 series, it will be converted to one with pd.to_datetime.

    drop: If true then the original date column will be removed.

    time: If true time features: Hour, Minute, Second will be added.

    Examples:

    ---------

    >>> df = pd.DataFrame({ 'A' : pd.to_datetime(['3/11/2000', '3/12/2000', '3/13/2000'], infer_datetime_format=False) })

    >>> df

        A

    0   2000-03-11

    1   2000-03-12

    2   2000-03-13

    >>> add_datepart(df, 'A')

    >>> df

        AYear AMonth AWeek ADay ADayofweek ADayofyear AIs_month_end AIs_month_start AIs_quarter_end AIs_quarter_start AIs_year_end AIs_year_start AElapsed

    0   2000  3      10    11   5          71         False         False           False           False             False        False          952732800

    1   2000  3      10    12   6          72         False         False           False           False             False        False          952819200

    2   2000  3      11    13   0          73         False         False           False           False             False        False          952905600

    >>>df2 = pd.DataFrame({'start_date' : pd.to_datetime(['3/11/2000','3/13/2000','3/15/2000']),

                            'end_date':pd.to_datetime(['3/17/2000','3/18/2000','4/1/2000'],infer_datetime_format=True)})

    >>>df2

        start_date	end_date    

    0	2000-03-11	2000-03-17

    1	2000-03-13	2000-03-18

    2	2000-03-15	2000-04-01

    >>>add_datepart(df2,['start_date','end_date'])

    >>>df2

    	start_Year	start_Month	start_Week	start_Day	start_Dayofweek	start_Dayofyear	start_Is_month_end	start_Is_month_start	start_Is_quarter_end	start_Is_quarter_start	start_Is_year_end	start_Is_year_start	start_Elapsed	end_Year	end_Month	end_Week	end_Day	end_Dayofweek	end_Dayofyear	end_Is_month_end	end_Is_month_start	end_Is_quarter_end	end_Is_quarter_start	end_Is_year_end	end_Is_year_start	end_Elapsed

    0	2000	    3	        10	        11	        5	            71	            False	            False	                False	                False	                False	            False	            952732800	    2000	    3	        11	        17	    4	            77	            False	            False	            False	            False	                False	        False	            953251200

    1	2000	    3	        11	        13	        0	            73	            False	            False	                False	                False               	False           	False           	952905600     	2000       	3	        11      	18  	5           	78          	False	            False           	False           	False               	False          	False           	953337600

    2	2000	    3	        11	        15	        2           	75          	False           	False               	False               	False               	False               False           	953078400      	2000    	4          	13      	1   	5           	92          	False           	True            	False           	True                	False          	False           	954547200

    """

    if isinstance(fldnames,str): 

        fldnames = [fldnames]

    for fldname in fldnames:

        fld = df[fldname]

        fld_dtype = fld.dtype

        if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):

            fld_dtype = np.datetime64



        if not np.issubdtype(fld_dtype, np.datetime64):

            df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True, errors=errors)

        targ_pre = re.sub('[Dd]ate$', '', fldname)

        attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',

                'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']

        if time: attr = attr + ['Hour', 'Minute', 'Second']

        for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())

        df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9

        if drop: df.drop(fldname, axis=1, inplace=True)

            

            

            

            

def train_cats(df):

    """Change any columns of strings in a panda's dataframe to a column of

    categorical values. This applies the changes inplace.

    Parameters:

    -----------

    df: A pandas dataframe. Any columns of strings will be changed to

        categorical values.

    Examples:

    ---------

    >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})

    >>> df

       col1 col2

    0     1    a

    1     2    b

    2     3    a

    note the type of col2 is string

    >>> train_cats(df)

    >>> df

       col1 col2

    0     1    a

    1     2    b

    2     3    a

    now the type of col2 is category

    """

    for n,c in df.items():

        if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()



def rf_feat_importance(m, df):

    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}

                       ).sort_values('imp', ascending=False)







def proc_df(df, y_fld=None, skip_flds=None, ignore_flds=None, do_scale=False, na_dict=None,

            preproc_fn=None, max_n_cat=None, subset=None, mapper=None):

  

    if not ignore_flds: ignore_flds=[]

    if not skip_flds: skip_flds=[]

    if subset: df = get_sample(df,subset)

    else: df = df.copy()

    ignored_flds = df.loc[:, ignore_flds]

    df.drop(ignore_flds, axis=1, inplace=True)

    if preproc_fn: preproc_fn(df)

    if y_fld is None: y = None

    else:

        if not is_numeric_dtype(df[y_fld]): df[y_fld] = pd.Categorical(df[y_fld]).codes

        y = df[y_fld].values

        skip_flds += [y_fld]

    df.drop(skip_flds, axis=1, inplace=True)



    if na_dict is None: na_dict = {}

    else: na_dict = na_dict.copy()

    na_dict_initial = na_dict.copy()

    for n,c in df.items(): na_dict = fix_missing(df, c, n, na_dict)

    if len(na_dict_initial.keys()) > 0:

        df.drop([a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis=1, inplace=True)

    if do_scale: mapper = scale_vars(df, mapper)

    for n,c in df.items(): numericalize(df, c, n, max_n_cat)

    df = pd.get_dummies(df, dummy_na=True)

    df = pd.concat([ignored_flds, df], axis=1)

    res = [df, y, na_dict]

    if do_scale: res = res + [mapper]

    return res



def fix_missing(df, col, name, na_dict):

    

    if is_numeric_dtype(col):

        if pd.isnull(col).sum() or (name in na_dict):

            df[name+'_na'] = pd.isnull(col)

            filler = na_dict[name] if name in na_dict else col.median()

            df[name] = col.fillna(filler)

            na_dict[name] = filler

    return na_dict



def numericalize(df, col, name, max_n_cat):



    if not is_numeric_dtype(col) and ( max_n_cat is None or len(col.cat.categories)>max_n_cat):

        df[name] = pd.Categorical(col).codes+1

        

def get_sample(df,n):

    """ Gets a random sample of n rows from df, without replacement.

    Parameters:

    -----------

    df: A pandas data frame, that you wish to sample from.

    n: The number of rows you wish to sample.

    Returns:

    --------

    return value: A random sample of n rows of df.

    Examples:

    ---------

    >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})

    >>> df

       col1 col2

    0     1    a

    1     2    b

    2     3    a

    >>> get_sample(df, 2)

       col1 col2

    1     2    b

    2     3    a

    """

    idxs = sorted(np.random.permutation(len(df))[:n])

    return df.iloc[idxs].copy()



def draw_tree(t, df, size=10, ratio=0.6, precision=0):

    """ Draws a representation of a random forest in IPython.

    Parameters:

    -----------

    t: The tree you wish to draw

    df: The data used to train the tree. This is used to get the names of the features.

    """

    s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True,

                      special_characters=True, rotate=True, precision=precision)

    IPython.display.display(graphviz.Source(re.sub('Tree {',

       f'Tree {{ size={size}; ratio={ratio}', s)))



def set_rf_samples(n):

    """ Changes Scikit learn's random forests to give each tree a random sample of

    n random rows.

    """

    forest._generate_sample_indices = (lambda rs, n_samples:

        forest.check_random_state(rs).randint(0, n_samples, n))

    

def reset_rf_samples():

    """ Undoes the changes produced by set_rf_samples.

    """

    forest._generate_sample_indices = (lambda rs, n_samples:

        forest.check_random_state(rs).randint(0, n_samples, n_samples))

    

def parallel_trees(m, fn, n_jobs=8):

        return list(ProcessPoolExecutor(n_jobs).map(fn, m.estimators_))

df_titanic_test = df_titanic_test.drop(['PassengerId'], axis=1)

df_titanic_test = df_titanic_test.drop(['Name'], axis=1)
X_train = df_titanic_train.drop("Survived", axis=1)

Y_train = df_titanic_train["Survived"]

X_test = df_titanic_test.copy()
# Algorithms

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB
SGD = linear_model.SGDClassifier(max_iter=5,tol=None)      # max_inter = Max number of iteration]

SGD.fit(X_train, Y_train)

Y_Prediction = SGD.predict(X_test)

SGD.score(X_train, Y_train)



Accuracy_SGD = round(SGD.score(X_train, Y_train) * 100, 2)

Accuracy_SGD
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)



Y_predict = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)



Y_pred = logreg.predict(X_test)



acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
#KNN 

knn = KNeighborsClassifier(n_neighbors = 3) 

knn.fit(X_train, Y_train)  

Y_pred = knn.predict(X_test)  

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn
gaussian = GaussianNB() 

gaussian.fit(X_train, Y_train)  

Y_pred = gaussian.predict(X_test)  

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian
linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)



Y_pred = linear_svc.predict(X_test)



acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

acc_linear_svc
decision_tree = DecisionTreeClassifier() 

decision_tree.fit(X_train, Y_train)  

Y_pred = decision_tree.predict(X_test)  

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
results = pd.DataFrame({'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression','Random Forest', 'Naive Bayes', 

                                  'Stochastic Gradient Decent','Decision Tree'],

                        'Score': [acc_linear_svc, acc_knn, acc_log, acc_random_forest, acc_gaussian, 

                                  Accuracy_SGD, acc_decision_tree]})

result_df = results.sort_values(by='Score', ascending=False)

result_df = result_df.set_index('Score')

result_df.head(10)