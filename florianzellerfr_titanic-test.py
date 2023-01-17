# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
#rcParams.update({'figure.autolayout': True})

import seaborn as sns
from functools import reduce
import pylab
import scipy.stats as scp

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
%matplotlib inline
# Any results you write to the current directory are saved as output.
#We have there all the functions

def ecdf(x):
    """
        Returns the ECDF of the data
    """
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    return xs, ys

def permutation_sample(data1, data2):
    """
    Generate a permutation sample from two data sets.
    """

    # Concatenate the data sets: data
    data = np.concatenate((data1, data2))

    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2

def draw_bs1d_func(data, func, size=10):
    """Generate bootstrap replicate of 1D data"""
    bs_replicates = np.empty(size)
    for i in range(size):
        bs_replicates[i] = func(np.random.choice(data, len(data)))
    return bs_replicates

def draw_bs_pairs_linreg(x, y, size=10):
    """Perform pairs bootstrap for linear regression."""

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, 1)

    return bs_slope_reps, bs_intercept_reps

def draw_perm_reps(data_1, data_2, func, size=10):
    """
    Generate multiple permutation replicates.
    """

    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(size)

    for i in range(size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)

        # Compute the test statistic
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)

    return perm_replicates

def diff_of_means(data_1, data_2):
    """
    Difference in means of two arrays.
    """

    # The difference of means of data_1, data_2: diff
    diff = np.mean(data_1) - np.mean(data_2)

    return diff

def p_value(test_sample, emp_val):
    """
    Return the p value between an empirical value and a data sample
    """
    p = np.sum(test_sample >= emp_val) / len(test_sample)
    return p

def data_cleaning(df, col_list):
    """
    Return a dataset with the col_list removed and without duplicates
    """
    df2 = df.drop(col_list, axis=1)
    df2.drop_duplicates(inplace=True)
    return df2

def mean_comparison(df, column, category1, category2, measure, any_or_but1 = True, any_or_but2 = True):
    """
    Take a dataframe, a column, two categories and two boolean values whether it is about the categories or anything but the categories
    On the input dataframe, selects a column and slices the dataframe to get all the data with the input category on the input column (if any_or_but is True)
    or all data but the ones with the category (if any_or_true is False) into two Series
    Perfoms a TTest (unequal variances) and bootstraps the two Series to get a distribution of the mean of the measure. Shows them.
    Returns of the TTest 
    """
    if any_or_but1 == True:
        s_1 = df[df[column] == category1][measure]
    elif any_or_but1 == False:
        s_1 = df[df[column] != category1][measure]
    if any_or_but2 == True:
        s_2 = df[df[column] == category2][measure]
    elif any_or_but2 == False:
        s_2 = df[df[column] != category2][measure]

    ttest = scp.ttest_ind(s_1, s_2, equal_var = False)

    boots_1 = draw_bs1d_func(s_1, np.mean, size=10000)
    boots_2 = draw_bs1d_func(s_2, np.mean, size=10000)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,6))
    if any_or_but1 == True:
        sns.distplot(boots_1, label = category1)
    elif any_or_but1 == False:
        sns.distplot(boots_1, label = 'Anything but ' + category1)
    if any_or_but2 == True:
        sns.distplot(boots_2, label = category2)
    elif any_or_but2 == False:
        sns.distplot(boots_2, label = 'Anything but ' + category2)
    plt.xlabel('Mean of the ' + measure)
    plt.title('Distribution of the ' + measure + ' through bootstrapping')
    plt.legend(ncol = 2)

    plt.show()
    plt.close(fig)
    
    return ttest
#Import the datasets
df_train = pd.read_csv("../input/train.csv", index_col = 'PassengerId')
df_test = pd.read_csv('../input/test.csv', index_col = 'PassengerId')

plt.style.use('seaborn')
df_train.info()
df_train.head()
sns.countplot(x='Survived', data=df_train)
print(df_train.Survived.sum()/df_train.Survived.count())
df_train["Pclass"] = df_train["Pclass"].astype('category')
df_train["Sex"] = df_train["Sex"].astype('category')
df_train["Embarked"] = df_train["Embarked"].astype('category')
df_train.describe()
df_train.describe(include=['O'])
df_survived = df_train[df_train['Survived'] == 1]
df_died = df_train[df_train['Survived'] == 0]
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 6))

plt.subplot(121)
plt.hist([df_survived.Fare.dropna(), df_died.Fare.dropna()], rwidth = 0.95, stacked = True, color = ['xkcd:blue', 'xkcd:red'], label = ['Survived', 'Died'])
plt.legend(loc='best')
plt.title("Histogram of the Fare column")

plt.subplot(122)
plt.hist([df_survived.Age.dropna(), df_died.Age.dropna()], rwidth = 0.95, stacked = True, color = ['xkcd:blue', 'xkcd:red'], label = ['Survived', 'Died'])
plt.legend(loc='best')
plt.title("Histogram of the Age column")
df_train.groupby(['Survived', 'Pclass']).mean()
df_train_stat = df_train.copy()
df_train_stat.Survived = df_train_stat.Survived.replace([0,1], ['Died', 'Survived'])
print(mean_comparison(df_train_stat, 'Survived', 'Died', 'Survived', 'Fare', any_or_but1 = True, any_or_but2 = True))
print(mean_comparison(df_train_stat.dropna(), 'Survived', 'Died', 'Survived', 'Age', any_or_but1 = True, any_or_but2 = True))
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# df_train_ml['Age'].fillna(np.random.normal(df_train_ml['Age'].mean(), df_train_ml['Age'].std()), inplace = True)
# df_train_ml['Age_Cat'] = pd.cut(df_train_ml['Age'], bins = 10)
# df_train_ml = pd.get_dummies(df_train, columns = ['Sex', 'Embarked', 'Age_Cat'], drop_first = True)
# df_train_ml.drop(['Cabin', 'Ticket', 'Name', 'Age'], axis = 1, inplace = True)
scaler = StandardScaler()
minmax = MinMaxScaler((-1,1))

df_train_ml = pd.get_dummies(df_train, columns = ['Sex', 'Embarked'], drop_first = True)
df_train_ml['Age'].fillna(np.random.normal(df_train_ml['Age'].mean(), df_train_ml['Age'].std()), inplace = True)
df_train_ml['Age_Cat'] = pd.cut(df_train_ml['Age'], bins = 10, labels = [i for i in range(10)])

df_train_ml['Title'] = df_train_ml['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
#Yes, sadly in 1900 society, women condition seemed mostly determined by their marital status
man_regular = 'Mr'
woman_married = ['Mrs', 'Countess', 'Mme', 'Lady', 'Dona']
man_upper = ['Master','Dr','Major','Sir','Don']
woman_unmarried = ['Miss', 'Mlle', 'Ms']
other = ['Rev','Col','Major','Capt','Jonkheer']

df_train_ml['Title'] = df_train_ml['Title'].replace(man_regular, 1)
df_train_ml['Title'] = df_train_ml['Title'].replace(woman_married, 2)
df_train_ml['Title'] = df_train_ml['Title'].replace(man_upper, 3)
df_train_ml['Title'] = df_train_ml['Title'].replace(woman_unmarried, 4)
df_train_ml['Title'] = df_train_ml['Title'].replace(other, 5)

df_train_ml = pd.get_dummies(df_train_ml, columns = ['Title'], drop_first = True)

df_train_ml.drop(['Cabin', 'Ticket', 'Name', 'Age'], axis = 1, inplace = True)

scaler.fit(df_train_ml[['Fare']])

df_train_ml[['Fare']] = scaler.transform(df_train_ml[['Fare']])

df_train_ml.head()
df_test_ml = pd.get_dummies(df_test, columns = ['Sex', 'Embarked'], drop_first = True)
df_test_ml['Age'].fillna(np.random.normal(df_test_ml['Age'].mean(), df_test_ml['Age'].std()), inplace = True)
df_test_ml['Fare'].fillna(np.random.normal(df_test_ml['Fare'].mean(), df_test_ml['Fare'].std()), inplace = True)
df_test_ml['Age_Cat'] = pd.cut(df_test_ml['Age'], bins = 10, labels = [i for i in range(10)])

df_test_ml['Title'] = df_test_ml['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df_test_ml['Title'] = df_test_ml['Title'].replace(man_regular, 1)
df_test_ml['Title'] = df_test_ml['Title'].replace(woman_married, 2)
df_test_ml['Title'] = df_test_ml['Title'].replace(man_upper, 3)
df_test_ml['Title'] = df_test_ml['Title'].replace(woman_unmarried, 4)
df_test_ml['Title'] = df_test_ml['Title'].replace(other, 5)
print(df_test_ml['Title'])

df_test_ml = pd.get_dummies(df_test_ml, columns = ['Title'], drop_first = True)

df_test_ml.drop(['Cabin', 'Ticket', 'Name', 'Age'], axis = 1, inplace = True)

df_test_ml.head()

scaler.fit(df_test_ml[['Fare']])
df_test_ml[['Fare']] = scaler.transform(df_test_ml[['Fare']])

df_test_ml['PassengerId'] = df_test.index.tolist()
df_test_ml.set_index('PassengerId', inplace=True)
corr = df_train_ml.corr()
sns.heatmap(corr)
X_train, X_test = train_test_split(df_train_ml, test_size=0.25)

used_features =[
    'Pclass'
    ,'Sex_male'
    ,'Age_Cat'
    ,'Fare'
    ,'Title_2'
    ,'Title_3'
    ,'Title_4'
    ,'Title_5'
    ,'Embarked_Q'
    ,'Embarked_S'
]

y_train = X_train["Survived"]
X_train = X_train[used_features].values
from sklearn.ensemble import RandomForestClassifier

# Setup the best parameters selection
n_estimators = [20, 25, 30, 35, 40, 45]
max_depth = [i for i in range (7, 11)]
min_samples_split = [2, 3, 5, 7]
param_grid = {'n_estimators' : n_estimators, 'max_depth' : max_depth, 'min_samples_split' : min_samples_split}

detc_cv = GridSearchCV(RandomForestClassifier(), param_grid, scoring='accuracy', cv=5)
detc_cv.fit(X_train, y_train)
print("Tuned Decision Tree Parameters: {}".format(detc_cv.best_params_)) 
print("Best score is {}".format(detc_cv.best_score_))

#Instantiate the classifier
rfc = RandomForestClassifier(
    n_estimators = detc_cv.best_params_['n_estimators'],
    max_depth = detc_cv.best_params_['max_depth'],
    min_samples_split = detc_cv.best_params_['min_samples_split'],
    min_impurity_decrease = 0.01)

# Train classifier
rfc.fit(
    X_train,
    y_train
)

y_pred = rfc.predict(X_test[used_features])

# Show features importance
for i in range(len(rfc.feature_importances_)):
    print("Importance of {} : {}".format(used_features[i], rfc.feature_importances_[i]))

# Print results
print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
          X_test.shape[0],
          (X_test["Survived"] != y_pred).sum(),
          100*(1-(X_test["Survived"] != y_pred).sum()/X_test.shape[0])
))

print(confusion_matrix(X_test['Survived'], y_pred))
print(classification_report(X_test['Survived'], y_pred))
X_predict_rfc = df_test_ml[used_features]
df_test_ml['SurvivedRFC'] = rfc.predict(X_predict_rfc).astype(int)
df_rfc = df_test_ml['SurvivedRFC'].rename('Survived')
df_rfc.to_csv('SubmissionRFC.csv', header=True)
df_rfc.head()
from sklearn.svm import SVC

# Setup the best parameters selection
c_space = [i for i in range (1, 2, 9)]
gamma = [0.25, 0.5, 1]
degree = [i for i in range (2, 5)]
kernel = ['poly','rbf','linear']
param_grid = {'C' : c_space, 'gamma' : gamma, 'degree' : degree, 'kernel': kernel}

svc_cv = GridSearchCV(SVC(), param_grid, scoring='accuracy', cv=5)
svc_cv.fit(X_train, y_train)
print("Tuned Support Vector Machine Parameters: {}".format(svc_cv.best_params_)) 
print("Best score is {}".format(svc_cv.best_score_))
                
#Instantiate the classifier
svc = SVC(
    C = svc_cv.best_params_['C'],
    gamma = svc_cv.best_params_['gamma'],
    degree = svc_cv.best_params_['degree'],
    kernel = svc_cv.best_params_['kernel'],
    probability = True
         )

# Train classifier
svc.fit(
    X_train,
    y_train
)

y_pred = svc.predict(X_test[used_features])

# Print results
print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
          X_test.shape[0],
          (X_test["Survived"] != y_pred).sum(),
          100*(1-(X_test["Survived"] != y_pred).sum()/X_test.shape[0])
))

print(confusion_matrix(X_test['Survived'], y_pred))
print(classification_report(X_test['Survived'], y_pred))
X_predict_svc = df_test_ml[used_features]
df_test_ml['SurvivedSVC'] = svc.predict(X_predict_svc).astype(int)
df_svc = df_test_ml['SurvivedSVC'].rename('Survived')
df_svc.to_csv('SubmissionSVC.csv', header=True)
df_svc.head()
from sklearn.linear_model import LogisticRegression

# Setup the best parameters selection
c_space = [1, 3, 5, 7, 10]
solvers = ['sag', 'saga', 'newton-cg', 'lbfgs', 'liblinear']
multi_class = ['ovr']
param_grid = {'solver': solvers, 'multi_class': multi_class, 'C' : c_space}

lre_cv = GridSearchCV(LogisticRegression(max_iter = 500), param_grid, scoring='accuracy', cv=5)
lre_cv.fit(X_train, y_train)
print("Tuned Logistic Regression Parameters: {}".format(lre_cv.best_params_)) 
print("Best score is {}".format(lre_cv.best_score_))

#Instantiate the classifier
lre = LogisticRegression(solver = lre_cv.best_params_['solver'], 
                         max_iter = 500, 
                         multi_class = lre_cv.best_params_['multi_class'], 
                         C = lre_cv.best_params_['C'])

# Train classifier
lre.fit(
    X_train,
    y_train
)
y_pred = lre.predict(X_test[used_features])

# Print results
print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
          X_test.shape[0],
          (X_test["Survived"] != y_pred).sum(),
          100*(1-(X_test["Survived"] != y_pred).sum()/X_test.shape[0])
))

print(confusion_matrix(X_test['Survived'], y_pred))
print(classification_report(X_test['Survived'], y_pred))
X_predict_lre = df_test_ml[used_features]
df_test_ml['SurvivedLRE'] = rfc.predict(X_predict_lre).astype(int)
df_lre = df_test_ml['SurvivedLRE'].rename('Survived')
df_lre.to_csv('SubmissionLRE.csv', header=True)
df_lre.head()
from sklearn.ensemble import VotingClassifier

eclf = VotingClassifier(estimators=[('svc', svc), ('rf', rfc), ('lr', lre)], voting='soft')

eclf.fit(X_train, y_train)

X_predict_eclf = df_test_ml[used_features]
df_test_ml['SurvivedECLF'] = rfc.predict(X_predict_eclf).astype(int)
df_eclf = df_test_ml['SurvivedECLF'].rename('Survived')
df_eclf.to_csv('SubmissionECLF.csv', header=True)
df_eclf.head()
import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import Adam, rmsprop, SGD
from keras.callbacks import EarlyStopping
from keras import regularizers

target = to_categorical(df_train_ml.Survived)
used_features =[
    'Pclass'
    ,'Sex_male'
    ,'Age_Cat'
    ,'Title_2'
    ,'Title_3'
    ,'Title_4'
    ,'Title_5'
    ,'Fare'  
    ,'Embarked_Q'
    ,'Embarked_S'
]
predictors = df_train_ml[used_features]
#predictors = df_train.drop('price_range', axis=1)
nb_features = len(predictors.columns)

def get_new_model():
    # Set up the model: model
    model = Sequential()
    model.add(Dense(50, kernel_initializer='uniform', activation='relu', input_shape=(nb_features,)))
    model.add(Dropout(0.2))
    model.add(Dense(50, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    return model
# Create list of learning rates: lr_to_test
lr_to_test = [0.000001, 0.0001, 0.01, 1]

# Loop over learning rates
for lr in lr_to_test:
    print('\n\nTesting model with learning rate: %f\n'%lr )
    
    # Build new model to test, unaffected by previous models
    model = get_new_model()
    
    # Create SGD optimizer with specified learning rate: my_optimizer
    my_optimizer = Adam(lr=lr)
    
    # Compile the model
    model.compile(optimizer = my_optimizer, loss = 'categorical_crossentropy')
    
    # Fit the model
    model.fit(predictors, target, validation_split = 0.3)
model = get_new_model()
my_optimizer = Adam(lr=0.01)
model.compile(my_optimizer, loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])

# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience = 3)

model.fit(predictors, target, epochs = 30, verbose = 1, validation_split = 0.25, callbacks = [early_stopping_monitor])
y_predict = model.predict_classes(df_test_ml[used_features])
df_test_ml['SurvivedNN'] = model.predict_classes(df_test_ml[used_features]).astype(int)
df_nn = df_test_ml['SurvivedNN'].rename('Survived')
df_nn.to_csv('SubmissionNN.csv', header=True)