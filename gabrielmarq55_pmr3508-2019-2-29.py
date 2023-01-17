#General

import pandas as pd

import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt

import statsmodels as sm

import math

from sklearn import metrics



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC



#Classifiers

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import RandomForestClassifier
column_names = ['Id', 'age', 'workclass', 'final_weight', 'education', 'education_num', 'marital_status',

                'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',

                'native_country', 'income']
adult = pd.read_csv("Databases/train_data.csv", names = column_names, na_values='?').drop(0, axis = 0).reset_index(drop = True)
adult.shape
adult.head()
adult.describe()
def count_null_values(data):

    '''

    Return a DataFrame with count of null values

    '''

    

    counts_null = []

    for column in data.columns:

        counts_null.append(data[column].isnull().sum())

    counts_null = np.asarray(counts_null)



    counts_null = pd.DataFrame({'feature': data.columns, 'count.': counts_null,

                                'freq. [%]': 100.0*counts_null/data.shape[0]}).set_index('feature', drop = True)

    counts_null = counts_null.sort_values(by = 'count.', ascending = False)

    

    return counts_null
count_null_values(adult).head()
fig = plt.figure(figsize=(20,15))

cols = 5

rows = math.ceil(float(adult.shape[1]) / cols)

for i, column in enumerate(adult.columns):

    ax = fig.add_subplot(rows, cols, i + 1)

    ax.set_title(column)

    if adult.dtypes[column] == np.object:

        adult[column].value_counts().plot(kind="bar", axes=ax)

    else:

        adult[column].hist(axes=ax)

        plt.xticks(rotation="vertical")

plt.subplots_adjust(hspace=0.7, wspace=0.2)
def work_missing_values(data):

    '''

    Return new data with no missing values for this problem

    '''

    

    aux = data.copy()

    # select index of rows that workclass is nan

    aux_index = aux[aux['workclass'].isna()].index

    

    # fill nan with 'unknown'

    aux['occupation'].loc[aux_index] = 'unknown'

    

    # complete missing of native_country and occupation with most frequent

    cols = ['native_country', 'workclass', 'occupation']

    for col in cols:

        top = aux[col].value_counts().index[0]

        aux[col] = aux[col].fillna(top)

    aux.reset_index(drop = True)

    

    return aux
adult = work_missing_values(adult)
count_null_values(adult).head()
# Encode the categorical features as numbers

def number_encode_features(df):

    result = df.copy()

    encoders = {}

    for column in result.columns:

        if result.dtypes[column] == np.object:

            encoders[column] = LabelEncoder()

            result[column] = encoders[column].fit_transform(result[column])

    return result, encoders



# Calculate the correlation and plot it

encoded_data, _ = number_encode_features(adult)

sns.heatmap(encoded_data.corr(), square=True)

plt.show()

testAdult = pd.read_csv("Databases/test_data.csv", names = column_names, na_values='?').drop(0, axis = 0).reset_index(drop = True)
testAdult.shape
testAdult = work_missing_values(testAdult)
testAdult.head()
Xadult = adult[["age","education_num","capital_gain", "capital_loss", "hours_per_week"]]
Yadult = adult.income
XtestAdult = testAdult[["age","education_num","capital_gain", "capital_loss", "hours_per_week"]]
YtestAdult = testAdult.income
LogClf = LogisticRegression(solver = 'lbfgs', C = 1.0, penalty = 'l2', warm_start =  True)



LogCV = cross_val_score(LogClf, Xadult, Yadult, cv = 10)



LogClf.fit(Xadult, Yadult)



cv_accuracy = [LogCV.mean()]

cv_std = [LogCV.std()]



cv_values = {}

cv_values['LogCV'] = LogCV

print('Logistic Regression CV accuracy: {0:1.4f} +-{1:2.5f}\n'.format(LogCV.mean(), LogCV.std()))
adb = AdaBoostClassifier(n_estimators=50,learning_rate=1)



adbCV = cross_val_score(adb, Xadult, Yadult, cv = 10)



adb.fit(Xadult, Yadult)



cv_accuracy.append(adbCV.mean())

cv_std.append(adbCV.std())

cv_values['adbCV'] = adbCV

print('Adaboost CV accuracy: {0:1.4f} +-{1:2.5f}\n'.format(adbCV.mean(), adbCV.std()))
rf = RandomForestClassifier(n_estimators = 700, max_depth = 12)



rfCV = cross_val_score(rf, Xadult, Yadult, cv = 10)



rf.fit(Xadult, Yadult)



cv_accuracy.append(rfCV.mean())

cv_std.append(rfCV.std())

cv_values['rfCV'] = rfCV

print('Random Forest CV accuracy: {0:1.4f} +-{1:2.5f}\n'.format(rfCV.mean(), rfCV.std()))
Ypred = LogClf.predict(XtestAdult)

savepath = "logPredictions.csv" 

prev = pd.DataFrame(Ypred, columns = ["income"]) 

prev.to_csv(savepath, index_label="Id") 

prev.head()
Ypred = adb.predict(XtestAdult)

savepath = "adbPredictions.csv" 

prev = pd.DataFrame(Ypred, columns = ["income"]) 

prev.to_csv(savepath, index_label="Id") 

prev.head()
Ypred = rf.predict(XtestAdult)

savepath = "rfPredictions.csv" 

prev = pd.DataFrame(Ypred, columns = ["income"]) 

prev.to_csv(savepath, index_label="Id") 

prev.head()