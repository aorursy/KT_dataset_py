# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objects as go

import plotly.express as px

import numpy as np

import sklearn

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



df = pd.read_csv("../input/india-ml-hiring-av/train.csv")



# Any results you write to the current directory are saved as output.
train_y = df['m13']

df.drop(columns = 'm13',inplace = True)

df.columns
df.dtypes
# months = ['m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12']

# list_nonobj = [df.dtypes.keys()[item] for item in range(len(df.dtypes.keys())) if df.dtypes[item]!='object' and df.dtypes.keys()[item] not in months]

# list_nonobj

# df2 = df[list_nonobj]
# list_nonobj
# scaler = StandardScaler().fit(df2)

# df2 = scaler.transform(df2)

# df2 = pd.DataFrame(df2, columns = list_nonobj)
def bank_dependency(df,status):

    dict = {}



    for i in range(len(train_y)):

        if train_y[i]==status:

            if df['financial_institution'][i] not in dict.keys():

                dict[df['financial_institution'][i]] = 1

            else:

                dict[df['financial_institution'][i]] = dict[df['financial_institution'][i]] + 1



    return dict
dict1 = bank_dependency(df,1)

labels = list(dict1.keys())

values = list(dict1.values())



fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.show()
dict2 = bank_dependency(df,0)

labels = list(dict2.keys())

values = list(dict2.values())



fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.show()
def response_time(df):

    diff = []

    for i in range(len(df['origination_date'])):

        temp = int(df['first_payment_date'][i].split('/')[0]) - int(df['origination_date'][i].split('-')[1])

        diff.append(temp)

    df.drop(columns = ['origination_date','first_payment_date'],inplace = True)

    df['response_time'] = diff

    return df
def one_hot(df,columns,prefix):

#     columns_encoding = ['source','financial_institution','loan_purpose']

    columns_encoding = columns

    df2 = pd.get_dummies(df,columns = columns_encoding, prefix = ['source','fi','lp'])

    return df2
df2 = response_time(df)

df2 = one_hot(df,['source','financial_institution','loan_purpose'],['source','fi','lp'])

df2.columns
i_rates_freq = {}

for i,j in zip(df2['interest_rate'],train_y):

    if j==0:

        if i not in i_rates_freq.keys():

               i_rates_freq[i] = 1

        else:

               i_rates_freq[i] = i_rates_freq[i] + 1



keys = list(sorted(i_rates_freq))

i_rates_f = [i_rates_freq[i] for i in keys]
# keys
def divide_interests(num):

    counter = 1

    sum = 0

    di = {}

    for i in range(len(keys)):

        counter = counter + 1

        if counter == num:

#             print(keys[i])

            counter = 0

            if keys[i] not in di.keys():

                di[keys[i]] = sum

            sum = 0

        sum = sum + i_rates_f[i]



    labels = list(di.keys())

    values = list(di.values())

    

    return labels,values
from plotly.subplots import make_subplots

specs = [[{'type':'domain'}, {'type':'domain'}], [{'type':'domain'}, {'type':'domain'}]]

fig = make_subplots(rows=2, cols=2, specs=specs)

night_colors = ['rgb(56, 75, 126)', 'rgb(18, 36, 37)', 'rgb(34, 53, 101)',

                'rgb(36, 55, 57)', 'rgb(6, 4, 4)']

# Define pie charts



ll = [(1,1),(1,2),(2,1),(2,2)]

vals = [46,92,115,230]

for i in range(len(vals)):

    labels,values = divide_interests(vals[i])

    fig.add_trace(go.Pie(labels=labels, values=values, marker_colors=night_colors), ll[i][0], ll[i][1])





fig = go.Figure(fig)

fig.show()
count = 0

count2 = 0

passed_positive = 0

passed_zero = 0

i=0

for item in df2['co-borrower_credit_score']:

    if item == 0:

        count = count + 1

        if train_y[i] == 1:

            passed_zero = passed_zero + 1

    else:

        count2 = count2 + 1

        if train_y[i] == 1:

            passed_positive = passed_positive + 1

    i = i+1
count,count2
passed_zero,passed_positive
train_y.value_counts()
unique_ids = df2['loan_id']

df2.drop(columns = "loan_id",inplace = True)
df2.columns
retained = []

from scipy.stats import pearsonr

for i in df2.columns:

    corr, _ = pearsonr(df2[i],train_y)

    if abs(corr)>0.015:

        retained.append(i)

        print(i+'\t\t\t\t '+str(corr))
retained
df3= df2[retained]
df3.columns
#Train 



from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score
import imblearn

from imblearn.over_sampling import SMOTE



smt = SMOTE()

X_train, y_train = smt.fit_sample(df3, train_y)
import tensorflow as tf

print(np.__version__)
X_train = pd.DataFrame(data = X_train, columns = df3.columns)
y_train = list(y_train)

np.bincount(y_train)
model = XGBClassifier(learning_rate = 0.065, max_depth = 4)

train_x, test_x, train_Y, test_y = train_test_split(X_train,y_train,train_size=0.7)

model.fit(train_x,train_Y)

xgb_predict = model.predict(test_x)

    

print( "Train Accuracy :: ", f1_score(train_Y, model.predict(train_x)))

print( "Test Accuracy  :: ", f1_score(test_y, xgb_predict))
model
test_X = pd.read_csv("../input/india-ml-hiring-av/test.csv")
months = {"Feb":2,"Mar":3,"Apr":4,"May":5}

diff2 = []

for i in range(len(test_X['origination_date'])):

    temp = months[test_X['first_payment_date'][i].split('-')[0]] - int(test_X['origination_date'][i].split('/')[1])

    diff2.append(temp)

test_X.drop(columns = ['origination_date','first_payment_date'],inplace = True)

test_X['response_time'] = diff2
test_X = one_hot(test_X,['source','financial_institution','loan_purpose'],['source','fi','lp'])
test_ids = test_X['loan_id']

test_X = test_X[retained]
df3.columns
test_X.columns
# model = XGBClassifier()

# model.fit(df2, train_y)

# joblib.dump(model, "xgb1.joblib.dat")

# loaded_model = joblib.load("xgb1.joblib.dat")

# y_pred = model.predict(test_X)

# submission = pd.DataFrame(test_ids)

# submission['m13'] = y_pred

# submission.to_csv("mysubmission2.csv",index = False)

# from sklearn.ensemble import RandomForestClassifier

# rfc = RandomForestClassifier()



# for i in range(len([0,1,2,3])):

#     train_x, test_x, train_Y, test_y = train_test_split(df2,train_y,train_size=0.7)

#     model.fit(train_x,train_Y)

#     rfc_predict = model.predict(test_x)

    

#     print( "Train Accuracy :: ", accuracy_score(train_Y, model.predict(train_x)))

#     print( "Test Accuracy  :: ", accuracy_score(test_y, rfc_predict))



xgb_predict_test = model.predict(test_X)
submission = pd.DataFrame(test_ids)

submission['m13'] = xgb_predict_test

submission.to_csv("mysubmission8.csv",index = False)
submission