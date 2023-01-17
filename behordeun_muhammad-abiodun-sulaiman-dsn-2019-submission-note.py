import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import numpy.random as nr

import sklearn.model_selection as ms

import seaborn as sns

from sklearn.model_selection import cross_validate

import sklearn.metrics as sklm

from sklearn import linear_model

from sklearn import preprocessing

import math

import scipy.stats as ss

from sklearn import feature_selection as fs



%matplotlib inline





import os

import warnings

warnings.filterwarnings("ignore")



from scipy import stats

from scipy.stats import norm, skew

import datetime as dt

import matplotlib.style as style

# Use a clean stylizatino for our charts and graphs

style.use('fivethirtyeight')

import lightgbm as lgb

import xgboost as xgb



import scipy.stats as stats

from imblearn.over_sampling import SMOTE



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, f1_score,precision_recall_fscore_support,classification_report
features = pd.read_csv("DSN_train.csv")

test = pd.read_csv('DSN_test.csv')

print(test.shape)

print(features.shape)
def new_name(data, cols):

    for col in cols:

        data[col + '_Str'] = data[col].astype(str)

    

cols = ['Year_of_birth', 'Year_of_recruitment']

new_name(features, cols)

new_name(test, cols)
features = features.drop(['Year_of_recruitment', 'Year_of_birth'], axis = 1)

test = test.drop(['Year_of_recruitment', 'Year_of_birth'], axis = 1)



print(features.shape)

print(test.shape)
def value_count(data, cols):

    for col in cols:

        value = data[col].value_counts().count()

        print('\n' + 'For column' + '                                     ' + col)

        print(value)

        

uniq_col = ['Division', 'Qualification', 'Gender', 'Channel_of_Recruitment',

       'Trainings_Attended', 'Year_of_birth_Str', 'Last_performance_score', 'Targets_met', 'Previous_Award',

       'Training_score_average', 'State_Of_Origin', 'Foreign_schooled',

       'Marital_Status', 'Past_Disciplinary_Action',

       'Previous_IntraDepartmental_Movement', 'No_of_previous_employers',

       'Promoted_or_Not', 'Year_of_birth_Str']





value_count(features, uniq_cols)
def plot_bar(data, cols):

    for col in cols:

        ax = data[col].value_counts().plot(kind='bar',

                                    figsize=(10,6),

                                    title="Frequency for each category", color= 'b')

        plt.xlabel(col)

        plt.ylabel('Frequency')

        plt.show()

    

plot_bar(features, colss)
year_categories = {'1991': '1990-2001', '1990': '1990-2001', '1989': '1980-1989', '1992': '1990-2001',

                   '1988': '1980-1989', '1987': '1980-1989', '1993': '1990-2001', '1994': '1990-2001',

                  '1986': '1980-1989', '1985': '1980-1989', '1984': '1980-1989', '1995': '1990-2001', 

                  '1983': '1980-1989', '1982': '1980-1989', '1981': '1980-1989', '1980': '1980-1989',

                  '1996': '1990-2001', '1979': '1970-1979', '1978': '1970-1979', '1977': '1970-1979',

                   '1976': '1970-1979', '1975': '1970-1979', '1997': '1990-2001', '1973': '1970-1979',

                   '1974': '1970-1979', '1971': '1970-1979', '1972': '1970-1979', '1970': '1970-1979',

                   '1969': '1950-1969', '1968': '1950-1969', '1998': '1990-2001', '1966': '1950-1969',

                   '1967': '1950-1969', '1965': '1950-1969', '1964': '1950-1969', '1963': '1950-1969',

                   '1961': '1950-1969', '1962': '1950-1969', '1999': '1990-2001', '2001': '1990-2001',

                   '2000': '1990-2001', '1957': '1950-1969', '1956': '1950-1969', '1955': '1950-1969',

                   '1950': '1950-1969', '1952':'1950-1969', '1958': '1950-1969', '1959': '1950-1969',

                   '1960': '1950-1969'}



features['Year_of_birth_Str'] = [year_categories[x] for x in features['Year_of_birth_Str']]

print(features['Year_of_birth_Str'].value_counts())



test['Year_of_birth_Str'] = [year_categories[x] for x in test['Year_of_birth_Str']]

print(test['Year_of_birth_Str'].value_counts())
year_categories1 = {'1982': '1982-2000', '1985': '1982-2000', '1986': '1982-2000', '1987': '1982-2000', '1988': '1982-2000', '1989': '1982-2000',

                    '1990': '1982-2000', '1991': '1982-2000', '1992': '1982-2000', '1993': '1982-2000', 

                   '1994': '1982-2000', '1995': '1982-2000', '1996': '1982-2000', '1997': '1982-2000', 

                   '1998': '1982-2000', '1999': '1982-2000', '2000': '1982-2000', '2001': '2001-2005', 

                   '2002': '2001-2005', '2003': '2001-2005', '2004': '2001-2005', '2005': '2001-2005', 

                   '2006': '2006-2010', '2007': '2006-2010', '2008': '2006-2010', '2009': '2006-2010', 

                   '2010': '2006-2010', '2011': '2011-2015', '2012': '2011-2015', '2013': '2011-2015', 

                   '2014': '2011-2015', '2015': '2011-2015', '2016': '2016-2018', '2017': '2016-2018', 

                   '2018': '2016-2018'}



features['Year_of_recruitment_Str'] = [year_categories1[x] for x in features['Year_of_recruitment_Str']]

print(features['Year_of_recruitment_Str'].value_counts())



test['Year_of_recruitment_Str'] = [year_categories1[x] for x in test['Year_of_recruitment_Str']]

print(test['Year_of_recruitment_Str'].value_counts())
states = {'ABIA': 'EAST', 'ANAMBRA': 'EAST', 'DELTA': 'EAST', 'BAYELSA': 'EAST',

         'ENUGU': 'EAST', 'EBONYI': 'EAST', 'RIVERS': 'EAST', 'CROSS RIVER': 'EAST',

         'AKWA IBOM': 'EAST', 'IMO': 'EAST', 'KATSINA': 'NORTH', 'KANO': 'NORTH', 'NIGER': 'NORTH',

         'SOKOTO': 'NORTH', 'KANO': 'NORTH', 'KADUNA': 'NORTH', 'BORNO': 'NORTH', 'TARABA':'NORTH',

         'YOBE': 'NORTH', 'ADAMAWA': 'NORTH', 'KEBBI': 'NORTH', 'JIGAWA': 'NORTH', 'ZAMFARA': 'NORTH',

         'KEBBI': 'NORTH', 'PLATEAU': 'NORTH', 'NASSARAWA': 'NORTH', 'FCT': 'NORTH', 'PLATEAU': 'NORTH',

         'BENUE': 'NORTH', 'KOGI':'NORTH', 'BAUCHI': 'NORTH', 'GOMBE': 'NORTH', 'OYO': 'WEST', 'LAGOS': 'WEST', 'OGUN': 'WEST',

         'OSUN': 'WEST', 'EKITI': 'WEST', 'ONDO': 'WEST', 'KWARA': 'WEST', 'EDO': 'WEST'}



features['State_Of_Origin'] = [states[x] for x in features['State_Of_Origin']]

print(features['State_Of_Origin'].value_counts())



test['State_Of_Origin'] = [states[x] for x in test['State_Of_Origin']]

print(test['State_Of_Origin'].value_counts())
label_count = features['Promoted_or_Not'].value_counts()

label_count
Labels = np.array(features['Promoted_or_Not'])
def encode_string(data):

    enc = preprocessing.LabelEncoder()

    enc.fit(data)

    enc_features = enc.transform(data)

    ohe = preprocessing.OneHotEncoder()

    encoded = ohe.fit(enc_features.reshape(-1,1))

    return encoded.transform(enc_features.reshape(-1,1)).toarray()

    

categorical_columns = ['Gender', 'Channel_of_Recruitment', 'State_Of_Origin',

                       'Foreign_schooled', 'Marital_Status', 'Past_Disciplinary_Action', 'Previous_IntraDepartmental_Movement',

                       'Year_of_birth_Str', 'Year_of_recruitment_Str']

Features_enc = encode_string(features['Division'])

for col in categorical_columns:

    temp = encode_string(features[col])

    Features_enc = np.concatenate([Features_enc, temp], axis = 1)

    

print(Features_enc.shape)



test_enc = encode_string(test['Division'])

for col in categorical_columns:

    temps = encode_string(test[col])

    test_enc = np.concatenate([test_enc, temps], axis = 1)

    

print(test_enc.shape)
Features_enc = np.concatenate([Features_enc, np.array(features[['Trainings_Attended', 'Last_performance_score',

                                                              'Targets_met', 'Previous_Award', 'Training_score_average']])], axis = 1)



print(Features_enc.shape)



test_enc = np.concatenate([test_enc, np.array(test[['Trainings_Attended', 'Last_performance_score',

                                                              'Targets_met', 'Previous_Award', 'Training_score_average']])], axis = 1)



print(test_enc.shape)
scaler = preprocessing.StandardScaler().fit(Features_enc[:, 36:])

Features_enc[:, 36:] = scaler.transform(Features_enc[:, 36:])

Features_enc[:, 31:]
test_enc[:, 36:] = scaler.transform(test_enc[:, 36:])

test_enc[:, 36]
birth_encoded = encode_string(features['Year_of_birth_Str'])

birth_encoded_test = encode_string(test['Year_of_birth_Str'])
Features_enc = np.concatenate([Features_enc, birth_encoded], axis = 1)

print(Features_enc.shape)



test_enc = np.concatenate([test_enc, birth_encoded_test], axis = 1)

print(test_enc.shape)
employers = {'0': '0-1', '1': '0-1', '2': '2-3', '3': '2-3',

             '4': 'Greater than 3', '5': 'Greater than 3', 'More than 5': 'Greater than 5'}



features['No_of_previous_employers'] = [employers[x] for x in features['No_of_previous_employers']]

test['No_of_previous_employers'] = [employers[x] for x in test['No_of_previous_employers']]



print(features['No_of_previous_employers'].value_counts())

print(test['No_of_previous_employers'].value_counts())
employers_encodedd = encode_string(features['No_of_previous_employers'])

employers_encoded_testt = encode_string(test['No_of_previous_employers'])
Features_enc = np.concatenate([Features_enc, employers_encodedd], axis = 1)

print(Features_enc.shape)



test_enc = np.concatenate([test_enc, employers_encoded_testt], axis = 1)

print(test_enc.shape)
nr.seed(9988)

indx = range(Features_enc.shape[0])

indx = ms.train_test_split(indx, test_size = 0.30)

x_train = Features_enc[indx[0],:]

y_train = np.ravel(Labels[indx[0]])

x_test = Features_enc[indx[1],:]

y_test = np.ravel(Labels[indx[1]])
import xgboost as xgb
model = xgb.XGBClassifier(objective = 'binary:logistic', n_estimators = 1000, seed = 123, max_depth = 8,

                           learning_rate=0.04, booster = 'gbtree', base_score = 0.7, subsample = 0.8,

                           reg_lambda = 0.03)
eval_set = [(x_train, y_train), (x_test, y_test)]

model.fit(x_train, y_train, eval_metric="auc", eval_set=eval_set, verbose=True, early_stopping_rounds = 1000)
model = model.predict(test_enc)
def score_model(probs, threshold):

    return np.array([1 if x > threshold else 0 for x in probs[:,1]])



def print_metrics(labels, probs, threshold):

    scores = score_model(probs, threshold)

    metrics = sklm.precision_recall_fscore_support(labels, scores)

    conf = sklm.confusion_matrix(labels, scores)

    print('                 Confusion matrix')

    print('                 Score positive    Score negative')

    print('Actual positive    %6d' % conf[0,0] + '             %5d' % conf[0,1])

    print('Actual negative    %6d' % conf[1,0] + '             %5d' % conf[1,1])

    print('')

    print('Accuracy        %0.2f' % sklm.accuracy_score(labels, scores))

    print('AUC             %0.2f' % sklm.roc_auc_score(labels, probs[:,1]))

    print('Macro precision %0.2f' % float((float(metrics[0][0]) + float(metrics[0][1]))/2.0))

    print('Macro recall    %0.2f' % float((float(metrics[1][0]) + float(metrics[1][1]))/2.0))

    print(' ')

    print('           Positive      Negative')

    print('Num case   %6d' % metrics[3][0] + '        %6d' % metrics[3][1])

    print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1])

    print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1])

    print('F1         %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1])

    

probabilities = model.predict_proba(x_test)

print_metrics(y_test, probabilities, 0.5)   
from sklearn.metrics import accuracy_score

print('the test accuracy is :{:.6f}'.format(accuracy_score(y_test, model.predict(x_test))))
my_sample = pd.read_csv('sample_submission2.csv')

my_sample.EmployeeNo = test.EmployeeNo

my_sample.Promoted_or_Not = model

my_sample.to_csv('muhammad1.csv', index = False)