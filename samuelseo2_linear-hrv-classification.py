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

        

def linear_HRV_Classification_Tree_Algorithm(lf, pnn50):

    if (lf<899.58 and pnn50>.9873) or (lf<277.28 and pnn50<.9873):

        return 'stress'

    if lf>899.58 or (lf>277.28 and pnn50<0.9783):

        return 'no stress'

    return 'uncertain state'



# Any results you write to the current directory are saved as output.

"""

from sklearn import tree

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

test = pd.read_csv('../input/swell-heart-rate-variability-hrv/hrv dataset/hrv dataset/data/final/test.csv')

train = pd.read_csv('../input/swell-heart-rate-variability-hrv/hrv dataset/hrv dataset/data/final/train.csv')

test.head(3)

right = 0;

wrong = 0;

for index, row in test.head(len(test.index)).iterrows():

     # access data using column names

     resultStr = linear_HRV_Classification_Tree_Algorithm(row['LF'], row['pNN50'])

     if resultStr == 'no stress' and row['condition'] == 'no stress':

        right+=1

     elif resultStr == 'stress' and row['condition'] != 'no stress':

        right+=1

     else:

        wrong+=1

print(right)

print(wrong)

print(right/(right+wrong))

"""



'''

import os

import sklearn.pipeline

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

#NOTE: This code is just a quick and dirty proof of concept. Our implimentation in the paper is completely different



def load_train_set():

    #Loading a hdf5 file is much much faster

    return pd.read_csv('../input/swell-heart-rate-variability-hrv/hrv dataset/hrv dataset/data/final/train.csv')

def load_test_set():

    #Loading a hdf5 file is much much faster

    return pd.read_csv('../input/swell-heart-rate-variability-hrv/hrv dataset/hrv dataset/data/final/train.csv')



def simple_model_evaluation():

    select = SelectKBest(k=20)

    train =load_train_set()

    test = load_test_set()

    target = 'condition'

    hrv_features = list(train)

    hrv_features = [x for x in hrv_features if x not in [target]]

    X_train= train[hrv_features]

    y_train= train[target]

    X_test = test[hrv_features]

    y_test = test[target]

    classifiers = [

                    DecisionTreeClassifier(),

                    RandomForestClassifier(n_estimators=100, max_features='log2', n_jobs=-1),

                    SVC(C=20, kernel='rbf'),   

                 ]

    for clf in classifiers:

        name = str(clf).split('(')[0]

        if 'SVC' == name:

            # Normalize the attribute values to mean=0 and variance=1

            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()

            scaler.fit(X_train)

            X_train = scaler.transform(X_train)

            X_test = scaler.transform(X_test)

        clf = RandomForestClassifier()

        steps = [('feature_selection', select),

             ('model', clf)]

        pipeline = sklearn.pipeline.Pipeline(steps)

        pipeline.fit(X_train, y_train)

        y_prediction = pipeline.predict(X_test)

        print("----------------------------{0}---------------------------".format(name))

        print(sklearn.metrics.classification_report(y_test, y_prediction))

        print()

        print()

        

simple_model_evaluation()

'''

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV 

from tpot import TPOTClassifier

%matplotlib inline

import matplotlib.pyplot as plt

from scipy import signal 

import pickle





import sklearn.metrics

from sklearn.model_selection import cross_val_score

from sklearn import svm

import numpy as np

import pandas as pd

from sklearn.metrics import precision_recall_fscore_support 



pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

pd.set_option('display.max_colwidth', -1)

dataframe_hrv = pd.read_csv("../input/swell-heart-rate-variability-hrv/hrv dataset/hrv dataset/data/final/train.csv")

dataframe_hrv = dataframe_hrv.reset_index(drop=True)

display(dataframe_hrv.head(5))
selected_x_columns = ['HR', 'RMSSD', 'pNN50', 'TP', 'VLF', 'LF', 'HF','LF_HF']

X = dataframe_hrv[selected_x_columns]

display(X.head(5))
def fix_stress_labels(df='',label_column='condition'):

    df['condition'] = np.where(df['condition']=='no stress', 0, 1)

    display(df["condition"].unique())

    return df

dataframe_hrv = fix_stress_labels(df=dataframe_hrv)

Y = dataframe_hrv['condition']

display(Y.head(5))
def do_tpot(generations=5, population_size=10,X='',y=''):



    X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.80,test_size=0.20)

    tpot = TPOTClassifier(generations=generations, population_size=population_size, verbosity=2,cv=3)

    tpot.fit(X_train, y_train)

    print(tpot.score(X_test, y_test))

    tpot.export('tpot_pipeline.py')

    return tpot



tpot_classifer = do_tpot(generations=5, population_size=20,X=X,y=Y)
newDataframe_hrv = pd.read_csv("../input/swell-heart-rate-variability-hrv/hrv dataset/hrv dataset/data/final/test.csv")

newdataframe_hrv = dataframe_hrv.reset_index(drop=True)

new_selected_x_columns = ['HR', 'RMSSD', 'pNN50', 'TP', 'VLF', 'LF', 'HF','LF_HF']

newX = newDataframe_hrv[selected_x_columns]

display(newX.head(5))
pred = tpot_classifer.predict_proba(newX)

dfpred = pd.DataFrame(pred)

display(dfpred.head(5))
display(newDataframe_hrv['condition'].head(5))
export_csv1 = dfpred.to_csv ('predicted_stress.csv', index = None, header=True)

export_csv2 = newDataframe_hrv['condition'].to_csv('actual_condition.csv', index = None, header=True)