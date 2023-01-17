# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
DATA_PATH = '/kaggle/input/rs6-attrition-predict/'

train = pd.read_csv(f'{DATA_PATH}/train.csv')

test = pd.read_csv(f'{DATA_PATH}/test.csv')
len(train), len(test)
train.head()
test.head()
train.columns
train.YearsAtCompany.unique()
id_col = 'user_id'

target_col = 'Attrition'



digital_cols = ['Age', 'DailyRate', 'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike',

                'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

category_cols = ['BusinessTravel', 'Department', 'DistanceFromHome', 'Education', 'EducationField',

                'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'JobInvolvement', 'JobLevel',

                'JobRole', 'JobSatisfaction', 'MaritalStatus', 'Over18', 'OverTime',

                'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel', 'TrainingTimesLastYear',

                'WorkLifeBalance']
for col in category_cols:

    nunique_tr = train[col].nunique()

    nunique_te = test[col].nunique()

    na_tr = len(train.loc[train[col].isna()]) / len(train)

    na_te = len(test.loc[test[col].isna()]) / len(test)

    print(f'Col name:{col:30}\tunique cate num in train:{nunique_tr:5}\tunique cate num in train:{nunique_te:5}\tnull sample in train:{na_tr:.2f}\tnull sample in test:{na_te:.2f}')

    
for col in digital_cols:

    min_tr = train[col].min()

    max_tr = train[col].max()

    mean_tr = train[col].mean()

    median_tr = train[col].median()

    std_tr = train[col].std()

    

    min_te = test[col].min()

    max_te = test[col].max()

    mean_te = test[col].mean()

    median_te = test[col].median()

    std_te = test[col].std()

    

    na_tr = len(train.loc[train[col].isna()]) / len(train)

    na_te = len(test.loc[test[col].isna()]) / len(test)

    print(f'Col name:{col:30}')

    print(f'\tIn train data: min value:{min_tr:.2f}\tmax value:{max_tr:.2f}\tmean value:{mean_tr:.2f}\tmedian value:{median_tr:.2f}\tstd value:{std_tr:.2f}\tnan sample rate:{na_tr:.2f}\t')

    print(f'\tIn  test data: min value:{min_te:.2f}\tmax value:{max_te:.2f}\tmean value:{mean_te:.2f}\tmedian value:{median_te:.2f}\tstd value:{std_te:.2f}\tnan sample rate:{na_te:.2f}\t')
train[target_col].unique()
from sklearn.preprocessing import MinMaxScaler



sacalar = MinMaxScaler()

train_digital = sacalar.fit_transform(train[digital_cols])

test_digital = sacalar.transform(test[digital_cols])
from sklearn.preprocessing import LabelEncoder, OneHotEncoder



train_category, test_category = None, None

drop_cols = ['EmployeeNumber', 'Over18', 'StandardHours']

for col in [var for var in category_cols if var not in drop_cols]:

    lbe, ohe = LabelEncoder(), OneHotEncoder()

    

    lbe.fit(pd.concat([train[col], test[col]]).values.reshape(-1, 1))

    train[col] = lbe.transform(train[col])

    test[col] = lbe.transform(test[col])

    

    ohe.fit(pd.concat([train[col], test[col]]).values.reshape(-1, 1))

    oht_train = ohe.transform(train[col].values.reshape(-1, 1)).todense()

    oht_test = ohe.transform(test[col].values.reshape(-1, 1)).todense()

    

    if train_category is None:

        train_category = oht_train

        test_category = oht_test

    else:

        train_category = np.hstack((train_category, oht_train))

        test_category = np.hstack((test_category, oht_test))
train_digital.shape, test_digital.shape, train_category.shape, test_category.shape
train_features = np.hstack((train_digital, train_category))

test_features = np.hstack((test_digital, test_category))

train_features.shape, test_features.shape
target_col_dict = {'Yes': 1, 'No': 0}

train_labels = train[target_col].map(target_col_dict).values

train_labels.shape
from sklearn.linear_model import LinearRegression



clf = LinearRegression()

clf.fit(train_features, train_labels)
predictions = clf.predict(test_features)

predictions.shape
predictions.mean()
sub = test[['user_id']].copy()

sub['Attrition'] = predictions

sub['Attrition'] = sub['Attrition'].apply(lambda x: x if x >=0 else 0.0005)

sub.to_csv('submission.csv', index=False)