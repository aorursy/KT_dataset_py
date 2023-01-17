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
import pandas as pd

from catboost import CatBoostRegressor, CatBoostClassifier
#数据读取

data_Path = '/kaggle/input/ccf-passenger-cars/first_round_training_data.csv'

dataset = pd.read_csv(data_Path)
#数据预处理

all_attrs = ['Parameter1','Parameter2','Parameter3','Parameter4','Parameter5','Parameter6','Parameter7','Parameter8',

            'Parameter9','Parameter10','Attribute1','Attribute2','Attribute3','Attribute4','Attribute5','Attribute6',

            'Attribute7','Attribute8','Attribute9','Attribute10','Quality_label']

unused_attrs = ['Attribute1','Attribute2','Attribute3','Attribute4','Attribute5','Attribute6',

            'Attribute7','Attribute8','Attribute9','Attribute10']

cat_attrs = ['Parameter5','Parameter6','Parameter7','Parameter8',

            'Parameter9','Parameter10']



dataset = dataset.drop(unused_attrs, axis=1)

quality_mapping = {

    'Excellent':1,

    'Good':2,

    'Pass':3,

    'Fail':4

}

dataset['Quality_label'] = dataset['Quality_label'].map(quality_mapping)
X_train = dataset.drop('Quality_label', axis=1)

y_train = dataset['Quality_label']

X_train.head(5)
test = pd.read_csv('/kaggle/input/ccf-passenger-cars/first_round_testing_data.csv')

submit = pd.read_csv('/kaggle/input/ccf-passenger-cars/submit_example.csv')

submit.head(5)
catboost_model = CatBoostClassifier(

    iterations=2000,

    od_type='Iter',

    od_wait=120,

    max_depth=8,

    learning_rate=0.02,

    l2_leaf_reg=9,

    random_seed=2019,

    metric_period=50,

    fold_len_multiplier=1.1,

    loss_function='MultiClass',

    logging_level='Verbose'

)
catboost_model.fit(X_train, y_train)# cat_features=cat_attrs

y_pred = catboost_model.predict(test)



test['Quality_label'] = y_pred



for i in range(0,120):

    lines = test[test['Group'] == i]

    failCount = 0

    passCount = 0

    goodCount = 0

    excelCount = 0

    totalCount = 0

    

    for index in lines.index:

        if lines.loc[index]['Quality_label'] == 1.0:

            failCount = failCount + 1

        if lines.loc[index]['Quality_label'] == 2.0:

            passCount = passCount + 1

        if lines.loc[index]['Quality_label'] == 3.0:

            goodCount = goodCount + 1

        if lines.loc[index]['Quality_label'] == 4.0:

            excelCount = excelCount + 1

        #print('-------------')

        #print(failCount)

        #print(passCount)

        #print(goodCount)

        #print(excelCount)

        totalCount = totalCount + 1

    submit.loc[i,'Excellent ratio'] = excelCount / totalCount

    submit.loc[i,'Good ratio'] = passCount / totalCount

    submit.loc[i,'Pass ratio'] = goodCount / totalCount

    submit.loc[i,'Fail ratio'] = failCount / totalCount

    print('-------------')

    submit.head(5)

submit.to_csv("result3.csv",index=False)