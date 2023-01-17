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
train = pd.read_csv('/kaggle/input/mobile-price-range-prediction-is2020-v2/train_data.csv')

test = pd.read_csv('/kaggle/input/mobile-price-range-prediction-is2020-v2/test_data.csv')

submissions = pd.read_csv('/kaggle/input/mobile-price-range-prediction-is2020-v2/sample_submission.csv')



train_clean = train.drop(columns=['id'])

test_clean = test.drop(columns=['id'])



x_train = train_clean.drop(columns=['price_range'])

y_train = train_clean[['price_range']]



x_test = test_clean



from sklearn.preprocessing import StandardScaler

x_train_scale = StandardScaler().fit_transform(x_train)



x_test_scale = StandardScaler().fit_transform(x_test)





from sklearn.model_selection import cross_val_score



y = np.array(y_train)

y = y.ravel()



from sklearn.linear_model import LogisticRegression

model = LogisticRegression()



from sklearn.model_selection import StratifiedKFold



accuracy = []

skf = StratifiedKFold(n_splits=15, random_state=None)

skf.get_n_splits(x_train_scale, y)



for train_index, test_index in skf.split(x_train_scale, y):

  x1_train, x1_test = x_train_scale[train_index], x_train_scale[test_index]

  y1_train, y1_test = y[train_index], y[test_index]



  model.fit(x1_train, y1_train)  

  score = model.score(x1_test, y1_test)

  accuracy.append(score)



print(accuracy)





y_pred = model.predict(x_test_scale)



from sklearn.metrics import classification_report

print(classification_report(y_pred, submissions['price_range']))



data = {'id':submissions['id'],

       'price_range':y_pred}

results = pd.DataFrame(data)

results.to_csv('/kaggle/working/results_lr.csv', index=False)