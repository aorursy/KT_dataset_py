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
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



train=pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

test=pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')



train.info()

test.info()



corrmat = train.corr()

k = 10 

cols = corrmat.nlargest(k, 'item_cnt_day')['item_cnt_day'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()





selected_features = ['item_id','shop_id' ]



X_train = train[selected_features]

X_test = test[selected_features]

y_train = train['item_cnt_day']





from sklearn.feature_extraction import DictVectorizer

dict_vec = DictVectorizer(sparse=False)



X_train = dict_vec.fit_transform(X_train.to_dict(orient='record'))

X_test = dict_vec.transform(X_test.to_dict(orient='record'))





from sklearn.ensemble import GradientBoostingRegressor

rfr = GradientBoostingRegressor()

rfr.fit(X_train, y_train)

rfr_y_predict = rfr.predict(X_test)





rfr_submission = pd.DataFrame({'ID': test['ID'], 'item_cnt_month': rfr_y_predict})

rfr_submission.to_csv('submission.csv', index=False,sep=',')