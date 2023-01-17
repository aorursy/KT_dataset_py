# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.







import pandas as pd

import numpy as np

import seaborn as sns

import sklearn

from sklearn.model_selection import StratifiedKFold 



#load the data into dataframes

train=pd.read_csv('../input/train.tsv',sep="\t")

test=pd.read_csv('../input/test.tsv',sep="\t")



#Explore the data

train.describe()

test.describe()



train.head()

test.head()



#About shipping detail

train['shipping'].unique()

test['shipping'].unique()





train['shipping'].value_counts()

test['shipping'].value_counts()



#About item_condiiton_id

train['item_condition_id'].value_counts()

test['item_condition_id'].value_counts()





train['brand_name'].value_counts()

test['brand_name'].value_counts()





train['category_name'].value_counts()





train['price'].describe()

train['price'].value_counts() 





###

target = 'price'

IDcol = ['train_id']



IDcol_test = ['test_id']



predictors = ['item_condition_id','shipping']



from sklearn.ensemble import RandomForestRegressor



alg=RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4)

alg.fit(train[predictors],train[target])



test[target]=alg.predict(test[predictors])



IDcol.append(target)

filename='sample_submission_rf_regression.csv'

submission = pd.DataFrame({ x: test[x] for x in IDcol_test})

submission.to_csv(filename, index=False)