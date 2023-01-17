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
ad_df = pd.read_csv('../input/internet-advertisements-data-set/add.csv', index_col=0,low_memory=False)

ad_df.head(5)
# Drop invalid

ad_df2 = ad_df.applymap(lambda val: np.nan if str(val).strip() == '?' else val)

ad_df2 = ad_df2.dropna()



ad_df2.head(5)
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()

y = lb.fit_transform(ad_df2.iloc[:, -1])
y[0:5]
X = ad_df2.iloc[:,:-1]

X.head(5)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)
X[0:5,:]
from sklearn import model_selection 

from sklearn.ensemble import BaggingClassifier 

from sklearn.tree import DecisionTreeClassifier 





def get_accuracy_with_bagging(base_cls = DecisionTreeClassifier(), bag_percent_size=10, splits=10, seed=1887, num_trees=50):

    kfold = model_selection.KFold(n_splits = splits)     



    # bagging classifier 

    model = BaggingClassifier(base_estimator = base_cls, 

                              n_estimators = num_trees, 

                              random_state = seed,

                              max_samples=(bag_percent_size/100.0)

                             ) 



    results = model_selection.cross_val_score(model, X, y, cv = kfold) 

    return results.mean() 
result = []



for bag_percent_size in range(10, 101, 10):

    accuracy = get_accuracy_with_bagging(bag_percent_size=bag_percent_size, splits=10)

    print ("Accuracy: ", accuracy, " Bag Percent Size: ", bag_percent_size)

    result.append([bag_percent_size, accuracy])
result
result_pd = pd.DataFrame(result)

result_pd.columns = ['BagPercentSize', 'Accuracy']

result_pd
result_pd.plot(x='BagPercentSize', y='Accuracy')