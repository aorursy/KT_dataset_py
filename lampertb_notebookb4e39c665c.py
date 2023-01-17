# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
house_df = pd.read_json('../input/test.json')

house_test_df = pd.read_json('../input/train.json')



house_df.info()

house_test_df.info()
#Test data to plot the effect of bathrooms with price



#For now, drop everything else

drop_list = ['building_id', 'created', 'description', 'display_address', 'features','latitude','listing_id',

             'longitude','manager_id','photos','street_address']

house_df = house_df.drop(drop_list,axis=1)

house_test_df = house_test_df.drop(drop_list,axis=1)

house_df.info()

sns.factorplot('bedrooms', 'price', order=[1,2,3], data=house_df,size=5)

sns.factorplot('bathrooms', 'price', order=[1,2,3], data=house_df,size=5)

#Log reg

X_train = house_df.drop("price",axis=1)

Y_train = house_df["price"]



logreg = LogisticRegression()

logreg.fit(X_train,Y_train)
