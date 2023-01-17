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
df = pd.read_csv('../input/chocolate-bar-2020/chocolate.csv')
dfn = df.drop(['ref', 'Unnamed: 0','company'], axis = 1) 

dfn
dfn.info()
hot_chocolate = pd.get_dummies(dfn,columns=['country_of_bean_origin','company_location','country_of_bean_origin','specific_bean_origin_or_bar_name','beans','cocoa_butter','vanilla','lecithin','salt','sugar','sweetener_without_sugar','first_taste','second_taste','third_taste','fourth_taste'])

hot_chocolate.shape
df_choc = pd.concat([dfn, hot_chocolate], axis=1)

df_choc.shape
df_finale = df_choc.drop(columns=['country_of_bean_origin','company_location','country_of_bean_origin','specific_bean_origin_or_bar_name','beans','cocoa_butter','vanilla','lecithin','salt','sugar','sweetener_without_sugar','first_taste','second_taste','third_taste','fourth_taste'],axis = 1)
df_finale.shape
a = df_finale.loc[:,~df_finale.columns.duplicated()]

a
a.to_csv('Finale Preprocess')
b = a.drop('review_date', axis = 1)


X = b.iloc[:,0:2800]  

y = a.iloc[:,-1]    





from sklearn.model_selection import train_test_split



X_train,y_train, X_test,y_test = train_test_split(X, y, test_size=0.3)
y
from sklearn.ensemble import ExtraTreesClassifier

import matplotlib.pyplot as plt

model = ExtraTreesClassifier()

model.fit(X,y)

print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(10).plot(kind='barh')

plt.show()

!pip install xverse


from xverse.ensemble import VotingSelector

clf = VotingSelector()

clf.fit(X, y)

clf.feature_importances_

clf.feature_importances_['Variable_Name'][0],clf.feature_importances_['Variable_Name'][12],clf.feature_importances_['Variable_Name'][11],clf.feature_importances_['Variable_Name'][2]
clf.feature_votes_

Xver= clf.transform(X)

Xver.head()


X_best = b.iloc[:,0:2800]  

y = a.iloc[:,-1]    





from sklearn.model_selection import train_test_split



X_train,y_train, X_test,y_test = train_test_split(Xver, y, test_size=0.3)
!pip install pycaret

from pycaret.classification import *

exp1 = setup(Xver, target = y)

compare_models()
adaboost = create_model('ada')

tuned_adaboost = tune_model('ada')
# creating a decision tree model

dt = create_model('dt')

# ensembling a trained dt model

dt_bagged = ensemble_model(dt)

from tpot import TPOTClassifier

tpot = TPOTClassifier(verbosity=2, max_time_mins=10)

tpot.fit(X_train, y_train)
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import PolynomialFeatures, RobustScaler





ss = StandardScaler() 

X_train_scaled = ss.fit_transform(X_train) 

X_test_scaled = ss.transform(X_test)



from tpot import TPOTClassifier 

tpot = TPOTClassifier(verbosity=2, max_time_mins=10)

tpot.fit(X_train_scaled, y_train)

tpot.fitted_pipeline_



tpot.score(X_test_scaled, y_test)
