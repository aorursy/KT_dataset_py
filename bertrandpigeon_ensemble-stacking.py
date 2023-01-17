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
import os

import random

import warnings



warnings.simplefilter(action='ignore', category=FutureWarning)





import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

#from sklearn.cross_validation import KFold

from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer, accuracy_score



from sklearn.model_selection import RandomizedSearchCV

import lightgbm

import matplotlib.pyplot as plt



from sklearn.mixture import GaussianMixture





from lightgbm import LGBMClassifier



from mlxtend.classifier import StackingCVClassifier



from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier



random_state = 1

random.seed(random_state)

np.random.seed(random_state)















# Read the data

X = pd.read_csv('/kaggle/input/learn-together/train.csv', index_col='Id')

X_test_full = pd.read_csv('/kaggle/input/learn-together/test.csv', index_col='Id')



y = X.Cover_Type

X.drop(['Cover_Type'], axis=1, inplace=True)



#X.drop(['Id'], axis=1, inplace=True)



train_X = X

train_y = y
def WH4(df):

    df['Hydro_high'] = df.Vertical_Distance_To_Hydrology.apply(lambda x: x > 3 )

    df['Hydro_Euclidean'] = (df['Horizontal_Distance_To_Hydrology']**2 +

                            df['Vertical_Distance_To_Hydrology']**2).apply(np.sqrt)

    #df.drop(['Vertical_Distance_To_Hydrology'], axis=1, inplace=True)

    #df.drop(['Horizontal_Distance_To_Hydrology'], axis=1, inplace=True)

    df['Hydro_Fire_road'] = (df.Horizontal_Distance_To_Roadways + df.Horizontal_Distance_To_Fire_Points)/(df.Hydro_Euclidean/20000+1)

    df['Hydro_Fire_sum'] = (df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Fire_Points'])

    df.drop(['Soil_Type15'], axis=1, inplace=True)

    df.drop(['Soil_Type7'], axis=1, inplace=True)

    df['Hydro_Elevation_diff'] = (df['Elevation'] - df['Vertical_Distance_To_Hydrology'])

    

    df['Hydro_Fire_diff'] = (df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Fire_Points']).abs()

    df['Hydro_Road_sum'] = (df['Horizontal_Distance_To_Hydrology'] +df['Horizontal_Distance_To_Roadways'])

    df['Hydro_Road_diff'] = (df['Horizontal_Distance_To_Hydrology'] -df['Horizontal_Distance_To_Roadways']).abs()

    df['Road_Fire_sum'] = (df['Horizontal_Distance_To_Roadways'] + df['Horizontal_Distance_To_Fire_Points'])

    df['Road_Fire_diff'] = (df['Horizontal_Distance_To_Roadways'] - df['Horizontal_Distance_To_Fire_Points']).abs()

    #df.loc[:, :] = np.floor(MinMaxScaler((0, 100)).fit_transform(df))

    #df = df.astype('int8')

    #df.fillna(0)

    



        



















WH4(X_test_full)

WH4(X)





gm = GaussianMixture(n_components  = 15)

gm.fit(X)

X['g_mixture'] = gm.predict(X)

X_test_full['g_mixture'] = gm.predict(X_test_full)
max_features = min(30, X.columns.size)



ab_clf = AdaBoostClassifier(n_estimators=200,

                            base_estimator=DecisionTreeClassifier(

                                min_samples_leaf=2,

                                random_state=random_state),

                            random_state=random_state)



et_clf = ExtraTreesClassifier(n_estimators=300,

                              min_samples_leaf=2,

                              min_samples_split=2,

                              max_depth=50,

                              max_features=max_features,

                              random_state=random_state,

                              n_jobs=1)



lg_clf = LGBMClassifier(n_estimators=300,

                        num_leaves=128,

                        verbose=-1,

                        random_state=random_state,

                        n_jobs=1)



rf_clf = RandomForestClassifier(n_estimators=300,

                                random_state=random_state,

                                n_jobs=1)



ensemble = [('AdaBoostClassifier', ab_clf),

            ('ExtraTreesClassifier', et_clf),

            ('LGBMClassifier', lg_clf),

            ('RandomForestClassifier', rf_clf)]
print('> Cross-validating classifiers')

for label, clf in ensemble:

    score = cross_val_score(clf, X, train_y,

                            cv=5,

                            scoring='accuracy',

                            verbose=0,

                            n_jobs=-1)



    print('  -- {: <24} : {:.3f} : {}'.format(label, np.mean(score), np.around(score, 3)))





print('> Fitting stack')

# - https://www.kaggle.com/itslek/stack-blend-lrs-xgb-lgb-house-prices-k-v17

# - https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard

# - https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python



stack = StackingCVClassifier(classifiers=[ab_clf, et_clf, lg_clf, rf_clf],

                             meta_classifier=rf_clf,

                             cv=5,

                             stratify=True,

                             shuffle=True,

                             use_probas=True,

                             use_features_in_secondary=True,

                             verbose=1,

                             random_state=random_state,

                             n_jobs=-1)



stack = stack.fit(X, train_y)

X_ID = pd.read_csv('/kaggle/input/learn-together/test.csv')

predictions = stack.predict(X_test_full)

predictions = predictions.astype(int)

#X_test_full.head()

ids = X_ID['Id']

output = pd.DataFrame({ 'ID' : ids, 'Cover_Type': predictions })



output.to_csv('submission2.csv' , index=False)