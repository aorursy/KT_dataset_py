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
df=pd.read_csv("../input/data-ready-for-model/csvfile.csv")

df.head()
from sklearn.model_selection import train_test_split, cross_val_score



df_=df[df.columns[~df.columns.isin(['default.payment.next.month'])]]#already does not have ID

# create X (data features) and y (target)

X = df_

target=df['default.payment.next.month']

y = target



# use train/test split with different random_state values

# we can change the random_state values that changes the accuracy scores

# the scores change a lot, this is why testing scores is a high-variance estimate

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 1)





sets=[X_train,X_test,y_train,y_test]



for s in sets:

    print(s.shape)
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.metrics import classification_report

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier

from sklearn.metrics import roc_auc_score





## List of ML Algorithms, we will use for loop for each algorithms.

models = [GradientBoostingClassifier()]





for model in models:

    m=model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    proba = model.predict_proba(X_test)

    roc_score = roc_auc_score(y_test, proba[:,1])

    cv_score = cross_val_score(model,X_train,y_train,cv=10).mean()

    score = accuracy_score(y_test,y_pred)

    bin_clf_rep = classification_report(y_test,y_pred, zero_division=1)

    name = str(model)

    print(name[0:name.find("(")])

    print("Accuracy :", score)

    print("CV Score :", cv_score)

    print("AUC Score : ", roc_score)

    print(bin_clf_rep)

    print(confusion_matrix(y_test,y_pred))

    print("------------------------------------------------------------")
def plot_feature_importances(clf, X_train, y_train=None, 

                             top_n=10, figsize=(8,8), print_table=False, title="Feature Importances"):

    '''

    plot feature importances of a tree-based sklearn estimator

    

    Note: X_train and y_train are pandas DataFrames

    

    Note: Scikit-plot is a lovely package but I sometimes have issues

              1. flexibility/extendibility

              2. complicated models/datasets

          But for many situations Scikit-plot is the way to go

          see https://scikit-plot.readthedocs.io/en/latest/Quickstart.html

    

    Parameters

    ----------

        clf         (sklearn estimator) if not fitted, this routine will fit it

        

        X_train     (pandas DataFrame)

        

        y_train     (pandas DataFrame)  optional

                                        required only if clf has not already been fitted 

        

        top_n       (int)               Plot the top_n most-important features

                                        Default: 10

                                        

        figsize     ((int,int))         The physical size of the plot

                                        Default: (8,8)

        

        print_table (boolean)           If True, print out the table of feature importances

                                        Default: False

        

    Returns

    -------

        the pandas dataframe with the features and their importance

        

    Author

    ------

        George Fisher

    '''

    

    __name__ = "plot_feature_importances"

    

    import pandas as pd

    import numpy  as np

    import matplotlib.pyplot as plt

    

    from xgboost.core     import XGBoostError

    from lightgbm.sklearn import LightGBMError

    

    try: 

        if not hasattr(clf, 'feature_importances_'):

            clf.fit(X_train.values, y_train.values.ravel())



            if not hasattr(clf, 'feature_importances_'):

                raise AttributeError("{} does not have feature_importances_ attribute".

                                    format(clf.__class__.__name__))

                

    except (XGBoostError, LightGBMError, ValueError):

        clf.fit(X_train.values, y_train.values.ravel())

            

    feat_imp = pd.DataFrame({'importance':clf.feature_importances_})    

    feat_imp['feature'] = X_train.columns

    feat_imp.sort_values(by='importance', ascending=False, inplace=True)

    feat_imp = feat_imp.iloc[:top_n]

    

    feat_imp.sort_values(by='importance', inplace=True)

    feat_imp = feat_imp.set_index('feature', drop=True)

    feat_imp.plot.barh(title=title, figsize=figsize)

    plt.xlabel('Feature Importance Score')

    plt.show()

    

    if print_table:

        from IPython.display import display

        print("Top {} features in descending order of importance".format(top_n))

        display(feat_imp.sort_values(by='importance', ascending=False))

        

    return feat_imp



model=GradientBoostingClassifier()



e = plot_feature_importances(model, X_train, y_train, top_n=20, title="Variable Importance: Scikit GBM")

print(e)