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
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier



import warnings

warnings.filterwarnings("ignore")
def evaluate_prediction(y_actual, y_pred):

    """

    Provides key metrics for evaluation of prediction.

    """

    df_pred = pd.DataFrame(data={'y_actual':y_actual,'y_pred':y_pred})

    df_pred['t'] = (df_pred['y_actual'] == df_pred['y_pred'])

    tp = sum(df_pred['t'] & (df_pred['y_actual']==1))

    fp = sum(~df_pred['t'] & (df_pred['y_pred']==1))

    tn = sum(df_pred['t'] & (df_pred['y_actual']==0))

    fn = sum(~df_pred['t'] & (df_pred['y_pred']==0))

    sensitivity = float(tp)/(tp+fn) #recall

    precision = float(tp)/(tp+fp)

    specificity = float(tn)/(tn+fp)

    return tp, fp, tn, fn, sensitivity, precision, specificity


df = pd.read_csv('../input/SalesKaggle3.csv')



# we are only interested in historical data for the model building 

df_historical = df.loc[df['File_Type']=='Historical']



# for the model, we transform the MarketingType into a numerical value. 

df_marketingtype = pd.get_dummies(df_historical['MarketingType'], prefix="MarketingType")

df_historical = pd.concat([df_historical,df_marketingtype], axis=1)



# subselecting relevant features and target in array format

X = df_historical[['MarketingType_S','New_Release_Flag','ReleaseNumber','StrengthFactor','PriceReg','ItemCount','LowUserPrice','LowNetPrice']].values

y = df_historical['SoldFlag'].values 



# train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

# grid search through Adaboost classifier on training set. Within the training set a 3-fold cross validation is applied.

parameters = {'n_estimators':[5,10,35,50], 'learning_rate':[0.1,0.3,0.5,0.7,0.9,1.0]}

abc = AdaBoostClassifier()#base_estimator = DecisionTreeClassifier())

clf = GridSearchCV(estimator=abc, param_grid=parameters, cv=3, verbose=0, scoring='f1')

clf.fit(X_train, y_train)
# model metrics of best identified model

best = clf.best_estimator_

y_pred = best.predict(X_test)

tp, fp, tn, fn, sensitivity, precision, specificity = evaluate_prediction(y_test, y_pred)

print('sensitivity: ' + str(sensitivity))

print('precision: ' + str(precision))

print('specificity: ' + str(specificity))
