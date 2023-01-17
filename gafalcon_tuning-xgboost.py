# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from xgboost import XGBClassifier
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn import model_selection, metrics
from sklearn.model_selection import GridSearchCV, cross_val_score
def add_features(df):
    df['HF1'] = df['Horizontal_Distance_To_Hydrology']+df['Horizontal_Distance_To_Fire_Points']
    df['HF2'] = abs(df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Fire_Points'])
    df['HR1'] = abs(df['Horizontal_Distance_To_Hydrology']+df['Horizontal_Distance_To_Roadways'])
    df['HR2'] = abs(df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Roadways'])
    df['FR1'] = abs(df['Horizontal_Distance_To_Fire_Points']+df['Horizontal_Distance_To_Roadways'])
    df['FR2'] = abs(df['Horizontal_Distance_To_Fire_Points']-df['Horizontal_Distance_To_Roadways'])
    df['ele_vert'] = df.Elevation-df.Vertical_Distance_To_Hydrology

    df['slope_hyd'] = (df['Horizontal_Distance_To_Hydrology']**2+df['Vertical_Distance_To_Hydrology']**2)**0.5
    df.slope_hyd=df.slope_hyd.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any

    #Mean distance to Amenities 
    df['Mean_Amenities']=(df.Horizontal_Distance_To_Fire_Points + df.Horizontal_Distance_To_Hydrology + df.Horizontal_Distance_To_Roadways) / 3 
    #Mean Distance to Fire and Water 
    df['Mean_Fire_Hyd']=(df.Horizontal_Distance_To_Fire_Points + df.Horizontal_Distance_To_Hydrology) / 2
    return df
def cv_score(clf, X, y, n_splits=5, scoring=None):
    # cv = ShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=0)
    scores = cross_val_score(clf, X, y, cv=n_splits, scoring=scoring)
    #print ("Scores with C=",C, scores)
    print("Scores: ", scores)
    print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std()))
df = pd.read_csv("../input/train.csv")
df = add_features(df)
#df.to_csv("train_extra_features.csv", index=False)
#df = pd.read_csv("train_extra_features.csv")
test = pd.read_csv("../input/test.csv")
test = add_features(test)
#test.to_csv("test_extra_features.csv", index=False)
#test = pd.read_csv("test_extra_features.csv")
y = df.Cover_Type
y = y - 1 #for xgb boost, classes must be in [0, num_class]
df.drop(["Id", "Cover_Type"], axis=1, inplace=True)

test_Ids = test.Id
test.drop(["Id"], axis=1, inplace=True)
def modelfit(alg, predictors, target, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(predictors.values, label=target.values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          early_stopping_rounds=early_stopping_rounds)#, show_progress=False)
        print ("Early stopping at n_estimators: ", cvresult.shape[0])
        alg.set_params(n_estimators=cvresult.shape[0])
    
    # Fit the algorithm on the data
    alg.fit(predictors, target)#, eval_metric='auc')

    # # Predict training set
    train_preds = alg.predict(predictors)
    train_predprob = alg.predict_proba(predictors)[:,1]

    # #Model report
    print ("Accuracy: %.4g" % metrics.accuracy_score(target.values, train_preds))
    # # print ("AUC SCORE (Train): %f" % metrics.roc_auc_score(target, train_predprob))

    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title="Feature Importances")
    plt.ylabel('Feature Importance Score')
xgb1 = XGBClassifier(
    learning_rate = 0.1,
    n_estimators=1000,
    max_depth=13,
    min_child_weight=1,
    gamma=0,
    subsample=0.9,
    colsample_bytree=0.8,
    nthread=4,
    scale_pos_weight=1,
    seed=7,
    objective='multi:softmax',
    num_class=7,
    reg_alpha=0
    )
%time modelfit(xgb1, df, y)
%time cv_score(xgb1, df, y)
def predict(clf, X_test, csv_name):
    global test_Ids
    predictions = clf.predict(X_test)
    pred_df = pd.DataFrame()
    pred_df["Id"] = test_Ids
    pred_df["Cover_Type"] = predictions + 1
    pred_df.to_csv(csv_name, index=False)
%time xgb1.fit(df, y)
predict(xgb1, test, "xgb_tuning.csv")