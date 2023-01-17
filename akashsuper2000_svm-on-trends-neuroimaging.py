import numpy as np

import pandas as pd

from sklearn.model_selection import KFold

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor as RFR

from sklearn.externals import joblib

from time import time

import pickle



def metric(y_true, y_pred):

    return np.mean(np.sum(np.abs(y_true - y_pred), axis=0)/np.sum(y_true, axis=0))
fnc_df = pd.read_csv("../input/trends-assessment-prediction/fnc.csv")

loading_df = pd.read_csv("../input/trends-assessment-prediction/loading.csv")



fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])

df = fnc_df.merge(loading_df, on="Id")





labels_df = pd.read_csv("../input/trends-assessment-prediction/train_scores.csv")

labels_df['is_train'] = True



df = df.merge(labels_df, on="Id", how="left")



test_df = df[df["is_train"] != True].copy()

df = df[df["is_train"] == True].copy()



df.shape, test_df.shape
for i in [500]:

    # Giving less importance to FNC features since they are easier to overfit due to high dimensionality.

    tic = time()

    FNC_SCALE = 1/i



    df[fnc_features] *= FNC_SCALE

    test_df[fnc_features] *= FNC_SCALE



    NUM_FOLDS = 3

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)





    features = loading_features + fnc_features



    overal_score = 0

    for target, c, w in [("age", 100, 0.3), ("domain1_var1", 10, 0.175), ("domain1_var2", 10, 0.175), ("domain2_var1", 10, 0.175), ("domain2_var2", 10, 0.175)]:    

        y_oof = np.zeros(df.shape[0])

        y_test = np.zeros((test_df.shape[0], NUM_FOLDS))



        for f, (train_ind, val_ind) in enumerate(kf.split(df, df)):

            train_df, val_df = df.iloc[train_ind], df.iloc[val_ind]

            train_df = train_df[train_df[target].notnull()]



            # model = SVR(C=c, cache_size=3000.0)

            model = RFR(n_estimators=100)

            model.fit(train_df[features], train_df[target])

            

            s = pickle.dumps(model)

            joblib.dump(model, 'model'+str(i)+'.pkl')



            y_oof[val_ind] = model.predict(val_df[features])

            y_test[:, f] = model.predict(test_df[features])



        df["pred_{}".format(target)] = y_oof

        test_df[target] = y_test.mean(axis=1)



        score = metric(df[df[target].notnull()][target].values, df[df[target].notnull()]["pred_{}".format(target)].values)

        overal_score += w*score

        print(target, np.round(score, 4))

        print()



    print("Overall score:", np.round(overal_score, 4))

    

    sub_df = pd.melt(test_df[["Id", "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]], id_vars=["Id"], value_name="Predicted")

    sub_df["Id"] = sub_df["Id"].astype("str") + "_" +  sub_df["variable"].astype("str")



    sub_df = sub_df.drop("variable", axis=1).sort_values("Id")

    assert sub_df.shape[0] == test_df.shape[0]*5

    sub_df.head(10)

    

    sub_df.to_csv("submission"+str(i)+".csv", index=False)

    

    toc = time()

    print("Time taken for "+str(i)+" "+str(toc-tic))