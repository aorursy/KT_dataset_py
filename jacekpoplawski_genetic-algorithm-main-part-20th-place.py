from sklearn.model_selection import StratifiedKFold

import pandas as pd

import lightgbm as lgb

import numpy as np

import pickle



train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")
lgb.__version__
def ginic(actual, pred):

    n = len(actual)

    a_s = actual[np.argsort(pred)]

    a_c = a_s.cumsum()

    giniSum = a_c.sum() / a_c[-1] - (n + 1) / 2.0

    return giniSum / n

 

def gini_normalizedc(a, p):

    if p.ndim == 2:

        p = p[:,1] 

    return ginic(a, p) / ginic(a, a)



def gini_lgb(preds, dtrain):  

    actuals = np.array(dtrain.get_label())   

    return 'gini', gini_normalizedc(actuals, preds), True

def perform_single_train(data, hyper):



    X_train = data["X_train"]

    y_train = data["y_train"]

    X_valid = data["X_valid"]

    y_valid = data["y_valid"]

    X_test = data["X_test"]

    

    lgb_pars = hyper["lgb_pars"]

    features = hyper["features"]

    

    rounds = hyper["rounds"]

    early = hyper["early"]

    noise_level = hyper["noise_level"]

    smoothing = hyper["smoothing"]

    min_samples_leaf= hyper["min_samples_leaf"]



    X_data = X_train.copy()

    X_data["target"] = y_train



    X_train_c=X_train.copy()

    X_valid_c=X_valid.copy()

    X_test_c=X_test.copy()



    for f in features:

        s = f.split("_add_")

        if (len(s) == 2):

            c1 = s[0]

            c2 = s[1]

            X_train[f] = X_train_c[c1] + X_train_c[c2]

            X_valid[f] = X_valid_c[c1] + X_valid_c[c2]

            X_test[f] = X_test_c[c1] + X_test_c[c2]



        s = f.split("_sub_")

        if (len(s) == 2):

            c1 = s[0]

            c2 = s[1]

            X_train[f] = X_train_c[c1] - X_train_c[c2]

            X_valid[f] = X_valid_c[c1] - X_valid_c[c2]

            X_test[f] = X_test_c[c1] - X_test_c[c2]



        s = f.split("_mul_")

        if (len(s) == 2):

            c1 = s[0]

            c2 = s[1]

            X_train[f] = X_train_c[c1] * X_train_c[c2]

            X_valid[f] = X_valid_c[c1] * X_valid_c[c2]

            X_test[f] = X_test_c[c1] * X_test_c[c2]



        s = f.split("_div_")

        if (len(s) == 2):

            c1 = s[0]

            c2 = s[1]

            X_train[f] = X_train_c[c1] / X_train_c[c2]

            X_valid[f] = X_valid_c[c1] / X_valid_c[c2]

            X_test[f] = X_test_c[c1] / X_test_c[c2]



        s = f.split("_mean_")                    

        if (len(s) > 1):

            if (s[0] == '0'):

                s.remove('0')



            averages = X_data.groupby(s)["target"].agg(["mean", "count"])

            smoothing_v = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))

            averages[f] = X_data["target"].mean() * (1 - smoothing_v) + averages["mean"] * smoothing_v

            averages.drop(["mean", "count"], axis=1, inplace=True)



            np.random.seed(42)

            noise = np.random.randn(len(averages[f])) * noise_level

            averages[f] = averages[f] + noise



            X_train = pd.merge(X_train, averages, how='left', left_on=s, right_index=True)

            X_valid = pd.merge(X_valid, averages, how='left', left_on=s, right_index=True)

            X_test = pd.merge(X_test, averages, how='left', left_on=s, right_index=True)                       

            

    X_train_subset=X_train[features]

    X_valid_subset=X_valid[features]

    X_test_subset=X_test[features]

    

    lgb_train = lgb.Dataset(X_train_subset, y_train)

    lgb_eval = lgb.Dataset(X_valid_subset, y_valid, reference=lgb_train)



    model = lgb.train(lgb_pars,

            lgb_train,

            num_boost_round=rounds,

            valid_sets=lgb_eval,

            early_stopping_rounds=early,

            feval=gini_lgb,

            verbose_eval=100)



    p_train = model.predict(X_train_subset, num_iteration=model.best_iteration)            

    p_valid = model.predict(X_valid_subset, num_iteration=model.best_iteration)            

    p_test = model.predict(X_test_subset, num_iteration=model.best_iteration)   



    train_score = gini_normalizedc(y_train, p_train) 

    valid_score = gini_normalizedc(y_valid, p_valid)     



    return [train_score, valid_score, p_test]

def perform_full(X, y, X_test, hyper, prefix):    

    scores = []   

    kfold = hyper["kfold"]

    

    skf = StratifiedKFold(n_splits=kfold, random_state=42)

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):

        X_train, X_valid = X.iloc[train_index], X.iloc[test_index]

        y_train, y_valid = y[train_index], y[test_index]

        

        X_test_c = X_test.copy()



        data = {"X_train": X_train,

                "y_train": y_train,

                "X_valid": X_valid,

                "y_valid": y_valid,

                "X_test": X_test_c

                }

        

        [train_score, valid_score, p_test] = perform_single_train(data, hyper)



        with open("test_"+prefix+str(i)+".pkl", 'wb') as f:

            pickle.dump(p_test,f)



        scores.append([train_score, valid_score])

        

    return scores     



best_features = ['0_mean_ps_car_05_cat',

 '0_mean_ps_car_10_cat',

 '0_mean_ps_car_12',

 '0_mean_ps_ind_04_cat',

 '0_mean_ps_ind_17_bin',

 '0_mean_ps_ind_18_bin',

 'ps_car_01_cat',

 'ps_car_01_cat_add_ps_car_02_cat',

 'ps_car_01_cat_mean_ps_car_07_cat',

 'ps_car_01_cat_mean_ps_ind_18_bin',

 'ps_car_02_cat',

 'ps_car_02_cat_add_ps_ind_12_bin',

 'ps_car_02_cat_mul_ps_ind_09_bin',

 'ps_car_02_cat_sub_ps_car_08_cat',

 'ps_car_03_cat',

 'ps_car_03_cat_add_ps_ind_05_cat',

 'ps_car_03_cat_div_ps_ind_11_bin',

 'ps_car_03_cat_mean_ps_ind_12_bin_mean_ps_ind_16_bin',

 'ps_car_04_cat',

 'ps_car_04_cat_mean_ps_ind_04_cat',

 'ps_car_04_cat_mean_ps_ind_14',

 'ps_car_05_cat',

 'ps_car_05_cat_add_ps_ind_10_bin',

 'ps_car_05_cat_mean_ps_ind_13_bin',

 'ps_car_06_cat',

 'ps_car_06_cat_add_ps_reg_02',

 'ps_car_07_cat',

 'ps_car_07_cat_mean_ps_car_10_cat_mean_ps_ind_10_bin',

 'ps_car_07_cat_mean_ps_ind_02_cat_mean_ps_ind_10_bin',

 'ps_car_07_cat_mean_ps_ind_14',

 'ps_car_07_cat_sub_ps_ind_09_bin',

 'ps_car_08_cat',

 'ps_car_08_cat_mul_ps_ind_17_bin',

 'ps_car_09_cat',

 'ps_car_09_cat_mean_ps_car_10_cat_mean_ps_ind_13_bin',

 'ps_car_09_cat_mean_ps_ind_05_cat',

 'ps_car_09_cat_mul_ps_car_10_cat',

 'ps_car_09_cat_sub_ps_ind_03',

 'ps_car_10_cat',

 'ps_car_10_cat_mul_ps_ind_07_bin',

 'ps_car_10_cat_sub_ps_ind_12_bin',

 'ps_car_11',

 'ps_car_11_cat',

 'ps_car_11_div_ps_ind_02_cat',

 'ps_car_11_mean_ps_ind_02_cat',

 'ps_car_11_mean_ps_ind_10_bin',

 'ps_car_11_mean_ps_ind_12_bin',

 'ps_car_11_mean_ps_ind_12_bin_mean_ps_ind_16_bin',

 'ps_car_12',

 'ps_car_13',

 'ps_car_14',

 'ps_car_15',

 'ps_car_15_add_ps_ind_01',

 'ps_car_15_add_ps_ind_11_bin',

 'ps_car_15_div_ps_ind_14',

 'ps_car_15_mul_ps_ind_01',

 'ps_ind_01',

 'ps_ind_01_div_ps_ind_16_bin',

 'ps_ind_01_mean_ps_ind_09_bin',

 'ps_ind_01_mean_ps_ind_11_bin',

 'ps_ind_01_sub_ps_ind_03',

 'ps_ind_02_cat',

 'ps_ind_02_cat_add_ps_reg_01',

 'ps_ind_02_cat_mean_ps_ind_11_bin',

 'ps_ind_02_cat_mean_ps_ind_12_bin',

 'ps_ind_02_cat_mean_ps_ind_14',

 'ps_ind_02_cat_mul_ps_ind_08_bin',

 'ps_ind_02_cat_mul_ps_ind_11_bin',

 'ps_ind_03',

 'ps_ind_03_mean_ps_ind_15',

 'ps_ind_03_mul_ps_ind_11_bin',

 'ps_ind_04_cat',

 'ps_ind_04_cat_mul_ps_ind_12_bin',

 'ps_ind_04_cat_mul_ps_ind_17_bin',

 'ps_ind_04_cat_sub_ps_ind_11_bin',

 'ps_ind_05_cat',

 'ps_ind_05_cat_add_ps_ind_14',

 'ps_ind_05_cat_add_ps_ind_17_bin',

 'ps_ind_05_cat_mean_ps_ind_13_bin',

 'ps_ind_05_cat_mean_ps_reg_01',

 'ps_ind_06_bin',

 'ps_ind_06_bin_div_ps_reg_02',

 'ps_ind_06_bin_mean_ps_ind_13_bin',

 'ps_ind_06_bin_sub_ps_ind_17_bin',

 'ps_ind_07_bin',

 'ps_ind_07_bin_mean_ps_ind_12_bin',

 'ps_ind_08_bin',

 'ps_ind_08_bin_add_ps_ind_17_bin',

 'ps_ind_08_bin_mean_ps_ind_13_bin_mean_ps_ind_14',

 'ps_ind_08_bin_mean_ps_reg_01',

 'ps_ind_09_bin',

 'ps_ind_09_bin_add_ps_ind_15',

 'ps_ind_09_bin_mul_ps_ind_17_bin',

 'ps_ind_10_bin',

 'ps_ind_11_bin',

 'ps_ind_11_bin_div_ps_ind_12_bin',

 'ps_ind_12_bin',

 'ps_ind_12_bin_mul_ps_reg_03',

 'ps_ind_13_bin',

 'ps_ind_13_bin_div_ps_reg_03',

 'ps_ind_13_bin_sub_ps_reg_03',

 'ps_ind_14',

 'ps_ind_14_div_ps_reg_01',

 'ps_ind_15',

 'ps_ind_16_bin',

 'ps_ind_17_bin',

 'ps_ind_18_bin',

 'ps_reg_01',

 'ps_reg_01_mean_ps_ind_18_bin',

 'ps_reg_02',

 'ps_reg_03']



X = train_data.drop(["id","target"],axis=1)

y = train_data["target"].values 

X_test = test_data.drop(["id"],axis=1)



lgb_pars = {

    'max_depth': 4,

    'min_data_in_leaf': 20,

    'min_sum_hessian_in_leaf': 1e-3,

    'feature_fraction': 0.47,

    'bagging_fraction': 0.87,

    'bagging_freq': 10,

    'lambda_l1': 8.0,    

    'lambda_l2': 13.0,    

    'min_split_gain': 0,

    'max_bin': 255,

    'min_data_in_bin': 3,

    'learning_rate': 0.08,

    'metric': {'gini_lgb'},

    'objective': "binary"

}



hyper = {"rounds": 1000,

         "early": 100,

         "lgb_pars": lgb_pars,

         "features": best_features,

         "noise_level": 0.1,

         "kfold" : 5,

         "smoothing": 30.0,

         "min_samples_leaf": 300}



#result = perform_full(X, y, X_test, hyper, "kaggle_kernel_")



predictions = []



kfold = 5



#for i in range(kfold):    

#    with open("test_kaggle_kernel_"+str(i)+".pkl", 'rb') as f:

#        pred = pickle.load(f)        

#    predictions.append(pred)



#final_prediction = np.zeros(predictions[0].shape[0])

#for i in range(kfold):

#    pred = predictions[i]

#    final_prediction += pred / kfold



submission=pd.DataFrame()

submission["id"] = test_data["id"]

#submission["target"] = final_prediction

submission.set_index("id", inplace=True)

submission.to_csv("kaggle_kernel_1.csv")  

#submission["target"].describe()



# Scores on LB:

# 0.29045

# 0.28330
base_features = [

 'ps_car_01_cat',

 'ps_car_02_cat',

 'ps_car_03_cat',

 'ps_car_04_cat',

 'ps_car_05_cat',

 'ps_car_06_cat',

 'ps_car_07_cat',

 'ps_car_08_cat',

 'ps_car_09_cat',

 'ps_car_10_cat',

 'ps_car_11',

 'ps_car_11_cat',

 'ps_car_12',

 'ps_car_13',

 'ps_car_14',

 'ps_car_15',

 'ps_ind_01',

 'ps_ind_02_cat',

 'ps_ind_03',

 'ps_ind_04_cat',

 'ps_ind_05_cat',

 'ps_ind_06_bin',

 'ps_ind_07_bin',

 'ps_ind_08_bin',

 'ps_ind_09_bin',

 'ps_ind_10_bin',

 'ps_ind_11_bin',

 'ps_ind_12_bin',

 'ps_ind_13_bin',

 'ps_ind_14',

 'ps_ind_15',

 'ps_ind_16_bin',

 'ps_ind_17_bin',

 'ps_ind_18_bin',

 'ps_reg_01',

 'ps_reg_02',

 'ps_reg_03']



X = train_data.drop(["id","target"],axis=1)

y = train_data["target"].values 

X_test = test_data.drop(["id"],axis=1)



hyper = {"rounds": 1000,

         "early": 100,

         "lgb_pars": lgb_pars,

         "features": base_features,

         "noise_level": 0.1,

         "kfold" : 5,

         "smoothing": 30.0,

         "min_samples_leaf": 300}



#result = perform_full(X, y, X_test, hyper, "kaggle_kernel_base_")



predictions = []



kfold = 5



#for i in range(kfold):    

#    with open("test_kaggle_kernel_base_"+str(i)+".pkl", 'rb') as f:

#        pred = pickle.load(f)        

#    predictions.append(pred)



#final_prediction = np.zeros(predictions[0].shape[0])

#for i in range(kfold):

#    pred = predictions[i]

#    final_prediction += pred / kfold



submission=pd.DataFrame()

submission["id"] = test_data["id"]

#submission["target"] = final_prediction

submission.set_index("id", inplace=True)

submission.to_csv("kaggle_kernel_2.csv")  

#submission["target"].describe()



#Scores on LB:

# 0.28885

# 0.28251

    