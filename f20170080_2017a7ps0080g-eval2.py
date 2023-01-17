import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import sklearn

import seaborn as sns

from sklearn.model_selection import train_test_split

import xgboost as xgb

train = pd.read_csv("Data/train.csv")

test = pd.read_csv("Data/test.csv")

test_index = test['id']
train.head()
test.head()
train.describe()
labels = np.array(train['class'])

unique, counts = np.unique(labels, return_counts=True)

    

for i in range(1,8):

    print("Class", i, ":", np.count_nonzero(train['class'] == i))
# def balance(data, limit):

#     original_len = data.shape[0]    

#     out = pd.DataFrame()

    

#     for i in range(original_len+1, limit+1):

#         random_idx = int(np.round(np.random.rand() * (original_len-1)))

#         random_row = data.iloc[random_idx]

#         out = out.append(random_row, ignore_index=True)

        

#     return out



# def augment_data(limit):

    

#     t = train

    

#     for i in [1,2,3,5,6,7]:

#         original_data = train.loc[train['class'] == i]

#         augmented_data = balance(original_data,limit)

#         # print(augmented_data.shape[0])

#         t = t.append(augmented_data)

    

#     return t

    

# train = augment_data(49)



# for i in range(1,8):

#     print("Class", i, ":", np.count_nonzero(train['class'] == i))
plt.figure(figsize=(12,10))

sns.heatmap(train.corr(), annot=True, cmap='RdYlGn', linewidths=0.2) 

plt.show()
#Correlation with output variable

cor = train.corr()

cor_target = abs(cor["class"])

#Selecting highly correlated features

relevant_features = cor_target[cor_target>0.45]

relevant_features
# chosen_features = ["chem_1", "chem_4"]

# chosen_features = ["chem_1", "chem_4", "chem_0"]

# chosen_features = ["chem_1", "chem_4", "attribute"]

# chosen_features = ["chem_0","chem_1","chem_4","chem_6"]

# chosen_features = ['chem_0', 'chem_1', 'chem_2', 'chem_3', 'chem_4']

# chosen_features = ['chem_0', 'chem_1', 'chem_2', 'chem_3', 'chem_4','chem_7']

# chosen_features = ['chem_0', 'chem_1', 'chem_2', 'chem_3', 'chem_4','chem_7', 'attribute']

# chosen_features = ['chem_0', 'chem_1', 'chem_2', 'chem_3', 'chem_4', 'chem_5', 'chem_7', 'attribute']

chosen_features = ['chem_0', 'chem_1', 'chem_2', 'chem_3', 'chem_4', 'chem_5', 'chem_6', 'chem_7', 'attribute']



# choose = train[chosen_features]

# print(choose.corr())


def randomize(train_size = 0.80):

    X_train, X_val, y_train, y_val = train_test_split(

                                            train[chosen_features], 

                                            train["class"], 

                                            train_size=train_size, 

    #                                         random_state=0

                                        )

    

    y_val = y_val.values

    y_train = y_train.values



    return X_train, X_val, y_train, y_val



X_train, X_val, y_train, y_val = randomize(119)

X_test = test[chosen_features]



# X_test = test
print(X_train.shape)

print(y_train.shape)

print(X_val.shape)

print(y_val.shape)

print(X_test.shape)
from sklearn.metrics import confusion_matrix



def get_acc(model, round=True):

    val_pred = model.predict(X_val)

    

    if(round): 

        val_pred = np.round(val_pred)

        val_pred = val_pred.astype(np.int) 

    

    acc = (val_pred == y_val).mean() * 100

    print("Accuracy :", acc)

    

    print("True :", y_val)

    print("Pred :", val_pred)

    

    plt.figure(figsize = (10,10))

    ax = sns.heatmap(

                confusion_matrix(y_val, val_pred),

                annot=True,

                fmt='2.0f',

                square = True,

                xticklabels=[1,2,3,5,6,7],

                yticklabels=[1,2,3,5,6,7],

                cbar = False

            )

    

# Feature Importance

def get_fi(model):

    print(model.feature_importances_)

    feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)

    feat_importances.nlargest(100).plot(kind='barh')

    plt.show()

    

def reg_score(X,y):

    pred = model.predict(X)

    pred = np.round(pred)

    pred = pred.astype(np.int) 

    acc = (pred == y).mean() * 100

    print("Accuracy :", acc)
# import lightgbm



# train_data = lightgbm.Dataset(X_train, label=y_train)

# val_data = lightgbm.Dataset(X_val, label=y_val)



# parameters = {

#     'application': 'binary',

#     'objective': 'binary',

#     'metric': 'auc',

#     'is_unbalance': 'true',

#     'boosting': 'gbdt',

#     'num_leaves': 31,

#     'feature_fraction': 0.5,

#     'bagging_fraction': 0.5,

#     'bagging_freq': 20,

#     'learning_rate': 0.05,

#     'verbose': 0

# }



# model = lightgbm.train(parameters,

#                        train_data,

#                        valid_sets=val_data,

#                        num_boost_round=5000,

#                        early_stopping_rounds=100)
# submission = pd.read_csv('../input/test.csv')

# ids = submission['id'].values

# submission.drop('id', inplace=True, axis=1)





# x = submission.values

# y = model.predict(x)



# output = pd.DataFrame({'id': ids, 'target': y})

# output.to_csv("submission.csv", index=False)
# model.predict(X_val)
# y_val
cum_train = 0

cum_val = 0

num_trials = 20



for trial in range(num_trials):



    X_train, X_val, y_train, y_val = randomize(0.80)



    model = lightgbm.LGBMClassifier(

                boosting_type='dart',

                silent = True, 

                num_estimators=500, 

                learning_rate=0.30,

                min_split_gain=0.03

#                 num_leaves=31,

#                 max_depth=-1,

#                 reg_alpha=0,

#                 reg_lambda=1

        )



    t = model.fit(

            X_train, y_train, 

            eval_set = (X_val, y_val), 

            eval_metric = 'multi_error',

            verbose=False

        )



    cur_train = model.score(X_train, y_train)

    cur_val = model.score(X_val, y_val)

    cum_train += cur_train

    cum_val += cur_val

#     print("%.4f \t %.4f" % (cur_train, cur_val))

    

cum_train /= num_trials 

cum_val /= num_trials    

    

# print('-----------------------------------')

print(model)

print('-----------------------------------')

print("Train score : ", cum_train)

print("Val score : ", cum_val)
get_acc(model)
get_fi(model)
from sklearn.model_selection import GridSearchCV



n_estimators = np.array([100,500,1000])

boosting_type = ['gbdt','dart','goss','rf']

learning_rate = [0.01, 0.03, 0.1, 0.3, 1]



grid = GridSearchCV(

                estimator = model, 

                param_grid = {

                        "n_estimators" : n_estimators,

                        "boosting_type" : boosting_type,

                        "learning_rate" : learning_rate,

                        },

                scoring = "accuracy"

            )



t = grid.fit(X_train, y_train)
print(grid.best_score_)

print(grid.best_estimator_.n_estimators)

print(grid.best_estimator_.booster)

print(grid.best_estimator_.learning_rate)

print(grid.best_estimator_.max_depth)
from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier(

                n_estimators=76, 

                max_depth=None, 

                random_state=0

            )



t = model.fit(X_train, y_train)
get_acc(model)
from sklearn.model_selection import GridSearchCV



# LightGBM

n_estimators = np.array([1000])

boosting_type = ['gbdt','dart','goss','rf']

learning_rate = [1]



# SKlearn

# criterion = ["entropy", "gini"]

# n_estimators = np.arange(70,91,2)





grid = GridSearchCV(

                estimator = model, 

                param_grid = {

                        "n_estimators" : n_estimators,

                        "criterion" : criterion,

                        "learning_rate" : learning_rate

                        },

                scoring = "accuracy"

            )



t = grid.fit(X_train, y_train)
print(grid.best_score_)

print(grid.best_estimator_.n_estimators)

print(grid.best_estimator_.boosting_type)

print(grid.best_estimator_.learning_rate)



# print(grid.best_estimator_.criterion)

grid.cv_results_
sorted(grid.cv_results_.keys())
grid.predict(X_val)

get_acc(grid)
# data_dmatrix = xgb.DMatrix(data=X,label=y)
cum_train = 0

cum_val = 0

num_trials = 10



for trial in range(num_trials):



    X_train, X_val, y_train, y_val = randomize(0.80)



    model = xgb.XGBClassifier(

                    max_depth=100,

                    learning_rate=0.30,

                    n_estimators=500,

                    booster='dart',

                    gamma=0.03,

#                     reg_alpha=3,

#                     reg_lambda=1

                )



    model.fit(

                X_train,

                y_train, 

                eval_set=[(X_val, y_val)],

                eval_metric="merror",

                early_stopping_rounds=300,

                verbose=False,

            )



    cur_train = model.score(X_train, y_train)

    cur_val = model.score(X_val, y_val)

    cum_train += cur_train

    cum_val += cur_val

    print("%.4f \t %.4f \t %.4f"%(cur_train, cur_val, 1 - model.best_score))

    

cum_train /= num_trials 

cum_val /= num_trials    

    

# print('-----------------------------------')

print(model)

# print('-----------------------------------')

print("Train score : ", cum_train)

print("Val score : ", cum_val)

# model.best_iteration
get_acc(model)
get_fi(model)
from sklearn.model_selection import GridSearchCV



max_depth = [1, 2, 3]

n_estimators = np.array([1500, 2000, 2500])

booster = ['dart']

learning_rate = [0.01, 0.03, 0.05, 0.1]



grid = GridSearchCV(

                estimator = model, 

                param_grid = {

                        "max_depth" : max_depth,

                        "n_estimators" : n_estimators,

                        "booster" : booster,

                        "learning_rate" : learning_rate,

                        },

                scoring = "accuracy"

            )



t = grid.fit(X_train, y_train)
print(grid.best_score_)

print(grid.best_estimator_.n_estimators)

print(grid.best_estimator_.booster)

print(grid.best_estimator_.learning_rate)

print(grid.best_estimator_.max_depth)
from sklearn.ensemble import GradientBoostingClassifier

import math



for i in range(1):

    

    n_estimators = 5000

    max_depth = 4

    learning_rate = 0.005

    warm_start = True

    subsample = 0.5



    params = {

                'n_estimators': n_estimators, 

                'max_depth': max_depth, 

                'learning_rate': learning_rate, 

                'warm_start' : warm_start,

                'subsample' : subsample,

                'loss': 'deviance',

             }

    

    model = GradientBoostingClassifier(**params)

    model.fit(X_train, y_train)



    pred_val = np.round(model.predict(X_val))



    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

    min_test_score = 100

    

    for ii, y_pred in enumerate(model.staged_predict(X_val)):

        test_score[ii] = model.loss_(y_val, y_pred)

        

        if(test_score[ii] < min_test_score):

            min_test_score = test_score[ii]

            actual_error = np.sqrt(min_test_score)

            print("Error %.4f at %d iteration." % (actual_error, ii))

            

#             if(ii > n_estimators * .8):

#                 generate_submission(model, 'temp'+str(i)+'.csv')

        

    error_min = math.sqrt(np.min(test_score))

    error_end = math.sqrt(test_score[-1])

    

    pred_test = model.predict(X_test)

    pred_test = np.round(pred_test)

    pred_test = np.maximum(np.zeros_like(pred_test), pred_test)



#     pred_test = pd.DataFrame(pred_test)

#     pred_test.index = test_index

   

    print("-----------------------------------------------------------------------------------------------------")

    print('i = ', i)

    print(params)

    print("End error: %.4f" % error_end)

    print("Min error: %.4f" % error_min)

    print("At iter: %d" % np.argmin(test_score))

    

    

    plt.figure(figsize=(12, 6))

        

    # compute test set deviance

    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)



    for i, y_pred in enumerate(model.staged_predict(X_val)):

        test_score[i] = model.loss_(y_val, y_pred)



    train_error = np.sqrt(model.train_score_)

    test_error = np.sqrt(test_score)





    plt.subplot(1, 2, 1)

    plt.title('Deviance')

    plt.plot(np.arange(params['n_estimators']) + 1, train_error, 'b-',

             label='Training Set Deviance')

    plt.plot(np.arange(params['n_estimators']) + 1, test_error, 'r-',

             label='Test Set Deviance')

    plt.legend(loc='upper right')

    plt.xlabel('Boosting Iterations')

    plt.ylabel('Deviance')

    plt.grid()

    plt.show()

    plt.close()

    

    

#     fname = 'Sample' + str(i) + '.csv'

#     pred_test.to_csv(fname)
def generate_submission(model, fname):

    pred_test = model.predict(X_test).astype(int)

    pred_data = np.array([test_index, pred_test]).T

    pred_data = pd.DataFrame(pred_data, columns=["id", "class"])

    pred_data.to_csv(fname,index=False)

    

    for i in range(1,8):

        print("Class", i, ":", np.count_nonzero(pred_data['class'] == i))
generate_submission(model, "sub14.csv")