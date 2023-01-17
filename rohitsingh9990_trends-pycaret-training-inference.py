!pip install pycaret --quiet
import numpy as np

import pandas as pd

from sklearn.model_selection import KFold



#import regression module

from pycaret.regression import *
BASE_PATH = '../input/trends-assessment-prediction'



fnc_df = pd.read_csv(f"{BASE_PATH}/fnc.csv")

loading_df = pd.read_csv(f"{BASE_PATH}/loading.csv")

labels_df = pd.read_csv(f"{BASE_PATH}/train_scores.csv")
fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])

df = fnc_df.merge(loading_df, on="Id")

labels_df["is_train"] = True

df = df.merge(labels_df, on="Id", how="left")



test_df = df[df["is_train"] != True].copy()

df = df[df["is_train"] == True].copy()

print(f'Shape of train data: {df.shape}, Shape of test data: {test_df.shape}')
target_cols = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']

df.drop(['is_train'], axis=1, inplace=True)

test_df = test_df.drop(target_cols + ['is_train'], axis=1)





# Giving less importance to FNC features since they are easier to overfit due to high dimensionality.

FNC_SCALE = 1/500

df[fnc_features] *= FNC_SCALE

test_df[fnc_features] *= FNC_SCALE
def get_train_data(target):

    other_targets = [tar for tar in target_cols if tar != target]

    train_df = df.drop( other_targets, axis=1)

    return train_df


target = 'age'



train_df = get_train_data(target)



setup_reg = setup(

    data = train_df,

    target = target,

    train_size=0.8,

    numeric_imputation = 'mean',

    silent = True

)
# There are few models which take a lot of time, ignoring those models to make demo faster.

blacklist_models = ['ransac', 'tr', 'rf', 'et', 'ada', 'gbr', 'xgboost', 'catboost']
compare_models(

    blacklist = blacklist_models,

    fold = 5,

    sort = 'MAE', ## competition metric

    turbo = True

)

lgbm_age = create_model(

    estimator='lightgbm',

    fold=5

)
# here we are tuning the above created model

tuned_lgbm_age = tune_model(

    estimator='lightgbm',

    fold=5

)
# plot_model(estimator = None, plot = ‘residuals’)

plot_model(estimator = tuned_lgbm_age, plot = 'learning')
plot_model(estimator = tuned_lgbm_age, plot = 'residuals')
plot_model(estimator = tuned_lgbm_age, plot = 'feature')
evaluate_model(estimator=tuned_lgbm_age)
interpret_model(

    estimator=tuned_lgbm_age,

    plot = 'summary',

    feature = None,

    observation = None

)
predictions =  predict_model(tuned_lgbm_age, data=test_df)
predictions.head()
target = target_cols[0]

train_df = get_train_data(target)



setup_reg = setup(

    data = train_df,

    target = target,

    train_size=0.8,

    numeric_imputation = 'mean',

    silent = True

)



compare_models(

    blacklist = blacklist_models,

    fold = 7,

    sort = 'MAE',

    turbo = True

)
target = target_cols[1]

train_df = get_train_data(target)



setup_reg = setup(

    data = train_df,

    target = target,

    train_size=0.8,

    numeric_imputation = 'mean',

    silent = True

)



compare_models(

    blacklist = blacklist_models,

    fold = 7,

    sort = 'MAE',

    turbo = True

)
target = target_cols[2]

train_df = get_train_data(target)



setup_reg = setup(

    data = train_df,

    target = target,

    train_size=0.8,

    numeric_imputation = 'mean',

    silent = True

)



compare_models(

    blacklist = blacklist_models,

    fold = 7,

    sort = 'MAE',

    turbo = True

)
target = target_cols[3]

train_df = get_train_data(target)



setup_reg = setup(

    data = train_df,

    target = target,

    train_size=0.8,

    numeric_imputation = 'mean',

    silent = True

)



compare_models(

    blacklist = blacklist_models,

    fold = 7,

    sort = 'MAE',

    turbo = True

)
target = target_cols[4]

train_df = get_train_data(target)



setup_reg = setup(

    data = train_df,

    target = target,

    train_size=0.8,

    numeric_imputation = 'mean',

    silent = True

)



compare_models(

    blacklist = blacklist_models,

    fold = 7,

    sort = 'MAE',

    turbo = True

)
# mapping targets to their corresponding models



models = []



target_models_dict = {

    'age': 'br',

    'domain1_var1':'ridge',

    'domain1_var2':'svm',

    'domain2_var1':'ridge',

    'domain2_var2':'svm',

}



def tune_and_ensemble(target):

    train_df = get_train_data(target)    

    exp_reg = setup(

        data = train_df,

        target = target,

        train_size=0.8,

        numeric_imputation = 'mean',

        silent = True

    )

    

    model_name = target_models_dict[target]

    tuned_model = tune_model(model_name, fold=7)

    model = ensemble_model(tuned_model, fold=7)

    return model

target = target_cols[0]

model = tune_and_ensemble(target)

models.append(model)
target = target_cols[1]

model = tune_and_ensemble(target)

models.append(model)
target = target_cols[2]

model = tune_and_ensemble(target)

models.append(model)
target = target_cols[3]

model = tune_and_ensemble(target)

models.append(model)
target = target_cols[4]

model = tune_and_ensemble(target)

models.append(model)
### create a pipelie or functio to run for all targets



def finalize_model_pipeline(model, target):

    # this will train the model on houldout data

    finalize_model(model)

    save_model(model, f'{target}_{target_models_dict[target]}', verbose=True)

    # making predictions on test data

    predictions = predict_model(model, data=test_df)

    test_df[target] = predictions['Label'].values
for index, target in enumerate(target_cols):

    model = models[index]

    finalize_model_pipeline(model,target)
sub_df = pd.melt(test_df[["Id", "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]], id_vars=["Id"], value_name="Predicted")

sub_df["Id"] = sub_df["Id"].astype("str") + "_" +  sub_df["variable"].astype("str")



sub_df = sub_df.drop("variable", axis=1).sort_values("Id")

assert sub_df.shape[0] == test_df.shape[0]*5



sub_df.to_csv("submission1.csv", index=False)
sub_df.head(10)