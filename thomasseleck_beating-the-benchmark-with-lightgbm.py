import os

import time

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

import lightgbm as lgb

import gc

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder



import riiideducation
st = time.time()

training_set_df = pd.read_csv("/kaggle/input/riiid-test-answer-prediction/train.csv", dtype = {"row_id": "int64", "timestamp": "int64", "user_id": "int32", "content_id": "int16", 

                                                                                               "content_type_id": "int8", "task_container_id": "int16", "user_answer": "int8", 

                                                                                               "answered_correctly": "int8", "prior_question_elapsed_time": "float32", 

                                                                                               "prior_question_had_explanation": "boolean"}, nrows = 5 * 10 ** 6)

questions_df = pd.read_csv("/kaggle/input/riiid-test-answer-prediction/questions.csv", dtype = {"question_id": "int16", "bundle_id": "int16", "correct_answer": "int8", "part": "int8"})

lectures_df = pd.read_csv("/kaggle/input/riiid-test-answer-prediction/lectures.csv", dtype = {"lecture_id": "int16", "tag": "int16", "part": "int8"})

gc.collect()



print("Loaded data in:", round(time.time() - st, 3), "secs")
training_set_df.head()
questions_df.head()
lectures_df.head()
# Drop index for train

training_set_df.drop("row_id", axis = 1, inplace = True)



# Cast boolean column

training_set_df["prior_question_had_explanation"] = training_set_df["prior_question_had_explanation"].fillna(0).astype(np.int8)



# Extract the target

original_shape = training_set_df.shape[0]

training_set_df = training_set_df.loc[training_set_df["answered_correctly"] != -1]

print("Deleted", original_shape - training_set_df.shape[0], "rows where the target was missing.")

target_sr = training_set_df["answered_correctly"]

training_set_df.drop(["answered_correctly", "user_answer"], axis = 1, inplace = True) # Remove also 'user_answer' to avoid leakage
# Prepare 'questions_df' and 'lectures_df' tables for merging with the main table

questions_df["content_type_id"] = 0

questions_df["content_type_id"] = questions_df["content_type_id"].astype(np.int8)

questions_df = questions_df.rename(columns = {"question_id": "content_id"})

lectures_df["content_type_id"] = 1

lectures_df["content_type_id"] = lectures_df["content_type_id"].astype(np.int8)

lectures_df = lectures_df.rename(columns = {"lecture_id": "content_id"})



# Merge 'questions_df' and 'lectures_df' tables to the main dataset

training_set_df = training_set_df.merge(questions_df, how = "left", on = ["content_id", "content_type_id"])

training_set_df = training_set_df.merge(lectures_df, how = "left", on = ["content_id", "content_type_id"])

training_set_df["part"] = (training_set_df["part_x"].fillna(0) + training_set_df["part_y"].fillna(0)).astype(np.int8)

training_set_df.drop(["part_x", "part_y"], axis = 1, inplace = True)

training_set_df.head()
# Remove constant features

tmp = training_set_df.nunique()

constant_features_lst = tmp.loc[tmp < 2].index.tolist()

if len(constant_features_lst) > 0:

    print("Found", len(constant_features_lst), "constant features:")

    for f in constant_features_lst:

        print("  - " + f)

        

    training_set_df.drop(constant_features_lst, axis = 1, inplace = True)
# Drop "tags" feature

training_set_df.drop("tags", axis = 1, inplace = True)
# Encode categorical features

categorical_features_lst = ["task_container_id", "prior_question_had_explanation", "bundle_id", "correct_answer", "part"] # "user_id"

label_encoders_dict = {}



for col in categorical_features_lst:

    le = LabelEncoder()

    training_set_df[col] = le.fit_transform(training_set_df[col])

    label_encoders_dict[col] = le
st = time.time()



# Hyperparameters for LightGBM

lgb_params = {

    "boosting_type": "gbdt",

    "metric": "auc",

    "objective": "binary",

    "n_jobs": 4,

    "seed": 42,

    "learning_rate": 0.03,

    "subsample": 0.75,

    "bagging_freq": 1,

    "colsample_bytree": 0.77,

    "max_depth": -1,

    "num_leaves": 40,

    "reg_alpha": 0.05,

    "reg_lambda": 0.05,

    "verbosity": -1

}



# Split training data into training and validation datasets

X_train, X_valid, y_train, y_valid = train_test_split(training_set_df, target_sr, test_size = 0.20, random_state = 42)

del training_set_df

gc.collect()



# Generate LightGBM datasets

lgb_train = lgb.Dataset(X_train, y_train, categorical_feature = categorical_features_lst)

lgb_eval = lgb.Dataset(X_valid, y_valid, categorical_feature = categorical_features_lst)



# Try to save some memory

gc.collect()



lgb_model = lgb.train(lgb_params, lgb_train, valid_sets = [lgb_train, lgb_eval], verbose_eval = 10, num_boost_round = 500, early_stopping_rounds = 50)

gc.collect()



# Try to save some memory

gc.collect()



print("Trained LightGBM in:", round(time.time() - st, 3), "secs")
# Free some memory by deletng training set

del X_train, X_valid, y_train, y_valid, lgb_train, lgb_eval



# Try to save some memory

gc.collect()
st = time.time()



# Generate the submission environment

submission_env = riiideducation.make_env()



# Actually make predictions

for (testing_set_df, sample_prediction_df) in submission_env.iter_test():    

    X_test = testing_set_df.drop(["prior_group_answers_correct", "prior_group_responses", "row_id"], axis = 1)

        

    # Cast boolean column

    X_test["prior_question_had_explanation"] = X_test["prior_question_had_explanation"].fillna(0).astype(np.int8)

    

    # Merge 'questions_df' and 'lectures_df' tables to the main dataset

    X_test = X_test.merge(questions_df, how = "left", on = ["content_id", "content_type_id"])

    X_test = X_test.merge(lectures_df, how = "left", on = ["content_id", "content_type_id"])

    X_test["part"] = (X_test["part_x"].fillna(0) + X_test["part_y"].fillna(0)).astype(np.int8)

    X_test.drop(["part_x", "part_y"], axis = 1, inplace = True)

    

    # Drop constant features

    if len(constant_features_lst) > 0:

        X_test.drop(constant_features_lst, axis = 1, inplace = True)

        

    # Drop "tags" feature

    X_test.drop("tags", axis = 1, inplace = True)

    

    # Encode categorical features

    for i, col in enumerate(categorical_features_lst):

        X_test[col] = label_encoders_dict[col].transform(X_test[col])

    

    # Make predictions using LightGBM

    predictions_npa = lgb_model.predict(X_test, num_iteration = lgb_model.best_iteration)

    testing_set_df["answered_correctly"] = predictions_npa

    submission_env.predict(testing_set_df.loc[testing_set_df["content_type_id"] == 0, ["row_id", "answered_correctly"]])

    

    # Try to save some memory

    gc.collect()

    

print("Made predictions in:", round(time.time() - st, 3), "secs")
importance = lgb_model.feature_importance(importance_type = "gain")

features_names = lgb_model.feature_name()

feature_importance_df = pd.DataFrame({"feature": features_names, "importance": importance}).sort_values(by = "importance", ascending = False).reset_index(drop = True)

feature_importance_df.to_csv("lgb_feature_importance.csv", index = False)

feature_importance_df.head(30)