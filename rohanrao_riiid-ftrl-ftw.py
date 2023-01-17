# installing datatable

!pip install ../input/python-datatable/datatable-0.11.0-cp37-cp37m-manylinux2010_x86_64.whl
## importing packages

import datatable as dt

import pandas as pd



from datatable.models import Ftrl

from sklearn.metrics import roc_auc_score



import riiideducation
## reading data

train = dt.fread("../input/riiid-train-data-multiple-formats/riiid_train.jay")

questions = dt.fread("../input/riiid-test-answer-prediction/questions.csv")
## viewing train data

train
## viewing questions data

questions
## merging questions metadata with train data

questions.key = "question_id"

train.names = {"content_id": "question_id"}



train = train[dt.f.content_type_id == 0, :]

train = train[:, :, dt.join(questions)]
## preparing train and validation data

train_features = ["user_id", "question_id", "prior_question_elapsed_time"]

question_features = ["bundle_id", "part", "tags"]



target = train[:, "answered_correctly"]

train = train[:, train_features + question_features]



X_train, X_valid = train[:90000000, :], train[90000000:, :]

y_train, y_valid = target[:90000000, :], target[90000000:, :]
## building and validating FTRL model

model_ftrl = Ftrl() # you can set hyper-parameters with: model_ftrl = Ftrl(alpha = 0.005, nepochs = 1)



model_ftrl.fit(X_train, y_train, X_validation=X_valid, y_validation=y_valid)

y_pred = model_ftrl.predict(X_valid)

    

print(f"Validation AUC: {roc_auc_score(y_valid.to_numpy(), y_pred.to_numpy())}")
## rebuilding FTRL model on entire dataset

model_ftrl = Ftrl()



model_ftrl.fit(train, target)
## initializing test environment

env = riiideducation.make_env()

iter_test = env.iter_test()
## inferencing and incremental learning

prev_test = pd.DataFrame()



for (current_test, current_prediction_df) in iter_test:



    # extracting previous batch's targets

    prev_target = eval(current_test["prior_group_answers_correct"].iloc[0])



    # incremental learning of FTRL model

    if prev_test.shape[0] > 0:

        prev_test["target"] = prev_target

        X_prev_test = dt.Frame(prev_test[prev_test.content_type_id == 0].rename(columns = {"content_id": "question_id"})[train_features + ["target"]])

        X_prev_test = X_prev_test[:, :, dt.join(questions)]



        y_prev_test = X_prev_test[:, "target"]

        X_prev_test = X_prev_test[:, train_features + question_features]



        model_ftrl.fit(X_prev_test, y_prev_test)



    # inferencing of current batch

    X_test = dt.Frame(current_test[current_test.content_type_id == 0].rename(columns = {"content_id": "question_id"})[train_features])

    X_test = X_test[:, :, dt.join(questions)]

    X_test = X_test[:, train_features + question_features]

    current_prediction_df.answered_correctly = model_ftrl.predict(X_test).to_numpy().ravel()

    env.predict(current_prediction_df)



    # retaining current batch data for next batch

    prev_test = current_test.copy(deep = True)
