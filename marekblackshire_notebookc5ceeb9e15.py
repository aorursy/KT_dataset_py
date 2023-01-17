# install packages and import libraries

print("Installing packages...")

!pip install -Uqqq --use-feature=2020-resolver pycaret

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pycaret.classification import * # machine learning library

from ipywidgets import interact, interactive

from IPython.display import display



print("Loading data...")

# read data

data=pd.read_csv('../input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv')



# clean data

# education

ed_filter = (data.EDUCATION == 5) | (data.EDUCATION == 6) | (data.EDUCATION == 0)

data.loc[ed_filter, 'EDUCATION'] = 4



# marriage

data.loc[data.MARRIAGE == 0, 'MARRIAGE'] = 3



# split dataset into learning and verification

dataset = data.sample(frac=0.999, random_state=8675309).reset_index(drop=True)

data_unseen = data.drop(dataset.index).reset_index(drop=True)



# do pycaret setup

classifier = setup(data=dataset, target="default.payment.next.month", ignore_features=["ID"], silent=True, verbose=False, profile=False, session_id=8675309)



print("Training Logistic Regression model...")

log_reg = create_model("lr", verbose=False)

tuned_log_reg = tune_model(log_reg, choose_better=True, verbose=False)

final_log_reg = finalize_model(tuned_log_reg)

log_reg_results = predict_model(final_log_reg,probability_threshold=0.46, data=data_unseen)



print("Training Light Gradient Boosting Machine...")

grad_boost = create_model("lightgbm", verbose=False)

tuned_grad_boost = tune_model(grad_boost, choose_better=True, verbose=False)

final_grad_boost = finalize_model(tuned_grad_boost)

grad_boost_results = predict_model(final_grad_boost,probability_threshold=0.34, data=data_unseen)



print("Setup complete.")
# Feature Importance

plot_model(final_log_reg, plot="feature")
# Accuracy

plot_model(final_log_reg, plot="error")
# This gets the SHAP value of different fields to determine which are most impactful to the final result.

# The SHAP value is usually different from Feature Importance. It cannot be calculated for the Logistic Regression model because it is only applicable to tree-based models.

interpret_model(final_grad_boost, plot="summary")
# Feature Importance

plot_model(final_grad_boost, plot="feature")
# Accuracy

plot_model(final_grad_boost, plot="error")
# Predictor

def f(Customer):

    i = data_unseen.iloc[Customer]

    display(i)

    display('Predictions:')

    display('Logistic Regression: ' + ('Default' if log_reg_results.take([Customer])['Label'][Customer] == '1' else 'No Default'))

    display('Light Gradient Boost: ' + ('Default' if grad_boost_results.take([Customer])['Label'][Customer] == '1' else 'No Default'))

    display('Actual: ' + ('Default' if i['default.payment.next.month'] == 1 else 'No Default'))

interact(f, Customer=[(str(r + 29971), r) for r in range(0,30)])
# Compare models to find best algorithm

# Ridge classifier excluded due to not being a binary classifier

# Gradient boost, extreme gradient boost, catboost, svm, and linear discriminant removed for time

compare_models(exclude = ['ridge', 'gbc', 'xgboost', 'catboost', 'svm', 'lda'])
# Calculate optimal threshold for Logistic Regression

# True positives and false negatives are more heavily weighted to give better results

optimize_threshold(final_log_reg, true_positive=10, true_negative=5, false_positive=-10, false_negative=-15)
# Calculate optimal threshold for Light Gradient Boosting Machine

# True positives and false negatives are more heavily weighted to give better results

optimize_threshold(final_grad_boost, true_positive=10, true_negative=5, false_positive=-10, false_negative=-15)
# This evaluates the Logistic Regression model by several useful metrics. Some of these, such as Feature Selection and Manifold Learning take a very long time to calculate.

evaluate_model(final_log_reg)
# This evaluates the Light Gradient Boosting Machine model by several useful metrics.

# Some of these, such as Feature Selection and Manifold Learning take a very long time to calculate.

evaluate_model(final_grad_boost)
# Test the model against the holdout data set to get accuracy information.

# Note that this is not the finalized model since that has no holdout and cannot be evaluated for accuracy.

predict_model(tuned_log_reg, probability_threshold=0.46)
# Test the model against the holdout data set to get accuracy information

# Note that this is not the finalized model since that has no holdout and cannot be evaluated for accuracy.

predict_model(tuned_grad_boost, probability_threshold=0.34)
# View final results from logistic regression against unseen data

log_reg_results
# View final results from light gradient boosting machine against unseen data

grad_boost_results