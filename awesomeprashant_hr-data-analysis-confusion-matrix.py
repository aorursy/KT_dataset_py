import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt









hr_df = pd.read_csv('../input/HR_comma_sep.csv')

hr_df['dept'] = hr_df['sales']

del hr_df['sales']



hr_df_label = hr_df.pop('left')

#print(hr_df_label)

from sklearn.model_selection import train_test_split

data_train, data_test, label_train, label_test  = train_test_split(hr_df, hr_df_label, test_size = 0.4, random_state = 1500)

from sklearn import linear_model



# Initialize logistic regression model

log_model = linear_model.LogisticRegression()



#consider only the parameter having "coefficient" > 1 

train_features = pd.DataFrame([data_train['satisfaction_level'],

                              data_train['Work_accident'],

                              data_train['promotion_last_5years'] 

                              ]).T



# Initialize logistic regression model

log_model = linear_model.LogisticRegression()



# Train the model

log_model.fit(X = train_features ,

              y = label_train)



# Check trained model intercept

print(log_model.intercept_)



# Check trained model coefficients

print(log_model.coef_)



# Make predictions

preds = log_model.predict(X= train_features)



train_score = log_model.score(X = train_features ,

                y = label_train)

print(train_score)



# Generate table of predictions vs actual

pd.crosstab(preds,label_train)

# Test the model

test_features = pd.DataFrame([data_test['satisfaction_level'],

                              data_test['Work_accident'],

                              data_test['promotion_last_5years'] 

                              ]).T



log_model.fit(X = test_features ,

              y = label_test)



# Check trained model intercept

print(log_model.intercept_)



# Check trained model coefficients

print(log_model.coef_)

# Make predictions

preds = log_model.predict(X= test_features)



test_score = log_model.score(X = test_features ,

                y = label_test)

print(test_score)

# Generate table of predictions vs actual

pd.crosstab(preds,label_test)