import os
print(os.listdir("../input"))
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv("../input/glass.csv")
df.head()
# Display class values
df.Type.value_counts().sort_index()
# glass_type 1, 2, 3 are window glass
# glass_type 5, 6, 7 are non-window glass
df['household'] = df.Type.map({1:0, 2:0, 3:0, 5:1, 6:1, 7:1})
df.head()
plt.scatter(df.Al, df.household)
plt.xlabel('Al')
plt.ylabel('household')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['Al']],df.household,train_size=0.7)
from sklearn.linear_model import LinearRegression
# Fit the model
linear_model = LinearRegression()
linear_model = linear_model.fit(X_train, y_train)
# Create a seperate table to store predictions
glass_df = X_train[['Al']]
glass_df['household_actual'] = y_train

# Predict with Linear Regression
glass_df['household_pred_linear'] = linear_model.predict(X_train)

# Examine the first 15 linear regression predictions
linear_model.predict(X_train)[0:15]
# Plot Linear Regression Line
sns.regplot(x='Al', y='household_actual', data=glass_df, logistic=False)
from sklearn.linear_model import LogisticRegression
# Fit logistic regression model
logistic_model = LogisticRegression(class_weight='balanced')
logistic_model = logistic_model.fit(X_train, y_train)
# Make class label predictions
logistic_model.predict(X_train)[:15]
# Make class probability predictions
logistic_model.predict_proba(X_train)[:15]
# Predict with Logistic Regression
glass_df['household_pred_log'] = logistic_model.predict(X_train)

# Predict Probability with Logistic Regression
glass_df['household_pred_prob_log'] = logistic_model.predict_proba(X_train)[:,1]
# Plot logistic regression line 
sns.regplot(x='Al', y='household_actual', data=glass_df, logistic=True, color='b')
# Examine the table
glass_df.head(10)
# Observe class predictions on test set
logistic_model.predict(X_test)
# Store predictions
predicted = logistic_model.predict(X_test)
from sklearn import metrics
# Print Confusion Matrix
print (metrics.confusion_matrix(y_test, predicted))
print (metrics.classification_report(y_test, predicted))
# Let's use the statsmodel library 
import statsmodels.api as sm

# Define independent variables
iv = ['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe']

# Fit the logistic regression function
logReg = sm.Logit(df.household,df[iv])
answer = logReg.fit()
# Display the parameter coefficients 
np.exp(answer.params)