import numpy as np
import pandas as pd
# Open file and inspect first five rows.
df_A = pd.read_csv('../input/creditcard.csv')
df_A.head()
# Check shape.
df_A.shape
# Check for nulls.
df_A.isnull().sum()
# Show class distribution.
print(df_A['Class'].value_counts(),'\n')

# Show proportion of fraudulent transactions.
print((df_A['Class'] == 1).mean())
# Set the variables.
X_B1 = df_A.iloc[:,1:30]
Y_B1 = df_A['Class']

# Split into train and test data.
from sklearn.cross_validation import train_test_split
X_train_B1, X_test_B1, y_train_B1, y_test_B1 = train_test_split(X_B1, Y_B1, test_size = .3, random_state=25)

# Standardize training features.
from sklearn.preprocessing import StandardScaler
scaler_B1 = StandardScaler().fit(X_train_B1)
X_train_B1_trans = scaler_B1.transform(X_train_B1)

# Fit the model.
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression()
logr.fit(X_train_B1_trans, y_train_B1)

# Standardize test features based on training set.
X_test_B1_trans = scaler_B1.transform(X_test_B1)

# Make predictions.
logr_pred_B1 = logr.predict(X_test_B1_trans)

# Accuracy score.
from sklearn.metrics import accuracy_score
print('Score:', accuracy_score(y_test_B1, logr_pred_B1))
# Confusion matrix.
from sklearn.metrics import confusion_matrix
confusion_matrix_B1 = confusion_matrix(y_test_B1, logr_pred_B1)
print(confusion_matrix_B1)
# Classification matrix.
from sklearn.metrics import classification_report
print(classification_report(y_test_B1, logr_pred_B1))
# Separate majority and minority classes.
df_B2_maj = df_A[df_A['Class']==0]
df_B2_min = df_A[df_A['Class']==1]

# Upsample minority class.
from sklearn.utils import resample
df_B2_min_up = resample(df_B2_min, replace=True, n_samples=284315, random_state=123)

# Combine majority class with upsampled minority class.
df_B2_up = pd.concat([df_B2_maj, df_B2_min_up])

# Display new class counts.
df_B2_up['Class'].value_counts()
# Set the variables.
X_B2 = df_B2_up.iloc[:,1:30]
Y_B2 = df_B2_up['Class']

# Split into train and test data.
X_train_B2, X_test_B2, y_train_B2, y_test_B2 = train_test_split(X_B2, Y_B2, test_size = .3, random_state=25)

# Standardize training features.
scaler_B2 = StandardScaler().fit(X_train_B2)
X_train_B2_trans = scaler_B2.transform(X_train_B2)

# Fit the model.
logr.fit(X_train_B2_trans, y_train_B2)

# Standardize test features based on training set.
X_test_B2_trans = scaler_B2.transform(X_test_B2)

# Make predictions.
logr_pred_B2 = logr.predict(X_test_B2_trans)

# Accuracy score.
print('Score:', accuracy_score(y_test_B2, logr_pred_B2))
# Confusion matrix.
confusion_matrix_B2 = confusion_matrix(y_test_B2, logr_pred_B2)
print(confusion_matrix_B2)
# Classification matrix.
print(classification_report(y_test_B2, logr_pred_B2))
# Downsample majority class.
df_B3_maj_down = resample(df_B2_maj, replace=False, n_samples=492, random_state=123)

# Combine minority class with downsampled majority class.
df_B3_down = pd.concat([df_B2_min, df_B3_maj_down])

# Display new class counts.
df_B3_down['Class'].value_counts()
# Set the variables.
X_B3 = df_B3_down.iloc[:,1:30]
Y_B3 = df_B3_down['Class']

# Split into train and test data.
X_train_B3, X_test_B3, y_train_B3, y_test_B3 = train_test_split(X_B3, Y_B3, test_size = .3, random_state=25)

# Standardize training features.
scaler_B3 = StandardScaler().fit(X_train_B3)
X_train_B3_trans = scaler_B3.transform(X_train_B3)

# Fit the model.
logr.fit(X_train_B3_trans, y_train_B3)

# Standardize test features based on training set.
X_test_B3_trans = scaler_B3.transform(X_test_B3)

# Make predictions.
logr_pred_B3 = logr.predict(X_test_B3_trans)

# Accuracy score.
print('Score:', accuracy_score(y_test_B3, logr_pred_B3))
# Confusion matrix.
confusion_matrix_B3 = confusion_matrix(y_test_B3, logr_pred_B3)
print(confusion_matrix_B3)
# Classification matrix.
print(classification_report(y_test_B3, logr_pred_B3))
# Fit the model on the original training set. 
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train_B1_trans, y_train_B1)

# Predict on the original test set.
pred_rfc_C1 = rfc.predict(X_test_B1_trans)

# Accuracy score.
print('Score:', accuracy_score(y_test_B1, pred_rfc_C1))
# Confusion matrix.
confusion_matrix_C1 = confusion_matrix(y_test_B1, pred_rfc_C1)
print(confusion_matrix_C1)
# Classification matrix.
print(classification_report(y_test_B1, pred_rfc_C1))
# Fit the model on the up-sample training set.
rfc.fit(X_train_B2_trans, y_train_B2)

# Predict on the up-sample test set.
pred_rfc_C2 = rfc.predict(X_test_B2_trans)

# Accuracy score.
print('Score:', accuracy_score(y_test_B2, pred_rfc_C2))
# Confusion matrix.
confusion_matrix_C2 = confusion_matrix(y_test_B2, pred_rfc_C2)
print(confusion_matrix_C2)
# Classification matrix.
print(classification_report(y_test_B2, pred_rfc_C2))
# Fit the model on the down-sample training set.
rfc.fit(X_train_B3_trans, y_train_B3)

# Predict on the down-sample test set.
pred_rfc_C3 = rfc.predict(X_test_B3_trans)

# Accuracy score.
print('Score:', accuracy_score(y_test_B3, pred_rfc_C3))
# Confusion matrix.
confusion_matrix_C3 = confusion_matrix(y_test_B3, pred_rfc_C3)
print(confusion_matrix_C3)
# Classification matrix.
print(classification_report(y_test_B3, pred_rfc_C3))
