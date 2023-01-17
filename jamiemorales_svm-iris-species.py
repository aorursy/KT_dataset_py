# Set-up libraries
import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from mpl_toolkits import mplot3d
from sklearn.metrics import classification_report
# Check input data source
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Read-in data
df = pd.read_csv('../input/iris/Iris.csv')
# Check some details
df.info()
# Check some records
df.head()
# Check for missing values
df.isna().sum()
# Check for duplicate values
df.duplicated().sum()
# Check breakdown of label
sns.countplot(df.Species)
df.Species.value_counts()
# Summarise
df.describe()
# Grab subset of data
df = df[['SepalLengthCm', 'SepalWidthCm', 'Species']]
# Split dataset into 80% train and 20% validation
X = df.drop('Species', axis=1)
y = df['Species']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
# Build and train the model(s)
C = 1.0
model_svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
model_svc_rbf = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X_train, y_train)
model_svc_pol = svm.SVC(kernel='poly', degree=3, C=C).fit(X_train, y_train)
model_svc_lin = svm.LinearSVC(C=C).fit(X_train, y_train)
# Apply model(s) to validation data
y_predict_svc = model_svc.predict(X_val)
y_predict_svc_rbf = model_svc_rbf.predict(X_val)
y_predict_svc_pol = model_svc_pol.predict(X_val)
y_predict_svc_lin = model_svc_lin.predict(X_val)
# Compare actual and predicted values
def get_df_name(df):
    name =[x for x in globals() if globals()[x] is df][0]
    return name

predictions = [y_predict_svc,
               y_predict_svc_rbf,
               y_predict_svc_pol,
               y_predict_svc_lin]

for pred in predictions:
    actual_vs_predict = pd.DataFrame({'Model': get_df_name(pred),
                                      'Actual': y_val,
                                      'Predict': pred
                                    })
    print(actual_vs_predict.sample(5))
# Evalute models
print('SVC with linear kernel: \n', classification_report(y_val, y_predict_svc))
print('Linear SVC with linear kernel: \n', classification_report(y_val, y_predict_svc_lin))
print('SVC rbf kernel: \n', classification_report(y_val, y_predict_svc_rbf))
print('SVC with polynomial kernel: \n', classification_report(y_val, y_predict_svc_pol))