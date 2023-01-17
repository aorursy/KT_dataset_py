# Set-up libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
# Check input data source
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Read-in data
df = pd.read_csv('../input/creditcardfraud/creditcard.csv')
# Look at some records
df.info()
# Look at some records
df.head()
# Check for missing values
df.isna().sum()
# Look at breakdown of label
df['Class'].value_counts()
sns.countplot(df['Class'])
# Explore correlations to label
df.corr()['Class'].sort_values(ascending=False)
# Summarise
df.describe()
# Split data into test and validation
X = df.drop('Class', axis=1)
y = df['Class']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
# Build model and train
model = RandomForestClassifier().fit(X_train, y_train)
# Apply model to validation data
y_predict = model.predict(X_val)
# Compare actual and predicted values
actual_vs_predict = pd.DataFrame({'Actual': y_val,
                                'Prediction': y_predict})
actual_vs_predict.sample(12)
# Evaluate model
print('Classification metrics: \n', classification_report(y_val, y_predict))
