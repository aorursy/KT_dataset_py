# Set-up libraries
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
# Check input data source
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Read-in data
df = pd.read_csv('../input/paysim1/PS_20174392719_1491204439457_log.csv')
# Look at some details
df.info()
# Look at some records
df.head()
# Check for missing values
df.isna().sum()
# Look at breakdown of label
df.isFlaggedFraud.value_counts()
sns.countplot(df.isFlaggedFraud)
# Look at breakdown of categorical feature(s)
df.type.value_counts()
sns.countplot(df.type)
# Summarise
df.describe()
# Transform categorical feature(s) to numeric
le = LabelEncoder()
df.type = le.fit_transform(df.type)
# Explore correlations to label
df.corr().isFlaggedFraud.sort_values(ascending=False)
# Explore correlations visually
f, ax = plt.subplots(figsize=(12,6))
sns.heatmap(df.corr(), annot=True, fmt='.2f')
# Split data into 80% train and 20% test
X = df.drop(columns=['isFlaggedFraud', 'nameDest', 'nameOrig'], axis=1)
y = df['isFlaggedFraud']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
# Build and train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
# Apply model to validation data
y_predict = model.predict(X_val)
# Compare actual and predicted values
actual_vs_predict = pd.DataFrame({'Actual': y_val,
                                'Prediction': y_predict})
actual_vs_predict.sample(12)
# Evaluate model
print('Classification metrics: \n', classification_report(y_val, y_predict))