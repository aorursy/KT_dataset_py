# Set-up libraries
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
# Check input data source
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Read-in data
df = pd.read_csv('../input/mushroom-classification/mushrooms.csv')
# Look at some details
df.info()
# Look at some records
df.head()
# Check for missing values
df.isna().sum()
# Check for duplicate values
df.duplicated().sum()
# Look at breakdown of label
sns.countplot(df['class'])
df['class'].value_counts()
# Summarise
df.describe()
# Grab some samples
df = df.sample(5000, random_state=0)
# Transform categorical feature(s) to numeric
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    
df.describe()
# Explore correlation to label
df.corr()['class'].sort_values(ascending=False)
# Explore correlations visually
f, ax = plt.subplots(figsize=(24,6))
sns.heatmap(df.corr(), annot=True, fmt='.2f')
# Split dataset into 80% train and 20% validation
X = df.drop('class', axis=1)
y = df['class']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
# Build model and train
model = SVC()
model.fit(X_train, y_train)
# Apply model to validation data
y_predict = model.predict(X_val)
# Compare actual and predicted values
actual_vs_predict = pd.DataFrame({'Actual': y_val,
                                 'Predict': y_predict
                                 })

actual_vs_predict.sample(12)
# Evaluate model
print('Classification metrics: \n', classification_report(y_val, y_predict))
