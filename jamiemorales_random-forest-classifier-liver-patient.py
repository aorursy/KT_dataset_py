# Set-up libraries needed
import os
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
# Check input data source
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Read-in data
df = pd.read_csv('../input/indian-liver-patient-records/indian_liver_patient.csv')
# Look at some details
df.info()
# Look at some records
df.head()
# Look at breakdown of label
df['Dataset'].value_counts()
sns.countplot(df['Dataset'])
# Look at breakdown of categorical feature(s)
df['Gender'].value_counts()
# Summarise
df.describe()
# Remove records with missing values
df.dropna(inplace=True)
df.info()
# Transform categorical feature(s) to numeric
le = LabelEncoder()
df.Gender = le.fit_transform(df.Gender)
# Explore correlations to label
df.corr().Dataset.sort_values(ascending=False)
# Explore correlations visually
f, ax = plt.subplots(figsize=(12,6))
sns.heatmap(df.corr(), annot=True, fmt='.2f')
# Split data into 80% train and 20% validation
X = df.drop('Dataset', axis=1)
y = df['Dataset']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
# Build and train the model
model = RandomForestClassifier(max_depth=2, random_state=0).fit(X,y)
# Apply model to validation data
y_predict = model.predict(X_val)
# Compare actual and predicted values
actual_vs_predict = pd.DataFrame({'Actual': y_val,
                                'Prediction': y_predict})
actual_vs_predict.sample(12)
# Evaluate model
print('Classification metrics: \n', classification_report(y_val, y_predict))
