# Set-up libraries
import os
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
# Check data input source
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Read-in data
df = pd.read_csv('../input/biomechanical-features-of-orthopedic-patients/column_3C_weka.csv')
# Look at some details
df.info()
# Look at some records
df.head()
# Check for missing values
df.isna().sum()
# Look at breakdown of label
df['class'].value_counts()
sns.countplot(df['class'])
# Explore data visually with multple scatter plots
sns.pairplot(df, hue='class')
# Summarise
df.describe()
# Split dataset into 80% train and 20% validation
X = df.drop('class', axis=1)
y = df['class']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
# Build model and train data
classifier = KNeighborsClassifier(n_neighbors=3)
knn = classifier.fit(X_train, y_train)
knn
# Apply model to validation data
y_predict = classifier.predict(X_val)
# Compare actual and predicted values
actual_vs_predict = pd.DataFrame({'Actual: ': y_val,
                     'Prediction: ': y_predict})
actual_vs_predict.head(10)
# Evaluate model
print('Classification metrics: \n', classification_report(y_val, y_predict))
