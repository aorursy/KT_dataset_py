# Set-up libraries
import os
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Check data input source
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Read-in data
df = pd.read_csv('../input/diamonds/diamonds.csv')
# Look at some information
df.info()
# Look at some records
df.head()
# Summarise data
df.describe()
# Check for missing values
df.isna().sum()
# Check for duplicate values
df.duplicated().sum()
# Explore data visually
sns.pairplot(df)
# Split dataset into 80% train and 20% validation
X = df['carat'].values.reshape(-1, 1)
y = df['price'].values.reshape(-1, 1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
# Build model
model = LinearRegression()
model.fit(X_train, y_train)
# Apply model to test data
y_predict = model.predict(X_val)

actual_vs_predict = pd.DataFrame({'Actual': y_val.flatten(), 'Prediction':y_predict.flatten()})
actual_vs_predict.sample(12)
# Evaluate model
print('Accuracy: %.2f' % (model.score(X_val, y_val)*100), '%' )
