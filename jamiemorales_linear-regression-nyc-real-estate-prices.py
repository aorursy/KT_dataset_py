# Set-up ibraries
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Check data input source
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Read-in data
df = pd.read_csv('../input/nyc-property-sales/nyc-rolling-sales.csv')
# Look at some information
df.info()
# Look at some records
df.head()
# Check for missing values
df.isna().sum()
# Check for duplicate values
df.duplicated().sum()
# Summarise data
df.describe()
# Transform data types
df['GROSS SQUARE FEET'] = pd.to_numeric(df['GROSS SQUARE FEET'], errors='coerce')
df['SALE PRICE'] = pd.to_numeric(df['GROSS SQUARE FEET'], errors='coerce')
# Handle missing values
mean_square_feet = df['GROSS SQUARE FEET'].mean()
df['GROSS SQUARE FEET'].fillna(mean_square_feet, inplace=True)
mean_sale_price = df['SALE PRICE'].mean()
df['SALE PRICE'].fillna(mean_sale_price, inplace=True)
# Split dataset into 80% train and 20% validation
X = df['GROSS SQUARE FEET'].values.reshape(-1, 1)
y = df['SALE PRICE'].values.reshape(-1, 1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
# Build and train model
model = LinearRegression()
model.fit(X_train, y_train)
# Apply model to validation
y_predict = model.predict(X_val)

actual_vs_predict = pd.DataFrame({'Actual': y_val.flatten(), 'Prediction':y_predict.flatten()})
actual_vs_predict.sample(12)
# Evaluate model
print('Accuracy: %.2f' % (model.score(X_val, y_val)*100), '%' )
