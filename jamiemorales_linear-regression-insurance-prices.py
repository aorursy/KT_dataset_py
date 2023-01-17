# Set-up libraries
import os
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
# Check data input source
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Read-in data
df = pd.read_csv('../input/insurance/insurance.csv')
# Look at some information
df.info()
# Look at some records
df.head()
# Check for missing values
df.isna().sum()
# Check for duplicate values
df.duplicated().sum()
# Show duplicated
df[df.duplicated(keep=False)]
# Show breakdown of categorical feature sex
df.sex.value_counts()
sns.countplot(df.sex)
# Show breakdown of categorical feature smoker
df.smoker.value_counts()
sns.countplot(df.smoker)
# Explore data visually
sns.pairplot(df)
# Show duplicated
df[df.duplicated(keep=False)]
# Remove duplicate
df.drop_duplicates(keep='first', inplace=True)
# Show duplicated
df[df.duplicated(keep=False)]
# Show remaining record
df.iloc[195]
# Transform categorical features
le = LabelEncoder()
le.fit(df['sex'].drop_duplicates())
df.sex = le.transform(df['sex'])

le.fit(df['smoker'].drop_duplicates())
df.smoker = le.transform(df['smoker'])
# Check transformations
print(df['sex'].unique())
print(df['smoker'].unique())
# Explore relationships of features against cost
df.corr()['charges']
# Summarise data 
df.describe()
# Split data into 80% train and 20% validation
X = df['smoker'].values.reshape(-1, 1)
y = df['charges'].values.reshape(-1, 1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
# Build and train model
model = LinearRegression()
model.fit(X_train, y_train)
# Apply model to validation
y_predict = model.predict(X_val)

actual_vs_predict = pd.DataFrame({'Actual': y_val.flatten(),
                                 'Prediction': y_predict.flatten()})
actual_vs_predict.sample(12)
# Evaluate model
print('Accuracy: %.2f'% (model.score(X_val, y_val)*100), '%' )
# Split data into 80% train and 20% validation
X = df.drop(['charges', 'region'], axis=1)
y = df['charges'].values.reshape(-1, 1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
# Build and train model
model = LinearRegression()
model.fit(X_train, y_train)
# Evaluate model
print('Accuracy: %.2f'% (model.score(X_val, y_val)*100), '%' )
