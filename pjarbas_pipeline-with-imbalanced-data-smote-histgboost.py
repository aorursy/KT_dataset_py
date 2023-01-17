import pandas as pd
df = pd.read_csv("/kaggle/input/churn-modeling-dataset/Churn_Modelling.csv")
df.head()
df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)
df.shape
df.isna().sum()
df['Exited'].value_counts(normalize=True)
import seaborn as sns

ax = sns.countplot(x="Geography", data=df)
X = df.drop('Exited', axis=1)
y = df['Exited']
from sklearn.model_selection import train_test_split

# Divide data into training and validation subsets
x_train, x_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0, stratify=y)
categorical_cols = ['Geography', 'Gender']

# Select numerical columns
numerical_cols = ['CreditScore',
                  'Age',
                  'Tenure',
                  'Balance',
                  'NumOfProducts',
                  'HasCrCard',
                  'IsActiveMember',
                  'EstimatedSalary']
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])
from imblearn.over_sampling import SMOTE

smt = SMOTE(random_state=42)
from sklearn.experimental import enable_hist_gradient_boosting  
from sklearn.ensemble import HistGradientBoostingClassifier

model = HistGradientBoostingClassifier()
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                            ('smote', smt),
                            ('model', model)
                          ])

# Preprocessing of training data, fit model 
pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_valid)
from sklearn.metrics import classification_report
print(classification_report(y_valid, y_pred))
new_data = pd.DataFrame({'CreditScore': 500, 'Geography': 'Spain', 'Gender': 'Female', 'Age': 30,
                  'Tenure': 1, 'Balance': 0., 'NumOfProducts': 2, 'HasCrCard': 0, 'IsActiveMember': 1, 
                  'EstimatedSalary': 10258.2}, index=[0])
new_data
pipeline.predict(new_data)