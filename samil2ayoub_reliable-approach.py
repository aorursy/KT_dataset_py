import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
df = pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.head()
df_copy = df.copy()
target_map = {'Yes':1, 'No':0}
y = df_copy['Attrition'].apply(lambda value: target_map[value]) # Encode target column
X = df_copy.drop(['Attrition'], axis = 1) # Separate predictor variables from predicted value

# Devide our dataframe into numerical dataframe and categorical dataframe
num_df = X.select_dtypes(exclude=object)  
cat_df = X.select_dtypes(include=object)  
df.isnull().values.any()
df.dtypes
plt.figure(figsize=(20,20))
for i, col in enumerate(num_df.columns, 1):
    plt.subplot(5, 6, i)
    sns.violinplot(x = df[col])
plt.figure(figsize=(20,20))
for i, col in enumerate(cat_df.columns, 1):
    plt.subplot(3, 3, i)
    sns.countplot(x = df[col])
plt.figure(figsize=(30,30))
for i, col in enumerate(num_df.columns, 1):
    plt.subplot(5, 6, i)
    sns.violinplot(x = df['Attrition'], y= df[col])

plt.figure(figsize=(30,30))
for i, col in enumerate(cat_df.columns, 1):
    plt.subplot(3, 3, i)
    sns.barplot(x = df[col], y= y)
encoder = Pipeline(steps=[
    ('encoder', OrdinalEncoder())])
scaler_transformer = Pipeline(
    steps=[
        ('scaler', MinMaxScaler()),
        ('transfomer', FunctionTransformer(np.log1p, validate=True))
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', encoder, cat_df),
        ('cat', scaler_transformer, num_df)
    ]
)
# drop unused columns
X.drop(['EmployeeCount','Over18','StandardHours','EmployeeNumber'], axis = 1, inplace = True) 
num_df = X.select_dtypes(exclude = object)
cat_df = X.select_dtypes(include = object)
sns.set(style="white")
mask = np.zeros_like(num_df.join(y).corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
fig, ax = plt.subplots(figsize=(15,10))
cmap = sns.diverging_palette(255, 10, as_cmap=True)
sns.heatmap(num_df.join(y).corr().round(2), mask=mask, annot=True,
            cmap=cmap , vmin=-1, vmax=1, ax=ax)