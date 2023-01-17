import numpy as np 
import pandas as pd

path_dataset = '/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(path_dataset)
df.head(3)
print(f'Dataset shape: {df.shape}')
# Check out a detailed description of the data.
# Mostly interested in the data types and any non-null values
df.info()
df = pd.read_csv(path_dataset, na_values=[' ', ''])
df.info()
df.dropna(inplace=True)
print(f'New dataset shape: {df.shape}')
# Check value ranges of data
for col in df:
    print(f'Feature: {col}')
    print(f'Values: {df[col].unique()[:5]}')
    print('---')
# Get a better overview of the numerical data
df.describe()
df = df.drop(columns=['customerID'])
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df[df.select_dtypes(['object']).columns] = (
    df[df.select_dtypes(['object']).columns].select_dtypes(['object']).apply(
        lambda x: label_encoder.fit_transform(x)
    ))
df.head()
# Check value ranges once more
df.describe()
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(16, 6))
plt.xticks(range(-10,100,2))
plt.title('Tenure')
sns.violinplot(x=['tenure'], data=df)
plt.figure(figsize=(16, 6))
plt.title('Monthly Charges')
plt.xticks(range(5,140,5))
sns.violinplot(x=['MonthlyCharges'], data=df)
plt.figure(figsize=(16, 6))
plt.title('Total Charges')
plt.xticks(range(0,10000,500))
sns.violinplot(x=['TotalCharges'], data=df)
# First grab the categorical subset of the data to make life easier
# To get the categorical data programmatically the data types are exploited.
df_original = pd.read_csv(path_dataset)
df_discrete = df_original[df_original.select_dtypes(['object']).columns]
df_discrete = df_discrete.drop(columns=['customerID', 'TotalCharges'])
feature_names = df_discrete.columns
fig = plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=.7, wspace=.4)

index = 0
for row in range(4):
    for col in range(4):
        ax = fig.add_subplot(4, 4, index+1)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
        feature_name = feature_names[index]
        ax = sns.countplot(x=feature_name, data=df_discrete)
        index+=1
sns.FacetGrid(df, col='Churn').map(sns.violinplot, 'tenure', order=[0,1])
sns.FacetGrid(df, col='Churn').map(sns.violinplot, 'MonthlyCharges', order=[0,1])
sns.FacetGrid(df, col='Churn').map(sns.violinplot, 'TotalCharges', order=[0,1])
df_categorical = df.drop(columns=['tenure', 'MonthlyCharges', 'TotalCharges'])
feature_names = df_categorical.columns

index = 0
for index in range(len(feature_names)):
    feature_name = feature_names[index]
    sns.FacetGrid(df_categorical, col='Churn').map(
        sns.countplot, feature_name, order=df_categorical[feature_name].unique())
correlations = df.corr()
fig = plt.figure(figsize=(12,10), dpi=80, facecolor='w', edgecolor='k')
ax = sns.heatmap(correlations, cmap='PiYG', vmin=-1, vmax=1,  annot=True)
correlations.Churn.sort_values()
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
values = df.values

X = values[:,0:19] # Features
y = values[:,19] # Targets

# Mutual Information for Classification
mutual_info = mutual_info_classif(X, y, discrete_features='auto', 
                                    n_neighbors=3, copy=True, random_state=None)

# chi square
chi_score, chi_pval = chi2(X,y)

# F-measure
f_score, f_pval = f_classif(X,y)

data_feature_selection = { 'MutualInfo': mutual_info,
                        'ChiSquaredScore': chi_score,
                        'ChiSquaredPVal': chi_pval,
                        'FScore': f_score,
                        'FPVal': f_pval
                        }

features = df.columns[0:19]

df_feature_selection = pd.DataFrame(data_feature_selection)
df_feature_selection.insert(0, 'Feature', features)

pd.set_option('display.float_format', lambda x: '%.3f' % x)
df_feature_selection
fig = plt.figure(figsize=(16, 5), dpi= 80, facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=.1, wspace=.6)

ax = fig.add_subplot(1, 2, 1)
ax = sns.barplot(x='MutualInfo', y="Feature", 
                 data=df_feature_selection.sort_values('MutualInfo', ascending=False))

ax = fig.add_subplot(1, 2, 2)
ax = sns.barplot(x='FScore', y="Feature", 
                 data=df_feature_selection.sort_values('FScore', ascending=False))
ax = sns.barplot(x='ChiSquaredScore', y="Feature", 
                 data=df_feature_selection.drop([4,17,18])
                 .sort_values('ChiSquaredScore', ascending=False))
fig = plt.figure(figsize=(16, 5), dpi= 80, facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=.1, wspace=.6)

ax = fig.add_subplot(1, 2, 1)
ax = sns.barplot(x='ChiSquaredPVal', y="Feature", 
                 data=df_feature_selection
                 .sort_values('ChiSquaredPVal', ascending=True))

ax = fig.add_subplot(1, 2, 2)
ax = sns.barplot(x='FPVal', y="Feature", data=df_feature_selection
                 .sort_values('FPVal', ascending=True))
selector = SelectKBest(score_func = mutual_info_classif, k = 8).fit(X,y)

feature_indices = selector.get_support(True)

print("Best 8 features (Mutual Information):")
for i in range(len(feature_indices)):
    index = feature_indices[i]
    print(df.columns[index])
# Extract cols with non binary data
df_object_dtypes = df_original.select_dtypes(include="object").copy()
df_object_dtypes.drop(columns = ['customerID', 'TotalCharges'], axis=1, inplace=True)
features_non_binary_categorical = []
for col in df_object_dtypes.columns:
    if(len(df_object_dtypes[col].value_counts()) > 2):
        features_non_binary_categorical.append(col)
    else:
        df_object_dtypes.drop([col], axis=1, inplace=True)
        
features_non_binary_categorical
from sklearn.preprocessing import OneHotEncoder
df_onehotencoded = df.copy()

for feature in features_non_binary_categorical:
    col_values = df_onehotencoded[feature].values.reshape(-1,1)
    col_values_one_hot = OneHotEncoder(sparse=False, categories='auto').fit_transform(col_values)
    col_values_one_hot = col_values_one_hot.tolist()
    
    df_onehotencoded[feature] = col_values_one_hot
    
df_onehotencoded.head()
# Standardized features
df_standardized = ((df-df.mean())/
                   df.std())
df_standardized.head()
from sklearn.model_selection import train_test_split
features = df.columns[0:19]
target = df.columns[19]

x_train, x_test, y_train, y_test = train_test_split(df[features],
                                                   df[target],
                                                   test_size = 0.3,
                                                   random_state = 10)

print(f'Shape of training set X: {x_train.shape}')
print(f'Shape of training set y: {y_train.shape}')
print('---')
print(f'Shape of test set X: {x_test.shape}')
print(f'Shape of test set y: {y_test.shape}')
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyClassifier
# Preparations needed for the pipelines
# We want to scale the numerical data and we want to onehot encode the categorical data.
# For this we make use of a ColumnTransformer
categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Generate a mask identifying the features that are supposed to be one hot encoded
one_hot_mask = (df_original.drop(columns=['Churn', 'customerID']).dtypes == object).values

#Define the pipeline steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(sparse=False, categories='auto'), categorical_features),
    remainder='passthrough'
)

pipeline = Pipeline([
    ('Preprocessor', preprocessor),
    ("KBest", SelectKBest(mutual_info_classif, k=8)),
    ('Classifier', LogisticRegression(solver ='liblinear'))
])
# Print a confusion matrix and values for accuracy, precision, recall and f1-measure
def calculate_results(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
        
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-Score: {f1}')
    
    return accuracy, precision, recall, f1
# Execute pipeline
pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)

pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True) 
_ = calculate_results(y_test, y_pred)
# Baseline - always predict 0
pipeline = Pipeline([
    ('Classifier', DummyClassifier(strategy='constant', constant=0))
]) 

pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)

#confusion_matrix(y_test, y_pred) - #Sklearn function to create a confusion matrix.
# pd confusion matrix -> better visualization
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True) 
_ = calculate_results(y_test, y_pred)
gridSearchParams = [
    {'C':[0.01, 0.03, 0.1, 0.3, 1.0, 1.1, 1.3, 1.33, 1.6],
     'class_weight':[{0:.1, 1:.9},{0:.2, 1:.8},{0:.3, 1:.7},{0:.4, 1:.6},
                    {0:.4, 1:.7}, {0:.6, 1:.8}, {0:.5, 1:.8}, 'balanced']
    }
]

classifier = GridSearchCV(LogisticRegression(solver ='liblinear'), gridSearchParams, cv=5)

#Define the pipeline steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(sparse=False, categories='auto'), categorical_features),
    remainder='passthrough'
)

pipeline = Pipeline([
    ('Preprocessor', preprocessor),
    ('Classifier', classifier)
])

# Execute pipeline
_ = pipeline.fit(x_train, y_train)

# Calculate results
y_pred = pipeline.predict(x_test)

pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True) 
p5_measures = calculate_results(y_test, y_pred)

print("Best Hyperparameters:",classifier.best_params_)
