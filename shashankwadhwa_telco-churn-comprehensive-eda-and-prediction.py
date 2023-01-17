import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
SEED = 1
sns.set(rc={'figure.figsize': (9, 6)})
sns.set_style('white')
# Load the data

df = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.shape
df.head()
df.columns
df.info()
# Find the ratio of churned customers

print(df['Churn'].value_counts(normalize=False))
print('\n')
print(df['Churn'].value_counts(normalize=True))
# Replace the text values in Churn column with boolean values of 0 and 1

df['Churn'].replace({'No': 0, 'Yes': 1}, inplace=True)
target_variable = 'Churn'
def plot_categorical_column(column_name):
    """
    A generic function to plot the distribution of a categorical column, and
    the ratio of Churn in each of the values of that column.
    """
    f, (ax1, ax2) = plt.subplots(2, figsize=(9, 12))
    sns.countplot(x=column_name, data=df, ax=ax1)
    sns.pointplot(x=column_name, y=target_variable, data=df, ax=ax2)
    ax2.set_ylim(0, 0.5)
def plot_continuous_column(column_name):
    """
    A generic function to plot the distribution of a continuous column, and
    boxplot of that column for each value of Churn
    """
    f, (ax1, ax2) = plt.subplots(2, figsize=(9, 12))
    sns.distplot(df[column_name], ax=ax1)
    sns.boxplot(x='Churn', y=column_name, data=df, ax=ax2)
plot_categorical_column('gender')
plot_categorical_column('SeniorCitizen')
plot_categorical_column('Partner')
plot_categorical_column('Dependents')
plot_continuous_column('tenure')
plot_categorical_column('PhoneService')
plot_categorical_column('MultipleLines')
plot_categorical_column('InternetService')
plot_categorical_column('OnlineSecurity')
plot_categorical_column('OnlineBackup')
plot_categorical_column('DeviceProtection')
plot_categorical_column('TechSupport')
plot_categorical_column('StreamingTV')
plot_categorical_column('StreamingMovies')
plot_categorical_column('Contract')
plot_categorical_column('PaperlessBilling')
plot_categorical_column('PaymentMethod')
plot_continuous_column('MonthlyCharges')
binary_columns_replace_dict = {
    'gender': {
        'Female': 0,
        'Male': 1
    },
    'Partner': {
        'No': 0,
        'Yes': 1
    },
    'Dependents': {
        'No': 0,
        'Yes': 1
    },
    'PhoneService': {
        'No': 0,
        'Yes': 1
    },
    'MultipleLines': {
        'No phone service': 0,
        'No': 0,
        'Yes': 1
    },
    'OnlineSecurity': {
        'No internet service': 0,
        'No': 0,
        'Yes': 1
    },
    'OnlineBackup': {
        'No internet service': 0,
        'No': 0,
        'Yes': 1
    },
    'DeviceProtection': {
        'No internet service': 0,
        'No': 0,
        'Yes': 1
    },
    'TechSupport': {
        'No internet service': 0,
        'No': 0,
        'Yes': 1
    },
    'StreamingTV': {
        'No internet service': 0,
        'No': 0,
        'Yes': 1
    },
    'StreamingMovies': {
        'No internet service': 0,
        'No': 0,
        'Yes': 1
    },
    'PaperlessBilling': {
        'No': 0,
        'Yes': 1
    }
}

for binary_col in binary_columns_replace_dict:
    df[binary_col].replace(binary_columns_replace_dict[binary_col], inplace=True)
df.info()
categorical_columns = ['InternetService', 'Contract', 'PaymentMethod']

for categorical_column in categorical_columns:
    dummy_df = pd.get_dummies(df[categorical_column], prefix=categorical_column, drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
# Create a feature for the number of internet services used

df['internet_services_count'] = df['OnlineSecurity'] + df['OnlineBackup'] + df['DeviceProtection'] \
                                + df['TechSupport'] + df['StreamingTV'] + df['StreamingMovies']
# Create a feature for checking if the payment is automatic or not

df['is_payment_automatic'] = df['PaymentMethod'].isin(['Bank transfer (automatic)', 'Credit card (automatic)'])
target_variable = 'Churn'
features = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'PaperlessBilling', 'MonthlyCharges', 'internet_services_count', 'is_payment_automatic'
]

for col in df.columns:
    if not col in features and col.startswith(('InternetService_', 'Contract_')):
        features.append(col)
features
scaler = MinMaxScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(df[features]))
scaled_df.columns = features
scaled_df[target_variable] = df[target_variable]
def evaluate_models(df, features):
    """
    Evaluate different models on the passed dataframe using the given features.
    """
    
    # Create testing and training data
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target_variable], test_size=0.2, random_state=SEED
    )

    results = {} # to store the results of the models
    models = [
        ('lr', LogisticRegression(random_state=SEED)),
        ('lda', LinearDiscriminantAnalysis()),
        ('svm', SVC(random_state=SEED)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('nb', GaussianNB()),
        ('dt', DecisionTreeClassifier(random_state=SEED)),
        ('rf', RandomForestClassifier(random_state=SEED, n_estimators=100)),
        ('et', ExtraTreesClassifier(random_state=SEED, n_estimators=100)),
        ('gb', GradientBoostingClassifier(random_state=SEED, n_estimators=100)),
        ('ada', AdaBoostClassifier(random_state=SEED)),
        ('xgb', xgb.XGBClassifier(random_state=SEED))
        
    ]

    for model_name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = model.score(X_test, y_test)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        results[model_name] = (model, accuracy, f1, cm)
        
    sorted_results = sorted(results.items(), key=lambda x: x[1][1], reverse=True)
    for model_name, (model, accuracy, f1, cm) in sorted_results:
        print(model_name, accuracy, f1)
        
    return results
results = evaluate_models(scaled_df, features)
model = xgb.XGBClassifier(random_state=SEED)
cross_val_scores = cross_val_score(model, scaled_df[features], scaled_df[target_variable], cv=5)
print(cross_val_scores.mean())