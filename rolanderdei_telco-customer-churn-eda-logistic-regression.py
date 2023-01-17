# Standard libaries for DataFrame, Dtypes & I/O

import numpy as np

import pandas as pd

from pandas.api.types import is_string_dtype

from pandas.api.types import is_numeric_dtype



# Plotting

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

import plotly.offline as py

%matplotlib inline

pd.set_option("display.precision", 2)



# Profile report

from pandas_profiling import ProfileReport



# Prediction libaries - Logistic regression

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.linear_model import LogisticRegression



# Warnings

import warnings

warnings.filterwarnings("ignore")
# Reading in the csv file

data = pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
# Checking the data's columns

data.info()
# Checking the shape of the data

print("Data shape:", data.shape)



# The raw data contains 7043 rows (customers) and 21 columns (features).
# Printing out the first 5 rows

display(data.head(6))
# Missing data



# Creating missing data table

# Total - total number of missing data

# Percent - percentage of the dataset

total = data.isnull().sum().sort_values(ascending=False)

percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(10)
# Unique values in each feature

for col in data.columns:

    unique_vals = data[col].unique()

    if len(unique_vals) < 10:

        print("Unique values for column {}: {}".format(col, unique_vals))

    else:

        if is_string_dtype(data[col]):

            print("Column {} has values string type".format(col))

        elif is_numeric_dtype(data[col]):

            print("Column {} is numerical".format(col))
# Reshaping data



# Renaming columns - Capital letters for better reading

data = data.rename(columns={'gender': 'Gender', 'tenure': 'Tenure'})



# Drop customerID - irrelevant for the analysis

del data['customerID']



# Convert integer type (0, 1) to categorical Yes/No

data['SeniorCitizen'] = data['SeniorCitizen'].map({True: 'Yes', False: 'No'})



# Count of online services used and creating a new feature

# Integer - 0, 1, 2, 3, 4, 5, 6

data['Count_PlusServices'] = (data[['OnlineSecurity', 'DeviceProtection', 'StreamingMovies', 'TechSupport', 'StreamingTV', 'OnlineBackup']] == 'Yes').sum(axis=1)



# Change columns to type category

catCols = ['Gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']

data[catCols] = data[catCols].astype('category')



# Change 'TotalCharges' to type float

dec_reg_exp = r'^[+-]{0,1}((\d*\.)|\d*)\d+$'

data = data[data.TotalCharges.str.contains(dec_reg_exp)]

# Dropped 11 rows

data['TotalCharges'] = data['TotalCharges'].astype(float)



data.head(6)
# Changing 'No internet service' to 'No'

# Easier to understand, simplify the data

colsForReplacement = [ 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport','StreamingTV', 'StreamingMovies']



for colName in colsForReplacement:

    data[colName] = data[colName].replace({'No internet service' : 'No'})

    data[colName] = data[colName].astype('category')



data.info()
# Using the interquartile range for outliers.

# IQR is a measure of statistical dispersion, being equal to the difference between 75th and 25th percentiles.

cont_features = ["Tenure", "MonthlyCharges", "TotalCharges", "Count_PlusServices"]

dataframe_num = data[cont_features]

dataframe_num.describe()



Q1 = dataframe_num.quantile(0.25)

Q3 = dataframe_num.quantile(0.75)

IQR = Q3 - Q1

IQR

((dataframe_num < (Q1 - 1.5 * IQR)) | (dataframe_num > (Q3 + 1.5 * IQR))).any()
# Generate descriptive statistics with Pandas Profiling

profile = ProfileReport(data, title='Telco Customer Churn - Pandas Profiling Report', progress_bar=False)

profile
# Correlation heatmap



plt.figure(figsize=(25, 10))



# Compute the correlation matrix

corr = data.apply(lambda x: pd.factorize(x)[0]).corr()

# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=bool))

# Draw the heatmap with the mask and correct aspect ratio

ax = sns.heatmap(corr, mask=mask, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, linewidths=.2, cmap='coolwarm', vmin=-1, vmax=1)



# Finding(s):

# Dependents and Partner features have inverse correlation relationship

# Churn and Contract have inverse correlation relationship

# Internet Service and Online Security have inverse correlation relationship

# Tech Support and Internet Service have inverse correlation relationship

# Count Plus Services and Online Backup have inverse correlation relationship



# Streaming TV and Streaming Movires have positive correlation relationship

# Multiple Lines and Phone Service have positive correlation relationship

# Count Plus Services and Contract have positive correlation relationship

# Device Protection and Streaming Movires have positive correlation relationship
trace = go.Pie(labels = ['Churn : no', 'Churn : yes'], values = data['Churn'].value_counts(), 

               textfont=dict(size=15), opacity = 0.8, marker=dict(colors=['lightblue','gold'], line=dict(color='#000000', width=1.5)))



layout = dict(title =  'Distribution of attrition variable',

                        autosize = False,

                        height  = 500,

                        width   = 800)

           

fig = dict(data = [trace], layout=layout)

py.offline.init_notebook_mode()

py.iplot(fig)



# Finding(s):

# Consumers are churning in alarming proportions
fig = plt.figure(figsize=(25,10))

ax = sns.countplot(data=data, x='Count_PlusServices', hue='Churn', palette='pastel')

plt.title('Number of services used by customers', fontsize=17)

ax.set_ylabel('Number of customers', fontsize=12)

ax.set_xlabel('Number of services', fontsize=12)



# Finding(s):

# Customers who does not avail any internet services are churning least.

# Customers who are availing just one Online Service are churning highest. 

# As the number of online services increases beyond one service, the less is the proportion of churn
fig = plt.figure(figsize=(25,10))

ax = sns.distplot(data['Tenure'], hist=True, kde=False, bins=int(200/6), hist_kws={'edgecolor':'black'}, kde_kws={'linewidth': 4})

ax.set_ylabel('Number of Customers')

ax.set_xlabel('Tenure in months')

ax.set_title('Distribution of Customers by Tenure')
feature = ['OnlineSecurity', 'DeviceProtection', 'StreamingMovies', 'TechSupport', 'StreamingTV', 'OnlineBackup']



fig = plt.figure(figsize=(25,20))

plt.subplots_adjust(hspace=0.45)



for i, item in enumerate(feature, 1):

    fig.add_subplot(3,3,i)

    ax = sns.countplot(data=data, x=item, order=["No", "Yes"], hue='Churn', palette='pastel')

    plt.title(f'{item}', fontsize=17)

    ax.set_ylabel('Number of customers', fontsize=12)

    ax.set_xlabel('Used by customers', fontsize=12)

    ax.set_ylim(0, 3750)
feature = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'PaperlessBilling']



fig = plt.figure(figsize=(25,20))

plt.subplots_adjust(hspace=0.45)



for i, item in enumerate(feature, 1):

    fig.add_subplot(2, 3, i)

    ax = sns.countplot(data=data, x=item, order=["No", "Yes"], hue='Churn', palette='pastel')

    plt.title(f'{item}', fontsize=17)

    ax.set_ylabel('Number of customers', fontsize=12)

    ax.set_xlabel('Used by customers', fontsize=12)

    ax.set_ylim(0, 3750)

    

# Finding(s):
agg = data.groupby('Count_PlusServices', as_index=False)[['MonthlyCharges']].mean()

agg[['MonthlyCharges']] = np.round(agg[['MonthlyCharges']], 0)



print(agg['MonthlyCharges'])



plt.figure(figsize=(25,10))

ax = sns.barplot(y='MonthlyCharges', x='Count_PlusServices', data=agg, palette='pastel')

ax.set_title('Average monthly Charges by number of services', fontsize=17)

ax.set_xlabel('Number of online services', fontsize=12)

ax.set_ylabel('Average monthly charges ($)',  fontsize=12)



# Finding(s):

# Customers who does not avail any internet service are paying just $33.

# While those with one service are paying double $66.

# As the number of services availed increases, the Average Monthly Charges are increasing linearly.
agg = agg.div(agg.sum())



plt.figure(figsize=(15,10))

ax = sns.boxplot(x='Churn', y='MonthlyCharges', palette='pastel', data=data)

ax.set_title('Monthly Charges', fontsize=17)

ax.set_ylabel('Monthly Charges ($)', fontsize=12)

ax.set_xlabel('Churn', fontsize=12)



# Finding(s):

# The higher the monthly charges, the higher possibility of Churn,

# Non churners are paying just over $60, while churners are paying nearly $80.
plt.figure(figsize=(15,10))

ax = sns.boxplot(x='Churn', y = 'Tenure', palette='pastel', data=data)

ax.set_title('Churn vs Tenure', fontsize=17)

ax.set_ylabel('Tenure (Months)', fontsize=12)

ax.set_xlabel('Churn', fontsize=12)



# Finding(s):

# Shorter the tenure, higher is the possibility of Churn.
plt.figure(figsize=(25,10))

ax = sns.countplot(x='Churn', hue='Contract', palette='pastel', data=data)

ax.set_title('Contract Type vs Churn', fontsize=17)

ax.set_ylabel('Number of Customers', fontsize=12)

ax.set_xlabel('Churn', fontsize=12)



# Finding(s):

# Customers with Month-to-Month contract are churning more, while two year contract customers are churning least.
plt.figure(figsize=(25,10))

ax = sns.countplot(x="Churn", hue="InternetService", palette='pastel', data=data)

ax.set_title('Churn By Internet Service Type', fontsize=17)

ax.set_ylabel('Number of Customers', fontsize=12)

ax.set_xlabel('Churn', fontsize=12)



# Finding(s):

# Customers with Fiber Optic internet service are churning in alarming proportions.
plt.figure(figsize=(25,6))

ax = sns.countplot(x="Churn", hue="PaymentMethod", palette='pastel', data=data)

ax.set_title('Churn by Payment Method', fontsize=17)

ax.set_ylabel('Number of Customers Churned', fontsize=12)

ax.set_xlabel('Churn', fontsize=12)



# Finding(s):

# Customers with Electronic Check as mode of payment are churning in higher proportion 
# Pandas Visual Analysis

# https://pypi.org/project/pandas-visual-analysis/

!pip install pandas-visual-analysis
# Pandas Visual Analysis

from pandas_visual_analysis import VisualAnalysis, DataSource



# The DataSource object provides the data itself to the plots and also manages the brushing between the plots.

datasource = DataSource(data)

VisualAnalysis(datasource)
# https://www.kaggle.com/alanchn31/churn-analysis-in-depth-eda-and-fe

non_numeric_features = ['Gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',

                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 

                       'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'Churn']



for feature in non_numeric_features:     

    # Encode target labels with value between 0 and n_classes-1

    data[feature] = LabelEncoder().fit_transform(data[feature])

    

data.info()
cat_features = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 

                'PaymentMethod']

encoded_features = []



for feature in cat_features:

    # Encode categorical features as a one-hot numeric array

    encoded_feat = OneHotEncoder().fit_transform(data[feature].values.reshape(-1, 1)).toarray()

    n = data[feature].nunique()

    cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]

    encoded_df = pd.DataFrame(encoded_feat, columns=cols)

    encoded_df.index = data.index

    encoded_features.append(encoded_df)

data = pd.concat([data, *encoded_features], axis=1)

    

print('Number of encoded feautes:', len(encoded_features))



# Drop columns that are unrelated and columns where we generate one-hot encoded variables earlier

data2 = data.copy()

drop_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']

data.drop(columns=drop_cols, inplace=True)
X = data.drop(columns=['Churn']).values

y = data["Churn"].values



# Splitting the data

# 75% train

# 25% test

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25,  stratify=y, random_state=22)

print('X_train shape: {}'.format(x_train.shape))

print('X_test shape: {}'.format(x_test.shape))
# Provides train/test indices to split data in train/test sets.

skf = StratifiedKFold(n_splits=4)

val_auc_scores = []



for train_index, valid_index in skf.split(x_train, y_train):

    x_pseudo_train, x_pseudo_valid = x_train[train_index], x_train[valid_index]

    y_pseudo_train, y_pseudo_valid = y_train[train_index], y_train[valid_index]

    # Standardize features by removing the mean and scaling to unit variance

    ss = StandardScaler()

    # Fit to data, then transform it.

    x_pseudo_train_scaled = ss.fit_transform(x_pseudo_train)

    # Perform standardization by centering and scaling

    x_pseudo_valid_scaled = ss.transform(x_pseudo_valid)

    # Logistic Regression

    lr = LogisticRegression() # Using default parameters

    # Fit the model according to the given training data

    lr.fit(x_pseudo_train_scaled, y_pseudo_train)

    # Predict logarithm of probability estimates.

    y_pred_valid_probs = lr.predict_proba(x_pseudo_valid_scaled)[:, 1]

    # Compute Receiver operating characteristic (ROC)

    val_fpr, val_tpr, val_thresholds = roc_curve(y_pseudo_valid, y_pred_valid_probs)

    # Compute Area Under the Curve (AUC) using the trapezoidal rule

    val_auc_score = auc(val_fpr, val_tpr)

    val_auc_scores.append(val_auc_score)
# Standardize features by removing the mean and scaling to unit variance

ss = StandardScaler()

# Fit to data, then transform it.

x_train_scaled = ss.fit_transform(x_train)

# Perform standardization by centering and scaling

x_test_scaled = ss.transform(x_test)



# Applying logistic regression classifier

lr = LogisticRegression()        # Using default parameters

lr.fit(x_train_scaled, y_train)  # Training the model with X_train, y_train



# Generate Confusion Matrix

y_pred = lr.predict(x_test_scaled)

y_pred = pd.Series(y_pred)

y_test = pd.Series(y_test)

pd.crosstab(y_pred, y_test, rownames=['Predicted'], colnames=['True'], margins=True)
# Checking overall accuracy

print("Overall Accuracy: {:%}".format(sum(y_pred == y_test)/len(y_test)))