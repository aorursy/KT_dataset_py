import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Install libraries
import pandas as pd
import numpy as np
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from joblib import dump, load

# Load the data
X_full = pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv', index_col='customerID')

print('The table has ', X_full.shape[0], ' unique customers')
print('The table has ', X_full.shape[1], ' features')

# Allow us to see all columns of our dataframe
pd.set_option('max_columns', None)
# Get a sense of our data
X_full.head()
# Summarize our data
X_full.info()
# Take a look at how many customers churned and how many stayed
target_dist = X_full['Churn'].value_counts()
print('Customers who stayed:  ', target_dist[0])
print('Customers who churned:  ', target_dist[1])

# Visualize the customer churn distribution
sns.set_style('whitegrid')
sns.barplot(x=target_dist.index, y=target_dist.values)
plt.title('Customer churned?', pad=20, fontsize=15, fontweight='bold')
plt.ylabel('Count')
# Check which datatypes are contained within 'TotalCharges'
X_full['TotalCharges'].apply(type).value_counts()
# Function to check if a string can be converted to a float
def is_convertible(value):
    """
    Checks to see whether a string can be converted to a float.
    Input:
        - A string
    Output:
        - A boolean indicating whether or not the string can be converted to a float
    """
    try:
        float(value)
        return True
    except ValueError:
        return False

    
    
# Initialize list of strings that can't be converted to a float
unconvertibles = []

# Iterate through an array of strings, and append unconvertible strings to a list
str_array = X_full['TotalCharges'].to_numpy()
for element in str_array:
    if is_convertible(element) == False:
        unconvertibles.append(element)

# See which strings (if any) are unconvertible
print("Unconvertibles:  ", unconvertibles)
print("Count of unconvertibles:  ", len(unconvertibles))
# Convert from 'object' to 'float' while replacing strings containing whitespace with NaN
X_full['TotalCharges'] = pd.to_numeric(X_full['TotalCharges'], errors='coerce')
# Get a sense of numerical data
X_full.describe()
# Set style
sns.set_style('whitegrid')

# Set up for graphs to be side-by-side
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25,5), sharex=False, sharey=False)
plt.suptitle('Probability Density Functions of Our Numerical Features', 
             fontsize=30, fontweight='bold', y=1.155)

# Create graph showing distribution of 'tenure'
tenure_distPlot = sns.kdeplot(data=X_full['tenure'], shade=True, ax=axes[0])
axes[0].set_title("'tenure'", fontsize=25, pad=15)
axes[0].set_xlabel("Number of months as a customer", fontsize=20, labelpad=15)

# Create graph showing distribution of 'MonthlyCharges'
monthlyCharge_distPlot = sns.kdeplot(data=X_full['MonthlyCharges'], shade=True, ax=axes[1])
axes[1].set_title("'MonthlyCharges'", fontsize=25, pad=15)
axes[1].set_xlabel("Monthly payment amount", fontsize=20, labelpad=15)

# Create graph showing distribution of 'TotalCharges'
totalCharge_distPlot = sns.kdeplot(data=X_full['TotalCharges'], shade=True, ax=axes[2])
axes[2].set_title("'TotalCharges'", fontsize=25, pad=15)
axes[2].set_xlabel("Total payment amount", fontsize=20, labelpad=15)
# Set up for graphs to be side-by-side
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25,5), sharex=False, sharey=False)
plt.suptitle('Distribution of Our Numerical Features by Churn Class', 
             fontsize=30, fontweight='bold', y=1.155)

# Create graph showing distribution of 'tenure'
tenure_boxPlot = sns.boxplot(x=X_full['Churn'], y=X_full['tenure'], 
                             ax=axes[0], palette='Blues_r')
axes[0].set_title("'tenure'", fontsize=25, pad=15)
axes[0].set_xlabel("Churn?", fontsize=20)
axes[0].set_ylabel("Months as a customer", fontsize=15)


# Create graph showing distribution of 'MonthlyCharges'
monthlyCharges_boxPlot = sns.boxplot(x=X_full['Churn'], y=X_full['MonthlyCharges'], 
                                     ax=axes[1], palette='Blues_r')
axes[1].set_title("'MonthlyCharges'", fontsize=25, pad=15)
axes[1].set_xlabel("Churn?", fontsize=20)
axes[1].set_ylabel("Monthly payment amount", fontsize=15)

# Create graph showing distribution of 'TotalCharges'
totalCharges_boxPlot = sns.boxplot(x=X_full['Churn'], y=X_full['TotalCharges'], 
                                   ax=axes[2], palette='Blues_r')
axes[2].set_title("'TotalCharges'", fontsize=25, pad=15)
axes[2].set_xlabel("Churn?", fontsize=20)
axes[2].set_ylabel("Total payment amount", fontsize=15)
# Find the 25th and 75th percentile values of 'TotalCharge' for customers who churned
lower_percentile = X_full[X_full['Churn']=='Yes'].TotalCharges.quantile(.25)
upper_percentile = X_full[X_full['Churn']=='Yes'].TotalCharges.quantile(.75)

# Calculate the inter-quartile range
iqr = upper_percentile - lower_percentile

# Calculate the cut-off point after which a datapoint becomes an outlier
outlier_limit = upper_percentile + (1.5 * iqr)

# Dataframe consisting only rows of outlierly, abnormally high 'TotalCharge' customers who churned
outlier_tot_charge_churn = X_full[ (X_full['Churn']=='Yes') & (X_full['TotalCharges'] > outlier_limit) ]

# Interesting statistics of abnormally high 'TotalCharge' customers who churned
num_totCharge_outliers = len(outlier_tot_charge_churn.index)
avg_spent_outliers = outlier_tot_charge_churn.TotalCharges.mean()
tot_spent_outliers = num_totCharge_outliers * avg_spent_outliers

print('Number of outlierly customers:  ', num_totCharge_outliers)
print("Average total amount spent per outlierly customer:  ", avg_spent_outliers)
print('Total amount spent by outlierly customers:  ', tot_spent_outliers)

# Generate the correlation matrix of our numerical features
corrMatrix = X_full[['tenure', 'MonthlyCharges', 'TotalCharges']].corr()

# Visualize the correlation matrix
plt.title('Correlation Matrix of Our Numerical Features', pad=20, fontweight='bold')
sns.heatmap(corrMatrix, cmap='Blues', annot=True)
# Set the style of our visualizations
sns.set_style('whitegrid')
sns.set(font_scale = 1.2)
plt.figure(figsize=(7,4))

# Create the barplot (in blue) depicting customer count--by gender
gender_dist = X_full['gender'].value_counts()
sns.barplot(x=gender_dist.index, y=gender_dist.values, color='lightskyblue')

# Create the barplot (in orange) depicting customer churn count--by gender
gender_churn_dist = X_full[X_full.Churn=='Yes']['gender'].value_counts()
sns.barplot(x=gender_churn_dist.index, y=gender_churn_dist.values, color='khaki')

# Label the graph
plt.title('Total customers (blue) and churn count (orange) by gender', 
          pad=20, fontsize=14, fontweight='bold')
plt.ylabel('Count', fontsize=14)
# Find and replace categorical values to make visualization more readable
X_full['SeniorCitizen'].astype('object')
X_full.replace(to_replace={'SeniorCitizen': {0:'No', 1:'Yes'}}, inplace=True)
X_full.replace(to_replace={'MultipleLines': {'No phone service':'N/A'}}, inplace=True)

# Get a list of attributes to visualize
cat_cols_viz = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 
                'PaperlessBilling', 'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']


# Set style
sns.set_style('whitegrid')

# Set up subplot to display graphs
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20,20), sharex=False, sharey=False)
plt.suptitle('Distribution of Our Categorical Features by Churn Class', 
             fontsize=30, fontweight='bold', y=1.032)

# Initialize row and column index iterators in preparation for filling in the subplot
row_iterator = 0
col_iterator = 0

# Fill in the subplot
for col in cat_cols_viz:
    # Adjust indices once we reach the end of a row (moving from left to right)
    if col_iterator == 4:
        col_iterator = 0
        row_iterator = row_iterator + 1
    
    
    # Initialize value count series
    valCount_series = X_full[col].value_counts()
    churn_valCount_series = X_full[X_full.Churn=='Yes'][col].value_counts()
    
    
    # Create the barplot (in blue) depicting customer count--by column
    sns.barplot(x=valCount_series.index, y=valCount_series.values, color='lightskyblue', ax=axes[row_iterator][col_iterator])
    
    # Create the barplot (in orange) depicting customer churn count--by column
    sns.barplot(x=churn_valCount_series.index, y=churn_valCount_series.values, color='khaki', ax=axes[row_iterator][col_iterator])

    # Label the graph
    axes[row_iterator][col_iterator].set_title('%s' % col, fontsize=20)
        
    # Rotate xlabels
    plt.sca(axes[row_iterator, col_iterator])
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')    
    
    # Increment row and column indices
    col_iterator = col_iterator + 1

    
# Adjust spacing of subplot
fig.tight_layout()
# Drop the 'gender' column from our dataset since it doesn't contribute to our model
X = X_full.drop('gender', axis=1)

# Remove whitespace from value classes and replace with '_'
whitespace_cols = cat_cols_viz[7:]
X[whitespace_cols] = X[whitespace_cols].stack().str.replace(' ', '_').unstack()

# Make value classes more descriptive
X.replace(to_replace={'SeniorCitizen': {'No':'Not_SenCit', 'Yes':'SeniorCitizen'}}, inplace=True)
X.replace(to_replace={'Partner': {'No':'No_Partner', 'Yes':'Partner'}}, inplace=True)
X.replace(to_replace={'Dependents': {'No':'No_Dependents', 'Yes':'Dependents'}}, inplace=True)
X.replace(to_replace={'PaperlessBilling': {'No':'No_PaperlessBill', 'Yes':'PaperlessBill'}}, inplace=True)
X.replace(to_replace={'PhoneService': {'No':'No_PhoneService', 'Yes':'PhoneService'}}, inplace=True)
X.replace(to_replace={'MultipleLines': {'No':'No_MultiLines', 'Yes':'MultiLines', 'N/A': 'No_PhoneService'}}, inplace=True)
X.replace(to_replace={'InternetService': {'No':'No_internet_service'}}, inplace=True)
X.replace(to_replace={'OnlineSecurity': {'No':'No_OnlineSecurity', 'Yes':'OnlineSecurity'}}, inplace=True)
X.replace(to_replace={'OnlineBackup': {'No':'No_OnlineBackup', 'Yes':'OnlineBackup'}}, inplace=True)
X.replace(to_replace={'DeviceProtection': {'No':'No_DeviceProtection', 'Yes':'DeviceProtection'}}, inplace=True)
X.replace(to_replace={'TechSupport': {'No':'No_TechSupport', 'Yes':'TechSupport'}}, inplace=True)
X.replace(to_replace={'StreamingTV': {'No':'No_StreamingTV', 'Yes':'StreamingTV'}}, inplace=True)
X.replace(to_replace={'StreamingMovies': {'No':'No_StreamingMov', 'Yes':'StreamingMov'}}, inplace=True)

# Using 'customerID', check to see if there are any duplicate entries
print('Number of duplicate entries:  ', X.index.duplicated().sum())
from sklearn.model_selection import train_test_split

# Separate target from predictors
y = X['Churn']
X.drop('Churn', axis=1, inplace=True)
# Check how many 0's are found within 'TotalCharges'
totCharges_zeroes = X[X['TotalCharges'] == 0].shape[0]
print('There are  ', totCharges_zeroes, "  representations of '0' found within 'TotalCharges'")
# DataFrame of customers whose 'TotalCharges' was the empty string
totCharges_nan = X[X['TotalCharges'].isnull()]

# Find how many customers there are whose 'TotalCharges' was the empty string
print("There are ", totCharges_nan.shape[0], " customers whose 'TotalCharges' was the empty string")

# Get average 'tenure' of customers whose 'TotalCharges' was the empty string
print("The average TENURE of this subset is:  ", 
      totCharges_nan['tenure'].mean())

# Get average 'MonthlyCharges' for customers whose 'TotalCharges' was the empty string
print("The average MONTHLY CHARGE of this subset is:  ", 
      totCharges_nan['MonthlyCharges'].mean())

# Check if any customers whose 'TotalCharges' was the empty string had a monthly charge of 0
print('Did any such customers have a monthly charge of 0?    ', 0 in totCharges_nan['MonthlyCharges'].values)
# Find and replace NaN with 0 in 'TotalCharges'
X.fillna({'TotalCharges': 0}, inplace=True)
# Generate new features by combining existing ones
X['SenCit_Dependents'] = X['SeniorCitizen'] + '_' + X['Dependents']
X['Partner_Dependents'] = X['Partner'] + '_' + X['Dependents']
X['SenCit_Partner'] = X['SeniorCitizen'] + '_' + X['Partner']
X['SenCit_Contract'] = X['SeniorCitizen'] + '_' + X['Contract']
X['SenCit_TechSupport'] = X['SeniorCitizen'] + '_' + X['TechSupport']
X['SenCit_PayMeth'] = X['SeniorCitizen'] + '_' + X['PaymentMethod']
# Create column giving the average of 'TotalCharges' by contract length
temp1 = X.groupby('Contract')['TotalCharges'].agg(['mean']).rename({'mean':'Contract_mean_totCharges'},axis=1)
X = pd.merge(X, temp1, on='Contract', how='left')

# Create column giving the difference in 'TotalCharges' and the average of 'TotalCharges' by contract length
X['Contract_totCharges_diff'] = X['TotalCharges'] - X['Contract_mean_totCharges']


# Create column giving the average of 'MonthlyCharges' by payment method
temp2 = X.groupby('PaymentMethod')['MonthlyCharges'].agg(['mean']).rename({'mean':'PayMeth_mean_monthCharges'},axis=1)
X = pd.merge(X, temp2, on='PaymentMethod', how='left')

# Create column giving the difference in 'MonthlyCharges' and the average of 'MonthlyCharges' by payment method
X['PayMeth_monthCharges_diff'] = X['MonthlyCharges'] - X['PayMeth_mean_monthCharges']
# Round values to two decimal places
X = X.round(2)
### Ordinal Encoding

# Ordinal encoding of 'MultipleLines'
multiLines_dict = {'No_PhoneService':0, 'No_MultiLines':1, 'MultiLines':2}
X['MultipleLines_Ordinal'] = X['MultipleLines'].map(multiLines_dict)

# Ordinal encoding of 'InternetService'
intServ_dict = {'No_internet_service':0, 'DSL':1, 'Fiber_optic':2}
X['InternetService_Ordinal'] = X['InternetService'].map(intServ_dict)

# Ordinal encoding of 'Contract'
contract_dict = {'Month-to-month':0, 'One_year':1, 'Two_year':2}
X['Contract_Ordinal'] = X['Contract'].map(contract_dict)

# Drop unnecessary columns that have been encoded
ordinal_drop_cols = ['MultipleLines', 'InternetService', 'Contract']
X.drop(ordinal_drop_cols, axis=1, inplace=True)
### One-hot Encoding

from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to the relevant columns
OH_col_names = ['SeniorCitizen', 'Partner', 'Dependents', 
           'PaperlessBilling', 'PhoneService', 'OnlineSecurity', 
           'OnlineBackup', 'DeviceProtection', 'TechSupport',
           'StreamingTV', 'StreamingMovies', 'PaymentMethod',
           'SenCit_Dependents', 'Partner_Dependents', 'SenCit_Partner',
           'SenCit_Contract', 'SenCit_TechSupport', 'SenCit_PayMeth']
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(X[OH_col_names]))

# Replace default column names with more descriptive ones
OH_cols.columns = OH_encoder.get_feature_names(OH_col_names)

# One-hot encoding removed index; put it back
OH_cols.index = X.index

# Remove categorical columns (will replace with one-hot encoding)
X.drop(OH_col_names, axis=1, inplace=True)

# Add one-hot encoded columns to numerical features
X = pd.concat([X, OH_cols], axis=1)

from imblearn.over_sampling import SMOTE

# Oversample our dataset using SMOTE to deal with class imbalance
oversample = SMOTE(sampling_strategy=0.5, random_state=42)
X, y = oversample.fit_resample(X, y)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Define the columns we wish to transform
scale_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 
              'Contract_mean_totCharges', 'Contract_totCharges_diff', 
              'PayMeth_mean_monthCharges', 'PayMeth_monthCharges_diff',]

# Scale the relevant columns
transformer = ColumnTransformer([('scaler', StandardScaler(), scale_cols)], 
                                remainder='passthrough')
scaled_X = pd.DataFrame(transformer.fit_transform(X))

# Transformation removed column names; put them back
scaled_X.columns = X.columns

## Function to reduce the DF size
def df_mem_reducer(df):
    """
    Reduces the memory usage of a given dataframe via conversion of numeric attribute datatypes
    Input:
        - df: a Pandas dataframe
    Output: 
        - the same Pandas dataframe that uses less memory
    """
    
    # Original memory usage of dataframe in MB
    start_mem = df.memory_usage().sum() / 1024**2    
    
    # Conversion of numerical datatypes
    num_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    for col in df.columns:
        col_dtype = df[col].dtypes            
        if col_dtype in num_dtypes:
            col_min = df[col].min()
            col_max = df[col].max()
            if str(col_dtype)[:3] == 'int':
                if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else: 
                    df[col] = df[col].astype(np.int64)  
            else:
                if col_min > np.finfo(np.float16).min and col_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif col_min > np.finfo(np.float32).min and col_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    
    end_mem = df.memory_usage().sum() / 1024**2
    
    print('Original memory usage:  ', start_mem, '  MB')
    print('Final memory usage:  ', end_mem, '  MB')
    print('Memory of DataFrame reduced by:  ', ((start_mem - end_mem) / start_mem) * 100, '%')
        
    return df
# Reduce memory usage of scaled_X
scaled_X = df_mem_reducer(scaled_X)
# Reduce memory usage of X
X = df_mem_reducer(X)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score


# Import linear ML algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Import non-linear ML algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Initialize our testing suite
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
results = []
model_names = []
all_models = []
all_models.append(('LR', LogisticRegression(max_iter=1000)))
all_models.append(('LDA', LinearDiscriminantAnalysis()))
all_models.append(('KNN', KNeighborsClassifier()))
all_models.append(('NB', GaussianNB()))
all_models.append(('SVM', SVC()))
all_models.append(('XGB', XGBClassifier()))

# Run the tests
for name, model in all_models:
    scores = cross_val_score(model, scaled_X, y, cv=cv, scoring='roc_auc')
    results.append(scores)
    model_names.append(name)
    print(name, ':  ', 'Mean =  ', scores.mean(), '  ', 'STD =   ', scores.std())
    
# Visualize the results
sns.set_style('whitegrid')
plt.figure(figsize=(10,6))
sns.boxplot(x=model_names, y=results, palette='Blues_r')
plt.title('Comparison of Algorithms', fontsize=20, pad=23, fontweight='bold')
plt.ylabel('AUC - ROC')
plt.show()
# Install Hyperband
!cp -r ../input/hyperbandcv/scikit-hyperband-master/* ./

# Import packages
from hyperband import HyperbandSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_randfloat 
from sklearn import metrics


# Define our model
fixed_params = {'dual': False, 
                'random_state': 42,
                'n_jobs': 1
               }
param_dist = {'max_iter': sp_randint(200,500),
              'solver': ['lbfgs', 'sag', 'newton-cg'],
              'penalty': ['l2'],
              'C': sp_randfloat(0.01, 10),
             }
lr = LogisticRegression(**fixed_params)



# Perform Hyperparameter Tuning
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
search_lr = HyperbandSearchCV(estimator = lr,
                              param_distributions = param_dist, 
                              resource_param='max_iter',
                              min_iter = 200,
                              max_iter = 1000,
                              cv = cv, 
                              scoring='roc_auc',
                              refit=True,
                              verbose = 0,
                              random_state = 42
                          )
search_lr.fit(scaled_X, y)

# Print ROC Curve
plt.figure(figsize=(8,5))
metrics.plot_roc_curve(search_lr.best_estimator_, scaled_X, y)
plt.title('ROC Curve of Our Logistic Regression Model', pad=20, fontweight='bold')
plt.xlabel('False Positive Rate', labelpad=15, fontsize=15)
plt.ylabel('True Positive Rate', labelpad=15, fontsize=15)
plt.show()

# Print results of hyperparameter tuning
print('Best parameters:  ', search_lr.best_params_)
print('AUC - ROC score:  ', search_lr.best_score_)
# Define our model
fixed_params = {'objective': 'binary:logistic', 
                'random_state': 42,
                'n_jobs': 1
               }
param_dist = {'max_depth': sp_randint(3,6),
              'learning_rate': sp_randfloat(0.01, 0.1),
              'n_estimators': sp_randint(100, 1000),
              'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
              'reg_alpha': sp_randfloat(0.0, 1.0),
              'reg_lambda': sp_randfloat(0.0, 1.0),
             }
clf = XGBClassifier(**fixed_params)



# Perform Hyperparameter Tuning
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
search = HyperbandSearchCV(estimator = clf,
                           param_distributions = param_dist, 
                           resource_param='n_estimators',
                           min_iter = 100,
                           max_iter = 1000,
                           cv = cv, 
                           scoring='roc_auc',
                           refit=True,
                           verbose = 0,
                           random_state = 42
                          )
search.fit(X, y)


# Print ROC Curve
plt.figure(figsize=(8,5))
metrics.plot_roc_curve(search.best_estimator_, X, y)
plt.title('ROC Curve of Our XGBoost Model', pad=20, fontweight='bold')
plt.xlabel('False Positive Rate', labelpad=15, fontsize=15)
plt.ylabel('True Positive Rate', labelpad=15, fontsize=15)
plt.show()

# Print results of hyperparameter tuning
print('Best parameters:  ', search.best_params_)
print('AUC - ROC score:  ', search.best_score_)
# Load our saved models
loaded_modelLR = load('../input/lrtelcochurnclassifier/lrTelcoChurnClassifier.joblib.dat')
loaded_modelXGB = load("../input/xgbtelcochurnclassifier/xgbTelcoChurnClassifier.joblib.dat")
# Get the feature weights of our Logistic Regression model
lr_coefficients = pd.DataFrame(loaded_modelLR.coef_)

# DataFrame lacks column names; put them back
lr_coefficients.columns = X.columns

# Reshape (wide --> long format) and 
# filter the data to get features our Logistic Regression model found important
lr_coefficients = pd.melt(lr_coefficients, var_name='Features', value_name='Coefficients')
lr_coefficients.sort_values('Coefficients', ascending=False, inplace=True)
lr_important_features = lr_coefficients[np.abs(lr_coefficients['Coefficients']) > 0.9]
lr_important_features

# Reshape (long --> wide format) in order to get our data in a form 
# such that Seaborn will be able to create a horizontal barplot
lr_important_features = lr_important_features.pivot_table(columns='Features', values='Coefficients')

# Manual ordering of columns to create visualization
col_order = ['MonthlyCharges', 'PayMeth_monthCharges_diff', 'PhoneService_No_PhoneService',
             'OnlineSecurity_OnlineSecurity', 'MultipleLines_Ordinal', 'StreamingMovies_StreamingMov',
             'StreamingTV_StreamingTV', 'tenure', 'PhoneService_PhoneService', 'InternetService_Ordinal' ]
lr_important_features = lr_important_features[col_order]

# Rename columns for enhanced readability
rename_mapping = {'PayMeth_monthCharges_diff': 'deviatFromPaymentMethodAvgMonthlyCharge',
                  'PhoneService_No_PhoneService': 'noPhoneService',
                  'OnlineSecurity_OnlineSecurity': 'hasOnlineSecurity',
                  'MultipleLines_Ordinal': 'numTelephoneLines',
                  'StreamingMovies_StreamingMov': 'streamsMovies',
                  'StreamingTV_StreamingTV': 'streamsTV',
                  'tenure': 'monthsWithCompany',
                  'PhoneService_PhoneService': 'hasPhoneService',
                  'InternetService_Ordinal': 'internetServiceQuality'}
lr_important_features.rename(columns=rename_mapping, inplace=True)


# Visualize the results
sns.set_style('whitegrid')
plt.figure(figsize=(7,7))
sns.barplot(data=lr_important_features, palette='Blues_r', orient='h')
plt.xlabel('Coefficient Values', labelpad=20)
plt.ylabel('')
plt.title('Feature Importance of Our Logistic Regression Model', 
          pad=25, fontsize=20, fontweight='bold')
import shap

# Code to fix a bug that prevents SHAP from interacting with the XGBoost library [3]
mybooster = loaded_modelXGB.get_booster()    
model_bytearray = mybooster.save_raw()[4:]
def myfun(self=None):
    return model_bytearray
mybooster.save_raw = myfun

# Get SHAP values
explainer = shap.TreeExplainer(mybooster)
shap_values = explainer.shap_values(X)
# Plot feature importance of our XGBoost Model
plt.figure()
plt.title('Feature Importance of Our XGBoost Model', pad=25, fontsize=20, fontweight='bold')
shap.summary_plot(shap_values, X, plot_type="bar")
# Get a summary plot showing groupings of data and correlations between features and the target variable.
plt.figure()
plt.title('Summary Plot of Our XGBoost Model', pad=25, fontsize=20, fontweight='bold')
shap.summary_plot(shap_values, X)
# Set up graphs to be displayed side-by-side
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,5), sharex=False, sharey=False)
fig.suptitle('Effect of Tenure, Monthly Charge, and Their Interactions on Model Output', 
             y=1.1, fontsize=23, fontweight='bold')

# Create graph showing effect of 'tenure' on model output
tenure_dependencePlot = shap.dependence_plot('tenure', shap_values, X, 
                                             ax=axes[0], show=False)

# Create graph showing effect of 'MonthlyCharges' on model output
monthlyCharges_dependencePlot = shap.dependence_plot('MonthlyCharges', shap_values, X, 
                                                     ax=axes[1], show=False)

plt.tight_layout()