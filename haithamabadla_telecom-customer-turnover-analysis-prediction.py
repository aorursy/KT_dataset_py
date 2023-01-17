import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_style('darkgrid')
pd.set_option('display.max_columns', None)
# Importing Dataset

df = pd.read_csv('../input/TelcomCustomerChurn.csv')
# Quick look at the DataFrame

df.head(2)
# Quick Overview

print('Number of Rows: {}'.format(df.shape[0]))
print('Number of Features / Columns: {}'.format(df.shape[1]))
print('Overall Missing Values: {}'.format(df.isnull().sum().values.sum()))
# Customized function to calculate the memory usage of the features in MB

def mem_usage(pandas_obj):
    
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    
    return "{:03.2f} MB".format(usage_mb)
# Display more details on DataFrame in more appealing way as DataFrame

pd.DataFrame({'Feature_Name': df.columns.tolist(), 
              'DataType': df.dtypes.values, 
              'UniqueValues': df.nunique().values,
              'MissingValues': [df[col].isnull().sum() for col in df.columns],
              'MemoryUsage': [mem_usage(df[col]) for col in df.columns]})
# Check TotalCharges column
#float(df.TotalCharges) # Can not convert to float type as it seems it contains str values or spaces
# Check if it contains text or spaces. Below is the quickest way to check
print('Number of observations that contain spaces in TotalCharges column: {}'.format(df.TotalCharges.str.contains(' ').sum()))
print('Number of observations that contain text in TotalCharges column: {}'.format(df.TotalCharges.str.isalpha().sum()))
# Replace ' ' with 'np.nan'
df.TotalCharges = df.TotalCharges.replace(' ', np.nan).astype('float')
df.dropna(inplace = True)

# Replace 'No phone service' with 'No'
df.MultipleLines.replace('No phone service', 'No', inplace= True)  
    
# Replace 'No internet service' with 'No'
list_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

for col in list_cols:
    df[col].replace('No internet service', 'No', inplace= True) 
    
# Replace '0,1' with 'No, Yes'
df.SeniorCitizen.replace({0: 'No', 1: 'Yes'}, inplace= True)
# Create groups based on number of months and assign them to customers

def tenure_grouping(x):
    if x <= 12:
        return '0-1 Year'
    elif (x > 12) & (x <= 24):
        return '1-2 Years'
    elif (x > 24) & (x <= 36):
        return '2-3 Years'
    elif (x > 36) & (x <= 48):
        return '3-4 Years'
    elif (x > 48) & (x <= 60):
        return '4-5 Years'
    elif x > 60:
        return '5+ Years'
    
df['TenureGroups'] = df.tenure.apply(lambda x: tenure_grouping(x))
# Create new dataframes to seperate the Churns

df_churn , df_not_churn = df[df.Churn == 'Yes'].copy(), df[df.Churn == 'No'].copy()
# Explsion

#explode = (0.05,0.05)
cols_soft = {0: 'gender', 1: 'SeniorCitizen', 2: 'Partner', 3: 'Dependents', 4: 'Contract', 5: 'PaymentMethod', 6: 'TenureGroups'}
cols_tech = {0: 'PhoneService', 1: 'MultipleLines', 2: 'InternetService', 3: 'OnlineSecurity', 4: 'OnlineBackup', 5: 'DeviceProtection', 6: 'TechSupport'}
cols_others = {0: 'StreamingTV', 1: 'StreamingMovies', 2: 'PaperlessBilling'}

# Plot customer's behavioral information
fig, ax = plt.subplots(ncols = 7, figsize= (50,6))
    
for indx, val in cols_soft.items():
    lbl  = df_churn[val].value_counts().index
    anot = (df_churn[val].value_counts(normalize= True).values * 100).round(1)
    
    ax[indx].pie(data = df_churn, x = anot, autopct='%1.1f%%', textprops = {'fontsize': 10, 'color': 'w', 'weight': 'bold'})# textprops = dict(color="w", size = 10))#, weight = 'bold')) # explode= explode, 
    ax[indx].add_patch(plt.Circle((0,0),0.35,fc='white'))
    ax[indx].legend(lbl, loc=2)
    ax[indx].set_title(val, size = 15)

    #draw circle
    #centre_circle = ax[0].Circle((0,0),0.40,fc='white')
    #fig_circle = ax[0].gcf()
    #fig_circle.gca().add_artist(centre_circle)

plt.show()

#Plot customer's technical information
fig, ax = plt.subplots(ncols = 7, figsize= (50,6))
    
for indx, val in cols_tech.items():
    lbl  = df_churn[val].value_counts().index
    anot = (df_churn[val].value_counts(normalize= True).values * 100).round(1)
    
    ax[indx].pie(data = df_churn, x = anot, autopct='%1.1f%%', textprops = {'fontsize': 10, 'color': 'w', 'weight': 'bold'}) 
    ax[indx].add_patch(plt.Circle((0,0),0.35,fc='white'))
    ax[indx].legend(lbl, loc=2)
    ax[indx].set_title(val, size = 15)

plt.show()

#Plot customer's other information
fig, ax = plt.subplots(ncols = 3, figsize= (18.5,5.5))
    
for indx, val in cols_others.items():
    lbl  = df_churn[val].value_counts().index
    anot = (df_churn[val].value_counts(normalize= True).values * 100).round(1)
    
    ax[indx].pie(data = df_churn, x = anot, autopct='%1.1f%%', textprops = {'fontsize': 10, 'color': 'w', 'weight': 'bold'}) 
    ax[indx].add_patch(plt.Circle((0,0),0.35,fc='white'))
    ax[indx].legend(lbl, loc=2)
    ax[indx].set_title(val, size = 15)

plt.show()
# Compare all features against Churn 

# Create a list of features that I want to plot 
cols_list = df.drop(['customerID', 'Churn', 'tenure', 'MonthlyCharges', 'TotalCharges'], axis = 1).columns.tolist()

# Define pie_plot function
def pie_plot(data_1, data_2, target, plot_title):

    fig, ax = plt.subplots(ncols = 2, figsize= (15,7))
    
    # Data preproccessing
    x = pd.DataFrame((data_1[target].value_counts(normalize=True).sort_index() * 100).round(1).reset_index().rename(columns = {'index': 'gender', 'gender': 'churn_yes'}))
    x['churn_no'] = (data_2[target].value_counts(normalize=True).sort_index() * 100).round(1).values

    cols_list = x.columns[1:3]
    hue_list = x.gender.unique()

    churn_list = ['Yes', 'No']

    for indx in range(0,2):

        ax[indx].pie(data = x, x = cols_list[indx], autopct='%1.1f%%', textprops = {'fontsize': 10, 'color': 'w', 'weight': 'bold'}) 
        ax[indx].add_patch(plt.Circle((0,0),0.35,fc='white'))
        label = ax[indx].annotate('Churn = {}'.format(churn_list[indx]), xy=(0, 0), fontsize=11, ha="center")
        ax[indx].legend(hue_list, loc=2)
        ax[indx].set_title(plot_title, size = 15)

        #draw circle
        #centre_circle = ax[0].Circle((0,0),0.40,fc='white')
        #fig_circle = ax[0].gcf()
        #fig_circle.gca().add_artist(centre_circle)

    plt.show()


for col in cols_list:
    
    pie_plot( data_1 = df_churn, 
              data_2 = df_not_churn, 
              target = col, 
              plot_title = '{} Distribution'.format(col.title()) ) # .title() to capitilize the first character
fig, ax = plt.subplots(ncols = 2, figsize= (25,6))

sns.distplot(df.MonthlyCharges, ax = ax[0])
sns.distplot(df.TotalCharges, ax = ax[1])

ax[0].set_title('Monthly Charges Distribution', pad = 10, weight= 'bold')
ax[0].set_xlabel('Monthly Charges')

ax[1].set_title('Total Charges Distribution', pad = 10, weight= 'bold')
ax[1].set_xlabel('Total Charges')

plt.show()
# Create annotation function
def annotate_perct(ax_plot, total, add_height):
    
    for p in ax_plot.patches:
        height = p.get_height()
        ax_plot.text(p.get_x() + p.get_width()/2., height + add_height, '{} - {}%'.format(height, round((round(height / total, 3) * 100), 1)), ha="center", va='center', fontsize=10)  # additional round() method to avoid the extra decimal points


#  Generate plot for the top countries based on their revenue, transactions, quantity ordered
plt.figure(figsize= (22,8))

groups_order = df.TenureGroups.value_counts().sort_index().keys()

_ = sns.countplot( data = df,  x = 'TenureGroups', hue = 'Churn', order = groups_order)

annotate_perct( ax_plot = _, total = len(df.TenureGroups), add_height = 20)

_.set_title('Customers Tenure Groups', pad = 10, weight= 'bold')
_.set_xlabel('Groups', weight= 'bold')
_.set_ylabel('Counts', weight= 'bold')

plt.show()
plt.figure(figsize= (22,8))

_ = sns.scatterplot( data = df, x = 'MonthlyCharges', y = 'TotalCharges', hue = 'TenureGroups',  hue_order = groups_order)

_.set_title('Monthly & Total Charges by Tenure Groups', pad = 10, weight= 'bold')
_.set_xlabel('Monthly Charges', weight= 'bold')
_.set_ylabel('Total Charges', weight= 'bold')

plt.show()
plt.figure(figsize= (22,8))

_ = sns.scatterplot( data = df, x = 'MonthlyCharges', y = 'TotalCharges', hue = 'Churn')

_.set_title('Monthly & Total Charges by Tenure Groups', pad = 10, weight= 'bold')
_.set_xlabel('Monthly Charges', weight= 'bold')
_.set_ylabel('Total Charges', weight= 'bold')

plt.show()
avg_charges = df.groupby(['TenureGroups', 'Churn']).agg({ 'MonthlyCharges': 'mean', 'TotalCharges': 'mean' }).reset_index()

# Create annotation function
def annotate_perct(ax_plot, total, add_height):
    
    for p in ax_plot.patches:
        height = p.get_height()
        ax_plot.text(p.get_x() + p.get_width()/2., height + add_height, '{}%'.format(round((round(height / total, 3) * 100), 1)), ha="center", va='center', fontsize=10)  # additional round() method to avoid the extra decimal points

fig, ax = plt.subplots(ncols = 2, figsize= (30,7))

sns.barplot(data = avg_charges, x = 'TenureGroups', y = 'MonthlyCharges', hue = 'Churn', ax = ax[0])
sns.barplot(data = avg_charges, x = 'TenureGroups', y = 'TotalCharges', hue = 'Churn', ax = ax[1])

annotate_perct( ax_plot = ax[0], total = avg_charges.MonthlyCharges.sum(), add_height = 2)
annotate_perct( ax_plot = ax[1], total = avg_charges.TotalCharges.sum(), add_height = 100)


ax[0].set_title('Average Monthly Charges By Tenure Groups', pad = 10, weight= 'bold')
ax[0].set_xlabel('Monthly Charges')

ax[1].set_title('Average Total Charges By Tenure Groups', pad = 10, weight= 'bold')
ax[1].set_xlabel('Total Charges')

plt.show()
from sklearn.preprocessing import StandardScaler
def process_data(df, drop_cols, unwanted_cols, target_col):
    
    data = df.copy()
    data.dropna(inplace = True)
    
    # Seperate the features from target 
    X = data.drop(drop_cols, axis = 1)    
    y = data.Churn.replace({'Yes': 1, 'No': 0})

    # Customer ID column
    customer_id = unwanted_cols

    # Target column
    target = target_col

    # Categorical columns for encoding
    cat_cols = X.nunique()[X.nunique() < 10].keys().tolist()

    #Binary columns with 2 values
    binary_val_cols   = X.nunique()[X.nunique() == 2].keys().tolist()

    #Columns more than 2 values
    multi_val_cols = [c for c in cat_cols if c not in binary_val_cols]

    # Numerical columns for scaling
    num_cols = [c for c in X.columns if c not in target + cat_cols + customer_id]

    # Convert categorical to numeric encoding
    X = pd.get_dummies(data = X, columns = binary_val_cols, drop_first = True)
    X = pd.get_dummies(data = X, columns = multi_val_cols)
    
    # Take a copy of X before scaling numeric variables
    X_original = X.copy()
    
    # Scaling numeric columns
    scaler = StandardScaler()
                        #scaled_cols = scaler.fit_transform(X[num_cols]) # There was issue seperating scaled columns and merge them. I get 11 Nan values for some reason
                        #scaled_cols = pd.DataFrame(scaled_cols, columns = num_cols) # There was issue seperating scaled columns and merge them. I get 11 Nan values for some reason
    X[num_cols] = scaler.fit_transform(X[num_cols])

    '''
    # Test ##########################################################################
    print('scaled_cols shape: {}'.format(scaled_cols.shape))
    print('scaled_cols missing values: \n{}\n\n'.format(scaled_cols.isnull().sum()))

    print('X shape: {}'.format(X.shape))
    print('X missing values: {}\n\n'.format(X.isnull().sum()))

    print('y shape: {}'.format(y.shape))
    print('y missing values: {}\n\n'.format(y.isnull().sum()))
    #################################################################################
    '''

    # Drop original columns as we are not going to use them for ML
    #X.drop(num_cols, axis = 1, inplace = True) # There was issue seperating scaled columns and merge them. I get 11 Nan values for some reason
    
    '''
    # Test ##########################################################################
    print('X shape: {}'.format(X.shape))
    print('X missing values: {}\n\n'.format(X.isnull().sum()))

    print('y shape: {}'.format(y.shape))
    print('y missing values: {}\n\n'.format(y.isnull().sum()))
    #################################################################################
    '''
        
    # Adding scaled columns to the dataframe
    #X = X.merge(scaled_cols, left_index = True, right_index = True, how = 'left')  # There was issue seperating scaled columns and merge them. I get 11 Nan values for some reason
    
    '''
    # Test ##########################################################################
    print('scaled_cols shape: {}'.format(scaled_cols.shape))
    print('scaled_cols missing values: \n{}\n\n'.format(scaled_cols.isnull().sum()))

    print('X shape: {}'.format(X.shape))
    print('X missing values: {}\n\n'.format(X.isnull().sum()))

    print('y shape: {}'.format(y.shape))
    print('y missing values: {}\n\n'.format(y.isnull().sum()))
    #################################################################################
    '''
    
    # Get features names
    features_names = X.columns    

    return X, y, X_original, features_names, num_cols, cat_cols
X, y, X_original, features_names, numeric_features, categorical_features = process_data( df = df, 
                                                                                         drop_cols = ['customerID', 'Churn'], 
                                                                                         unwanted_cols = ['customerID'], 
                                                                                         target_col = ['Churn'] )
# Check columns to ensure X matches X_original
pd.DataFrame({'X': X.columns.tolist(), 'X_or': X_original.columns.tolist()})
# Display shapes and types
print('X Original (without scaling):\n 1. type: {}\n 2. Converted type: {}\n 3.shape: {}\n'.format(type(X_original), type(X_original.values), X_original.shape))
print('X type:\n 1. type: {}\n 2. Converted type: {}\n 3.shape: {}\n'.format(type(X), type(X.values), X.shape))
print('y type:\n 1. type: {}\n 2. Converted type: {}\n 3.shape: {}\n'.format(type(y), type(y.values), y.shape))
# Display null values if existed in X features
pd.DataFrame(X.isnull().sum()).T
# Display null values if existed in X features
pd.DataFrame(X_original.isnull().sum()).T
X.describe().T
plt.figure(figsize= (15,10))

sns.heatmap(X.corr(), cmap="Blues") 

plt.show()
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve, scorer, f1_score, precision_score, recall_score, cohen_kappa_score

# For Visualization
from yellowbrick.classifier import DiscriminationThreshold 

# To ignore future warning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
def telecom_churn_prediction(algorithm, X_train, X_test, y_train, y_test, cols, cf, threshold_plot):
    
    #model
    algorithm.fit(X_train, y_train)
    predictions   = algorithm.predict(X_test)
    probabilities = algorithm.predict_proba(X_test)

    if cf != 'None':
        #coeffs
        if   cf == "coefficients" :
             coefficients  = pd.DataFrame(algorithm.coef_.ravel())
        elif cf == "features" :
             coefficients  = pd.DataFrame(algorithm.feature_importances_)

        column_df           =  pd.DataFrame(cols)
        coef_sumry          =  (pd.merge(coefficients,column_df,left_index= True, right_index= True, how = "left"))
        coef_sumry.columns  =  ["coefficients","features"]
        coef_sumry          =  coef_sumry.sort_values(by = "coefficients",ascending = False)
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, predictions)
    
    # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    model_roc_auc = roc_auc_score(y_test, predictions) 
    
    # Compute Receiver operating characteristic (ROC)
    fpr,tpr,thresholds = roc_curve(y_test, probabilities[:,1])

    # Prints
    print("Algorithm : \n", algorithm, '\n')   # Display algrithm used
    print("---------------------------------------------------------------------------------------------------------------------\n")
    print("Classification report : \n\n",classification_report(y_test, predictions))   # Display Classification Report
    print("---------------------------------------------------------------------------------------------------------------------\n")
    print("Accuracy Score : ",accuracy_score(y_test, predictions))                        # Display Accuracy Score
    print("Area under curve : ",model_roc_auc,"\n")                                       # Display ROC AUC 
    print("---------------------------------------------------------------------------------------------------------------------\n")
    
    # Plot Confusion Martix and Receiver Operating Characteristic
    fig, ax = plt.subplots(ncols = 2, figsize= (16,4))

    sns.heatmap(data = conf_matrix, xticklabels = ["Not churn","Churn"], yticklabels = ["Churn", "Not churn"], annot = True, fmt='g', ax= ax[0])
    sns.scatterplot(x = fpr, y = tpr, ax= ax[1])
    sns.lineplot(x = [0,1], y = [0,1], color = 'red', ax= ax[1])

    ax[0].set_title('Confusion Matrix Heatmap', pad = 10, weight= 'bold')

    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=90)
    ax[1].set_title('Area Under The Curve\nReceiver Operating Characteristic', pad = 10, weight= 'bold')
    ax[1].set_xlabel('False Positive Rate', weight= 'bold')
    ax[1].set_ylabel('True Positive Rate', weight= 'bold')

    plt.show()
    
    print("---------------------------------------------------------------------------------------------------------------------\n")
    
    if cf != 'None':
        # Plot Features Importances
        plt.figure(figsize = (16, 8))

        _ = sns.barplot( data = coef_sumry, x = 'features', y = 'coefficients')

        _.set_title('Features Importances Based on Coefficients', pad = 10, weight= 'bold')
        _.set_xlabel('Features', weight= 'bold')
        _.set_xticklabels(_.get_xticklabels(), rotation=90)
        _.set_ylabel('Coefficients', weight= 'bold')

        plt.show()
    
        print("---------------------------------------------------------------------------------------------------------------------\n")

    # Plot Threshold of the selected algorithm
    if threshold_plot == True : 
        visualizer = DiscriminationThreshold(algorithm)
        visualizer.fit(X_train, y_train)
        visualizer.poof()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20, random_state = 42)

print('X_train - shape: {}'.format(X_train.shape))
print('X_test  - shape: {}'.format(X_test.shape))
print('y_train - shape: {}'.format(y_train.shape))
print('y_test  - shape: {}'.format(y_test.shape))
# Initiate classifier
logreg = LogisticRegression()

# Get results
telecom_churn_prediction( algorithm = logreg,
                          X_train = X_train.values, 
                          X_test = X_test.values, 
                          y_train = y_train.values, 
                          y_test = y_test.values, 
                          cols = features_names, 
                          cf = 'coefficients',  
                          threshold_plot = True )  
# Check imbalanced target classes
print('Original Dataset Classes: \n\n{}'.format(pd.DataFrame({'full_train': y})['full_train'].value_counts()))
print('\n\ny_train Dataset Classes: \n\n{}'.format(pd.DataFrame({'y_train': y_train})['y_train'].value_counts()))
from imblearn.over_sampling import SMOTE
# Oversampling minority class using SMOTE
smote = SMOTE(random_state = 0)

smote_X_train, smote_y_train = smote.fit_sample(X_train.values, y_train.values)
# Notice - Classes are balanced and samples number increased from 7032 to 8260 samples

smote_X_train_df = pd.DataFrame(data = smote_X_train, columns = features_names)
smote_y_train_df = pd.DataFrame(data = smote_y_train, columns = ['Churn'])

# Check shape
print('os_smote_X shape: {}'.format(smote_X_train.shape))
print('os_smote_y shape: {}\n'.format(smote_y_train.shape))

# Check class / target balance
print('os_smote_y class balance: \n{}'.format(smote_y_train_df.Churn.value_counts(normalize= True)))
print("\n---------------------------------------------------------------------------------------------------------------------\n")

# Initiate classifier
logreg_smote = LogisticRegression()

# Get results
telecom_churn_prediction( algorithm = logreg_smote,
                          X_train = smote_X_train, 
                          X_test = X_test.values, 
                          y_train = smote_y_train, 
                          y_test = y_test.values, 
                          cols = features_names, 
                          cf = 'coefficients',  
                          threshold_plot = True )  
from sklearn.feature_selection import RFE
# Initiate algorithm
logreg_rfe = LogisticRegression()

rfe = RFE(estimator = logreg_rfe, n_features_to_select = 10) # Total features number is 32
rfe = rfe.fit(smote_X_train, smote_y_train)

rfe_selection_df = pd.DataFrame({ 'RFE_Supprt': rfe.support_,
                                  'Features': features_names,
                                  'Ranking': rfe.ranking_     })

# Extract the columns with RFE_Supprt == True where there ranking is 1
rfe_selection_cols = rfe_selection_df[rfe_selection_df.RFE_Supprt == True]['Features'].tolist()

#print('RFE Features Performance Details: \n {}\n'.format(rfe_selection_df))
#print('Selected Features: \n {}'.format(rfe_selection_cols))
rfe_selection_df.sort_values('Ranking')
# Create new X,y traing/ test datasets with the selected features 
rfe_X_train = smote_X_train_df[rfe_selection_cols]
rfe_y_train = smote_y_train
rfe_X_test = X_test[rfe_selection_cols]
rfe_y_test = y_test

# Initiate classifier
logreg_rfe = LogisticRegression()

# Get results
telecom_churn_prediction( algorithm = logreg_rfe,
                          X_train = rfe_X_train.values, 
                          X_test = rfe_X_test.values, 
                          y_train = rfe_y_train, 
                          y_test = rfe_y_test.values, 
                          cols = rfe_selection_cols, 
                          cf = 'coefficients',  
                          threshold_plot = True )  
from sklearn.feature_selection import chi2, SelectKBest
# Create new X, y without scaling numeric variables as this test takes non negative values

excluded_cols = ['customerID', 'Churn'] # Exclude customer id column and the target column 'Churn'
skb_cols = [col for col in features_names if col not in excluded_cols]

# Create new dataframes for X, y
skb_X_df = X_original[skb_cols]
skb_y_df = y

# Create arrays for X, y
skb_X = skb_X_df.values
skb_y = skb_y_df.values

# Fit SelectKBest model with k = 3
skb_model = SelectKBest(score_func = chi2, k = 3)
skb_model_fit = skb_model.fit(skb_X, skb_y)

# Scores summary
print('Score:\n{}\n'.format(skb_model_fit.scores_))
print('P - Values:\n{}\n'.format(skb_model_fit.pvalues_))

# Create dataframes
skb_scores_df = pd.DataFrame({ 'Features': skb_cols, 
                               'Scores': skb_model_fit.scores_, 
                               'P-Values': skb_model_fit.pvalues_,
                               'Type': np.where(skb_scores_df.Features.isin(numeric_features), 'Numerical', 'Categorical')}).sort_values(by = 'Scores', ascending = False)

# Display scatter plot for numerical features and barplot for categorical features
fig, ax = plt.subplots(ncols = 2, figsize= (25,6))

sns.scatterplot( data = skb_scores_df[skb_scores_df.Type == 'Categorical'], x = 'Features', y = 'Scores', ax = ax[0])
sns.barplot( data = skb_scores_df[skb_scores_df.Type == 'Numerical'], x = 'Features', y = 'Scores', ax = ax[1])

ax[0].set_title('Categorical Features Scores', pad = 10, weight= 'bold')
ax[0].set_xlabel('Features', weight= 'bold')
ax[0].set_xticklabels(skb_scores_df.loc[skb_scores_df.Type == 'Categorical', 'Features'].tolist(), rotation = 90)
ax[0].set_ylabel('Scores', weight= 'bold')

ax[1].set_title('Numerical Features Scores', pad = 10, weight= 'bold')
ax[1].set_xlabel('Features', weight= 'bold')
ax[1].set_ylabel('Scores', weight= 'bold')

plt.show()
from sklearn.neighbors import  KNeighborsClassifier
# Initiate KNN classifier
knn = KNeighborsClassifier()

# Get results
telecom_churn_prediction( algorithm = knn,
                          X_train = smote_X_train, 
                          X_test = X_test.values, 
                          y_train = smote_y_train, 
                          y_test = y_test.values, 
                          cols = features_names, 
                          cf = 'None',  
                          threshold_plot = True )  
from sklearn.naive_bayes import GaussianNB
# Initiate GaussianNB classifier
gnb = GaussianNB()

# Get results
telecom_churn_prediction( algorithm = gnb,
                          X_train = smote_X_train, 
                          X_test = X_test.values, 
                          y_train = smote_y_train, 
                          y_test = y_test.values, 
                          cols = features_names, 
                          cf = 'None',  
                          threshold_plot = True )  
from sklearn.svm import SVC
# Initiate SVC classifier
svc_linear = SVC(kernel = 'linear', probability = True)

# Get results
telecom_churn_prediction( algorithm = svc_linear,
                          X_train = smote_X_train, 
                          X_test = X_test.values, 
                          y_train = smote_y_train, 
                          y_test = y_test.values, 
                          cols = features_names, 
                          cf = 'coefficients',  
                          threshold_plot = False )  
from lightgbm import LGBMClassifier
# Initiate LGBM classifier
lgbm_clf = LGBMClassifier(objective = 'binary') # Default: 'regression' for LGBMRegressor, 'binary' or 'multiclass' for LGBMClassifier

# Get results
telecom_churn_prediction( algorithm = lgbm_clf,
                          X_train = smote_X_train, 
                          X_test = X_test.values, 
                          y_train = smote_y_train, 
                          y_test = y_test.values, 
                          cols = features_names, 
                          cf = 'features',  
                          threshold_plot = True )  
from xgboost import  XGBClassifier
# Initiate XGBoost classifier
xgb_clf = XGBClassifier()

# Get results
telecom_churn_prediction( algorithm = xgb_clf,
                          X_train = smote_X_train, 
                          X_test = X_test.values, 
                          y_train = smote_y_train, 
                          y_test = y_test.values, 
                          cols = features_names, 
                          cf = 'features',  
                          threshold_plot = True )  
# Initiate lists 
model_name_list = []
predictions_list = []
accuracyScore_list = []
recallScore_list = []
precisionScore_list = []
rocAucScore_list = []
f1Score_list = []
kappaScore_list = []

# Extract scores for each algorithm and append results to lists
def extract_algorithm_scores(model_name, model, X_train, X_test, y_train, y_test):
    
    model.fit(X_train, y_train)
    
    predictions    = model.predict(X_test)
    accuracyScore  = accuracy_score(y_test, predictions)
    recallScore    = recall_score(y_test, predictions)
    precisionScore = precision_score(y_test, predictions)
    rocAucScore    = roc_auc_score(y_test, predictions)
    f1Score        = f1_score(y_test, predictions)
    kappaScore     = cohen_kappa_score(y_test, predictions)

    model_name_list.append(model_name)
    predictions_list.append(predictions)
    accuracyScore_list.append(accuracyScore)
    recallScore_list.append(recallScore)
    precisionScore_list.append(precisionScore)
    rocAucScore_list.append(rocAucScore)
    f1Score_list.append(f1Score)
    kappaScore_list.append(kappaScore)

# Assemble scores from lists to dataframe
def assemble_scores():

    results_dict = { 'Model': model_name_list,
                     'Accuracy_Score': accuracyScore_list,
                     'Recall_Score': recallScore_list,
                     'Precision_Score': precisionScore_list,
                     'ROC_AUC_Score': rocAucScore_list,
                     'F1_Score': f1Score_list,
                     'Cohen_Kappa_Score': kappaScore_list }

    results_df = pd.DataFrame(results_dict)
    
    return results_df

# Call extract_algorithm_scores function to extract scores 
extract_algorithm_scores('Logistic Regression   (Baseline)'       , logreg       , X_train.values     , X_test.values     , y_train.values , y_test.values)
extract_algorithm_scores('Logistic Regression   (SMOTE)'          , logreg_smote , smote_X_train      , X_test.values     , smote_y_train  , y_test.values)
extract_algorithm_scores('Logistic Regression   (RFE)'            , logreg_rfe   , rfe_X_train.values , rfe_X_test.values , rfe_y_train    , rfe_y_test.values)
extract_algorithm_scores('KNeighbors Classifier (SMOTE)'          , knn          , smote_X_train      , X_test.values     , smote_y_train  , y_test.values)
extract_algorithm_scores('Gaussian Naive Bayes  (SMOTE)'          , gnb          , smote_X_train      , X_test.values     , smote_y_train  , y_test.values)
extract_algorithm_scores('Support Vector Classification  (SMOTE)' , svc_linear   , smote_X_train      , X_test.values     , smote_y_train  , y_test.values)
extract_algorithm_scores('Light GBM Classifier  (SMOTE)'          , lgbm_clf     , smote_X_train      , X_test.values     , smote_y_train  , y_test.values)
extract_algorithm_scores('XGBoost classifier    (SMOTE)'          , xgb_clf      , smote_X_train      , X_test.values     , smote_y_train  , y_test.values)

# Call assemble_scoes function to assemble scores into dataframe
results_df = assemble_scores()
# Display dataframe based on accuracy score
results_df.sort_values(by = 'Accuracy_Score', ascending = False)
# Change the shape of the dataframe using pd.melt
results_melted_df = pd.melt(frame = results_df, id_vars = ['Model'], value_vars = ['Accuracy_Score', 'Recall_Score', 'Precision_Score', 'ROC_AUC_Score', 'F1_Score', 'Cohen_Kappa_Score'], var_name = 'Score_Type', value_name = 'Score')
# Plot scores againt models
plt.figure(figsize= (30,12))

_ = sns.barplot( data = results_melted_df, x = 'Score', y = 'Model', hue = 'Score_Type', palette= "Paired")

_.set_title('Models Metrics / Scores', pad = 10, weight= 'bold')
_.set_xlabel('Performance Score', weight= 'bold')
_.set_ylabel('Model', weight= 'bold')

plt.show()