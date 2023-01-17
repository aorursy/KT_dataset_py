#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
import os
# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder
# List files available
print(os.listdir("../input/"))
#Importing the dataset
df_train = pd.read_csv('../input/application_train.csv')
df_test=pd.read_csv('../input/application_test.csv')
#Shape of dataset
df_train.shape
df_train.head(8)
df_train.describe()
df_train['NAME_FAMILY_STATUS'].value_counts()
df_test.head(5)
df_test.describe()
df_train['TARGET'].value_counts()
df_train['TARGET'].astype(int).plot.hist();
# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
# Missing values statistics
missing_values = missing_values_table(df_train)
missing_values.head(20)
# Number of each type of column
df_train.dtypes.value_counts()
# Number of unique classes in each object column
df_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
from sklearn.preprocessing import LabelEncoder
# Create a label encoder object
le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in df_train:
    if df_train[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(df_train[col].unique())) <= 2:
            # Train on the training data
            le.fit(df_train[col])
            # Transform both training and testing data
            df_train[col] = le.transform(df_train[col])
            df_test[col] = le.transform(df_test[col])
            
            # Keep track of how many columns were label encoded
            le_count += 1
            
print('%d columns were label encoded.' % le_count)
# one-hot encoding of categorical variables
df_train = pd.get_dummies(df_train)
df_test = pd.get_dummies(df_test)

print('Training Features shape: ', df_train.shape)
print('Testing Features shape: ', df_test.shape)
train_labels = df_train['TARGET']

# Align the training and testing data, keep only columns present in both dataframes
df_train, df_test = df_train.align(df_test, join = 'inner', axis = 1)

# Add the target back in
df_train['TARGET'] = train_labels

print('Training Features shape: ', df_train.shape)
print('Testing Features shape: ', df_test.shape)
(df_train['DAYS_BIRTH'] / -365).describe()
df_train['DAYS_EMPLOYED'].describe()
df_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');
plt.xlabel('Days Employment');
# Missing values statistics
missing_values = missing_values_table(df_train)
missing_values.head(20)
#Omitting TARGET from Column list
train= df_train.drop(columns = ['TARGET'])
# Replace Nulls with NaN
# mark zero values as missing or NaN
train.iloc[:, :] = train.iloc[:, :].replace('' , np.NaN)
# count the number of NaN values in each column
print(train.isnull().sum())
# print the first 20 rows of data
print(df_train.head(20))
print (train.head(10))
# drop rows with missing values
train.dropna(inplace=True)
# summarize the number of rows and columns in the dataset
#Add in the TARGET
train['TARGET']=df_train['TARGET']
print(train.shape)
train.head(10)
#Deploying Logistic Regression
#Splitting the dataset

from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

X = train.iloc[:, :1]
y = train.iloc[:,1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
from sklearn.metrics import classification_report,accuracy_score
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
df_train = pd.read_csv('../input/application_train.csv')
missing_values = missing_values_table(df_train)
print (df_test.shape)
# Number of unique classes in each object column
df_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
print (list(df_train[col].unique()))
df_train = pd.get_dummies(df_train)
print('Training Features shape: ', df_train.shape)
train_labels = df_train['TARGET']

# Align the training and testing data, keep only columns present in both dataframes
df_train, df_test = df_train.align(df_test, join = 'inner', axis = 1)

# Add the target back in
df_train['TARGET'] = train_labels

print('Training Features shape: ', df_train.shape)
print('Testing Features shape: ', df_test.shape)
# Missing values statistics
missing_values = missing_values_table(df_train)
missing_values.head(20)
df_train.head(10)
# Replace Nulls with NaN
# mark zero values as missing or NaN

df_train.iloc[:, :1] = df_train.iloc[:, :1].replace('', np.NaN)
# count the number of NaN values in each column
print(df_train.isnull().sum())

# fill missing values with mean column values
df_train.fillna(df_train.mean(), inplace=True)
# count the number of NaN values in each column
print(df_train.isnull().sum())
df_train.head(10)
#LDA

from sklearn.preprocessing import Imputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# split dataset into inputs and outputs
#values = dataset.values
X = df_train.iloc[:, :-1]
y = df_train.iloc[:, -1]
# fill missing values with mean column values
imputer = Imputer()
transformed_X = imputer.fit_transform(X)
# evaluate an LDA model on the dataset using k-fold cross validation
model = LinearDiscriminantAnalysis()
kfold = KFold(n_splits=3, random_state=7)
result = cross_val_score(model, transformed_X, y, cv=kfold, scoring='accuracy')
print(result.mean())
# Deploying Logistic Regression
#Splitting the dataset
#Keep the following 6 features (variables) which are important
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

X = df_train.iloc[:, :1]
y = df_train.iloc[:,1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
from sklearn.metrics import classification_report,accuracy_score
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
from sklearn.preprocessing import MinMaxScaler, Imputer

# Drop the target from the training data
if 'TARGET' in df_train:
    train = df_train.drop(columns = ['TARGET'])
else:
    train = df_train.copy()
    
# Feature names
features = list(train.columns)

# Copy of the testing data
test = df_test.copy()

# Median imputation of missing values
imputer = Imputer(strategy = 'median')

# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (0, 1))

# Fit on the training data
imputer.fit(train)

# Transform both training and testing data
train = imputer.transform(train)
test = imputer.transform(df_test)

# Repeat with the scaler
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

print('Training data shape: ', train.shape)
print('Testing data shape: ', test.shape)
from sklearn.linear_model import LogisticRegression

# Make the model with the specified regularization parameter
log_reg = LogisticRegression(C = 0.0001)

# Train on the training data
log_reg.fit(train, train_labels)
# Make predictions
# Make sure to select the last column only
log_reg_pred = log_reg.predict_proba(test)[:, -1]
# Submission dataframe
submit = df_test[['SK_ID_CURR']]
submit['TARGET'] = log_reg_pred

submit.head()
# Save the submission to a csv file
submit.to_csv('log_reg_baseline.csv', index = False)
def numeric(col):
    plt.figure(figsize=(12,5))
    plt.title("Distribution of "+col)
    ax = sns.distplot(df_train[col].dropna())
numeric("AMT_CREDIT")
numeric("AMT_INCOME_TOTAL")
numeric("AMT_ANNUITY")
anom = df_train[df_train['DAYS_EMPLOYED'] == 365243]
non_anom = df_train[df_train['DAYS_EMPLOYED'] != 365243]
print('The non-anomalies default on %0.2f%% of loans' % (100 * non_anom['TARGET'].mean()))
print('The anomalies default on %0.2f%% of loans' % (100 * anom['TARGET'].mean()))
print('There are %d anomalous days of employment' % len(anom))
# Create an anomalous flag column
df_train['DAYS_EMPLOYED_ANOM'] = df_train["DAYS_EMPLOYED"] == 365243

# Replace the anomalous values with nan
df_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

df_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');
plt.xlabel('Days Employment');
df_test['DAYS_EMPLOYED_ANOM'] = df_test["DAYS_EMPLOYED"] == 365243
df_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)

print('There are %d anomalies in the test data out of %d entries' % (df_test["DAYS_EMPLOYED_ANOM"].sum(), len(df_test)))
(df_train['DAYS_BIRTH']/365.0).describe()
from plotly.offline import init_notebook_mode, iplot
from wordcloud import WordCloud
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.plotly as py
from plotly import tools
from datetime import date
import pandas as pd
import numpy as np 
import seaborn as sns
import random 
import warnings
warnings.filterwarnings("ignore")
init_notebook_mode(connected=True)

path = "../input/"

def bar_hor(df, col, title, color, w=None, h=None, lm=0, limit=100, return_trace=False, rev=False, xlb = False):
    cnt_srs = df[col].value_counts()
    yy = cnt_srs.head(limit).index[::-1] 
    xx = cnt_srs.head(limit).values[::-1] 
    if rev:
        yy = cnt_srs.tail(limit).index[::-1] 
        xx = cnt_srs.tail(limit).values[::-1] 
    if xlb:
        trace = go.Bar(y=xlb, x=xx, orientation = 'h', marker=dict(color=color))
    else:
        trace = go.Bar(y=yy, x=xx, orientation = 'h', marker=dict(color=color))
    if return_trace:
        return trace 
    layout = dict(title=title, margin=dict(l=lm), width=w, height=h)
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)

def bar_hor_noagg(x, y, title, color, w=None, h=None, lm=0, limit=100, rt=False):
    trace = go.Bar(y=x, x=y, orientation = 'h', marker=dict(color=color))
    if rt:
        return trace
    layout = dict(title=title, margin=dict(l=lm), width=w, height=h)
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)


def bar_ver_noagg(x, y, title, color, w=None, h=None, lm=0, rt = False):
    trace = go.Bar(y=y, x=x, marker=dict(color=color))
    if rt:
        return trace
    layout = dict(title=title, margin=dict(l=lm), width=w, height=h)
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)
    
def gp(col, title):
    df1 = df_train[df_train["TARGET"] == 1]
    df0 = df_train[df_train["TARGET"] == 0]
    a1 = df1[col].value_counts()
    b1 = df0[col].value_counts()
    
    total = dict(df_train[col].value_counts())
    x0 = a1.index
    x1 = b1.index
    
    y0 = [float(x)*100 / total[x0[i]] for i,x in enumerate(a1.values)]
    y1 = [float(x)*100 / total[x1[i]] for i,x in enumerate(b1.values)]

    trace1 = go.Bar(x=a1.index, y=y0, name='Target : 1', marker=dict(color="#96D38C"))
    trace2 = go.Bar(x=b1.index, y=y1, name='Target : 0', marker=dict(color="#FEBFB3"))
    return trace1, trace2 


target_distribution = df_train['TARGET'].value_counts()
target_distribution.plot.pie(figsize=(10, 10),
                             title='Target Distribution',
                             fontsize=15, 
                             legend=True, 
                             autopct=lambda v: "{:0.1f}%".format(v))
# Find correlations with the target and sort
correlations = df_train.corr()['TARGET'].sort_values()

# Display correlations
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))
# Find the correlation of the positive days since birth and target
df_train['DAYS_BIRTH'] = abs(df_train['DAYS_BIRTH'])
df_train['DAYS_BIRTH'].corr(df_train['TARGET'])
# Set the style of plots
plt.style.use('fivethirtyeight')

# Plot the distribution of ages in years
plt.hist(df_train['DAYS_BIRTH'] / 365, edgecolor = 'k', bins = 25)
plt.title('Age of Client'); plt.xlabel('Age (years)'); plt.ylabel('Count');
import seaborn as sns
plt.figure(figsize = (10, 8))

# KDE plot of loans that were repaid on time
sns.kdeplot(df_train.loc[df_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'target == 0')

# KDE plot of loans which were not repaid on time
sns.kdeplot(df_train.loc[df_train['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label = 'target == 1')

# Labeling of plot
plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');
# Age information into a separate dataframe
age_data = df_train[['TARGET', 'DAYS_BIRTH']]
age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365

# Bin the age data
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
age_data.head(10)
# Group by the bin and calculate averages
age_groups  = age_data.groupby('YEARS_BINNED').mean()
age_groups
plt.figure(figsize = (8, 8))

# Graph the age bins and the average of the target as a bar plot
plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])

# Plot labeling
plt.xticks(rotation = 75); plt.xlabel('Age Group (years)'); plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by Age Group');
# Extract the EXT_SOURCE variables and show correlations
ext_data = df_train[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'NAME_EDUCATION_TYPE_Higher education', 'CODE_GENDER_F']]
ext_data_corrs = ext_data.corr()
ext_data_corrs

plt.figure(figsize = (25, 36))
# Heatmap of correlations
sns.heatmap(ext_data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.02, annot = True, vmax = 0.5)
plt.title('Correlation Heatmap');
plt.figure(figsize = (10, 12))
# iterate through the sources
for i, source in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):
    
    # create a new subplot for each source
    plt.subplot(3, 1, i + 1)
    # plot repaid loans
    sns.kdeplot(df_train.loc[df_train['TARGET'] == 0, source], label = 'target == 0')
    # plot loans that were not repaid
    sns.kdeplot(df_train.loc[df_train['TARGET'] == 1, source], label = 'target == 1')
    
    # Label the plots
    plt.title('Distribution of %s by Target Value' % source)
    plt.xlabel('%s' % source); plt.ylabel('Density');
    
plt.tight_layout(h_pad = 2.5)

# Copy the data for plotting
plot_data = ext_data.drop(columns = ['DAYS_BIRTH']).copy()

# Add in the age of the client in years
plot_data['YEARS_BIRTH'] = age_data['YEARS_BIRTH']

# Drop na values and limit to first 100000 rows
plot_data = plot_data.dropna().loc[:100000, :]

# Function to calculate correlation coefficient between two columns
def corr_func(x, y, **kwargs):
    r = np.corrcoef(x, y)[0][1]
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),
                xy=(.2, .8), xycoords=ax.transAxes,
                size = 20)

# Create the pairgrid object
grid = sns.PairGrid(data = plot_data, size = 3, diag_sharey=False,
                    hue = 'TARGET', 
                    vars = [x for x in list(plot_data.columns) if x != 'TARGET'])

# Upper is a scatter plot
grid.map_upper(plt.scatter, alpha = 0.2)

# Diagonal is a histogram
grid.map_diag(sns.kdeplot)

# Bottom is density plot
grid.map_lower(sns.kdeplot, cmap = plt.cm.OrRd_r);

plt.suptitle('Ext Source and Age Features Pairs Plot', size = 32, y = 1.05);
n= len(df_train)
print(n)
x = np.sort(df_train['TARGET'])
y = np.arange(1,len(x)+1)/float(len(x)) 
print (y)

_= plt.plot(x, y, marker = '.', linestyle = 'none')
_= plt.xlabel('Customer Credibility to Repay Loan')
_= plt.ylabel('ECDF')
_= plt.margins(.02)

plt.show()
# Checking ECDF Distribution of Ability to Repay Loan across the real data and theoretical samples of data
def ecdf(data):
    x= np.sort(data)
    n= float(len(data))
    y = np.arange(1, n+1)/n
    return x,y

%matplotlib inline
plt.figure(figsize=(25,20))

# Seed the random number generator:
np.random.seed(15)
#Sample data for theortical normal dist
samples = np.random.normal(np.mean(df_train.TARGET), np.std(df_train.TARGET), size=10000)
samples
#find ecdf of data
x_count, y_count = ecdf(df_train.TARGET)
x_theor, y_theor = ecdf(samples)

fig = plt.plot(x_count, y_count, marker='.', linestyle='none')
fig = plt.plot(x_theor, y_theor, marker='.', linestyle='none')

# Label axes and add legend and a title:
fig = plt.title('Customer Credibility to Repay Loan VS Theoretical Normal Dist')
fig = plt.xlabel('Customer to Repay Loan')
fig = plt.ylabel('ECDF')

# Save and display the plots:
#plt.savefig('reports/figures/cdf_body_temps.png')
plt.show()
np.percentile(df_train['TARGET'], [25, 50, 75, 90, 98, 100])
pd.DataFrame.hist(df_train, column='TARGET')
np.var(df_train['TARGET'])
np.std(df_train['TARGET'])
np.cov(df_train['TARGET'], df_train['EXT_SOURCE_1'])
np.cov(df_train['TARGET'], df_train['DAYS_BIRTH'])
np.corrcoef(df_train['TARGET'], df_train['DAYS_BIRTH'])
def bootstrap_replicate_1d(data, func):
    return func(np.random.choice(df_train['TARGET'], size=len(df_train['TARGET'])))
#np.random.choice() works on linear model
def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates
# Take 10,000 bootstrap replicates of the mean: bs_replicates
bs_replicates = draw_bs_reps(df_train['TARGET'], np.mean, 10000)

# Compute and print SEM Standard Error of the Mean
sem = np.std(df_train['TARGET']) / np.sqrt(len(df_train['TARGET']))
print(sem)

# Compute and print standard deviation of bootstrap replicates
bs_std = np.std(bs_replicates)
print(bs_std)

# Make a histogram of the results
_ = plt.hist(bs_replicates, bins=50, normed=True)
_ = plt.xlabel('Credit Loan Default Risk')
_ = plt.ylabel('ECDF')

# Show the plot
plt.show()
np.percentile(bs_replicates, [2.5, 97.5])
#Finding pairs bootstrap for slope & intercept of a linear function between Bike REntal Count and Registered User Type
def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""

    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, 1)

    return bs_slope_reps, bs_intercept_reps

# Generate replicates of slope and intercept using pairs bootstrap
bs_slope_reps, bs_intercept_reps = draw_bs_pairs_linreg(df_train['EXT_SOURCE_1'], df_train['TARGET'], 1000)

# Compute and print 95% CI for slope
print(np.percentile(bs_slope_reps, [2.5, 97.5]))
# Plot the histogram
_ = plt.hist(bs_slope_reps, bins=50, normed=True)
_ = plt.xlabel('slope')
_ = plt.ylabel('PDF')
plt.show()
def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""

    # Concatenate the data sets: data
    data = np.concatenate((data1, data2))

    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2

for _ in range(50):
    # Generate permutation samples
    perm_sample_1, perm_sample_2 = permutation_sample(
                                    df_train['EXT_SOURCE_1'], df_train['EXT_SOURCE_2'])

    # Compute and plot ECDF from permutation sample 1 
    x1 = np.sort(perm_sample_1)
    y1 = np.arange(1,len(x1)+1)/float(len(x1)) 
    
    # Compute and plot ECDF from permutation sample 2
    x2 = np.sort(perm_sample_2)
    y2 = np.arange(1,len(x2)+1)/float(len(x2))


    # Plot ECDFs of permutation sample
    _ = plt.plot(x1, y1, marker='.', linestyle='none',
                 color='red', alpha=0.02)
    _ = plt.plot(x2, y2, marker='.', linestyle='none',
                 color='blue', alpha=0.02)
# Compute and plot ECDF from original 'registered'
x11 = np.sort(df_train['EXT_SOURCE_1'])
y11 = np.arange(1,len(x11)+1)/float(len(x11)) 

_ = plt.plot(x11, y11, marker='.', color= 'red')

# Compute and plot ECDF from original 'casual'
x22 = np.sort(df_train['EXT_SOURCE_2'])
y22 = np.arange(1,len(x22)+1)/float(len(x22)) 

_ = plt.plot(x22, y22, marker='.', color= 'blue')
# Make margins and label axes
plt.margins(0.02)
_ = plt.xlabel('External Data Source Influence')
_ = plt.ylabel('ECDF')

# Show the plot
plt.show()