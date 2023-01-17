# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

five_thirty_eight = [
    "#30a2da",
    "#fc4f30",
    "#e5ae38",
    "#6d904f",
    "#8b8b8b",
]


plt.style.use('fivethirtyeight')
#sns.set_palette(five_thirty_eight)
#sns.palplot(sns.color_palette())

%matplotlib inline 


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go

init_notebook_mode(connected=True) #do not miss this line
from plotly import tools


# For model estimation
from sklearn.preprocessing import LabelEncoder,MinMaxScaler, Imputer
from sklearn.linear_model import LogisticRegression
from sklearn import svm


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
PATH = "../input"
# Any results you write to the current directory are saved as output.
application_train = pd.read_csv(PATH+"/application_train.csv")
application_test = pd.read_csv(PATH+"/application_test.csv")
bureau = pd.read_csv(PATH+"/bureau.csv")
bureau_balance = pd.read_csv(PATH+"/bureau_balance.csv")
credit_card_balance = pd.read_csv(PATH+"/credit_card_balance.csv")
installments_payments = pd.read_csv(PATH+"/installments_payments.csv")
previous_application = pd.read_csv(PATH+"/previous_application.csv")
POS_CASH_balance = pd.read_csv(PATH+"/POS_CASH_balance.csv")
correlations = application_train.corr()['TARGET'].sort_values()

# Display correlations
print('Most Positive Correlations: \n', correlations.tail(15))
print('\nMost Negative Correlations: \n', correlations.head(15))
application_train.head()
application_train.columns.values
bureau.head()
def missing(data):
    miss = data.isnull().sum().sort_values(ascending = False)
    perc = 100*(data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
    return pd.concat([miss, perc], axis = 1, keys = ['Total', 'Percent'])
missing(application_train).head(20)
missing(bureau).head(20)
missing(bureau_balance)
missing(credit_card_balance)
missing(previous_application)
missing(POS_CASH_balance)
missing(installments_payments)
plt.figure(figsize=(10,6))
sns.countplot(x='TARGET', data =  application_train)
plt.title("Number of Loans that were repayed and not repayed")


# prop = (temp.values/(temp.values[0]+temp.values[1]))*100
# print("Propotion of loans not repayed: %.2f" %prop[1]+"%")
def count_plots(feature, label_rotation=False):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,9))
    plot1 = sns.countplot(ax = ax1, x=feature, hue = 'TARGET',data =  application_train)
    if(label_rotation):
        plot1.set_xticklabels(plot1.get_xticklabels(),rotation=90)
    plt.tick_params(axis='both', which='major', labelsize=10)
    # since 1 means not repayed the mean will give us the proportion of non repayed loans
    perc_grouped = application_train[[feature, 'TARGET']].groupby(feature).mean().sort_values(by="TARGET", ascending= False)
    plot2 = sns.barplot(ax=ax2, x=perc_grouped['TARGET'], y = perc_grouped.index, orient="h")

    plt.tight_layout()
    plt.show()
count_plots("CODE_GENDER")
count_plots('NAME_EDUCATION_TYPE', label_rotation=True)
count_plots("NAME_FAMILY_STATUS")
count_plots("CNT_CHILDREN")
# * OCCUPATION_TYPE
# * FLAG_OWN_CAR (own car)
# * FLAG_OWN_REALTY (own house)
count_plots("OCCUPATION_TYPE",  label_rotation=True)
count_plots("NAME_INCOME_TYPE", True)
count_plots("FLAG_OWN_CAR")
count_plots("FLAG_OWN_REALTY")
count_plots("NAME_CONTRACT_TYPE")
count_plots("ORGANIZATION_TYPE", True)
def plot_distribution(data, feature, log_transform = False):
    plt.figure(figsize=(12,8))
    if log_transform:
        plt.title("Distribution of log of %s" % feature)
        sns.distplot(np.log(data[feature]).dropna(), kde=True,bins=100)
    else:
        plt.title("Distribution of %s" % feature)
        sns.distplot(data[feature].dropna(), kde=True,bins=100)
    plt.show() 
def plot_target(data, feature, xlab= '', ylab= '', title= ""):
    plt.figure(figsize=(12,9))
    sns.kdeplot(data.loc[data['TARGET'] == 0, feature], label = 'target == 0')

    # KDE plot of loans which were not repaid on time
    sns.kdeplot(data.loc[data['TARGET'] == 1, feature], label = 'target == 1')
    
    # Labeling of plot
    plt.xlabel(feature); plt.ylabel('Density'); plt.title("Distribution of %s"%(feature));
plot_distribution(application_train, "AMT_INCOME_TOTAL")
plot_target(application_train, "AMT_INCOME_TOTAL")
plot_distribution(application_train, "AMT_INCOME_TOTAL", True)
plot_distribution(application_train,"AMT_CREDIT")
plot_target(application_train, "AMT_CREDIT")
plot_distribution(application_train,"AMT_CREDIT", True)
plot_distribution(application_train,"AMT_ANNUITY")
plot_target(application_train, "DAYS_BIRTH", "Days since Birth")
# The younger you are the less likely you are to repay your loan
plot_distribution(application_train,"AMT_ANNUITY", True)
plot_distribution(application_train,"AMT_GOODS_PRICE")
plot_distribution(application_train,"AMT_GOODS_PRICE", True)
plot_distribution(application_train,"DAYS_EMPLOYED")
plot_distribution(application_train,"EXT_SOURCE_1")
plot_target(application_train, "EXT_SOURCE_1")
plot_distribution(application_train,"EXT_SOURCE_2")
plot_target(application_train, "EXT_SOURCE_2")
plot_distribution(application_train,"EXT_SOURCE_3")
plot_target(application_train, "EXT_SOURCE_3")
application_train['DAYS_EMPLOYED'].describe()
print("Longest time unemployed is %.0f" % (application_train['DAYS_EMPLOYED'].min()/365 * -1) + " years")
print("Longest time employed is %.0f" % (application_train['DAYS_EMPLOYED'].max()/365) + " years")
plot_distrubtion(application_train,"DAYS_REGISTRATION")
bureau.info()
app_bur_train = application_train.merge(bureau, left_on = 'SK_ID_CURR', right_on = 'SK_ID_CURR', how = 'inner')
print(application_train.shape, bureau.shape, app_bur_train.shape)
def count_plots2(data, feature, label_rotate = False):
    plt.figure(figsize=(12,9))
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,9))
    plot1 = sns.countplot(ax=ax1, x = feature, hue = 'TARGET', data = data)
    if(label_rotate):
            plot1.set_xticklabels(plot1.get_xticklabels(),rotation=90)
    plt.tick_params(axis='both', which='major', labelsize=10)
    perc_grouped = data[[feature, 'TARGET']].groupby(feature).mean().sort_values(by="TARGET", ascending= False)
    plot2 = sns.barplot(ax=ax2, x=perc_grouped['TARGET'], y = perc_grouped.index, orient="h")

    plt.tight_layout()
    plt.show()
count_plots2(app_bur_train, "CREDIT_ACTIVE" ,True)
app_bur_train["CREDIT_ACTIVE"].value_counts()
count_plots2(app_bur_train, "CREDIT_CURRENCY", True)
import plotly.graph_objs as go
x = app_bur_train["CREDIT_ACTIVE"].value_counts()

trace1 = go.Bar(
                x = x.index,
                y = x.values,
                name = "Credit")
data = [trace1]

fig = go.Figure(data = data)
py.offline.iplot(fig)
def plotly_plot(df,feature):
    x = df[feature].value_counts().sort_values(ascending=False)
    trace = go.Bar(
                x = x.index,
                y = x.values,
                name = feature)
    data = [trace]

    fig = go.Figure(data = data)
    py.offline.iplot(fig)
plotly_plot(app_bur_train,"CREDIT_TYPE")
def plotly_plots_percs(data, feature):
    # groupby and mean gives us average default rate by that particular feature
    x = pd.DataFrame(data[[feature, "TARGET"]].groupby(feature).mean().sort_values(by = "TARGET",ascending=False))
    trace1 = go.Bar(
                x = x.index,
                y = x['TARGET'],
                name = feature)

    data = [trace1]

    fig = go.Figure(data = data)
    py.offline.iplot(fig)
plotly_plot(app_bur_train, "CREDIT_TYPE")
# x = pd.DataFrame(app_bur_train[["CREDIT_TYPE", "TARGET"]].groupby("CREDIT_TYPE").mean().sort_values(by = "TARGET",ascending=False))
bureau_balance.columns.values
# number of months that each loan has had the balance outstanding
bureau_balance_size = bureau_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].size()

# longest number of months they have had the balance outstanding
bureau_balance_max = bureau_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].max()

# shortest number of months they had balance outstanding
bureau_balance_min = bureau_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].min()
plot_distribution(bureau_balance,"MONTHS_BALANCE")
# It looks like the majoroty of loans are repayed quite quickly
credit_card_balance.info()
print("Number of rows in the dataset: %d" %len(credit_card_balance['SK_ID_CURR']))
print("Number of unique credit cards: %d" %len(credit_card_balance['SK_ID_CURR'].unique()))
plot_distribution(credit_card_balance, "MONTHS_BALANCE")
plot_distribution(credit_card_balance, "AMT_BALANCE")
plot_distribution(credit_card_balance, "AMT_CREDIT_LIMIT_ACTUAL")
print("Max credit limit: %d" %credit_card_balance['AMT_CREDIT_LIMIT_ACTUAL'].max())
print("Min credit limit: %d" %credit_card_balance['AMT_CREDIT_LIMIT_ACTUAL'].min())
temp1 = credit_card_balance["NAME_CONTRACT_STATUS"].value_counts().sort_values(ascending=False)
print(temp1)
plt.figure(figsize=(10,8))
temp_perc = 100*(credit_card_balance["NAME_CONTRACT_STATUS"].value_counts().sort_values(ascending=False)/temp1.sum())
print(temp_perc)
sns.countplot(x="NAME_CONTRACT_STATUS", data=credit_card_balance)
le = LabelEncoder()
le_count = 0

# only label encode those variables with 2 or less categories
for col in application_train:
    if application_train[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(application_train[col].unique())) <= 2:
            # Train on the training data
            le.fit(application_train[col])
            # Transform both training and testing data
            application_train[col] = le.transform(application_train[col])
            application_test[col] = le.transform(application_test[col])
            
            # Keep track of how many columns were label encoded
            le_count += 1
            
print('%d columns were label encoded.' % le_count)
application_train = pd.get_dummies(application_train)
application_test = pd.get_dummies(application_test)

print('Training Features shape: ', application_train.shape)
print('Testing Features shape: ', application_test.shape)
target = application_train['TARGET']

application_train, application_test = application_train.align(application_test, join = 'inner', axis = 1)

print('Training Features shape: ', application_train.shape)
print('Testing Features shape: ', application_test.shape)
# put target back in training set
#application_train['TARGET'] = target
from sklearn.pipeline import Pipeline
# Drop the target from the training data
if 'TARGET' in application_train:
    train = application_train.drop(columns = ['TARGET'])
else:
    train = application_train.copy()
features = list(train.columns)

test = application_test.copy()

# Median imputation of missing values
imputer = Imputer(strategy = 'median')

# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (0, 1))

# Fit on the training data
imputer.fit(train)

# Transform both training and testing data
train = imputer.transform(train)
test = imputer.transform(application_test)

# Repeat with the scaler
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

print('Training data shape: ', train.shape)
print('Testing data shape: ', test.shape)

# Make the model with the specified regularization parameter
log_reg = LogisticRegression(C = 0.0005)

# Train on the training data
log_reg.fit(train, target)

# Make predictions
# Make sure to select the second column only
log_reg_pred = log_reg.predict_proba(test)[:, 1]
# Submission dataframe
submit = application_test[['SK_ID_CURR']]
submit['TARGET'] = log_reg_pred

submit.head()
# Save the submission to a csv file
submit.to_csv('logistic_baseline.csv', index = False)