# to handle datasets
import pandas as pd
import numpy as np

# for plotting
import matplotlib.pyplot as plt
% matplotlib inline

# to display all the columns of the dataframe in the notebook
pd.pandas.set_option('display.max_columns', None)
# load dataset
data = pd.read_csv('diabetic_data.csv')

# rows and columns of the data
print(data.shape)     

# visualise the dataset
data.head()
# make a list of the variables that contain missing values
vars_with_na = [var for var in data.columns if data[var].isnull().sum()>1]

# print the variable name and the percentage of missing values
for var in vars_with_na:
    print(var, np.round(data[var].isnull().mean(), 3),' % missing values')
def analyse_na_value(df, var):
    df = df.copy()
    
    # let's make a variable that indicates 1 if the observation was missing or zero otherwise
    df[var] = np.where(df[var].isnull(), 1, 0)
    
      # let's calculate the mean diabetesMed where the information is missing or present
    df.groupby(var)['diabetesMed'].median().plot.bar()
    plt.title(var)
    plt.show()
    
for var in vars_with_na:
    analyse_na_value(data, var)
# list of numerical variables
num_vars = [var for var in data.columns if data[var].dtypes != 'O']

print('Number of numerical variables: ', len(num_vars))

# visualise the numerical variables
data[num_vars].head()
print('Number of  Id labels: ', len(data.encounter_id.unique()))
print('Number of data in the Dataset: ', len(data))
#  list of discrete variables
discrete_vars = [var for var in num_vars if len(data[var].unique())<20 and var not in year_vars+['Id']]

print('Number of discrete variables: ', len(discrete_vars))
# let's visualise the discrete variables
data[discrete_vars].head()
#converting  target to  numerical 
###converting target using Label Encoder 

from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
data["diabetesMed"] = lb_make.fit_transform(data["diabetesMed"])
data[["diabetesMed"]].head()
def analyse_discrete(df, var):
    df = df.copy()
    df.groupby(var)['diabetesMed'].median().plot.bar()
    plt.title(var)
    plt.ylabel('diabetesMed')
    plt.show()
    
for var in discrete_vars:
    analyse_discrete(data, var)
# list of continuous variables
cont_vars = [var for var in num_vars if var not in discrete_vars+year_vars+['Id']]

print('Number of continuous variables: ', len(cont_vars))
# let's visualise the continuous variables
data[cont_vars].head()
# Let's go ahead and analyse the distributions of these variables
def analyse_continous(df, var):
    df = df.copy()
    df[var].hist(bins=20)
    plt.ylabel('Number of rows')
    plt.xlabel(var)
    plt.title(var)
    plt.show()
    
for var in cont_vars:
    analyse_continous(data, var)
# Let's go ahead and analyse the distributions of these variables
def analyse_transformed_continous(df, var):
    df = df.copy()
    
    # log does not take negative values, so let's be careful and skip those variables
    if 0 in data[var].unique():
        pass
    else:
        # log transform the variable
        df[var] = np.log(df[var])
        df[var].hist(bins=20)
        plt.ylabel('Number of houses')
        plt.xlabel(var)
        plt.title(var)
        plt.show()
    
for var in cont_vars:
    analyse_transformed_continous(data, var)
# let's explore the relationship between the house price and the transformed variables
# with more detail
def transform_analyse_continous(df, var):
    df = df.copy()
    
    # log does not take negative values, so let's be careful and skip those variables
    if 0 in data[var].unique():
        pass
    else:
        # log transform
        df[var] = np.log(df[var])
        df['diabetesMed'] = np.log(df['diabetesMed'])
        plt.scatter(df[var], df['diabetesMed'])
        plt.ylabel('diabetesMed')
        plt.xlabel(var)
        plt.show()
    
for var in cont_vars:
    if var !='diabetesMed':
        transform_analyse_continous(data, var)
# let's make boxplots to visualise outliers in the continuous variables 

def find_outliers(df, var):
    df = df.copy()
    
    # log does not take negative values, so let's be careful and skip those variables
    if 0 in data[var].unique():
        pass
    else:
        df[var] = np.log(df[var])
        df.boxplot(column=var)
        plt.title(var)
        plt.ylabel(var)
        plt.show()
    
for var in cont_vars:
    find_outliers(data, var)
### Categorical variables

cat_vars = [var for var in data.columns if data[var].dtypes=='O']

print('Number of categorical variables: ', len(cat_vars))
# let's visualise the values of the categorical variables
data[cat_vars].head()
for var in cat_vars:
    print(var, len(data[var].unique()), ' categories')
def analyse_rare_labels(df, var, rare_perc):
    df = df.copy()
    tmp = df.groupby(var)['diabetesMed'].count() / len(df)
    return tmp[tmp<rare_perc]

for var in cat_vars:
    print(analyse_rare_labels(data, var, 0.01))
    print()

for var in cat_vars:
    analyse_discrete(data, var)
# to divide train and test set
from sklearn.model_selection import train_test_split

# feature scaling
from sklearn.preprocessing import MinMaxScaler

# Let's separate into train and test set
# Remember to set the seed (random_state for this sklearn function)

X_train, X_test, y_train, y_test = train_test_split(data, data.diabetesMed,
                                                    test_size=0.2,
                                                    random_state=0) # we are setting the seed here
X_train.shape, X_test.shape
# this function will assign discrete values to the strings of the variables, 
# so that the smaller value corresponds to the smaller mean of target

def replace_categories(train, test, var, target):
    ordered_labels = train.groupby([var])[target].mean().sort_values().index
    ordinal_label = {k:i for i, k in enumerate(ordered_labels, 0)} 
    train[var] = train[var].map(ordinal_label)
    test[var] = test[var].map(ordinal_label)
for var in cat_vars:
    replace_categories(X_train, X_test, var, 'diabetesMed')
# check absence of na
[var for var in X_train.columns if X_train[var].isnull().sum()>0]
# check absence of na
[var for var in X_test.columns if X_test[var].isnull().sum()>0]
# let me show you what I mean by monotonic relationship between labels and target
def analyse_vars(df, var):
    df = df.copy()
    df.groupby(var)['diabetesMed'].median().plot.bar()
    plt.title(var)
    plt.ylabel('diabetesMed')
    plt.show()
    
for var in cat_vars:
    analyse_vars(X_train, var)
# check absence of missing values
X_train.isnull().sum().sum()