# # Imports



# Pandas and numpy for data manipulation

import pandas as pd

import numpy as np

import seaborn as sns

import os

sns.set(font_scale = 3)

# No warnings about setting value on copy of slice

pd.options.mode.chained_assignment = None



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





# Matplotlib visualization

import matplotlib.pyplot as plt

%matplotlib inline



# Set default font size

# plt.rcParams['font.size'] = 24



# Internal ipython tool for setting figure size

from IPython.core.pylabtools import figsize



# Seaborn for visualization





# Splitting data into training and testing

from sklearn.model_selection import train_test_split
# # # Data Cleaning and Formatting



# # Load in the Data and Examine



# Read in data into a dataframe 

df = pd.read_csv('/kaggle/input/loan-default-prediction/train_v2.csv.zip')



# Display top of dataframe

df.head()
df.shape

# df.info()
df.select_dtypes(include=['object']).head()
# Statistics for each column

df.describe()

# # Missing Values



# Function to calculate missing values by column

def total_missing_values(df):

        # Total missing values

        count_nulls = df.isnull().sum()

        

        # Percentage of missing values

        null_percentage = 100 * df.isnull().sum() / len(df)

        

        # Make a table with the results

        null_data = pd.concat([count_nulls, null_percentage], axis=1)

        

        # Rename the columns

        null_data_ren_columns = null_data.rename(columns = {0 : 'null values', 1 : 'null_percent'})

        

        # Sort the table by percentage of missing descending

        null_data_ren_columns = null_data_ren_columns[

            null_data_ren_columns.iloc[:,1] != 0].sort_values('null_percent', ascending=False).round(1)

        

        # Print some summary information

        print ('df has' + str(df.shape[1]) + " columns"      

            " and " + str(null_data_ren_columns.shape[0]) +

              " columns with null values.")

        

        # Return the dataframe with missing information

        return null_data_ren_columns
total_missing_values(df).head(50)
df.fillna(df.mean(), inplace=True)
total_missing_values(df).head(50)
df.dropna(inplace=True)

total_missing_values(df)
df.shape
# # # Exploratory Data Analysis



for i in df.select_dtypes(include=['object']).columns:

    df.drop(labels=i, axis=1, inplace=True)
# # Correlations between Features and Target



# Find all correlations and sort 

corr = df.corr()['loss'].sort_values()



# Print the most negative correlations

print(corr.head(10), '\n')



# Print the most positive correlations

print(corr.tail(10))
for i in df.columns:

    if len(set(df[i]))==1:

        df.drop(labels=[i], axis=1, inplace=True)
# Find all correlations and sort 

corr = df.corr()['loss'].sort_values()



# Print the most negative correlations

print(corr.head(15), '\n')



# Print the most positive correlations

print(corr.tail(15))
df.shape
# # # Feature Engineering and Selection



def remove_collinear_features(x, threshold):

    #remove outer effecting data

    # Dont want to remove correlations between loss

    y = x['loss']

    x = x.drop(columns = ['loss'])

    

    # Calculate the correlation matrix

    corr_matrix = x.corr()

    iters = range(len(corr_matrix.columns) - 1)

    drop_cols = []



    # comparinf correlations

    for i in iters:

        for j in range(i):

            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]

            col = item.columns

            row = item.index

            val = abs(item.values)

            

            # corr > threshold, then drop

            if val >= threshold:

                # Print the correlated features and the correlation value

                # print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))

                drop_cols.append(col.values[0])



    # Drop one of each pair of correlated columns

    drops = set(drop_cols)

    x = x.drop(columns = drops)

    

    # Add the score back in to the data

    x['loss'] = y

               

    return x
# Remove the collinear features above a specified correlation coefficient

df = remove_collinear_features(df, 0.6);
df.shape
# # # Split Into Training and Testing Sets



# Separate out the features and targets

features = df.drop(columns='loss')

targets = pd.DataFrame(df['loss'])



# Split into 80% training and 20% testing set

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size = 0.2, random_state = 42)



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
# # Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# Convert y to one-dimensional array (vector)

y_train = np.array(y_train).reshape((-1, ))

y_test = np.array(y_test).reshape((-1, ))
X_train
X_test
# # # Models to Evaluate



# We will compare five different machine learning Cassification models:



# 1 - Logistic Regression

# 2 - K-Nearest Neighbors Classification

# 3 - Suport Vector Machine

# 4 - Naive Bayes

# 5 - Random Forest Classification



# Function to calculate mean absolute error

def cross_val(X_train, y_train, model):

    # Applying k-Fold Cross Validation

    from sklearn.model_selection import cross_val_score

    accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 5)

    return accuracies.mean()



# Takes in a model, trains the model, and evaluates the model on the test set

def eval_fit(model):

    

    # Train the model

    model.fit(X_train, y_train)

    

    # Make predictions and evalute

    model_pred = model.predict(X_test)

    model_cross = cross_val(X_train, y_train, model)

    

    # Return the performance metric

    return model_cross
# # Naive Bayes

from sklearn.naive_bayes import GaussianNB

naive = GaussianNB()

naive_cross = eval_fit(naive)



print('Naive Bayes Performance on the test set: Cross Validation Score = %0.4f' % naive_cross)
# # Random Forest Classification

from sklearn.ensemble import RandomForestClassifier

random = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')

random_cross = eval_fit(random)



print('Random Forest Performance on the test set: Cross Validation Score = %0.4f' % random_cross)
# # Gradiente Boosting Classification

from xgboost import XGBClassifier

xc = XGBClassifier()

gb = eval_fit(xc)



print('Gradiente Boosting Classification Performance on the test set: Cross Validation Score = %0.4f' % gb)