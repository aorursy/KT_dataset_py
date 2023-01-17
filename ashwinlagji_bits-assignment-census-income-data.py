import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import r2_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn.model_selection import train_test_split



from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix

from sklearn.naive_bayes import MultinomialNB



%matplotlib inline 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



if not os.path.exists("../input/dataset.csv"):

    os.symlink("/kaggle/input/adult-census-income/adult.csv", "../input/dataset.csv")



    

print("Setup Complete")
# Data filepath

data_filepath = '../input/dataset.csv'



# Read the data

adult_census_data = pd.read_csv(data_filepath,

                                header=0, 

                                sep=',', 

                                na_values=['.', '??','?', '', ' ', 'NA', 'na', 'Na', 'N/A', 'N/a', 'n/a']

                               )

# Print the top 10 entries

adult_census_data.head(10)
# Describe the data set

adult_census_data.describe().T


print('Shape of dataset: {}'.format(adult_census_data.shape))



# Data Types of all the variables

print('Feature Type: ')

print('{}'.format(adult_census_data.dtypes))



# Number of Unique values present in each variable

adult_census_data.nunique()
# unique values in each columns

for attribute in adult_census_data.columns:

    print("{} ".format(attribute))

    print("{}".format(adult_census_data[attribute].unique()),"\n")


# categorising the variables in two category " Continuos" and "Categorical"

continuous_attributes = [attribute for attribute in adult_census_data.columns if adult_census_data[attribute].dtypes !='object']  

categorical_attributes = [attribute for attribute in adult_census_data.columns if adult_census_data[attribute].dtypes =='object']



print ( continuous_attributes)

print ( categorical_attributes)
# Heatmap to check the missing values in the dataset

plt.figure(figsize=(18,8))

sns.heatmap(adult_census_data.isnull())
# Number of Unique values present in each variable

# employee_data.nunique()



# Make a copy of employee dataframe

df = adult_census_data.copy()



print(df[df.isnull().any(axis=1)])



#Creating dataframe with number of missing values

null_data_rows = df[df.isnull().any(axis=1)]

null_data_columns = df.columns[df.isnull().any()]

print("number of missing data ::", null_data_rows.count())

print("Percentage Of missing data (All tuples)",(null_data_rows.count()/df.count())*100 )



missing_val = pd.DataFrame(df.isnull().sum())



#Reset the index to get row names as columns

missing_val = missing_val.reset_index()



#Rename the columns

missing_val = missing_val.rename(columns = {'index': 'Variables', 0: 'Missing_percntage'})

missing_val



#Calculate percentage

missing_val['Missing_percntage'] = (missing_val['Missing_percntage']/len(df))*100





#Sort the rows according to decreasing missing percentage

missing_val = missing_val.sort_values('Missing_percntage', ascending = False).reset_index(drop = True)



#Save output to csv file

missing_val.to_csv("Missing_percntage.csv", index = False)



# Return the percentage of missing data in the original dataset

def PerOfMissing(d1,d2):# d1--data by droping the NAN value d2--Original data

    percent_of_missing_data = round( 100 - ((len(d1)/len(d2))*100), 2)

    percent_of_missing_data = str(percent_of_missing_data) + '% of data has Missing value'

    return percent_of_missing_data



# droping all the NAN value from the data and saving the data in data_without_NAN

data_without_NAN = adult_census_data.dropna()

print (PerOfMissing(data_without_NAN,adult_census_data))



print(null_data_rows[null_data_columns].head())

print(null_data_columns)

missing_val
# get names of columns with missing values

cols_with_missing = [col for col in adult_census_data.columns

                     if adult_census_data[col].isnull().any()] 

print(cols_with_missing)
# replacing with the MODE of the data

from sklearn_pandas import CategoricalImputer

imputer = CategoricalImputer()



for col in cols_with_missing:

    adult_census_data[col] = imputer.fit_transform(adult_census_data[col])



# get names of columns with missing values

cols_with_missing = [col for col in adult_census_data.columns

                     if adult_census_data[col].isnull().any()] 

print(cols_with_missing)
print(" CATEGORICAL ATTRIBUTES ")

print(categorical_attributes)
adult_census_data['workclass'].value_counts()
self_employed = ['Self-emp-not-inc','Self-emp-inc']

govt_employees = ['Local-gov','State-gov','Federal-gov']



# replace elements in list.

adult_census_data['workclass'].replace(to_replace = self_employed ,value = 'self-employed',inplace = True)

adult_census_data['workclass'].replace(to_replace = govt_employees,value = 'govt-employee',inplace = True)



adult_census_data['workclass'].value_counts()
elementary_school = ['1st-4th','5th-6th']

high_school = ['7th-8th','10th','9th']

higher_secondary_school = ['HS-grad','11th','12th']





# replace elements in list.

adult_census_data['education'].replace(to_replace = elementary_school,value = 'elementary_school',inplace = True)

adult_census_data['education'].replace(to_replace = high_school,value = 'high_school',inplace = True)

adult_census_data['education'].replace(to_replace = higher_secondary_school,value = 'HS-grad',inplace = True)

married= ['Married-spouse-absent','Married-civ-spouse','Married-AF-spouse']

separated = ['Separated','Divorced']



#replace elements in list.

adult_census_data['marital.status'].replace(to_replace = married ,value = 'Married',inplace = True)

adult_census_data['marital.status'].replace(to_replace = separated,value = 'Separated',inplace = True)



adult_census_data['marital.status'].value_counts()
self_employed = ['Self-emp-not-inc','Self-emp-inc']

govt_employees = ['Local-gov','State-gov','Federal-gov']



#replace elements in list.

adult_census_data['workclass'].replace(to_replace = self_employed ,value = 'Self_employed',inplace = True)

adult_census_data['workclass'].replace(to_replace = govt_employees,value = 'Govt_employees',inplace = True)



adult_census_data['workclass'].value_counts()
for i, attribute in enumerate(categorical_attributes):

    

    # Set the width and height of the figure

    plt.figure(figsize=(16,6))

    plt.figure(i)

    sns.countplot(adult_census_data[attribute])

    plt.xticks(rotation=90)



plt.show()
for i, attribute in enumerate(categorical_attributes):

    

    if attribute == 'income':

        continue

    # Set the width and height of the figure

    plt.figure(i)

    plt.figure(figsize=(16,6))

    table_ct = pd.crosstab(adult_census_data[attribute], adult_census_data['income'])

    table_ct.plot.bar(stacked=False)

    plt.legend(title='Salary')

    plt.xlabel(attribute,fontsize = 14)

    plt.xticks(rotation=90)





plt.show()
adult_census_data['education'].value_counts()

# Set the width and height of the figure

plt.figure(figsize=(16,6))

plt.title("Count of the people in different workclass")

sns.countplot(adult_census_data['education'])

plt.ylabel("Count")
table_workclass = pd.crosstab(adult_census_data['education'], adult_census_data['income'])

fig = plt.figure(figsize = (17,6))



table_workclass.plot.bar(stacked=False)

plt.legend(title='Salary')

plt.xlabel("education",fontsize = 14)

plt.show()
# checking the corellation between all the attributes

plt.figure(figsize = (12,12))

correlation_matrix = adult_census_data.corr().round(2)

sns.heatmap(data=correlation_matrix, annot=True)
del_cols = ['relationship','education.num']

adult_census_data.drop(labels = del_cols,axis = 1,inplace = True)

continuous_attributes = [ele for ele in continuous_attributes if ele not in del_cols] 

categorical_attributes =  [ele for ele in categorical_attributes if ele not in del_cols] 
# Check for outliers using boxplots

# Replace that with MEAN



for i in continuous_attributes:

    # Getting 75 and 25 percentile of variable "i"

    Q3, Q1 = np.percentile(adult_census_data[i], [75,25])

    MEAN = adult_census_data[i].mean()

    

    # Calculating Interquartile range

    IQR = Q3 - Q1

    

    # Calculating upper extream and lower extream

    minimum = Q1 - (IQR*1.5)

    maximum = Q3 + (IQR*1.5)

    

    # Replacing all the outliers value to Mean

    adult_census_data.loc[adult_census_data[i]< minimum,i] = MEAN

    adult_census_data.loc[adult_census_data[i]> maximum,i] = MEAN
for i in continuous_attributes:

    adult_census_data[i]=(adult_census_data[i]-min(adult_census_data[i]))/(max(adult_census_data[i])-min(adult_census_data[i]))

from sklearn.pipeline import Pipeline

from sklearn.base import TransformerMixin

from sklearn.preprocessing import MinMaxScaler,StandardScaler

scaler = MinMaxScaler()

pd.DataFrame(scaler.fit_transform(adult_census_data[continuous_attributes]),columns = continuous_attributes).head(3)



class DataFrameSelector(TransformerMixin):

    def __init__(self,attribute_names):

        self.attribute_names = attribute_names

                

    def fit(self,X,y = None):

        return self

    

    def transform(self,X):

        return X[self.attribute_names]

    

    

class num_trans(TransformerMixin):

    def __init__(self):

        pass

    

    def fit(self,X,y=None):

        return self

    

    def transform(self,X):

        df = pd.DataFrame(X)

        df.columns = continuous_attributes 

        return df

    

pipeline = Pipeline([('selector',DataFrameSelector(continuous_attributes)),  

                     ('scaler',MinMaxScaler()),

                    ('transform',num_trans())])



num_df = pipeline.fit_transform(adult_census_data)

print(num_df.shape)

class dummies(TransformerMixin):

    def __init__(self,cols):

        self.cols = cols

    

    def fit(self,X,y = None):

        return self

    

    def transform(self,X):

        df = pd.get_dummies(X)

        df_new = df[df.columns.difference(cols)] 

#difference returns the original columns, with the columns passed as argument removed.

        return df_new

# columns which I don't need after creating dummy variables dataframe

cols = ['workclass_Govt_employess','education_Some-college',

        'marital.status_Never-married','occupation_Other-service',

        'race_Black','sex_Male','income_>50K']

pipeline_cat=Pipeline([('selector',DataFrameSelector(categorical_attributes)),

                      ('dummies',dummies(cols))])

cat_df = pipeline_cat.fit_transform(adult_census_data)

cat_df.shape
cat_df['id'] = pd.Series(range(cat_df.shape[0]))

num_df['id'] = pd.Series(range(num_df.shape[0]))
df = pd.merge(cat_df,num_df,how = 'inner', on = 'id')

print(f"Number of observations in final dataset: {df.shape}")
print(df.columns)

df.to_excel("/adult-cencus-raw.xlsx")
# Step 1 cleaning

# Remove data with missing target value .. the missing y value

df.dropna(axis=0, subset=['income_<=50K'], inplace=True)

y = df['income_<=50K']

df.drop(['income_<=50K'], axis=1, inplace=True)



# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(df, y, train_size=0.8, test_size=0.2, random_state=0)
# Accuracy score and other parameters

from sklearn.metrics import accuracy_score

def Print_Analysis(y_true, y_pred):

    print ("Accuracy score ", accuracy_score(y_true, y_pred))
# MultinomialNB



def MultinomialNB_Classffier(X_train, X_valid, y_train, y_valid):

    MultinomialNB_clf = MultinomialNB()

    MultinomialNB_clf.fit(X_train, y_train)



    y_pred = MultinomialNB_clf.predict(X_valid)

    Print_Analysis(y_valid, y_pred)

    confusion_matrix(y_valid, y_pred)



MultinomialNB_Classffier(X_train, X_valid, y_train, y_valid)



def GausianNB_Classffier(X_train, X_valid, y_train, y_valid):

    Gausian_clf = GaussianNB()

    Gausian_clf.fit(X_train, y_train)



    y_pred = Gausian_clf.predict(X_valid)

    Print_Analysis(y_valid, y_pred)

    confusion_matrix(y_valid, y_pred)



GausianNB_Classffier(X_train, X_valid, y_train, y_valid)
MultinomialNB_clf = MultinomialNB()

MultinomialNB_clf.fit(X_train, y_train)



y_pred = MultinomialNB_clf.predict(X_valid)

Print_Analysis(y_valid, y_pred)

print("Confusion Matrix")

print(confusion_matrix(y_valid, y_pred))



Gausian_clf = GaussianNB()

Gausian_clf.fit(X_train, y_train)



y_pred = Gausian_clf.predict(X_valid)

Print_Analysis(y_valid, y_pred)

print("Confusion Matrix")

print(confusion_matrix(y_valid, y_pred))