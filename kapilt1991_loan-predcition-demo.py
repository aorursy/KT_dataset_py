## Importing Kaggle directory for getting input dataset so that we can use the path to read the data. 

##The below code prints the path where the loan data set is hosted. 

## We will use that path to read data set using python code



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

## Lot of tasks associated with building a model or data processing comes in form of libraries/packages which are essentially

## curated set of codes. Rather than building them from scratch, We use a lot of these to do tasks related to Model dev, data processing etc.

## Below we import some of these packages
import numpy as np  # for mathematical operations

import pandas as pd  # for data operations and processing

import seaborn as sns # for visualization

from matplotlib import pyplot as plt  #for visualization

from pylab import plot, show, subplot, specgram, imshow, savefig #for visualization

from matplotlib.pyplot import rcParams

rcParams['figure.figsize'] = 10,6



#Sklearn is the one of the most important library having lot of modules for data processing, ML models development and validation

#Below we import some of these

from sklearn.preprocessing import Normalizer

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn import metrics

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import Normalizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score,f1_score, confusion_matrix , recall_score , precision_score

from sklearn import tree





%matplotlib inline
# Here we are reading the data set from the location we printed earlier. We are using pandas library (imported in the earlier cell)
df = pd.read_csv("/kaggle/input/loan-data-set/loan_data_set.csv") # To read csv using pandas. df is the variable in which we are storing the contents of the file



df.sample(5) # To see a sample of 5 records from the data set
# Here we are deriving some data statistics to get a sense of loan data
# What does the data consist of- (No. of rows, No. of columns)

df.shape

# So we have 614 data points and 13 columns in the loan data set
# What are the data types? Whether a content in a column is string or object or integer or float etc.

df.dtypes
# Get descriptive stats like count, No. of unique values, mean etc

df.describe(include='all')
# Majority of the times, values in the data set are missing and understanding those missing values really becomes important. 

# For building a model we have to really be sure how to treat them, whether to replace them or whether to completely remove the rows

# or columns where quite a lot of missing values are there
# # Missing Values



# Function to calculate missing values by column

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

        print ("Your selected dataframe has " + str(df.shape[1]) + "columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              "columns that have missing values")

        

        # Return the dataframe with missing information

        return mis_val_table_ren_columns
missing_values_table(df)
# There are very less %age of values missing from every column. So we can use some of the popluar to techniques to handle them. 

# While there is a rigorous academic material to analyse the missing values and treating them, here we are using simple imputation of 

# missing values by mean values if the column is numeric and ignore or remove from the ones where column is categorical
# Let's see the data points where loan amount term is missing

df[df['Loan_Amount_Term'].isnull()]
# The below two lines of code replace the Loan Amount term and Loan Amount with their respective column means

df.LoanAmount.fillna(df.LoanAmount.mean(),inplace=True)

df.Loan_Amount_Term.fillna(df.Loan_Amount_Term.mean(),inplace=True)
# If we check the Loan Amount term column for Null values again- we see there are no rows having null values

df[df['Loan_Amount_Term'].isnull()]
# Dropping the Rows where any of the columns are still null

df.dropna(how="any",inplace=True)
df.drop("Loan_ID",axis=1,inplace=True)
df.dtypes
# Checking top 5 rows of data

df.head(5)

# See the Columns which had categorical values have been mapped to numeric values. We can always derive what value were mapped 

# to what numeric values using inverse_transform(y)



le.get_params()
numerical_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term","Credit_History"]
df[numerical_cols].hist(figsize=(30,15), bins=20)

plt.suptitle("Histograms of numerical values")

plt.show()
df.head()
as_fig = sns.FacetGrid(df,hue='Loan_Status',aspect=5)



as_fig.map(sns.kdeplot,'ApplicantIncome',shade=True)



oldest = df['ApplicantIncome'].max()



as_fig.set(xlim=(0,oldest))



as_fig.add_legend()
as_fig = sns.FacetGrid(df,hue='Loan_Status',aspect=5)



as_fig.map(sns.kdeplot,'CoapplicantIncome',shade=True)



oldest = df['CoapplicantIncome'].max()



as_fig.set(xlim=(0,oldest))



as_fig.add_legend()
as_fig = sns.FacetGrid(df,hue='Loan_Status',aspect=5)



as_fig.map(sns.kdeplot,'LoanAmount',shade=True)



oldest = df['LoanAmount'].max()



as_fig.set(xlim=(0,oldest))



as_fig.add_legend()
sns.factorplot('Gender',kind='count',data=df,hue='Loan_Status')

sns.factorplot('Dependents',kind='count',data=df,hue='Loan_Status')

sns.factorplot('Education',kind='count',data=df,hue='Loan_Status')

sns.factorplot('Self_Employed',kind='count',data=df,hue='Loan_Status')





sns.factorplot('Credit_History',kind='count',data=df,hue='Loan_Status')

#Encoding the categorical values (which are non numeric) into corresonding numeric values



df["Gender"] = df["Gender"].map({'Male':1, 'Female':0})

df["Married"] = df["Married"].map({'Yes':1, 'No':0})

df["Dependents"] = df["Dependents"].map({'0':0, '1':0,'2':2,'3+':3})

df["Self_Employed"] = df["Self_Employed"].map({'Yes':1, 'No':0})

df["Education"] = df["Education"].map({'Graduate':1, 'Not Graduate':0})

df["Property_Area"] = df["Property_Area"].map({'Semiurban':0, 'Rural':0,'Urban':2})

df["Loan_Status"] = df["Loan_Status"].map({'Y':1, 'N':0})
fig,ax=plt.subplots(figsize=(4,5))

sns.countplot(x = "Education", data=df, order = df["Education"].value_counts().index)

plt.show()
sns.relplot(x="ApplicantIncome", y="LoanAmount", data=df, col="Gender",color="Blue",alpha=0.3)

plt.show()
g=sns.relplot(x="Loan_Amount_Term", y="LoanAmount", data=df,kind="line",hue="Education",ci=None)

g.fig.set_size_inches(15,7)

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

sns.heatmap(data=df.corr().round(2),annot=True,linewidths=0.5,cmap="Blues")

plt.show()
def plot_feature_importance(importance,names,model_type):



    #Create arrays from feature importance and feature names

    feature_importance = np.array(importance)

    feature_names = np.array(names)



    #Create a DataFrame using a Dictionary

    data={'feature_names':feature_names,'feature_importance':feature_importance}

    fi_df = pd.DataFrame(data)



    #Sort the DataFrame in order decreasing feature importance

    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)



    #Define size of bar plot

    plt.figure(figsize=(10,8))

    #Plot Searborn bar chart

    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])

    #Add chart labels

    plt.title(model_type + ' FEATURE IMPORTANCE')

    plt.xlabel('FEATURE IMPORTANCE')

    plt.ylabel('FEATURE NAMES')
rf_model = RandomForestClassifier().fit(df.drop("Loan_Status",axis=1),df["Loan_Status"])
plot_feature_importance(rf_model.feature_importances_,df.drop("Loan_Status",axis=1).columns,'RANDOM FOREST')
gbc_model = GradientBoostingClassifier().fit(df.drop("Loan_Status",axis=1),df["Loan_Status"])
plot_feature_importance(gbc_model.feature_importances_,df.drop("Loan_Status",axis=1).columns,'GRADIENT BOOSTING')
X = df.drop('Loan_Status',axis=1)

y = df['Loan_Status']
model=DecisionTreeClassifier()

Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.25,random_state=0)

model.fit(Xtrain,ytrain)

plt.figure(figsize=(20,20))

tree.plot_tree(model.fit(Xtrain,ytrain))

dt_predict = model.predict(Xtest)

print ('Accuracy:', accuracy_score(ytest, dt_predict)*100,"%")

print ('Precision:', precision_score(ytest, dt_predict,average='weighted')*100,"%")

print ('Recall:', recall_score(ytest, dt_predict,average='weighted')*100,"%")

print ('F1 score:', f1_score(ytest, dt_predict,average='weighted')*100,"%")
X = df[['Credit_History', 'ApplicantIncome', 'LoanAmount', 'CoapplicantIncome', 'Loan_Amount_Term']]

model=DecisionTreeClassifier()

Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.25,random_state=0)

model.fit(Xtrain,ytrain)

plt.figure(figsize=(20,20))

tree.plot_tree(model.fit(Xtrain,ytrain))

dt_predict = model.predict(Xtest)

print ('Accuracy:', accuracy_score(ytest, dt_predict)*100,"%")

print ('Precision:', precision_score(ytest, dt_predict,average='weighted')*100,"%")

print ('Recall:', recall_score(ytest, dt_predict,average='weighted')*100,"%")

print ('F1 score:', f1_score(ytest, dt_predict,average='weighted')*100,"%")
rf_model = RandomForestClassifier()

Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.25,random_state=0)

rf_model.fit(Xtrain,ytrain)

rf_model_pred = rf_model.predict(Xtest)

print ('Accuracy:', accuracy_score(ytest, rf_model_pred)*100,"%")

print ('Precision:', precision_score(ytest, rf_model_pred,average='weighted')*100,"%")

print ('F1 score:', f1_score(ytest, rf_model_pred,average='weighted')*100,"%")

print ('Recall:', recall_score(ytest, rf_model_pred,average='weighted')*100,"%")
abc_model = AdaBoostClassifier()

Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.25,random_state=0)

abc_model.fit(Xtrain,ytrain)

abc_model_pred = abc_model.predict(Xtest)

print ('Accuracy:', accuracy_score(ytest, abc_model_pred)*100,"%")

print ('Precision:', precision_score(ytest, abc_model_pred,average='weighted')*100,"%")

print ('Recall:', recall_score(ytest, abc_model_pred,average='weighted')*100,"%")

print ('F1 score:', f1_score(ytest, abc_model_pred,average='weighted')*100,"%")
ax = plt.subplot()

sns.heatmap(confusion_matrix(ytest,abc_model_pred), annot=True, ax=ax);

ax.set_xlabel('Predicted')

ax.set_ylabel('True')

ax.set_title('Confusion Matrix')
ytest.shape