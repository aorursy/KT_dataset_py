## Importing Kaggle directory for input dataset

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

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
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 10,6 # To set the figure size
# Import the data and take a first look 
df = pd.read_csv("/kaggle/input/loan-data-set/loan_data_set.csv") # To read csv using pandas. df is the variable in which we are storing the contents of the file
df.sample(5)# To see a sample of 5 records from the data set
# What does the data consist of- (No. of rows, No. of columns)
df.shape
# So we have 614 data points and 13 columns in the loan data set
# What are the data types?
df.dtypes
# Get descriptive stats
df.describe(include='all')
numerical_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term","Credit_History"]
df[numerical_cols].hist(figsize=(12,10), bins=20)
plt.suptitle("Histograms of numerical values")
plt.show()
fig,ax=plt.subplots(figsize=(4,5))
sns.countplot(x = "Education", data=df, order = df["Education"].value_counts().index)
plt.show()
fig,ax=plt.subplots(figsize=(4,5))
sns.countplot(x = "Gender", data=df, order = df["Gender"].value_counts().index)
plt.show()
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
sns.factorplot('Dependents',kind='count',data=df,hue='Loan_Status')
print(pd.crosstab(df["Married"],df["Loan_Status"]))
Married=pd.crosstab(df["Married"],df["Loan_Status"])
Married.div(Married.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.xlabel("Married")
plt.ylabel("Percentage")
plt.show()
sns.relplot(x="ApplicantIncome", y="LoanAmount", data=df, col="Gender",color="Blue",alpha=0.3)
plt.show()
g=sns.relplot(x="Loan_Amount_Term", y="LoanAmount", data=df,kind="line",hue="Education",ci=None)
g.fig.set_size_inches(15,7)
plt.show()
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
df[df['Loan_Amount_Term'].isnull()]
df.LoanAmount.fillna(df.LoanAmount.mean(),inplace=True)
df.Loan_Amount_Term.fillna(df.Loan_Amount_Term.mean(),inplace=True)
df[df['Loan_Amount_Term'].isnull()]
df[df['Loan_ID']=='LP001041']
df.dropna(how="any",inplace=True)
df.drop("Loan_ID",axis=1,inplace=True)
df.dtypes # Check data types
fig,ax=plt.subplots(figsize=(15,8))
sns.heatmap(data=df.corr().round(2),annot=True,linewidths=0.5,cmap="Blues")
plt.show()
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])
df["Married"] = le.fit_transform(df["Married"])
df["Dependents"] = le.fit_transform(df["Dependents"])
df["Self_Employed"] = le.fit_transform(df["Self_Employed"])
df["Education"] = le.fit_transform(df["Education"])
df["Property_Area"] = le.fit_transform(df["Property_Area"])
df["Loan_Status"] = le.fit_transform(df["Loan_Status"])
df[["Gender","Married","Dependents","Self_Employed"]].head()
df["TotalIncome"]=df["ApplicantIncome"]+df["CoapplicantIncome"]
df["EMI"]=df["LoanAmount"]/df["Loan_Amount_Term"]
df["Balance_Income"] = df["TotalIncome"]-df["EMI"]*1000 # To make the units equal we multiply with 1000
fig,ax=plt.subplots(figsize=(15,8))
sns.heatmap(data=df.corr().round(2),annot=True,linewidths=0.5,cmap="Blues")
plt.show()
X = df.drop(["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term",'Loan_Status'],axis=1)
y = df['Loan_Status']
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.25,random_state=0)
X.head()
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model=LogisticRegression(random_state=1)
model.fit(Xtrain,ytrain)
plt.figure(figsize=(20,20))
dt_predict = model.predict(Xtest)
print ('Accuracy:', accuracy_score(ytest, dt_predict)*100,"%")
print ('Precision:', precision_score(ytest, dt_predict,average='weighted')*100,"%")
print ('Recall:', recall_score(ytest, dt_predict,average='weighted')*100,"%")
print ('F1 score:', f1_score(ytest, dt_predict,average='weighted')*100,"%")
model=DecisionTreeClassifier()
model.fit(Xtrain,ytrain)
plt.figure(figsize=(20,20))
#tree.plot_tree(model.fit(Xtrain,ytrain))
dt_predict = model.predict(Xtest)
print ('Accuracy:', accuracy_score(ytest, dt_predict)*100,"%")
print ('Precision:', precision_score(ytest, dt_predict,average='weighted')*100,"%")
print ('Recall:', recall_score(ytest, dt_predict,average='weighted')*100,"%")
print ('F1 score:', f1_score(ytest, dt_predict,average='weighted')*100,"%")
rf_model=RandomForestClassifier()
rf_model.fit(Xtrain,ytrain)
plt.figure(figsize=(20,20))
dt_predict = rf_model.predict(Xtest)
print ('Accuracy:', accuracy_score(ytest, dt_predict)*100,"%")
print ('Precision:', precision_score(ytest, dt_predict,average='weighted')*100,"%")
print ('Recall:', recall_score(ytest, dt_predict,average='weighted')*100,"%")
print ('F1 score:', f1_score(ytest, dt_predict,average='weighted')*100,"%")
model=GradientBoostingClassifier()
model.fit(Xtrain,ytrain)
plt.figure(figsize=(20,20))
dt_predict = model.predict(Xtest)
print ('Accuracy:', accuracy_score(ytest, dt_predict)*100,"%")
print ('Precision:', precision_score(ytest, dt_predict,average='weighted')*100,"%")
print ('Recall:', recall_score(ytest, dt_predict,average='weighted')*100,"%")
print ('F1 score:', f1_score(ytest, dt_predict,average='weighted')*100,"%")
ax = plt.subplot()
rf_model_pred = rf_model.predict(Xtest)
sns.heatmap(confusion_matrix(ytest,rf_model_pred), annot=True, ax=ax);
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix')
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
plot_feature_importance(rf_model.feature_importances_,X.columns,'RANDOM FOREST')