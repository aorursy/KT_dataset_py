## Importing Kaggle directory for input dataset

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from pylab import plot, show, subplot, specgram, imshow, savefig
from sklearn import preprocessing
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,f1_score, confusion_matrix , recall_score , precision_score
from sklearn import tree
%matplotlib inline
df = pd.read_csv("/kaggle/input/loan-data-set/loan_data_set.csv")
df.sample(5)
df.shape
df.dtypes
df.describe(include='all')
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
df.LoanAmount.fillna(df.LoanAmount.mean(),inplace=True)
df.Loan_Amount_Term.fillna(df.Loan_Amount_Term.mean(),inplace=True)
df.dropna(how="any",inplace=True)

df.isna().sum()

df.drop("Loan_ID",axis=1,inplace=True)
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])
df["Married"] = le.fit_transform(df["Married"])
df["Dependents"] = le.fit_transform(df["Dependents"])
df["Self_Employed"] = le.fit_transform(df["Self_Employed"])
df["Education"] = le.fit_transform(df["Education"])
df["Property_Area"] = le.fit_transform(df["Property_Area"])
df["Loan_Status"] = le.fit_transform(df["Loan_Status"])
df.dtypes
numerical_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term","Credit_History"]

df[numerical_cols].hist(figsize=(12,10), bins=20)
plt.suptitle("Histograms of numerical values")
plt.show()

print("Skewness of numerical columns:")
df[numerical_cols].skew()
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
abc_model = AdaBoostClassifier().fit(df.drop("Loan_Status",axis=1),df["Loan_Status"])
plot_feature_importance(abc_model.feature_importances_,df.drop("Loan_Status",axis=1).columns,'ADA BOOST')
X = df.drop('Loan_Status',axis=1)
y = df['Loan_Status']
model=DecisionTreeClassifier()
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.25,random_state=0)
model.fit(Xtrain,ytrain)
plt.figure(figsize=(20,20))
tree.plot_tree(model.fit(Xtrain,ytrain))
dt_model = model.predict(Xtest)
print ('Accuracy:', accuracy_score(ytest, dt_model)*100,"%")
print ('F1 score:', f1_score(ytest, dt_model,average='weighted')*100,"%")
print ('Recall:', recall_score(ytest, dt_model,average='weighted')*100,"%")
print ('Precision:', precision_score(ytest, dt_model,average='weighted')*100,"%")
rf_model = RandomForestClassifier()
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.25,random_state=0)
rf_model.fit(Xtrain,ytrain)
rf_model_pred = rf_model.predict(Xtest)
print ('Accuracy:', accuracy_score(ytest, rf_model_pred)*100,"%")
print ('F1 score:', f1_score(ytest, rf_model_pred,average='weighted')*100,"%")
print ('Recall:', recall_score(ytest, rf_model_pred,average='weighted')*100,"%")
print ('Precision:', precision_score(ytest, rf_model_pred,average='weighted')*100,"%")
abc_model = AdaBoostClassifier()
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.25,random_state=0)
abc_model.fit(Xtrain,ytrain)
abc_model_pred = abc_model.predict(Xtest)
print ('Accuracy:', accuracy_score(ytest, abc_model_pred)*100,"%")
print ('F1 score:', f1_score(ytest, abc_model_pred,average='weighted')*100,"%")
print ('Recall:', recall_score(ytest, abc_model_pred,average='weighted')*100,"%")
print ('Precision:', precision_score(ytest, abc_model_pred,average='weighted')*100,"%")
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = SVC()
clf = GridSearchCV(svc, parameters)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=30)
clf.fit(Xtrain, ytrain)
clf.best_params_
pred = clf.best_estimator_.predict(Xtest)
confusion_matrix(ytest,pred)
print ('Accuracy:', accuracy_score(ytest, pred)*100,"%")
print ('F1 score:', f1_score(ytest, pred,average='weighted')*100,"%")
print ('Recall:', recall_score(ytest, pred,average='weighted')*100,"%")
print ('Precision:', precision_score(ytest, pred,average='weighted')*100,"%")
X=df[['Credit_History','ApplicantIncome','LoanAmount','CoapplicantIncome','Loan_Amount_Term']]





rf_model = RandomForestClassifier()
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.25,random_state=0)
rf_model.fit(Xtrain,ytrain)
rf_model_pred = rf_model.predict(Xtest)
print ('Accuracy:', accuracy_score(ytest, rf_model_pred)*100,"%")
print ('F1 score:', f1_score(ytest, rf_model_pred,average='weighted')*100,"%")
print ('Recall:', recall_score(ytest, rf_model_pred,average='weighted')*100,"%")
print ('Precision:', precision_score(ytest, rf_model_pred,average='weighted')*100,"%")
model=DecisionTreeClassifier()
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.25,random_state=0)
model.fit(Xtrain,ytrain)
plt.figure(figsize=(20,20))
tree.plot_tree(model.fit(Xtrain,ytrain))
dt_model = model.predict(Xtest)
print ('Accuracy:', accuracy_score(ytest, dt_model)*100,"%")
print ('F1 score:', f1_score(ytest, dt_model,average='weighted')*100,"%")
print ('Recall:', recall_score(ytest, dt_model,average='weighted')*100,"%")
print ('Precision:', precision_score(ytest, dt_model,average='weighted')*100,"%")