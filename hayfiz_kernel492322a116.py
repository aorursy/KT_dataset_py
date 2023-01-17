#import

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



#set font size to 20

plt.rc("font", size=20)



#set seaborn styles

sns.set(style="white")

sns.set(style="whitegrid", color_codes=True)



df = pd.read_csv('../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv', index_col=0)

df.head()
df.get_dtype_counts()
df = df.dropna()
#Convert values in Education, Gender, Married, Self_Employed, Loan_Status to integer values and replace in dataframe

df = df.replace({"Graduate": 1, "Not Graduate": 0})

df = df.replace({"Female": 1, "Male": 0})

df = df.replace({"Yes": 1, "No" : 0})

df = df.replace({"Y": 1, "N" : 0})

df = df.replace({"Rural": 0, "Semiurban": 1, "Urban": 2})

df = df.replace({"3+": 3})



#Education should now be 0 or 1 values in the dataframe

df.head()
df['Loan_Status'].value_counts()
sns.countplot(x='Loan_Status', data=df, palette='hls')
count_no_loan = len(df[df['Loan_Status']==0])

count_loan = len(df[df['Loan_Status']==1])

pct_of_loans = count_loan/(count_no_loan+count_loan)

print("Percentage of loans granted: ", pct_of_loans*100)

print("Percentage of loans not granted: ", (1-pct_of_loans)*100)
df.groupby('Loan_Status').mean()
%matplotlib inline

#Analyse whether property area is apredictor of loan approval

pd.crosstab(df.Property_Area,df.Loan_Status).plot(kind='bar')

plt.title('Approval frequency per property area')

plt.xlabel('Property Area')

plt.ylabel('Loan Status Frequency')
#Analyse whether marital status is apredictor of loan approval

pd.crosstab(df.Married,df.Loan_Status).plot(kind='bar')

plt.title('Approval frequency per marital status')

plt.xlabel('Married')

plt.ylabel('Loan Status Frequency')
pd.crosstab(df.Gender,df.Loan_Status).plot(kind='bar', stacked=True)

plt.title('Approval frequency per gender')

plt.xlabel('Gender')

plt.ylabel('Loan Status Frequency')



gender_df = df[['Gender', 'Loan_Status']]



male_approved_df = gender_df.loc[(gender_df['Gender'] == 0) & (gender_df['Loan_Status'] == 1)]

female_approved_df = gender_df.loc[(gender_df['Gender'] == 1) & (gender_df['Loan_Status'] == 1)]



male_approval_rate = len(male_approved_df) / len(df[df['Gender']==0])

female_approval_rate = len(female_approved_df) / len(df[df['Gender']==1])



print("Percentage of male loans granted: ", male_approval_rate*100)

print("Percentage of female loans granted: ", female_approval_rate*100)
pd.crosstab(df.Education,df.Loan_Status).plot(kind='bar')

plt.title('Approval frequency per education')

plt.xlabel('Education')

plt.ylabel('Loan Status Frequency')
df.groupby('Loan_Status').ApplicantIncome.hist()
df.groupby('Loan_Status').CoapplicantIncome.hist()
df.groupby('Loan_Status').LoanAmount.hist()
pd.crosstab(df.Loan_Amount_Term,df.Loan_Status).plot(kind='bar')

plt.title('Approval frequency per education')

plt.xlabel('Loan Term (Days)')

plt.ylabel('Loan Status Frequency')
pd.crosstab(df.Credit_History,df.Loan_Status).plot(kind='bar')

plt.title('Approval frequency per education')

plt.xlabel('Credit History')

plt.ylabel('Loan Status Frequency')
#Split the data into features and target variables

feature_cols = ['Property_Area','Married','Dependents','Education','Gender','ApplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']

X = df[feature_cols]

y = df.Loan_Status
# split X and y into training and testing sets

from sklearn.cross_validation import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
# import LogisticRegression from sklearn

from sklearn.linear_model import LogisticRegression



# instantiate the model (using the default parameters)

logreg = LogisticRegression()



# fit the model with data

logreg.fit(X_train,y_train)



y_pred=logreg.predict(X_test)
# import the metrics class

from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

cnf_matrix
%matplotlib inline

class_names=[0,1] # name  of classes

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)





# create heatmap

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
#Print accuracy, precision and recall of prediction

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Precision:",metrics.precision_score(y_test, y_pred))

print("Recall:",metrics.recall_score(y_test, y_pred))
y_pred_proba = logreg.predict_proba(X_test)[::,1]

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()
#Load and convert test data

test_df = pd.read_csv('../input/loan-prediction-problem-dataset/test_Y3wMUE5_7gLdaTN.csv', index_col=0)

#Convert values in Education, Gender, Married, Self_Employed, Loan_Status to integer values and replace in dataframe

test_df = df.replace({"Graduate": 1, "Not Graduate": 0})

test_df = df.replace({"Female": 1, "Male": 0})

test_df = df.replace({"Yes": 1, "No" : 0})

test_df = df.replace({"Y": 1, "N" : 0})

test_df = df.replace({"Rural": 0, "Semiurban": 1, "Urban": 2})

df = df.replace({"3+": 3})
X_test = test_df[feature_cols]

y_pred=logreg.predict(X_test)