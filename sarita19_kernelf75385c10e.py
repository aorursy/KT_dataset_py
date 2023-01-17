import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn import metrics

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

from scipy.stats import boxcox

import seaborn as sns

import os



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
##Reading the dataset

df_borrower = pd.read_csv('C:/Users/sarita.machado/OneDrive - Accenture/Sarita_BEL/myLearning/Projects/Assessment Test/OneDrive_2019-07-04/To be communicated to the candidate/Borrower Information.csv')

df_borrower.head(3)
df_loan = pd.read_csv('C:/Users/sarita.machado/OneDrive - Accenture/Sarita_BEL/myLearning/Projects/Assessment Test/OneDrive_2019-07-04/To be communicated to the candidate/Loan Classification Information.csv')

df_loan.head(3)
df_pay = pd.read_csv('C:/Users/sarita.machado/OneDrive - Accenture/Sarita_BEL/myLearning/Projects/Assessment Test/OneDrive_2019-07-04/To be communicated to the candidate/Loan Payment Information.csv')

df_pay.head(3)
# Joining Borrower and Loan Datasets on "member_id"

df_inter=pd.merge(df_loan,df_borrower, on="member_id")
df_inter.head(3)
# Joining Intermediate DataSet and Payment Datasets on "id"

df=pd.merge(df_inter,df_pay, on="id")
df_pay.head(3)
df.head(3)
df.shape
df_null=pd.DataFrame({'Count': df.isnull().sum(), 'Percent': 100*df.isnull().sum()/len(df)})

df_null.sort_values(by=['Count','Percent'],ascending=False)

#pd.DataFrame({'Count': df.isnull().sum(), 'Percent': 100*df.isnull().sum()/len(df)})
df = df.dropna(axis=1, thresh=int(0.80*len(df)))
df.shape
df.head(3)
#df_numerical = df.select_dtypes(include ={'float64','int64'}) 

#df_numerical.describe()


# Create correlation matrix

corr_matrix = df.corr().abs()

# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



# Find index of feature columns with correlation greater than 0.95

to_drop = [column for column in upper.columns if any(upper[column] > 0.65)]

df.drop(columns=to_drop, axis=1, inplace=True )

#df.drop(df_numerical[to_drop], axis=1, inplace=True )
df.shape
df.dtypes
df.drop(['Unnamed: 0_x','zip_code','emp_title','addr_state','earliest_cr_line','issue_d','purpose','collections_12_mths_ex_med','last_credit_pull_d','last_pymnt_d','earliest_cr_line','sub_grade','title','pymnt_plan','application_type'],axis=1,inplace=True)
df.shape
df.dropna(axis=0,inplace=True)
df.shape
df['term'] = [x[:2] for x in df['term']]

df['term'] = pd.to_numeric(df['term'],errors='coerce')
df['emp_length'] = df['emp_length'].replace({'10+ years':'10'})

df['emp_length'] = df['emp_length'].replace({'< 1 year':'1'})

df['emp_length'] = df['emp_length'].replace({'1 year':'1'})

df['emp_length'] = df['emp_length'].replace({'2 years':'2'})

df['emp_length'] = df['emp_length'].replace({'3 years':'3'})

df['emp_length'] = df['emp_length'].replace({'4 years':'4'})

df['emp_length'] = df['emp_length'].replace({'5 years':'5'})

df['emp_length'] = df['emp_length'].replace({'6 years':'6'})

df['emp_length'] = df['emp_length'].replace({'7 years':'7'})

df['emp_length'] = df['emp_length'].replace({'8 years':'8'})

df['emp_length'] = df['emp_length'].replace({'9 years':'9'})

df.emp_length.value_counts()
df['emp_length'] = pd.to_numeric(df['emp_length'],errors='coerce')

df.emp_length.value_counts()
df.emp_length=df.emp_length.astype(int)
plt.figure(figsize = [15, 15])

sns.heatmap(df.corr(), annot = True, fmt = '.3f',cmap = 'vlag_r', center = 0)

plt.show()
plt.figure(figsize=(15,5))

loan_status=df['loan_status'].value_counts()

sns.barplot(loan_status.index,loan_status,palette='Pastel1')

plt.xlabel('Count')

plt.title("Loan Value Status")

plt.xticks(rotation=90)

loan_status
df.drop(df[(df['loan_status'] == 'Late (16-30 days)')].index,inplace = True)

df.drop(df[(df['loan_status'] == 'Late (31-120 days)')].index,inplace = True)

df.drop(df[(df['loan_status'] == 'Issued')].index,inplace = True)

df.drop(df[(df['loan_status'] == 'Does not meet the credit policy. Status:Fully Paid')].index,inplace = True)

df.drop(df[(df['loan_status'] == 'Does not meet the credit policy. Status:Charged Off')].index,inplace = True)

df.drop(df[(df['loan_status'] == 'In Grace Period')].index,inplace = True)

df.drop(df[(df['loan_status'] == 'Current')].index,inplace = True)
plt.figure(figsize=(15,5))

loan_status=df['loan_status'].value_counts()

sns.barplot(loan_status.index,loan_status,palette='Pastel1')

plt.xlabel('Count')

plt.title("Loan Value Status")

plt.xticks(rotation=90)

loan_status
df['loan_status'] = df['loan_status'].replace({'Charged Off':'Default'})

df['loan_status'].value_counts()

df.loan_status=df.loan_status.astype('category').cat.codes
df.shape
numerical = df.columns[df.dtypes == {'float64','int64'}]

for i in numerical:

    if df[i].min() > 0:

        transformed, lamb = boxcox(df.loc[df[i].notnull(), i])

        if np.abs(1 - lamb) > 0.02:

            df.loc[df[i].notnull(), i] = transformed
df_string = df.select_dtypes(include ='object') 

dummy_grade=pd.get_dummies(df["grade"], prefix="grade")

dummy_ownership=pd.get_dummies(df["home_ownership"], prefix="ownership")

df=df.join(dummy_grade.iloc[:,:])

df=df.join(dummy_ownership.iloc[:,:])

df.drop(columns={'grade','home_ownership'},inplace=True )
X = df.iloc[:, df.columns != "loan_status"]

y = df["loan_status"]

X_dup = X.copy()
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_curve,classification_report

from sklearn.preprocessing import RobustScaler
robust_scaler = RobustScaler()

X = robust_scaler.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.30, random_state=123, stratify=y)
print("X_train is {} ,X_test is {} ,Y_train is {} ,Y_test is {} ".format(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape))
def CMatrix(cm):

    fig, axes = plt.subplots(figsize=(8,6))

    cm = cm.astype('float')/cm.sum(axis=0)

    ax = sns.heatmap(cm, annot=True, cmap='Blues');

    ax.set_xlabel('True Label')

    ax.set_ylabel('Predicted Label')

    ax.axis('equal')
logistic_regression = LogisticRegression(C=1)

logistic_regression.fit(X_train, Y_train)

Y_pred_test = logistic_regression.predict(X_test)

CM = confusion_matrix(y_pred=Y_pred_test, y_true=Y_test)

CMatrix(CM)
from sklearn.tree import DecisionTreeClassifier

class_tree = DecisionTreeClassifier(min_samples_split=30, min_samples_leaf=10, random_state=10)

class_tree.fit(X_train, Y_train)

Y_pred_test = class_tree.predict(X_test)

#CM = confusion_matrix(y_pred=Y_pred_test, y_true=Y_test)

CM = confusion_matrix(Y_test, Y_pred_test)

CMatrix(CM)

print(classification_report(Y_test, Y_pred_test)) 

print(confusion_matrix(Y_test, Y_pred_test))  
"""

from sklearn.externals.six import StringIO  

from IPython.display import Image 

from sklearn.tree import export_graphviz

import pydotplus

from pydotplus.graphviz import graph_from_dot_data

from sklearn.tree import export_graphviz

from sklearn.datasets import load_iris



dot_data = StringIO()

export_graphviz(class_tree, out_file=dot_data,  

                filled=True, rounded=True,

                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

Image(graph.create_png())

"""
from sklearn.naive_bayes import GaussianNB

NBC = GaussianNB()

NBC.fit(X_train, Y_train)

Y_pred_test = NBC.predict(X_test)

CM = confusion_matrix(y_pred=Y_pred_test, y_true=Y_test)

CMatrix(CM)

print(classification_report(Y_test, Y_pred_test)) 
from sklearn.ensemble import RandomForestClassifier



#Create a Gaussian Classifier

clf=RandomForestClassifier(n_estimators=100)



#Train the model using the training sets y_pred=clf.predict(X_test)

clf.fit(X_train,Y_train)

Y_pred_test= clf.predict(X_test)

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

            max_depth=None, max_features='auto', max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=2,

            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,

            oob_score=False, random_state=None, verbose=0,

            warm_start=False)

#clf.feature_importances_

list(df.columns)

feature_imp = pd.Series(clf.feature_importances_,index=X_dup.columns).sort_values(ascending=False)

CM = confusion_matrix(y_pred=Y_pred_test, y_true=Y_test)

CMatrix(CM)

#print("Accuracy:",metrics.accuracy_score(Y_test, y_pred=Y_pred_test))

print(classification_report(Y_test, Y_pred_test)) 

"""sns.barplot(x=feature_imp, y=feature_imp.index)

plt.xlabel('Feature Importance Score')

plt.ylabel('Features')

plt.title("Visualizing Important Features")

plt.legend()

plt.show()"""

feature_imp*100
def createROC(models, X, y, Xte, yte):

    false_p, true_p = [], [] ##false postives and true positives



    for i in models.keys():  ##dict of models

        models[i].fit(X, y)



        fp, tp, threshold = roc_curve(yte, models[i].predict_proba(Xte)[:,1]) ##roc_curve function



        true_p.append(tp)

        false_p.append(fp)

    return true_p, false_p ##returning the true postive and false positive
models = {'NB': GaussianNB(),

          'DT': DecisionTreeClassifier(min_samples_split=30, min_samples_leaf=10, random_state=10),

          'LR': LogisticRegression(C=1),

        'CLF': RandomForestClassifier(n_estimators=100)}

unbalset = {}

for i in models.keys():

    scores = cross_val_score(models[i], X_train - np.min(X_train) + 1,Y_train, cv=3)

    unbalset[i] = scores

    

print(unbalset)
tp_unbalset, fp_unbalset = createROC(models, X_train - np.min(X_train) + 1, Y_train, X_test - np.min(X_test) + 1, Y_test)
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18,5))

#predict = model.predict(X_train)

ax = pd.DataFrame(unbalset).boxplot(widths=(0.9,0.9,0.9,0.9), grid=False, vert=False, ax=axes[0])

ax.set_ylabel('Classifier')

ax.set_xlabel('Cross-Validation Score')



for i in range(0, len(tp_unbalset)):

    axes[1].plot(fp_unbalset[i], tp_unbalset[i], lw=1)



axes[1].plot([0, 1], [0, 1], '--k', lw=1)

axes[1].legend(models.keys())

axes[1].set_ylabel('True Positive Rate')

axes[1].set_xlabel('False Positive Rate')

axes[1].set_xlim(0,1)

axes[1].set_ylim(0,1)



cm = confusion_matrix(y_pred=Y_pred_test, y_true=Y_test).T

cm = cm.astype('float')/cm.sum(axis=0)



ax = sns.heatmap(cm, annot=True, cmap='Blues', ax=axes[2]);

ax.set_xlabel('True Value')

ax.set_ylabel('Predicted Value')

ax.axis('equal')