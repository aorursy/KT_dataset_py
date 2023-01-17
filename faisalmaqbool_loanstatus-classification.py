import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.cross_validation import train_test_split

from sklearn import tree

from sklearn import metrics

from sklearn.metrics import precision_recall_fscore_support

from sklearn.cross_validation import cross_val_score, LeaveOneOut

from scipy.stats import sem

from sklearn.cross_validation import cross_val_score, LeaveOneOut

from scipy.stats import sem
data = pd.read_csv("../input/loan.csv")
data.head()
data.shape
colsRem = ["id","member_id","desc","emp_title","policy_code","pymnt_plan","title","url"]

data.drop(colsRem,axis=1,inplace=True)

data.shape
df = data 
def getYear(year):

    yearNo = year.split(" ")[0]

    return int(yearNo)
df["emp_length"].replace("10+ years","10",inplace=True)

df["emp_length"].replace("< 1 year","0",inplace=True)

df["emp_length"].replace("n/a","-1",inplace=True)

df["emp_length"] = df["emp_length"].apply(getYear)

df["emp_length"].unique()
def getTerm(term):

    month = term.split(" ")[1]

    return int(month)
df["term"] = df["term"].apply(getTerm)

df["term"].unique()
def getZip(zipcode):

    zipcd = zipcode.split("x")[0]

    return int(zipcd)
df["zip_code"] = df["zip_code"].apply(getZip)

df["zip_code"].head()
cols = ["earliest_cr_line","issue_d","last_credit_pull_d","last_pymnt_d","next_pymnt_d"]

for col in cols:

    df[col] = pd.to_datetime(df[col],format="%b-%Y")

df[cols].head()
df["loan_status"].replace("Late (31-120 days)","Late",inplace=True)

df["loan_status"].replace("Late (16-30 days)","Late",inplace=True)

df["loan_status"].replace("Does not meet the credit policy. Status:Fully Paid","Fully Paid",inplace=True)

df["loan_status"].replace("Does not meet the credit policy. Status:Charged Off","Charged Off",inplace=True)

df["loan_status"].unique()
df["loan_status_num"] = df["loan_status"]

loanStat = {'Default':0, 'Charged Off':1,'Late':2,'In Grace Period':3,'Issued':4,'Current':5,'Fully Paid':6}

df['loan_status_num'] = df['loan_status_num'].map(loanStat)
corr =df.corr()
plt.figure(figsize=(10, 10))

plt.imshow(corr, cmap='RdYlGn', interpolation='none', aspect='auto')

plt.colorbar()

plt.xticks(range(len(corr)), corr.columns, rotation='vertical')

plt.yticks(range(len(corr)), corr.columns);

plt.suptitle('Correlations Heat Map', fontsize=15, fontweight='bold')

plt.show()
corrvalues = corr.tail(1)

import numpy as np

corrvalues = np.round(corrvalues, decimals=2)
c =[]

for cols in corrvalues:

    if corrvalues[cols][0] <= -0.1 or corrvalues[cols][0] >= 0.1:

        c.append(cols)

print (c)



dfCl = df[c]
dfCl.drop("annual_inc_joint",axis=1,inplace=True)

dfCl.shape
df_norm = (dfCl - dfCl.mean()) / (dfCl.max() - dfCl.min())
dfCl['loan_status_num'].unique()
corr = df_norm.corr()

plt.figure(figsize=(10, 10))

plt.imshow(df_norm.corr(), cmap='RdYlGn', interpolation='none', aspect='auto')

plt.colorbar()

plt.xticks(range(len(corr)), corr.columns, rotation='vertical')

plt.yticks(range(len(corr)), corr.columns);

plt.suptitle('Correlations Heat Map of Normalized Data', fontsize=15, fontweight='bold')

plt.show()
df_norm.isnull().sum()
loan = dfCl

loanX = loan.drop("loan_status_num",axis=1)

loanX = np.array(loanX)

loanY = loan["loan_status_num"]

loanY = np.array(loanY)

X_train, X_test, y_train, y_test = train_test_split(loanX, loanY, test_size=0.25, random_state=33)
classifier = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3,min_samples_leaf=5)

classifier = classifier.fit(X_train,y_train)
def performance_measure(X,y,classifier, show_accuracy=True, show_precision = True, show_classification_report=True, show_confusion_matrix=True):

    y_pred=classifier.predict(X)   

    if show_accuracy:

        print("Accuracy:{0:.3f}".format(metrics.accuracy_score(y,y_pred)),"\n")

        

    if show_precision:

        print("Precision Report")

        print("Precision,Recall,F-score")

        print(precision_recall_fscore_support(y, y_pred, average='weighted'))

        

    if show_confusion_matrix:

        print("Confusion matrix")

        print(metrics.confusion_matrix(y,y_pred),"\n")

        

performance_measure(X_train,y_train,classifier, show_classification_report=True, show_confusion_matrix=True)
def LeaveOneOut(X_train,y_train,classifier):

    # Perform Leave-One-Out cross validation

    # We are preforming 1313 classifications!

    loo = LeaveOneOut(X_train[:].shape[0])

    scores=np.zeros(X_train[:].shape[0])

    for train_index,test_index in loo:

        X_train_cv, X_test_cv= X_train[train_index], X_train[test_index]

        y_train_cv, y_test_cv= y_train[train_index], y_train[test_index]

        classifier = classifier.fit(X_train_cv,y_train_cv)

        y_pred=classifier.predict(X_test_cv)

        scores[test_index]=metrics.accuracy_score(y_test_cv.astype(int), y_pred.astype(int))

    print (("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores)))
LeaveOneOut(X_train, y_train,classifier)