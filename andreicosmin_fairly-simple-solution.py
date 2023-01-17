import csv

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv('../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')

test = pd.read_csv('../input/loan-prediction-problem-dataset/test_Y3wMUE5_7gLdaTN.csv')
train.describe()
train.info()
train.head()
obj_data = []

for col in train:

    if train[col].dtype ==object:

        obj_data.append(col)



obj_data.pop(0)#remove the loan_id 

obj_data.append('Credit_History')
f, axes =plt.subplots(3,3, figsize=(19,17))



x=0 #the axes coordinates

y=0

for col in obj_data:

    ax_ = sns.countplot(train[col], hue=train.Loan_Status, ax=axes[x,y])

    ax_.set(ylabel='')

    if y<2:

        y+=1

    else:

        y=0

        x+=1

        

axes[2,2].remove()# there's only 8 plots on this 3x3 grid
num_data = []

for col in train:

    if col != 'Loan_ID' and col not in obj_data:

        num_data.append(col)
num_data.remove('Loan_Amount_Term')# this is giving a bandwith error so it'll be plotted separately
fig= sns.FacetGrid(train, hue='Loan_Status',aspect=4)

# we need to provide a bandwith because the kde function is unable to find one. It is obvious that there is no correlation here

fig.map(sns.kdeplot,'Loan_Amount_Term',shade=True, bw=1.2)

oldest=train['Loan_Amount_Term'].max()

#fig.set(xlim=(0,oldest))

fig.add_legend()
for col in num_data:

    fig= sns.FacetGrid(train, hue='Loan_Status',aspect=4) 

    fig.map(sns.kdeplot,col,shade=True)

    oldest=train[col].max()

    fig.set(xlim=(0,oldest), xlabel = col)

    fig.add_legend()
tt =[train, test] #creating a list so we can clean the data for both test and train at the same time
train['Loan_Status'].loc[train['Loan_Status'] =='Y'] = 1

train['Loan_Status'].loc[train['Loan_Status'] =='N'] = 0

train['Loan_Status'] = train['Loan_Status'].astype(int)#change the dtype to integer
for col in num_data:

    sns.lmplot(col,'Loan_Status', train, height=6,aspect=2)

corr = train[num_data+['Loan_Status']].corr()
sns.heatmap(corr, annot = True)
to_remove = num_data

to_remove.append('Loan_ID')

to_remove.append('Loan_Amount_Term')
for df in tt:

    df.drop(to_remove, axis =1,  inplace = True)
f, axes =plt.subplots(3,3, figsize=(17,14))



x=0

y=0

for col in obj_data:

    if col!='Loan_Status':

        ax_ = sns.pointplot(col, 'Loan_Status', data=train, kind = 'point', ax=axes[x,y])

        ax_.set(ylabel='')

    if y<2:

        y+=1

    else:

        y=0

        x+=1

axes[2,2].remove()

axes[2,0].remove()
for df in tt:

    df.drop(['Gender','Self_Employed'], axis =1, inplace = True)
train.head()
from scipy.stats import spearmanr



corr, _ = spearmanr(train.Property_Area, train.Loan_Status)

print(f'Spearmans correlation between Proeprty Area and loan status: {corr:.3f} ')
#Defining a function that finds the most common value, in order to replace nan with it

def find_most_common(col):

    l = train[col].tolist() #get the column values to a list

    common_len = l.count(l[0])#the count of the first element of the list

    common_value = l[0]

    l = [element for element in l if element!=common_value]#delete all occurences of that element

    while len(set(l))>1: #when the sets length is 1, it means only the nan values left in it

        current = l[0] 

        if l.count(current) > common_len:# check for each value in the column if it's the most common one

            common_len = l.count(current)

            common_value = current

        l = [element for element in l if element!=current]# delete it and move on the next



    return common_value      
for data in tt:

    for col in test.columns:

        if data[col].dtype==object:

            replacement =find_most_common(col) 

            data[col].fillna(replacement, inplace = True)

        else:

            data[col].fillna(train[col].median(), inplace = True)
for data in tt:



    data['Married'].loc[data['Married'] =='No'] = 0

    data['Married'].loc[data['Married'] =='Yes'] = 1



    data['Dependents'].loc[data['Dependents'] =='3+'] = 3



    data['Education'].loc[data['Education'] =='Not Graduate'] = 0

    data['Education'].loc[data['Education'] =='Graduate'] = 1



    data['Property_Area'].loc[data['Property_Area'] =='Urban'] = 0

    data['Property_Area'].loc[data['Property_Area'] =='Rural'] = 1

    data['Property_Area'].loc[data['Property_Area'] =='Semiurban'] = 2

    

    for column in data:

        data[column] = data[column].astype(int)
train.head()
obj_data_correlation = [col for col in train.columns if col!= 'Property_Area']
corr = train[obj_data_correlation].corr()
sns.heatmap(corr, annot = True)
train
for data in tt:

    data.drop('Dependents', axis = 1, inplace = True)
import sklearn

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

#from sklearn import metrics

#import statsmodels.api as sm



classifiers = [

    KNeighborsClassifier(4),

    SVC(probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    LogisticRegression()]

#A function to test our models accuracy and precision 



from sklearn.metrics import precision_score , recall_score, f1_score, log_loss, accuracy_score, matthews_corrcoef



def test_model(y_test, y_pred):    

    pre = precision_score(y_test, y_pred)

    rec = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    loss = log_loss(y_test, y_pred)

    acc = accuracy_score(y_test, y_pred)

    matc = matthews_corrcoef(y_test, y_pred)

    

    

    print('  pre: %.3f\n  rec: %.3f\n  f1: %.3f\n  loss: %.3f\n  acc: %.3f\n  matc: %.3f' % (pre, rec, f1, loss, acc, matc))
def model_and_test(X, y):

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)



    for train_, test_ in sss.split(X, y):

        X_train, X_test = X.iloc[train_], X.iloc[test_]

        y_train, y_test = y.iloc[train_], y.iloc[test_]

    for model in classifiers:

        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)# splitting the data into 70%/30%

        this_model = model.__class__.__name__ #get the name of the classifier

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        

        print(f'{this_model} results:')

        test_model(y_test, y_pred)

        print('\n')
X =train.drop('Loan_Status', axis = 1)

y=train.Loan_Status

model_and_test(X, y)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)



for train_, test_ in sss.split(X, y):

    X_train, X_test = X.iloc[train_], X.iloc[test_]

    y_train, y_test = y.iloc[train_], y.iloc[test_]



y_pred = LogisticRegression().fit(X_train, y_train).predict(X_test)
y_test_ = np.ravel(y_test)
cred =np.ravel(X_test['Credit_History'])
print(cred == y_pred)
train0 = pd.read_csv('../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')
train0.head()
c = 0

for i in range(len(train0)):

    if train0.Credit_History[i] == 1.0 and train0.Loan_Status[i] =='N':

        c+=1

    elif train0.Credit_History[i] ==0 and train0.Loan_Status[i] =='Y':

        c+=1



p = c/len(train0) *100

print(f'{c} False predictions({p:.2f}%)')
no_cred = 0

got_loan=0

for i in range(len(train0)):

    if train0.Credit_History[i] ==0: 

        if train0.Loan_Status[i] =='Y':

            got_loan+=1

        no_cred += 1



false_negatives = got_loan/no_cred

#print('  pre: %.3f\n  rec: %.3f\n  f1: %.3f\n  loss: %.3f\n  acc: %.3f' % (pre, rec, f1, loss, acc))

print(f'{no_cred} applicants with no credit history: ; only {got_loan} of them got a loan ({false_negatives:.2f}%)')



cred = 0

no_loan=0

for i in range(len(train0)):

    if train0.Credit_History[i] ==1: 

        if train0.Loan_Status[i] =='N':

            no_loan+=1

        cred += 1

false_positives = no_loan/ cred        

print(f'{cred} applicants with credit history: ; {no_loan} of them did not get a loan({false_positives:.2f}%)')
false_neg = train0[(train0['Credit_History'] == 0) & (train0['Loan_Status'] == 'Y')]

false_neg
from scipy.stats import spearmanr



def spearman(col):

    corr, _ = spearmanr(train0[col], train0.Loan_Status)

    print(f'Spearmans correlation between {col} and loan status: {corr:.3f} ')



#num_data.remove('Loan_ID')

for col in num_data:

    spearman(col)

    

    
loan_score = []

for i in range(len(train0)):

    scr = (train0.ApplicantIncome[i] + train0.CoapplicantIncome[i])/train0.LoanAmount[i]

    loan_score.append(scr)
corr, _ = spearmanr(loan_score, train0.Loan_Status)

print(f'Spearmans correlation between {col} and loan status: {corr:.3f} ')
loan_score0 = []

for i in range(len(train0)):

    scr = (train0.ApplicantIncome[i] + train0.CoapplicantIncome[i]) - train0.LoanAmount[i]*train0.Loan_Amount_Term[i]

    loan_score0.append(scr)
corr, _ = spearmanr(loan_score0, train0.Loan_Status)

print(f'Spearmans correlation between thisnew score and loan status: {corr:.3f} ')
train0['loan_score'] = loan_score0
train0.info()
data = train0.copy()

data.drop(['Gender','Self_Employed', 'Loan_ID' ,'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',

          'Loan_Amount_Term','Dependents'], axis =1, inplace = True)

data['Loan_Status'].loc[data['Loan_Status'] =='Y'] = 1

data['Loan_Status'].loc[data['Loan_Status'] =='N'] = 0

data['Loan_Status'] = data['Loan_Status'].astype(int)#change the dtype to integer



for col in data.columns:

    if data[col].dtype==object:

        replacement =find_most_common(col) 

        data[col].fillna(replacement, inplace = True)

    elif col=='Credit_History':

        data[col].fillna(data[col].mean(), inplace = True)

    else:

        data[col].fillna(data[col].median(), inplace = True)



data['Married'].loc[data['Married'] =='No'] = 0

data['Married'].loc[data['Married'] =='Yes'] = 1



data['Education'].loc[data['Education'] =='Not Graduate'] = 0

data['Education'].loc[data['Education'] =='Graduate'] = 1





data['Property_Area'].loc[data['Property_Area'] =='Urban'] = 0

data['Property_Area'].loc[data['Property_Area'] =='Rural'] = 1

data['Property_Area'].loc[data['Property_Area'] =='Semiurban'] = 2



for column in data:

    if column!='Credit_History':

        data[column] = data[column].astype(int)

    else:

        data[col] = 100* data[col]
data.head()
acc_dict = {}

log_cols = ["Classifier", "Accuracy"]

log  = pd.DataFrame(columns=log_cols)





for clf in classifiers:

    name = clf.__class__.__name__

    clf.fit(X_train, y_train)

    train_predictions = clf.predict(X_test)

    acc = matthews_corrcoef(y_test, y_pred)

    if name in acc_dict:

        acc_dict[name] += acc

    else:

        acc_dict[name] = acc

        

for clf in acc_dict:

    acc_dict[clf] = acc_dict[clf] / 10.0

    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)

    log = log.append(log_entry)

    

plt.xlabel('Accuracy')

plt.title('Classifier Accuracy')



sns.set_color_codes("muted")

sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")