import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder


warnings.filterwarnings('ignore')
df = pd.read_csv('../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')
df.drop('Loan_ID',axis=1, inplace = True)
df['Credit_History'] = df['Credit_History'].astype('O')
df.shape
df.info()
df.duplicated().any()
plt.figure(figsize=(8,6))
sns.countplot(df['Loan_Status'])

df.select_dtypes('O').columns
grid = sns.FacetGrid(df, col = 'Loan_Status', size = 3.2, aspect = 1.6)
grid.map(sns.countplot, 'Gender')
#most male got loan and male got higher chance to got loan than female
#more male asked for loan too
import matplotlib.patches as mpatches

fig, axes = plt.subplots(2,1,sharex= True )

axes[0].barh([0],[len(df[(df['Gender'] == 'Female') & (df['Loan_Status'] == 'Y')])], color = '#b5ffb9', edgecolor='white')
axes[0].barh([0],[(df.Gender == 'Female').sum()- len(df[(df['Gender'] == 'Female') & (df['Loan_Status'] == 'Y')])],
         left = [len(df[(df['Gender'] == 'Female') & (df['Loan_Status'] == 'Y')])],color = '#f9bc86', edgecolor='white')


leg1 = mpatches.Patch (color = '#b5ffb9', label= 'Y')
leg2 = mpatches.Patch (color = '#f9bc86', label = 'N')
axes[0].legend(handles = [leg1,leg2], title = 'Loan Status')
axes[0].text(5,0, str(np.round(len(df[(df['Gender'] == 'Female') & (df['Loan_Status'] == 'Y')])/(df.Gender == 'Female').sum()*100,2)) +str('%'), fontsize = 12)
axes[0].text(80,0, str(np.round(len(df[(df['Gender'] == 'Female') & (df['Loan_Status'] == 'N')])/(df.Gender == 'Female').sum()*100,2)) +str('%'), fontsize = 12)
#axes[0].axes.get_yaxis().set_visible(False)
axes[0].set_ylabel('Female')
axes[1].barh([0],[len(df[(df['Gender'] == 'Male') & (df['Loan_Status'] == 'Y')])], color = '#b5ffb9', edgecolor='white')
axes[1].barh([0],[(df.Gender == 'Male').sum()- len(df[(df['Gender'] == 'Male') & (df['Loan_Status'] == 'Y')])],
         left = [len(df[(df['Gender'] == 'Male') & (df['Loan_Status'] == 'Y')])],color = '#f9bc86', edgecolor='white')
axes[1].set_ylabel('Female')
axes[1].text(200,0, str(np.round(len(df[(df['Gender'] == 'Male') & (df['Loan_Status'] == 'Y')])/(df.Gender == 'Male').sum()*100,2)) +str('%'), fontsize = 12)
axes[1].text(390,0, str(np.round(len(df[(df['Gender'] == 'Male') & (df['Loan_Status'] == 'N')])/(df.Gender == 'Male').sum()*100,2)) +str('%'), fontsize = 12)
grid = sns.FacetGrid(df, col = 'Loan_Status', size = 3.2, aspect = 1.6)
grid.map(sns.countplot, 'Dependents')

#if Loan_Status is 1, peole got higher chance to got rejected for a a loan
# when Loan_Status = +3, people got higher chance to get a loan
# when Loan_Status = 1, people got the highest chance to get a loan
grid = sns.FacetGrid(df, col = 'Loan_Status', size = 3.2, aspect = 1.6)
grid.map(sns.countplot, 'Married')

#if you are married, you got a lower chance to get a loan
grid = sns.FacetGrid(df, col = 'Loan_Status',size=3.2, aspect = 1.6)
grid.map(sns.countplot, 'Education')

# most graduate student got a loan (p)
grid = sns.FacetGrid (df, col = 'Loan_Status', size=3.2, aspect = 1.6)
grid.map(sns.countplot,'Self_Employed')
grid = sns.FacetGrid (df, col = 'Loan_Status', size = 3.2, aspect=1.6)
grid.map(sns.countplot, 'Credit_History')
grid = sns.FacetGrid (df, col = 'Loan_Status', size = 3.2, aspect = 1.6)
grid.map(sns.countplot, 'Property_Area')
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold

models = {
    "Logistic Regression" : LogisticRegression(random_state = 42),
    "K Neighbors Classifier": KNeighborsClassifier(),
    "SVC" : SVC(random_state=42),
    "DecisionTreeClassifier" : DecisionTreeClassifier(max_depth=1, random_state=42)
}
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss, accuracy_score
#(y_true, y_pred)
def check_result (y_true, y_pred):
    pre = precision_score(y_true,y_pred)
    rec = recall_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)
    log = log_loss(y_true, y_pred)
    acc = accuracy_score (y_true, y_pred)
    return pre, rec, f1, log, acc



def implement_model (X,y, models ):
    skt=StratifiedKFold(n_splits = 10, random_state =42, shuffle = True)
    for name, model in models.items():
        ls=[]
        title_name=['pre', 'rec', 'f1', 'log', 'acc']
        for train, test in skt.split(X,y):
            model.fit(X.iloc[train],y.iloc[train])
            y_pred = model.predict(X.iloc[test])
            ls.append(check_result(y.iloc[test],y_pred))
        print(name + ' : ')
        print(pd.DataFrame(ls, columns= title_name).mean(axis=0))
        print('-'*30)
    

df.isnull().sum().sort_values(ascending = False)
#drop Loan_ID
test=df.copy()

#Put two different category of data types into list
cate_col = [col for col in test.columns if test[col].dtypes == 'O']
num_col = [col for col in test.columns if test[col].dtype in ['int64','float64']]

#replace numerical missing data by mean()
for col in num_col:
    test[col] = test[col].fillna(test[col].mean())
    
#replace categorical missing data by most popular
for col in cate_col:
    test[col] = test[col].fillna(value = test[col].value_counts().index[0])

#label encode Categorical data
le = LabelEncoder()
for col in cate_col:
    test[col] = le.fit_transform(test[col])
    
y=test['Loan_Status']
X=test.drop('Loan_Status',axis = 1)
    
implement_model(X,y,models)
#drop Loan_ID
test=df.copy()

cate_col = [col for col in test.columns if test[col].dtypes == 'O']
num_col = [col for col in test.columns if test[col].dtype in ['int64','float64']]

#replace numerical missing data by mean()
for col in num_col:
    test[col] = test[col].fillna(test[col].mean())
#replace categorical missing data by most popular
for col in cate_col:
    test[col] = test[col].fillna(value = test[col].value_counts().index[0])
    
le = LabelEncoder()
for col in [col for col in cate_col if col not in ['Property_Area', 'Dependents']]:
    test[col] = le.fit_transform(test[col])

from sklearn.preprocessing import OneHotEncoder
OH_encoder = OneHotEncoder(handle_unknown = 'ignore',sparse = False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(test[['Property_Area','Dependents']]))

test.drop(['Property_Area','Dependents'],axis=1,inplace=True)

df_final = pd.concat([test, OH_cols_train],axis=1)

y=df_final['Loan_Status']
X=df_final.drop('Loan_Status',axis = 1)
    
implement_model(X,y,models)



#One hot encoder give a slightly more positive to our result so I will keep this method
#drop Loan_ID
test=df.copy()

cate_col = [col for col in test.columns if test[col].dtypes == 'O']
num_col = [col for col in test.columns if test[col].dtype in ['int64','float64']]

#replace numerical missing data by mean()
for col in num_col:
    test[col] = test[col].fillna(test[col].mean())
#replace categorical missing data by most popular
for col in cate_col:
    test[col] = test[col].fillna(value = test[col].value_counts().index[0])
    
le = LabelEncoder()
for col in [col for col in cate_col if col not in ['Property_Area', 'Dependents']]:
    test[col] = le.fit_transform(test[col])

test['%_of_App_Coapp'] = test['CoapplicantIncome'] / test['ApplicantIncome']
test['Total_loan'] = test['LoanAmount'] * test['Loan_Amount_Term']


fig, ax = plt.subplots (1,2, figsize=(15,5))
fig.tight_layout(pad=5.0)
a=test[['CoapplicantIncome','ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term','Loan_Status']].corr()
sns.heatmap(a, annot = True, ax = ax[0])
ax[0].set_title('Correlation of original columns')

b=test[['%_of_App_Coapp','Total_loan', 'Loan_Status']].corr()
sns.heatmap(b, annot = True, ax = ax[1])
ax[1].set_title('Correlation of new columns')


#New columns give better correlation with the Loan Status, therefore we gonna drop old columns

test.drop(['CoapplicantIncome', 'ApplicantIncome', 'LoanAmount','Loan_Amount_Term'],axis = 1,inplace=True)

from sklearn.preprocessing import OneHotEncoder
OH_encoder = OneHotEncoder(handle_unknown = 'ignore',sparse = False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(test[['Property_Area','Dependents']]))
test.drop(['Property_Area','Dependents'],axis=1,inplace=True)
test = pd.concat([test, OH_cols_train],axis=1)
from scipy.stats import norm

fig, ax = plt.subplots (2,2, figsize=(10,7))
fig.tight_layout(pad=5.0)

sns.distplot(test['Total_loan'], ax=ax[0,0], fit=norm)
ax[0,0].set_title('new_col_2_before log')

test['New_total_2'] = np.log(test['Total_loan'])
sns.distplot(test['New_total_2'], ax=ax[0,1],fit=norm)
ax[0,1].set_title('New_col_after_log')

sns.boxplot(test['New_total_2'],ax=ax[1,0])
ax[1,0].set_title('Dispersion of new col after log')

threshold = 0.1
q25, q75 = np.percentile(test['New_total_2'],25), np.percentile(test['New_total_2'],75)
iqr = q75 - q25
cut = iqr * threshold
lower, upper = q25 - cut, q75 + cut
sns.boxplot(test[test['New_total_2']>lower][test['New_total_2']<upper]['New_total_2'], ax = ax[1,1])
ax[1,1].set_title('Dispersion of new col after log and drop outliner')
test=test[test['New_total_2']>lower][test['New_total_2']<upper]
fig, ax =plt.subplots(1,2, figsize =(10,5))
sns.distplot(test['%_of_App_Coapp'],ax= ax[0])

test['Bool_new_column']=(test['%_of_App_Coapp']==0).astype(int)
sns.distplot(test['Bool_new_column'],ax= ax[1])


#most of data equal to 0 so I put it into bool type
fig, axes = plt.subplots(1,2, figsize = (15,5))
a=test[['Total_loan','%_of_App_Coapp','Loan_Status']].corr()
sns.heatmap(a , ax=axes[0], annot = True)
axes[0].set_label('Not preprocessing 2 columns')

b=test[['New_total_2','Bool_new_column','Loan_Status']].corr()
sns.heatmap(b , ax = axes[1], annot = True)
axes[1].set_label('After preprocessing 2 columns')



#Our feature engineers on two new columns has a higher correlation with Loan Status
test.drop(['%_of_App_Coapp','Total_loan'],axis=1,inplace=True)
X=test.drop('Loan_Status',axis=1)
y=test['Loan_Status']
implement_model(X,y,models)

df1 = pd.read_csv('../input/loan-prediction-problem-dataset/test_Y3wMUE5_7gLdaTN.csv')
df1.drop('Loan_ID',axis=1, inplace = True)
df1['Credit_History'] = df1['Credit_History'].astype('O')
cate_col = [col for col in df1.columns if df1[col].dtypes == 'O']
num_col = [col for col in df1.columns if df1[col].dtype in ['int64','float64']]

#replace numerical missing data by mean()
for col in num_col:
    df1[col] = df1[col].fillna(df1[col].mean())
#replace categorical missing data by most popular
for col in cate_col:
    df1[col] = df1[col].fillna(value = df1[col].value_counts().index[0])
    
le = LabelEncoder()
for col in [col for col in cate_col if col not in ['Property_Area', 'Dependents']]:
    df1[col] = le.fit_transform(df1[col])


from sklearn.preprocessing import OneHotEncoder
OH_encoder = OneHotEncoder(handle_unknown = 'ignore',sparse = False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(df1[['Property_Area','Dependents']]))
df1.drop(['Property_Area','Dependents'],axis=1,inplace=True)

df1 = pd.concat([df1, OH_cols_train],axis=1)

df1['%_of_App_Coapp'] = df1['CoapplicantIncome'] / df1['ApplicantIncome']
df1['Total_loan'] = df1['LoanAmount'] * df1['Loan_Amount_Term']
df1['New_total_2'] = np.log(df1['Total_loan'])
df1.drop(['CoapplicantIncome', 'ApplicantIncome', 'LoanAmount','Loan_Amount_Term'],axis = 1,inplace=True)
threshold = 0.1
q25, q75 = np.percentile(df1['New_total_2'],25), np.percentile(df1['New_total_2'],75)
iqr = q75 - q25
cut = iqr * threshold
lower, upper = q25 - cut, q75 + cut

df1=df1[df1['New_total_2']>lower][df1['New_total_2']<upper]

df1['Bool_new_column']=(df1['%_of_App_Coapp']==0).astype(int)
df1.drop(['%_of_App_Coapp','Total_loan'],axis=1,inplace=True)
df1[:5]
#Choose Logistic Regression as result in highest precision, recall, f1, accuracy, log
model =  LogisticRegression(random_state = 42)
model.fit(X,y)
model.predict(df1)