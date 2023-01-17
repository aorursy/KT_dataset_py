import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/credit-risk-loan-eliginility/train_split.csv')

print('Shape of Dataframe:',df.shape)
df.columns
df.head(10)
print(df.info())

print('\n\nNo of columns:',len(df.columns))
df.isnull().sum()
sns.heatmap(df.isnull())
dfcopy = df.copy()
toomanynull = ['mths_since_last_delinq','mths_since_last_record',

               'mths_since_last_major_derog','pymnt_plan','desc',

               'verification_status_joint']

df.drop(toomanynull,inplace=True,axis=1)
## getting numeric columns

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

num = df.select_dtypes(include=numerics).columns
num
## getting categorical columns

cat = df.drop(num,axis=1)

cat = cat.columns
cat
df[cat].head(3)
df[num].head(3)
df.corr()
plt.figure(figsize=(20,20))

sns.heatmap(df[num].corr(),annot=True,square=True,cmap='Set2')
df[num].isnull().sum()
## revol_util
plt.figure(figsize = (14,10)) 

plt.subplot(221)

sns.boxplot(df['revol_util'])

plt.subplot(222)

sns.violinplot(df['revol_util'])

plt.subplot(223)

df['revol_util'].plot.hist()

plt.suptitle('revol_util columns',size=20)
## checking mean nd median and imputing median

print(df['revol_util'].mean())

print(df['revol_util'].median())

df['revol_util'].fillna(value=df['revol_util'].median(),inplace=True)
sns.distplot(df['revol_util'])
# tot_coll_amt  (total collected amount)
plt.figure(figsize = (14,10)) 

plt.subplot(221)

sns.boxplot(df['tot_coll_amt'])

plt.subplot(222)

sns.violinplot(df['tot_coll_amt'])

plt.subplot(223)

df['tot_coll_amt'].plot.hist()

plt.suptitle('tot_coll_amt columns',size=20)



print('Mean :',df['tot_coll_amt'].mean())

print('Median :',df['tot_coll_amt'].median())
df.tot_coll_amt.value_counts()
df.drop('tot_coll_amt',axis=1,inplace=True)
## tot_cur_bal 
plt.figure(figsize = (14,10)) 

plt.subplot(221)

sns.boxplot(df['tot_cur_bal'])

plt.subplot(222)

sns.violinplot(df['tot_cur_bal'])

plt.subplot(223)

df['tot_cur_bal'].plot.hist()

plt.suptitle('tot_cur_bal (Total Current Balance of user) columns',size=20)



print('Mean :',df['tot_cur_bal'].mean())

print('Median :',df['tot_cur_bal'].median())
#### Data is totally biased and has too many outliers so imputing Median
df['tot_cur_bal'].fillna(value=df['tot_cur_bal'].median(),inplace=True) 
## total_rev_hi_lim
plt.figure(figsize = (14,10)) 

plt.subplot(221)

sns.boxplot(df['total_rev_hi_lim'])

plt.subplot(222)

sns.violinplot(df['total_rev_hi_lim'])

plt.subplot(223)

df['total_rev_hi_lim'].plot.hist()

plt.suptitle('total_rev_hi_lim columns',size=20)



print('Mean :',df['total_rev_hi_lim'].mean())

print('Median :',df['total_rev_hi_lim'].median())
df['total_rev_hi_lim'].fillna(value=df['total_rev_hi_lim'].median(),inplace=True) 
df['collections_12_mths_ex_med'].value_counts()
df['collections_12_mths_ex_med'].plot.hist()
## dropping column as its just 0

df.drop('collections_12_mths_ex_med',axis=1,inplace=True)
df[cat]
## Dropping Useless columns

print(df['batch_enrolled'].head(5))
df['title']
# 1. batch_enrolled >> it doesn't concern which batch the user was from

# 2. desc >> too many null values 

# 5. zip_code >> not a significant column

## the columns is no significance so we drop it

temp = ['batch_enrolled','zip_code']

df.drop(temp,axis=1,inplace=True)
df['emp_title'].value_counts()
df.purpose
df.title
#we drop 'title' as its serves  same pupose as 'purpose'

df.drop('title',axis=1,inplace=True)
## Replaceing Nan Employment Type with 'Unknown' as we cannot mode it and guess it(impute)

df['emp_title'].fillna(value="Unknown",inplace=True)
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

num = df.select_dtypes(include=numerics).columns

cat = df.drop(num,axis=1)

cat = cat.columns
## emp_length

df['emp_length'].head()
## extracts the number form emp_length

df['emp_length'] = df['emp_length'].astype(str)

df['emp_length'].replace("[^0-9]","",regex=True,inplace=True)

df['emp_length'].replace("","-1",regex=True,inplace=True)

df['emp_length'] = df['emp_length'].apply(lambda x: x.strip())
df.emp_length = df.emp_length.astype(int)
## here -1 stands for unknown

df['emp_length'].fillna(value='-1',inplace=True)
### remoing moths tag from term

df.term = df.term.apply(lambda x: x.split(' ')[0])

df.term = df.term.astype(int)
df[cat].isnull().sum()
df.verification_status.value_counts()
df[cat]
## serves no relevance

df.drop('addr_state',inplace=True,axis=1)
## extracts the number form 'last_week_pay'

df['last_week_pay'] = df['last_week_pay'].astype(str)

df['last_week_pay'].replace("[^0-9]","",regex=True,inplace=True)

df['last_week_pay'].replace("","-1",regex=True,inplace=True)

df['last_week_pay'] = df['last_week_pay'].apply(lambda x: x.strip())

df.last_week_pay = df.last_week_pay.astype(int)
df['last_week_pay']
import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots
## Making seprate df for Visualization

df1 = df.copy()

df1.drop('member_id',inplace=True,axis=1)

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

num = df1.select_dtypes(include=numerics).columns

cat = df1.drop(num,axis=1)

cat = cat.columns
df1.loan_status.value_counts().values
### getting ratio of target variable to check balance between values

labels = ['Loan Granted','Loan Not Granted']

fig = px.pie(names=labels,values = df1.loan_status.value_counts().values,title='Percentage Loan Granted of Total Application')

fig.update_traces(hole=.4, hoverinfo="label+percent+name")

fig.show()
# data prepararion

from wordcloud import WordCloud 

x2011 = df1.emp_title

plt.subplots(figsize=(8,8))

wordcloud = WordCloud(

                          background_color='white',

                          width=512,

                          height=384

                         ).generate(" ".join(x2011))

plt.title('Employment Types Word Cloud',size=25)

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
px.histogram(df1,x='loan_amnt',color='loan_status',title='Loam amount W.R.t Loan Status',

             labels = labels)
## lets see successs rate for loan pass for each profession people

temp = pd.DataFrame()

temp['emp'] = df1.emp_title

temp['loan_status'] = df1.loan_status



list1 = temp.emp.value_counts().head(25).index



labels = ['Loan Granted','Loan Not Granted']

for i in list1:

    temp1 = temp[temp.emp == i].loan_status.value_counts()

    fig1 = make_subplots(rows=1, cols=2)

    fig1.add_trace(go.Pie(labels=labels,values=temp1))

    fig1.update_traces(hole=.4, hoverinfo="label+percent+name")

    fig1.update_layout(

        title_text="Percentage Loan Pass Success according to Employment -- "+ i,

        # Add annotations in the center of the donut pies.

        annotations=[dict(text=i, x=0.50,y=0.5, font_size=20, showarrow=False)])

    fig1.show()
plt.figure(figsize=(20,10))

sns.barplot(df1.purpose,df1.loan_amnt)

plt.title('loan amount passed for each purpose',size=20)
plt.figure(figsize=(20,10))

sns.barplot(df1.emp_length,df1.loan_amnt)

plt.title('Loan amnt wrt Experience',size=20)

plt.xlabel('NO of Year Experience')
plt.figure(figsize=(10,5))

sns.barplot(df1.term,df1.loan_amnt)

plt.title('Term period wrt loan amount',size=20)
plt.figure(figsize=(20,10))

sns.barplot(df1.grade,df1.loan_amnt)

plt.title('Grade wrt Loan amount',size=20)
temp = ['loan_amnt','funded_amnt','funded_amnt_inv']

sns.heatmap(df[temp].corr(),annot=True)
df.drop('funded_amnt',axis=1,inplace=True)

df.drop('funded_amnt_inv',axis=1,inplace=True)
temp = ['total_rec_int','total_rec_late_fee','recoveries','collection_recovery_fee']
for i in temp:

    sns.distplot(df[i])

    plt.show()
for i in temp:

    print(df[i].value_counts())
temp = ['total_rec_late_fee','recoveries','collection_recovery_fee']

df.drop(temp,axis=1,inplace=True)
df.acc_now_delinq.value_counts()
## has only 0 in it mostly so we drop it

df.drop('acc_now_delinq',axis=1,inplace=True)
df.delinq_2yrs.value_counts()
sns.distplot(df.delinq_2yrs)
df.drop('delinq_2yrs',axis=1,inplace=True)
df.pub_rec.value_counts()
sns.distplot(df.pub_rec)
df.drop('pub_rec',axis=1,inplace=True)
sns.countplot(df.application_type)
## its one sided data so we drop column

df.drop('application_type',axis=1,inplace=True)
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

num = df.select_dtypes(include=numerics).columns

cat = df.drop(num,axis=1)

cat = cat.columns
df[cat]
df[num]
for i in df[cat].columns:

    print(i,":\n\n",df[i].value_counts())
from sklearn.preprocessing import LabelEncoder
df[cat] = df[cat].apply(LabelEncoder().fit_transform)
df.home_ownership.value_counts()
dfcopy.home_ownership.value_counts()
df[cat].head(10)
plt.figure(figsize=(20,20))

sns.heatmap(df.corr(),annot=True,square=True,cmap='ocean')
toohightcorr = ['grade','sub_grade','total_rev_hi_lim','total_acc']
df.drop(toohightcorr,axis=1,inplace=True)
## Storing member id 

ids = df['member_id']

df.drop('member_id',axis=1,inplace=True)
plt.figure(figsize=(20,20))

sns.heatmap(df.corr(),annot=True,square=True,cmap='ocean')

plt.title('After Removing Highly Co-related Columns')
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import cross_val_score

from sklearn import metrics

from sklearn.metrics import roc_auc_score
X = df.drop('loan_status',axis=1)

y = df['loan_status']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=101)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100,max_depth=8, random_state=101,class_weight='balanced')

rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
print('AUC-ROC Score :',roc_auc_score(y_test, y_pred))

print('Report:\n',classification_report(y_test, y_pred))

print('confusion Matrix:\n',confusion_matrix(y_pred,y_test))

print('cross validation:',cross_val_score(rfc, X, y, cv=3).mean())
importances=rfc.feature_importances_

feature_importances=pd.Series(importances, index=X_train.columns).sort_values(ascending=False)

plt.figure(figsize=(10,7))

sns.barplot(x=feature_importances[0:10], y=feature_importances.index[0:10])

plt.title('Feature Importance',size=20)

plt.ylabel("Features")

plt.show()
from sklearn.feature_selection import RFE
rfe = RFE(rfc, 8) 

rfe.fit(X_train,y_train)
rfecols = X_train.columns[rfe.support_]
rfecols
rfc = RandomForestClassifier(n_estimators=200,random_state=101,class_weight='balanced')

rfc.fit(X_train[rfecols],y_train)

y_pred = rfc.predict(X_test[rfecols])

print('AUC-ROC Score :',roc_auc_score(y_test, y_pred))

print('Report:\n',classification_report(y_test, y_pred))

print('confusion Matrix:\n',confusion_matrix(y_test,y_pred))

#print('cross validation:',cross_val_score(rfc, X, y, cv=3).mean())
X_train['laon_status'] = y_train

X_train.laon_status.value_counts()

temp = X_train[X_train.laon_status == 0].sample(12000)

X_train = X_train[X_train.laon_status==1]

X_train = X_train.append(temp)

X_train.laon_status.value_counts()

X_train = X_train.sample(frac=1)
y_train = X_train.laon_status

X_train.drop('laon_status',axis=1,inplace=True)
X_train['laon_status'] = y_train

temp = X_train[X_train.laon_status==1]

X_train = X_train.append(temp)

X_train = X_train.append(temp)
X_train.laon_status.value_counts()
X_train = X_train.sample(frac=1)
y_train = X_train.laon_status

X_train.drop('laon_status',axis=1,inplace=True)
rfc = RandomForestClassifier(n_estimators=200,random_state=101,class_weight='balanced')

rfc.fit(X_train[rfecols],y_train)

y_pred = rfc.predict(X_test[rfecols])

print('AUC-ROC Score :',roc_auc_score(y_test, y_pred))

print('Report:\n',classification_report(y_test, y_pred))

print('confusion Matrix:\n',confusion_matrix(y_test,y_pred))

#print('cross validation:',cross_val_score(rfc, X, y, cv=3).mean())
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimator=100,max_depth=12,class_weight='balanced',refit='AUC')
xgb.fit(X_train[rfecols],y_train)
y_pred = xgb.predict(X_test[rfecols])
print('AUC-ROC Score :',roc_auc_score(y_test, y_pred))

print('Report:\n',classification_report(y_test, y_pred))

print('confusion Matrix:\n',confusion_matrix(y_pred,y_test))

#print('cross validation:',cross_val_score(xgb, X, y, cv=3).mean())
import lightgbm as lgb
#y_train = y_train.values
model = lgb.LGBMClassifier(n_estimators=600,random_state=101,max_depth=8,class_weight='balanced')

model.fit(X_train[rfecols], y_train)
y_pred = model.predict(X_test[rfecols])
print('AUC-ROC Score :',roc_auc_score(y_test, y_pred))

print('Report:\n',classification_report(y_test, y_pred))

print('confusion Matrix:\n',confusion_matrix(y_pred,y_test))

print('cross validation:',cross_val_score(model, X, y, cv=5).mean())
fig, ax = plt.subplots(figsize=(12,8))

lgb.plot_importance(model, max_num_features=10, height=0.8, ax=ax)

ax.grid(False)

plt.title("LightGBM - Feature Importance", fontsize=15)

plt.show()
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)

roc_auc = metrics.auc(fpr, tpr)



# method I: plt

import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt='d')

plt.title('Confusion Matrix',size=20)
tempval = pd.Series(y_test).value_counts()

tempvalpred = pd.Series(y_pred).value_counts()
labels = ['Loan Granted','Loan Not Granted']



fig1 = make_subplots(rows=1, cols=2)

fig1.add_trace(go.Pie(labels=labels, values=tempval))

fig2 = make_subplots(rows=1, cols=2)

fig2.add_trace(go.Pie(labels=labels,values=tempvalpred))



# Use `hole` to create a donut-like pie chart

fig1.update_traces(hole=.4, hoverinfo="label+percent+name")

fig2.update_traces(hole=.4, hoverinfo="label+percent+name")



fig1.update_layout(

    title_text="Predicted Vs Actual Loan Granted Ratio Comparision",

    # Add annotations in the center of the donut pies.

    annotations=[dict(text='Actual Loan Status Ratio', x=0.25, y=0.5, font_size=20, showarrow=False)])

fig2.update_layout(

    # Add annotations in the center of the donut pies.

    annotations=[dict(text='Predicted Loan Status Ratio', x=0.25,y=0.5, font_size=20, showarrow=False)])





fig1.show()

fig2.show()
