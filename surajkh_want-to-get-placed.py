import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


df1=pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

df1.head()
df1.shape
df1.isnull().sum()
df1.drop(columns=['sl_no'],axis=1,inplace=True)
X_train, X_test, y_train, y_test = train_test_split(
    df1.drop(columns=['status'],axis=1),  # predictors
    df1['status'],  # target
    test_size=0.3,  # percentage of obs in test set
    random_state=0)  # seed to ensure reproducibility

X_train.shape, X_test.shape


df_ssc=df1[['gender','ssc_p','ssc_b','status']]
import plotly.express as px

fig2=px.violin(df_ssc,y='status',x='ssc_p', color='gender',
          orientation='h',title='"SSC_PERCENTAGE_EFFCET(WITH RESPECT TO GENDER) PLACED OR NOT PLACED"').update_traces(side='positive',width=2)
fig2.show()
sns.catplot(x='ssc_b',y='ssc_p',hue='status',data=df_ssc,height=8,aspect=1.3,kind='box')
plt.tick_params(labelsize=15)

df_hsc=df1[['gender','hsc_p','hsc_b','hsc_s','status']]
fig1=px.violin(df_hsc,y='status',x='hsc_p', color='gender',
          orientation='h',title='"HSC_PERCENTAGE_EFFCET(WITH RESPECT TO GENDER) PLACED OR NOT PLACED"').update_traces(side='positive',width=2)
fig1.show()
sns.catplot(x='hsc_s',y='hsc_p',hue='status',data=df_hsc,height=8,aspect=1.3,kind='boxen')
plt.tick_params(labelsize=15)

sns.catplot(x='hsc_b',y='hsc_p',hue='status',data=df_hsc,height=8,aspect=1.3,kind='box')
plt.tick_params(labelsize=15)

df_emp=df1[['degree_p','degree_t','workex','status','gender']]
fig3=px.violin(df_emp,y='status',x='degree_p', color='gender',
          orientation='h',title='"Graduation_PERCENTAGE_EFFCET(WITH RESPECT TO GENDER) PLACED OR NOT PLACED"').update_traces(side='positive',width=2)
fig3.show()
sns.catplot(x='degree_t',y='degree_p',hue='status',data=df_emp,height=8,aspect=1.3,kind='box')
plt.tick_params(labelsize=15)

sns.catplot(x='specialisation',y='mba_p',data=df1,hue='status',height=8,aspect=1.3,kind='boxen')
plt.tick_params(labelsize=15)


sns.catplot(x='specialisation',y='etest_p',data=df1,hue='status',height=8,aspect=1.3,kind='bar')
plt.tick_params(labelsize=15)

sns.catplot(x='workex',y='mba_p',data=df1,hue='status',height=8,aspect=1.3,kind='violin')
plt.tick_params(labelsize=15)


plt.figure(figsize=(20,10))
b=sns.countplot(x='workex',data=df1,hue='status',lw=3,edgecolor=sns.color_palette("dark", 3))
b.axes.set_title("T",fontsize=30)
b.set_xlabel("Workex",fontsize=20)
b.set_ylabel("Count",fontsize=20)
b.tick_params(labelsize=20)


plt.figure(figsize=(15,7))
df1['gender'].value_counts().plot(kind='bar')
plt.tick_params(labelsize=15)
plt.figure(figsize=(20,10))
sns.countplot(x='gender',data=df1,hue='status')
plt.tick_params(labelsize=25)

sns.pairplot(df1)
plt.show()
num=df1.dtypes[df1.dtypes=='float64'].index
for i in num:
    fig = px.histogram(df1, x=i, color="status",title='Histogram of '+i+' with respect to target variable')
    fig.show()
df1_sal=df1[~df1['salary'].isnull()]
df1_sal_obj=df1_sal.dtypes[df1_sal.dtypes=='object'].index
for i in df1_sal_obj:
    fig = px.box(df1, x="status", y="salary", color=i,title='Salary comparision with respect to '+i,width=550,height=350)
    fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
    fig.show()


df1['Percentage_change_scc_to_hsc']=(df1['hsc_p']-df1['ssc_p'])/(df1['ssc_p'])*100
fig = px.box(df1, x="status", y="Percentage_change_scc_to_hsc",color='gender',title='Percentage change from SSC to HSC, how does it influence status with respect to gender')
fig.show()

X_train.shape, X_test.shape
df1.dtypes[df1.dtypes=='object']
sns.countplot(x='gender',data=df1,hue='status')
plt.figure(figsize=(15,8))
sns.countplot(x='degree_t',data=df1,hue='status')
plt.title('Is ssc_board makes any differnece for status variable')
plt.show()
for i in df1.columns:
    if df1[i].dtypes=='object':
        print('For column ',i)
        print(df1[i].value_counts())
        print('----------------------------------')
!pip install feature_engine
from feature_engine.categorical_encoders import OneHotCategoricalEncoder
from feature_engine import missing_data_imputers as mdi
X_train.shape,X_test.shape
imputer = mdi.ArbitraryNumberImputer(arbitrary_number = 0,
                                     variables=['salary'])

imputer.fit(X_train)
X_train = imputer.transform(X_train)

# let's check null values are gone
X_train[imputer.variables].isnull().mean()
X_test=imputer.transform(X_test)
ohe_enc = OneHotCategoricalEncoder(
    top_categories=None, # we can select which variables to encode
    drop_last=True) # to return k-1, false to return k


ohe_enc.fit(X_train)

X_train_one_hot = ohe_enc.transform(X_train)

X_train_one_hot.head()
X_test_one_hot = ohe_enc.transform(X_test)

X_test_one_hot.head()
ohe_tr = OneHotCategoricalEncoder(
    top_categories=None, # we can select which variables to encode
    drop_last=True) # to return k-1, false to return k


ohe_tr.fit(pd.DataFrame(y_train))
y_train_one_hot=ohe_tr.transform(pd.DataFrame(y_train))
y_train_one_hot
y_test_one_hot=ohe_tr.transform(pd.DataFrame(y_test))
y_test_one_hot
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train_one_hot,y_train_one_hot)
predictions = logmodel.predict(X_test_one_hot)
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
accuracy = accuracy_score(y_test_one_hot,predictions)
accuracy

# Confusion Matrix
conf_mat = confusion_matrix(y_test_one_hot,predictions)
conf_mat
true_positive = conf_mat[0][0]
false_positive = conf_mat[0][1]
false_negative = conf_mat[1][0]
true_negative = conf_mat[1][1]
Accuracy = (true_positive + true_negative) / (true_positive +false_positive + false_negative + true_negative)
Accuracy
Precision = true_positive/(true_positive+false_positive)
Precision
Recall = true_positive/(true_positive+false_negative)
Recall
F1_Score = 2*(Recall * Precision) / (Recall + Precision)
F1_Score
auc = roc_auc_score(y_test_one_hot, predictions)
auc
fpr, tpr, thresholds = roc_curve(y_test_one_hot, predictions)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
X=X_train_one_hot.copy()
x=X_test_one_hot.copy()
Y=y_train_one_hot.copy()
y=y_test_one_hot.copy()
X.drop('salary',inplace=True,axis=1)
x.drop('salary',inplace=True,axis=1)
sal_train=X_train_one_hot['salary']
Y=Y.merge(pd.DataFrame(sal_train),left_index=True,right_index=True)
train_status=Y['status_Not Placed']
X=X.merge(pd.DataFrame(train_status),left_index=True,right_index=True)

sal_test=X_test_one_hot['salary']
y=y.merge(pd.DataFrame(sal_test),left_index=True,right_index=True)
test_status=y['status_Not Placed']
x=x.merge(pd.DataFrame(test_status),left_index=True,right_index=True)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X,Y)
predictions = lm.predict(x)
plt.scatter(y,predictions)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y, predictions))
print('MSE:', metrics.mean_squared_error(y, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y, predictions)))
lm.score(X,Y)
def adj_r2(x,y):
    r2 = lm.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2
adj_r2(X,Y)
