#Importing basic libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



#Silencing warnings 



import warnings

warnings.filterwarnings('ignore')

pd.set_option('mode.chained_assignment',None) #Silencing the Setting with Copying Warning
%matplotlib inline
filepath = '../input/adult-census-income/adult.csv'

df = pd.read_csv(filepath)
df.head()
df.info()
df.describe()
# df.rename(columns={'income':'salary'},inplace=True)
df['income'].value_counts()
df['income'].value_counts(normalize=True)
cols_df = pd.DataFrame(df.dtypes)

num_cols = list(cols_df[cols_df[0]=='int64'].index)

cat_cols = list(cols_df[cols_df[0]=='object'].index)[:-1] #excluding target column of income 

print('Numeric variables includes:','\n',num_cols)

print('\n')

print('Categorical variables includes','\n',cat_cols)
# Data description mentions that unknown values are repalced with '?'. Let us check counts for it



for i in list(df.columns):

    print(i,df[df[i]=='?'][i].count())
#Replacing '?' string with NaN values

df.replace(to_replace='?',value=np.nan,inplace=True)
df.isna().sum()
df.dropna(axis=0,inplace=True)
plt.figure(figsize=(5,5))

correlation = df.corr()

matrix = np.triu(correlation)

sns.heatmap(correlation,cmap='coolwarm',square=True,linecolor='black',linewidths=1,

            mask=matrix,annot=True)

plt.show()
for i in num_cols:

    plt.figure(figsize=(6,3))

    df[df['income']=='<=50K'][i].hist(color='mediumblue')

    df[df['income']=='>50K'][i].hist(color='firebrick')

    plt.title(i)

    plt.show()
# Instead of using sns.catplot() we use the below loop to create a cross tab 

# which will also include the total column for better comparison



for i in cat_cols:

    ct = pd.crosstab(df[i],df['income'],margins=True, margins_name="Total")

    ct.drop(labels='Total',axis=0,inplace=True) #Removing subtotal row 

    ct.sort_values(by='Total',ascending=False,inplace=True) #Sorting based on total column

    #Selecting only top 6 categories for plotting

    ct.iloc[:6,:].plot(kind='bar',colormap='viridis',edgecolor='black')  

    plt.xlabel(' ')

    plt.title(str(i).capitalize())

    plt.legend(loc=1)

    plt.show()
df['income'].replace(to_replace='<=50K',value=0,inplace=True)

df['income'].replace(to_replace='>50K',value=1,inplace=True)
#Identifying categorical columns where more than 90% of observations belong only to one categroy



cat_drop = []

for i in cat_cols:

    if (df[i].value_counts(normalize=True)[0]) > 0.9:

        cat_drop.append(i)

        

print(cat_drop)
#Similarly for numerical columns



num_drop = []

for i in num_cols:

    if df[i].value_counts(normalize=True).iloc[0] > 0.9:

        num_drop.append(i)

        

print(num_drop)
X = df.drop(labels = cat_drop + num_drop + ['income'],axis=1)

y = df['income']
X.head(2)
y.value_counts()
sns.boxplot('education.num','education',data=df)

plt.show()
ed_cross = pd.crosstab(df['education.num'],df['education'])
ed_cross
X.drop('education',axis=1,inplace=True)
X.drop('fnlwgt',axis=1,inplace=True)
X['workclass'].value_counts(normalize=True)*100
#Listing all options other than private

to_replace = list(X['workclass'].unique())

to_replace.remove('Private')



#Placing all other categories under one bracket

X.replace(to_replace,'Non-Private',inplace=True)

X['workclass'].value_counts(normalize=True)*100
X['race'].value_counts(normalize=True)*100
#Listing all options other than white

to_replace = list(X['race'].unique())

to_replace.remove('White')



#Placing all other categories under one bracket

X.replace(to_replace,'Other',inplace=True)

X['race'].value_counts(normalize=True)*100
X['marital.status'].value_counts(normalize=True)*100
#Let us consolidate all options where individuals were married at least once (i.e. all options other than never-married)

to_replace = list(X['marital.status'].unique())

to_replace.remove('Never-married')



#Placing all other categories under one bracket

X.replace(to_replace,'Married',inplace=True)



#Renaming the 'Never-married' category to 'Single'

X.replace('Never-married','Single',inplace=True)



#Checking the final output

X['marital.status'].value_counts(normalize=True)*100
X.head(2)
X.shape
#Separating the categorical variables in feature matrix that need to be encoded 



cols_X = pd.DataFrame(X.dtypes)

X_cat_cols = list(cols_X[cols_X[0]=='object'].index)

X_num_cols = list(cols_X[cols_X[0]=='int64'].index)
X_num_cols
X = pd.get_dummies(data=X,prefix=X_cat_cols,drop_first=True)
X.head(2)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.25,random_state=101)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train.head(2)
X_train[X_num_cols] = sc.fit_transform(X_train[X_num_cols])

X_val[X_num_cols] = sc.transform(X_val[X_num_cols])
X_train.head()
from sklearn import __version__ 

print(__version__)

from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix, roc_curve, log_loss, brier_score_loss
from sklearn.linear_model import LogisticRegression

log = LogisticRegression(random_state=101)

log.fit(X_train,y_train)

log_y_pred = log.predict_proba(X_val)

log_roc = roc_auc_score(y_val,log_y_pred[:,-1])

print('ROC AUC score : ',log_roc)

print(log.get_params())
d = {'Baseline Logistic Regression' : [log_roc]}



results = pd.DataFrame(d,index=['ROC AUC Score'])

results = results.transpose()

results.sort_values(by='ROC AUC Score',ascending=False,inplace=True)

results
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=101)

dt.fit(X_train, y_train)

dt_y_pred = dt.predict_proba(X_val)

dt_roc = roc_auc_score(y_val,dt_y_pred[:,-1])

print('ROC AUC score : ',dt_roc)

print(dt.get_params())
results.loc['Baseline Decision Tree'] = dt_roc

results.sort_values(by='ROC AUC Score',ascending=False,inplace=True)

results
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=101)

rf.fit(X_train,y_train)

rf_y_pred = rf.predict_proba(X_val)

rf_roc = roc_auc_score(y_val,rf_y_pred[:,-1])

print(rf_roc)

print(rf.get_params())
results.loc['Baseline Random Forest'] = rf_roc

results.sort_values(by='ROC AUC Score',ascending=False,inplace=True)

results
#Linear kernel



from sklearn.svm import SVC



svc_l = SVC(kernel='linear',random_state=101,probability=True)

svc_l.fit(X_train,y_train)

svcl_y_pred = svc_l.predict_proba(X_val)

svcl_roc = roc_auc_score(y_val,svcl_y_pred[:,-1])

print('ROC AUC score : ',svcl_roc)

print(svc_l.get_params)
results.loc['Baseline SVC Linear'] = svcl_roc

results.sort_values(by='ROC AUC Score',ascending=False,inplace=True)

results
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=35)

knn.fit(X_train,y_train)

knn_y_pred = knn.predict_proba(X_val)

knn_roc = roc_auc_score(y_val,knn_y_pred[:,-1])

print('ROC AUC score : ',knn_roc)

print(knn.get_params())
results.loc['Baseline KNN'] = knn_roc

results.sort_values(by='ROC AUC Score',ascending=False,inplace=True)

results
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train,y_train)

nb_y_pred = nb.predict_proba(X_val)

nb_roc = roc_auc_score(y_val,nb_y_pred[:,-1])

print('ROC AUC score : ',nb_roc)

print(nb.get_params())
results.loc['Baseline Naive Bayes'] = nb_roc

results.sort_values(by='ROC AUC Score',ascending=False,inplace=True)

results
from xgboost import XGBClassifier

xg = XGBClassifier(random_state=101)

xg.fit(X_train,y_train)

xg_y_pred = xg.predict_proba(X_val)

xg_roc = roc_auc_score(y_val,xg_y_pred[:,-1])

print('ROC AUC score : ',xg_roc)

print(xg.get_params())
results.loc['Baseline XGBoost'] = xg_roc

results.sort_values(by='ROC AUC Score',ascending=False,inplace=True)

results
from catboost import CatBoostClassifier

cat = CatBoostClassifier(silent=True)

cat.fit(X_train,y_train)

cat_y_pred = cat.predict_proba(X_val)

cat_roc = roc_auc_score(y_val,cat_y_pred[:,-1])

print('ROC AUC score : ',cat_roc)

print()

print(cat.get_all_params())
results.loc['Baseline Cat Boost'] = cat_roc

results.sort_values(by='ROC AUC Score',ascending=False,inplace=True)

results