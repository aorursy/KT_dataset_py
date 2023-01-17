import numpy as np 

import pandas as pd

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(font_scale=1)
df_train=pd.read_csv('../input/club-data-set/club_churn_train.csv',encoding = "ISO-8859-1")
df_train.head()
df_test=pd.read_csv('../input/club-data-set/club_churn_test.csv',encoding = "ISO-8859-1")
def des_stat_analyze(df_input):

    # check number of rows, cols

    no_rows = df_input.shape[0]

    no_cols = df_input.shape[1]

    print("No. observations:", no_rows )

    print("No. features:", no_cols )

  

    # checking type of features

    name = []

    cols_type = []

    for n,t in df_input.dtypes.iteritems():

        name.append(n)

        cols_type.append(t)



    # checking distinction (unique values) of features

    ls_unique = []

    for cname in df_input.columns:

        try:

            nunique = df_input[cname].nunique()

            pct_unique = nunique*100.0/ no_rows

            ls_unique.append("{} ({:0.2f}%)".format(nunique, pct_unique))

        except:

            ls_unique.append("{} ({:0.2f}%)".format(0,0))

            continue



    # checking missing values of features

    ls_miss = []

    for cname in df_input.columns:

        try:

            nmiss = df_input[cname].isnull().sum()

            pct_miss = nmiss*100.0/ no_rows

            ls_miss.append("{} ({:0.2f}%)".format(nmiss, pct_miss))

        except:

            ls_miss.append("{} ({:0.2f}%)".format(0,0))

            continue 

      

    # checking zeros

    ls_zeros = []

    for cname in df_input.columns:

        try:

            nzeros = (df_input[cname] == 0).sum()

            pct_zeros = nzeros * 100.0/ no_rows

            ls_zeros.append("{} ({:0.2f}%)".fornat(nzeros, pct_zeros))

        except:

            ls_zeros.append("{} ({:0.2f}%)".format(0,0))

            continue

      

    # checking negative values

    ls_neg = []

    for cname in df_input.columns:

        try:

            nneg = (df_input[cname].astype("float")<0).sum()

            pct_neg =nneg * 100.0 / no_rows

            ls_neg.append("{} ({:0.2f}%)".format(nneg, pct_neg))

        except:

            ls_neg.append("{} ({:0.2f}%)".format(0,0))

            continue

      

    # extracting the output

    data = {

      "name": name,

      "col_type": cols_type,

      "n_unique": ls_unique,

      "n_miss": ls_miss,

      "n_zeros":ls_zeros,

      "n_neg":ls_neg      

    }

  

    # statistical info

    df_stats = df_input.describe().transpose()

    ex_stats = pd.concat([df_input.median(),df_input.kurtosis(),df_input.skew()], axis = 1)

    ex_stats = ex_stats.rename(columns={0:"median",1:"kurtosis",2:"skew"})

    df_stats = df_stats.merge(ex_stats, left_index=True, right_index=True)

    #ls_stats = []

    for stat in df_stats.columns:

        data[stat] = []

        for cname in df_input.columns:

            try:

                data[stat].append(df_stats.loc[cname, stat])

            except:

                data[stat].append("NaN")

        

    # take samples

    df_sample = df_input.sample(frac = .5).head().transpose()

    df_sample.columns = ["sample_{}".format(i) for i in range(5)]

  

    # repair the output

    col_ordered = ["name","col_type","count","n_unique","n_miss","n_zeros","n_neg",

                "25%","50%","75%","max","min","mean","median","std","kurtosis","skew"]

    df_data = pd.DataFrame(data, columns = col_ordered).set_index("name")

    df_data = pd.concat([df_data, df_sample], axis = 1)



    return df_data.reset_index().sort_values(by=['col_type'])  
des_stat_analyze(df_train)

des_stat_analyze(df_test)
df_train['MEMBERSHIP_STATUS_new'] = df_train.MEMBERSHIP_STATUS.astype('category').cat.codes
df_train['MEMBER_AGE_AT_ISSUE_bin']=pd.cut(x=df_train['MEMBER_AGE_AT_ISSUE'],bins=[0,28,39,50,54,65,92])

df_test['MEMBER_AGE_AT_ISSUE_bin']=pd.cut(x=df_test['MEMBER_AGE_AT_ISSUE'],bins=[0,28,39,50,54,65,92])
df_train['MEMBERSHIP_TERM_YEARS_bin']=pd.cut(x=df_train['MEMBERSHIP_TERM_YEARS'],bins=[0,12,20,50,100])

df_test['MEMBERSHIP_TERM_YEARS_bin']=pd.cut(x=df_test['MEMBERSHIP_TERM_YEARS'],bins=[0,12,20,50,100])
labels = ['Poor','Medium','Rich','VeryRich']

df_train['MEMBER_ANNUAL_INCOME_bin'] = pd.cut(df_train['MEMBER_ANNUAL_INCOME'], bins=[9995.999,290000,1000000,18000000,999999996], labels=labels, right=False)

labels = ['Poor','Medium','Rich','VeryRich']

df_test['MEMBER_ANNUAL_INCOME_bin'] = pd.cut(df_test['MEMBER_ANNUAL_INCOME'], bins=[9995.999,290000,1000000,18000000,999999996], labels=labels, right=False)
labels = ['VeryLow','Medium','High','VeryHigh']

df_train['ANNUAL_FEES_bin'] = pd.cut(df_train['ANNUAL_FEES'], bins=[99999.989,125000,200000,500000,10100000], labels=labels, right=False)

df_test['ANNUAL_FEES_bin'] = pd.cut(df_test['ANNUAL_FEES'], bins=[99999.989,125000,200000,500000,10100000], labels=labels, right=False)
print("Training set - MEMBERSHIP_TERM_YEARS")

train_MEMBERSHIP_TERM_YEARS = df_train[['MEMBERSHIP_TERM_YEARS_bin','MEMBERSHIP_NUMBER']].groupby(['MEMBERSHIP_TERM_YEARS_bin']).count()

train_MEMBERSHIP_TERM_YEARS = train_MEMBERSHIP_TERM_YEARS / train_MEMBERSHIP_TERM_YEARS.sum()

train_MEMBERSHIP_TERM_YEARS.columns = ['Proportions']

print(train_MEMBERSHIP_TERM_YEARS.round(2))



print("Testing set - MEMBERSHIP_TERM_YEARS")

test_MEMBERSHIP_TERM_YEARS = df_test[['MEMBERSHIP_TERM_YEARS_bin','MEMBERSHIP_NUMBER']].groupby(['MEMBERSHIP_TERM_YEARS_bin']).count()

test_MEMBERSHIP_TERM_YEARS = test_MEMBERSHIP_TERM_YEARS / test_MEMBERSHIP_TERM_YEARS.sum()

test_MEMBERSHIP_TERM_YEARS.columns = ['Proportions']

print(test_MEMBERSHIP_TERM_YEARS.round(2))
print("Training set - MEMBER_AGE_AT_ISSUE")

train_MEMBER_AGE_AT_ISSUE = df_train[['MEMBER_AGE_AT_ISSUE_bin','MEMBERSHIP_NUMBER']].groupby(['MEMBER_AGE_AT_ISSUE_bin']).count()

train_MEMBER_AGE_AT_ISSUE = train_MEMBER_AGE_AT_ISSUE / train_MEMBER_AGE_AT_ISSUE.sum()

train_MEMBER_AGE_AT_ISSUE.columns = ['Proportions']

print(train_MEMBER_AGE_AT_ISSUE.round(2))



print("Testing set - MEMBER_AGE_AT_ISSUE")

test_MEMBER_AGE_AT_ISSUE = df_test[['MEMBER_AGE_AT_ISSUE_bin','MEMBERSHIP_NUMBER']].groupby(['MEMBER_AGE_AT_ISSUE_bin']).count()

test_MEMBER_AGE_AT_ISSUE = test_MEMBER_AGE_AT_ISSUE / test_MEMBER_AGE_AT_ISSUE.sum()

test_MEMBER_AGE_AT_ISSUE.columns = ['Proportions']

print(test_MEMBER_AGE_AT_ISSUE.round(2))
print("Training set - ADDITIONAL_MEMBERS")

train_ADDITIONAL_MEMBERS = df_train[['ADDITIONAL_MEMBERS','MEMBERSHIP_NUMBER']].groupby(['ADDITIONAL_MEMBERS']).count()

train_ADDITIONAL_MEMBERS = train_ADDITIONAL_MEMBERS / train_ADDITIONAL_MEMBERS.sum()

train_ADDITIONAL_MEMBERS.columns = ['Proportions']

print(train_ADDITIONAL_MEMBERS.round(2))



print("Testing set - ADDITIONAL_MEMBERS")

test_ADDITIONAL_MEMBERS = df_test[['ADDITIONAL_MEMBERS','MEMBERSHIP_NUMBER']].groupby(['ADDITIONAL_MEMBERS']).count()

test_ADDITIONAL_MEMBERS = test_ADDITIONAL_MEMBERS / test_ADDITIONAL_MEMBERS.sum()

test_ADDITIONAL_MEMBERS.columns = ['Proportions']

print(test_ADDITIONAL_MEMBERS.round(2))
print("Training set - ANNUAL_FEES")

train_ANNUAL_FEES_bin = df_train[['ANNUAL_FEES_bin','MEMBERSHIP_NUMBER']].groupby(['ANNUAL_FEES_bin']).count()

train_ANNUAL_FEES_bin = train_ANNUAL_FEES_bin / train_ANNUAL_FEES_bin.sum()

train_ANNUAL_FEES_bin.columns = ['Proportions']

print(train_ANNUAL_FEES_bin.round(2))



print("Testing set - ANNUAL_FEES")

test_ANNUAL_FEES_bin = df_test[['ANNUAL_FEES_bin','MEMBERSHIP_NUMBER']].groupby(['ANNUAL_FEES_bin']).count()

test_ANNUAL_FEES_bin = test_ANNUAL_FEES_bin / test_ANNUAL_FEES_bin.sum()

test_ANNUAL_FEES_bin.columns = ['Proportions']

print(test_ANNUAL_FEES_bin.round(2))
print("Training set - MEMBER_ANNUAL_INCOME")

train_MEMBER_ANNUAL_INCOME_bin = df_train[['MEMBER_ANNUAL_INCOME_bin','MEMBERSHIP_NUMBER']].groupby(['MEMBER_ANNUAL_INCOME_bin']).count()

train_MEMBER_ANNUAL_INCOME_bin = train_MEMBER_ANNUAL_INCOME_bin / train_MEMBER_ANNUAL_INCOME_bin.sum()

train_MEMBER_ANNUAL_INCOME_bin.columns = ['Proportions']

print(train_MEMBER_ANNUAL_INCOME_bin.round(4))



print("Testing set - MEMBER_ANNUAL_INCOME")

test_MEMBER_ANNUAL_INCOME_bin = df_test[['MEMBER_ANNUAL_INCOME_bin','MEMBERSHIP_NUMBER']].groupby(['MEMBER_ANNUAL_INCOME_bin']).count()

test_MEMBER_ANNUAL_INCOME_bin = test_MEMBER_ANNUAL_INCOME_bin / test_MEMBER_ANNUAL_INCOME_bin.sum()

test_MEMBER_ANNUAL_INCOME_bin.columns = ['Proportions']

print(test_MEMBER_ANNUAL_INCOME_bin.round(4))
print("Training set - MEMBER_OCCUPATION_CD")

train_MEMBER_OCCUPATION_CD = df_train[['MEMBER_OCCUPATION_CD','MEMBERSHIP_NUMBER']].groupby(['MEMBER_OCCUPATION_CD']).count()

train_MEMBER_OCCUPATION_CD = train_MEMBER_OCCUPATION_CD / train_MEMBER_OCCUPATION_CD.sum()

train_MEMBER_OCCUPATION_CD.columns = ['Proportions']

print(train_MEMBER_OCCUPATION_CD.round(4))



print("Testing set - MEMBER_OCCUPATION_CD")

test_MEMBER_OCCUPATION_CD = df_test[['MEMBER_OCCUPATION_CD','MEMBERSHIP_NUMBER']].groupby(['MEMBER_OCCUPATION_CD']).count()

test_MEMBER_OCCUPATION_CD = test_MEMBER_OCCUPATION_CD / test_MEMBER_OCCUPATION_CD.sum()

test_MEMBER_OCCUPATION_CD.columns = ['Proportions']

print(test_MEMBER_OCCUPATION_CD.round(4))
print("Training set - MEMBER_MARITAL_STATUS")

train_MEMBER_MARITAL_STATUS = df_train[['MEMBER_MARITAL_STATUS','MEMBERSHIP_NUMBER']].groupby(['MEMBER_MARITAL_STATUS']).count()

train_MEMBER_MARITAL_STATUS = train_MEMBER_MARITAL_STATUS / train_MEMBER_MARITAL_STATUS.sum()

train_MEMBER_MARITAL_STATUS.columns = ['Proportions']

print(train_MEMBER_MARITAL_STATUS.round(4))



print("Testing set - MEMBER_MARITAL_STATUS")

test_MEMBER_MARITAL_STATUS = df_test[['MEMBER_MARITAL_STATUS','MEMBERSHIP_NUMBER']].groupby(['MEMBER_MARITAL_STATUS']).count()

test_MEMBER_MARITAL_STATUS = test_MEMBER_MARITAL_STATUS / test_MEMBER_MARITAL_STATUS.sum()

test_MEMBER_MARITAL_STATUS.columns = ['Proportions']

print(test_MEMBER_MARITAL_STATUS.round(4))
print("Training set - MEMBER_GENDER")

train_MEMBER_GENDER = df_train[['MEMBER_GENDER','MEMBERSHIP_NUMBER']].groupby(['MEMBER_GENDER']).count()

train_MEMBER_GENDER = train_MEMBER_GENDER / train_MEMBER_GENDER.sum()

train_MEMBER_GENDER.columns = ['Proportions']

print(train_MEMBER_GENDER.round(4))



print("Testing set - MEMBER_GENDER")

test_MEMBER_GENDER = df_test[['MEMBER_GENDER','MEMBERSHIP_NUMBER']].groupby(['MEMBER_GENDER']).count()

test_MEMBER_GENDER = test_MEMBER_GENDER / test_MEMBER_GENDER.sum()

test_MEMBER_GENDER.columns = ['Proportions']

print(test_MEMBER_GENDER.round(4))
print("Training set - MEMBERSHIP_PACKAGE")

train_MEMBERSHIP_PACKAGE = df_train[['MEMBERSHIP_PACKAGE','MEMBERSHIP_NUMBER']].groupby(['MEMBERSHIP_PACKAGE']).count()

train_MEMBERSHIP_PACKAGE = train_MEMBERSHIP_PACKAGE / train_MEMBERSHIP_PACKAGE.sum()

train_MEMBERSHIP_PACKAGE.columns = ['Proportions']

print(train_MEMBERSHIP_PACKAGE.round(4))



print("Testing set - MEMBERSHIP_PACKAGE")

test_MEMBERSHIP_PACKAGE = df_test[['MEMBERSHIP_PACKAGE','MEMBERSHIP_NUMBER']].groupby(['MEMBERSHIP_PACKAGE']).count()

test_MEMBERSHIP_PACKAGE = test_MEMBERSHIP_PACKAGE / test_MEMBERSHIP_PACKAGE.sum()

test_MEMBERSHIP_PACKAGE.columns = ['Proportions']

print(test_MEMBERSHIP_PACKAGE.round(4))
print("Training set - PAYMENT_MODE")

train_PAYMENT_MODE = df_train[['PAYMENT_MODE','MEMBERSHIP_NUMBER']].groupby(['PAYMENT_MODE']).count()

train_PAYMENT_MODE = train_PAYMENT_MODE / train_PAYMENT_MODE.sum()

train_PAYMENT_MODE.columns = ['Proportions']

print(train_PAYMENT_MODE.round(4))



print("Testing set - PAYMENT_MODE")

test_PAYMENT_MODE = df_test[['PAYMENT_MODE','MEMBERSHIP_NUMBER']].groupby(['PAYMENT_MODE']).count()

test_PAYMENT_MODE = test_PAYMENT_MODE / test_PAYMENT_MODE.sum()

test_PAYMENT_MODE.columns = ['Proportions']

print(test_PAYMENT_MODE.round(4))
print("Training set - MEMBERSHIP_STATUS")

train_MEMBERSHIP_STATUS = df_train[['MEMBERSHIP_STATUS','MEMBERSHIP_NUMBER']].groupby(['MEMBERSHIP_STATUS']).count()

train_MEMBERSHIP_STATUS = train_MEMBERSHIP_STATUS / train_MEMBERSHIP_STATUS.sum()

train_MEMBERSHIP_STATUS.columns = ['Proportions']

print(train_MEMBERSHIP_STATUS.round(4))

df_train.corr()
df_train = df_train.rename({'MEMBER_ANNUAL_INCOME_bin': 'Income', 'ANNUAL_FEES_bin': 'Fee', 'MEMBERSHIP_STATUS_new':'Status','MEMBER_OCCUPATION_CD':'Occupation','MEMBER_GENDER':'Gender','ADDITIONAL_MEMBERS':'Additional','PAYMENT_MODE':'Mode'}, axis=1)
from matplotlib.pyplot import show

ax = sns.countplot(x="Status", hue="Additional", data=df_train)

total = float(len(df_train)) # one person per row 

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format(height/total),

            ha="center") 

show()

"""

Less additional less cancelled

"""
from matplotlib.pyplot import show

ax = sns.countplot(x="Status", hue="MEMBER_MARITAL_STATUS", data=df_train)

total = float(len(df_train)) # one person per row 

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format(height/total),

            ha="center") 

show()

"""

Married people much more 'cancelled'

"""
from matplotlib.pyplot import show

ax = sns.countplot(x="Status", hue="Gender", data=df_train)

total = float(len(df_train)) # one person per row 

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format(height/total),

            ha="center") 

show()

"""

Male much more 'cancelled'

"""



from matplotlib.pyplot import show

ax = sns.countplot(x="Status", hue="MEMBERSHIP_PACKAGE", data=df_train)

total = float(len(df_train)) # one person per row 

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format(height/total),

            ha="center") 

show()



"""

Type A less 'Cancelled' in percentage compared to TypeB

"""
df_train.columns
sourceCount = df_train.groupby(['MEMBERSHIP_PACKAGE','Income','Fee','Status'])['MEMBERSHIP_NUMBER'].count()

print(sourceCount)

"""

The fees are customized for every member's personal package not logically

"""
"""

Annual payment mode is more 'cancelled, think about the promotion for this mode'

"""

from matplotlib.pyplot import show

ax = sns.countplot(x="Status", hue="Mode", data=df_train)

total = float(len(df_train)) # one person per row 

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format(height/total),

            ha="center") 

show()
"""

Customers mostly from 30 above

"""

from matplotlib.pyplot import show

ax = sns.countplot(x="Status", hue="MEMBER_AGE_AT_ISSUE_bin", data=df_train)

total = float(len(df_train)) # one person per row 

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format(height/total),

            ha="center") 

show()
"""

Medium income dominated the customers of the club- the highest rate of cancelled

"""

from matplotlib.pyplot import show

ax = sns.countplot(x="Status", hue="Income", data=df_train)

total = float(len(df_train)) # one person per row 

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format(height/total),

            ha="center") 

show()
# g = sns.FacetGrid(df_train, row="Income", col="Fee")

# g=(g.map(sns.distplot, "Status",color="r")

#       .set(xlim=(0, 40), ylim=(0, 6),

#            xticks=[10, 30, 50], yticks=[2, 6, 10]))
# g = sns.FacetGrid(df_train, row="Gender", col="Income")

# g=(g.map(sns.distplot, "Status",color="r")

#       .set(xlim=(0, 40), ylim=(0, 6),

#            xticks=[10, 30, 50], yticks=[2, 6, 10]))
# g = sns.FacetGrid(df_train, row="Additional", col="Mode")

# g=(g.map(sns.distplot, "Status",color="r")

#       .set(xlim=(0, 40), ylim=(0, 6),

#            xticks=[10, 30, 50], yticks=[2, 6, 10]))
df_train1=pd.read_csv('../input/club-data-set/club_churn_train.csv',encoding = "ISO-8859-1")
df_test1=pd.read_csv('../input/club-data-set/club_churn_test.csv',encoding = "ISO-8859-1")
df_train1.info()
agent_cancelled = df_train1.loc[df_train1.MEMBERSHIP_STATUS=='CANCELLED','AGENT_CODE']
df_train1['agent_cancelled'] = agent_cancelled
count_member_by_agent=df_train1.groupby(['agent_cancelled'])['AGENT_CODE'].count()
count_member_by_agent = pd.DataFrame(count_member_by_agent)
count_member_by_agent=count_member_by_agent.reset_index()
count_member_by_agent=count_member_by_agent.rename(columns={"AGENT_CODE": "total_cancelled_by_agents", "agent_cancelled": "AGENT_CODE"})
count_member_by_agent = pd.DataFrame(count_member_by_agent)

df_train_2 = pd.merge(df_train1, count_member_by_agent, how='left', on='AGENT_CODE')

df_train_2.head()
df_test1['AGENT_CODE']=df_test1['AGENT_CODE'].apply(str)
df_test_2 = pd.merge(df_test1, count_member_by_agent, how='left', on='AGENT_CODE')

df_test_2.head()
df_train_2.info()
df_test_2.info()
df_train_2['Type_data'] = 'Train'

df_test_2['Type_data'] = 'Test'
def concat_df(train_data, test_data):

    # Returns a concatenated df of training and test set on axis 0

    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)
df_all = concat_df(df_train_2, df_test_2)
print(df_all.isnull().sum())
df_all['total_cancelled_by_agents'] = df_all['total_cancelled_by_agents'].fillna(0)
df_all['MEMBER_ANNUAL_INCOME'] = df_all['MEMBER_ANNUAL_INCOME'].fillna(np.nanmedian(df_all['MEMBER_ANNUAL_INCOME']))

df_all['MEMBER_GENDER'] = df_all['MEMBER_GENDER'].fillna(df_all['MEMBER_GENDER'].mode()[0])

df_all['MEMBER_MARITAL_STATUS'] = df_all['MEMBER_MARITAL_STATUS'].fillna(df_all['MEMBER_MARITAL_STATUS'].mode()[0])

df_all['MEMBER_OCCUPATION_CD'] = df_all['MEMBER_OCCUPATION_CD'].fillna(df_all['MEMBER_OCCUPATION_CD'].mode()[0])
df_all.info()
df_train3 = df_all.loc[df_all.Type_data=='Train']

df_test3 = df_all.loc[df_all.Type_data=='Test']
df_train3=df_train3.reset_index()

df_test3=df_test3.reset_index()
df_train_final = df_train3.drop(['index','AGENT_CODE','END_DATE','MEMBERSHIP_NUMBER','START_DATE','Type_data','agent_cancelled'],axis=1)
df_train_final.info()
df_train_final['MEMBERSHIP_PACKAGE'] = df_train_final.MEMBERSHIP_PACKAGE.astype('category').cat.codes

df_train_final['MEMBERSHIP_STATUS'] = df_train_final.MEMBERSHIP_STATUS.astype('category').cat.codes

df_train_final['MEMBER_GENDER'] = df_train_final.MEMBER_GENDER.astype('category').cat.codes

df_train_final['MEMBER_MARITAL_STATUS'] = df_train_final.MEMBER_MARITAL_STATUS.astype('category').cat.codes

df_train_final['MEMBER_OCCUPATION_CD'] = df_train_final.MEMBER_OCCUPATION_CD.astype('category').cat.codes

df_train_final['PAYMENT_MODE'] = df_train_final.PAYMENT_MODE.astype('category').cat.codes
df_train_final['MEMBERSHIP_STATUS'].value_counts()
df_train_final['ANNUAL_FEES']=df_train_final['ANNUAL_FEES'].astype(int)

df_train_final['MEMBER_ANNUAL_INCOME']=df_train_final['MEMBER_ANNUAL_INCOME'].astype(int)

df_train_final['total_cancelled_by_agents']=df_train_final['total_cancelled_by_agents'].astype(int)
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
df_train_final.columns
from sklearn.model_selection import train_test_split

y=df_train_final['MEMBERSHIP_STATUS']

X= df_train_final[['ADDITIONAL_MEMBERS', 'ANNUAL_FEES', 'MEMBERSHIP_PACKAGE',

       'MEMBERSHIP_TERM_YEARS', 'MEMBER_AGE_AT_ISSUE',

       'MEMBER_ANNUAL_INCOME', 'MEMBER_GENDER']]



x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25 , random_state=0)
# X_test

# Create the parameter grid based on the results of random search 

param_grid = {

    'min_samples_leaf': [1,3],

    'min_samples_split': [3,5,7],

    'n_estimators': [100, 150, 200, 250]

}

rf = GridSearchCV(estimator = RandomForestClassifier(n_estimators=150, min_samples_split= 2,n_jobs=-1,verbose=1),

                  param_grid = param_grid, 

                  cv = 3, n_jobs = -1, verbose = 2)
%%time

rf.fit(x_train,y_train)
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,roc_auc_score,roc_curve,auc,recall_score

pred_prob=rf.predict_proba(x_test)

pred_var=rf.predict(x_test)



precision_score(pred_var,y_test),recall_score(pred_var,y_test),accuracy_score(pred_var,y_test)
from sklearn.model_selection import train_test_split

y=df_train_final['MEMBERSHIP_STATUS']

X= df_train_final.drop(['MEMBERSHIP_STATUS'],axis=1)



x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25 , random_state=0)
X.info()
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
# X_test

# Create the parameter grid based on the results of random search 

param_grid = {

    'min_samples_leaf': [1,3],

    'min_samples_split': [3,5,7],

    'n_estimators': [100, 150, 200, 250]

}

rf = GridSearchCV(estimator = RandomForestClassifier(n_estimators=150, min_samples_split= 2,n_jobs=-1,verbose=1),

                  param_grid = param_grid, 

                  cv = 3, n_jobs = -1, verbose = 2)
%%time

rf.fit(x_train,y_train)
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,roc_auc_score,roc_curve,auc,recall_score

pred_prob=rf.predict_proba(x_test)

pred_var=rf.predict(x_test)
print(pd.DataFrame(confusion_matrix(y_test, pred_var, labels=[1,0]), index=['true:1', 'true:0'], columns=['pred:1', 'pred:0']))
precision_score(pred_var,y_test),recall_score(pred_var,y_test),accuracy_score(pred_var,y_test)
labels = df_train_final['MEMBERSHIP_STATUS']

features = df_train_final.drop('MEMBERSHIP_STATUS', axis=1)

# List of features for later use

feature_list = list(features.columns)



# Get numerical feature importances

importances = list(rf.best_estimator_.feature_importances_)

# List of tuples with variable and importance

feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

# Sort the feature importances by most important first

feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 

[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]





import matplotlib.pyplot as plt



feat_importances = pd.Series(rf.best_estimator_.feature_importances_, index=feature_list)

plt.ylabel('Features')

plt.xlabel('Importance---->')

plt.title('Feature Importance plot')

feat_importances.nlargest(9).plot(kind='barh', color=['orange', 'red', 'green', 'blue', 'cyan','yellow'])
from sklearn.externals import joblib 

  

# Save the model as a pickle in a file 

joblib.dump(rf, 'rf_model_01Mar_test.pkl') 

  

# Load the model from the file 

rf_from_joblib = joblib.load('rf_model_01Mar_test.pkl')  
rf
from sklearn.model_selection import KFold



kf = KFold(n_splits=10)

outcomes = []

    

fold = 0

for train_index, test_index in kf.split(X):

    fold += 1

    X_train, X_test = X.values[train_index], X.values[test_index]

    y_train, y_test = y.values[train_index], y.values[test_index]

    rf.fit(X_train, y_train)

    predictions = rf.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    outcomes.append(accuracy)

    print("Fold {0} accuracy: {1}".format(fold, accuracy))     

mean_outcome = np.mean(outcomes)

print("\n\nMean Accuracy: {0}".format(mean_outcome)) 
df_test_final = df_test3.drop(['MEMBERSHIP_STATUS','index','AGENT_CODE','END_DATE','MEMBERSHIP_NUMBER','START_DATE','Type_data','agent_cancelled'],axis=1)
df_test_final.info()
df_test_final['MEMBERSHIP_PACKAGE'] = df_test_final.MEMBERSHIP_PACKAGE.astype('category').cat.codes

df_test_final['MEMBER_GENDER'] = df_test_final.MEMBER_GENDER.astype('category').cat.codes

df_test_final['MEMBER_MARITAL_STATUS'] = df_test_final.MEMBER_MARITAL_STATUS.astype('category').cat.codes

df_test_final['MEMBER_OCCUPATION_CD'] = df_test_final.MEMBER_OCCUPATION_CD.astype('category').cat.codes

df_test_final['PAYMENT_MODE'] = df_test_final.PAYMENT_MODE.astype('category').cat.codes
pred2=rf_from_joblib.predict(df_test_final)    

probs_=rf_from_joblib.predict_proba(df_test_final)
df=pd.DataFrame()

df['p']=probs_[:,1]

df['pred2'] =df['p'].map(lambda x: 'CANCELLED' if x > 0.9 else 'INFORCE')
df['MEMBERSHIP_NUMBER']=df_test3['MEMBERSHIP_NUMBER']
df.info()
df.head()
df_sub = pd.merge(df_test3, df, how='inner', on='MEMBERSHIP_NUMBER')

df_sub.head()
df.to_csv(r'C:\Users\347997\Desktop\sub.csv')