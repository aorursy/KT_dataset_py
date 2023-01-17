import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
%matplotlib inline
pd.pandas.set_option('display.max_columns', None)
df_train = pd.read_csv('../input/term-deposit-prediction-data-set/train.csv')
df_test = pd.read_csv('../input/term-deposit-prediction-data-set/test.csv')
df_train.head()
df_test.head()
df_train.shape, df_test.shape
df_train.columns
df_test.columns
data_types_train = pd.DataFrame(df_train.dtypes, columns = ['Train'])
data_types_test = pd.DataFrame(df_test.dtypes, columns = ['Test'])
data_types = pd.concat([data_types_train, data_types_test], axis = 1)
data_types
missing_values_train = pd.DataFrame(df_train.isna().sum(), columns = ['Train'])
missing_values_test = pd.DataFrame(df_test.isna().sum(), columns = ['Test'])
missing_values = pd.concat([missing_values_train, missing_values_test], axis = 1)
missing_values
y_df = df_train[df_train['subscribed'] == 'yes']
df_train[['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']].describe()
df_test[['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']].describe()
fig, ax = plt.subplots()
fig.set_size_inches(20,10)
sns.heatmap(df_train.corr(), vmax = 1, vmin = -1, square = False, annot = True)
sns.countplot(x = 'subscribed', data = df_train)
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Subscribed', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
age_group_train = []
for i in df_train['age']:
    if (i >= 18 and i <= 25):
        age_group_train.append('18-25')
    elif (i >= 26 and i <= 33):
        age_group_train.append('26-33')
    elif (i >= 34 and i <= 41):
        age_group_train.append('34-41')
    elif (i >= 42 and i <= 49):
        age_group_train.append('42-49')
    elif (i >= 50 and i <= 57):
        age_group_train.append('50-57')
    elif (i >= 58 and i <= 65):
        age_group_train.append('58-65')
    elif (i >= 66 and i <= 73):
        age_group_train.append('66-73')
    elif (i >= 74 and i <= 81):
        age_group_train.append('74-81')
    elif (i >= 82 and i <= 89):
        age_group_train.append('82-89')
    elif (i >= 90 and i <= 97):
        age_group_train.append('90-97')
    else:
        pass
df_train['age_group'] = age_group_train
age_group_test = []
for i in df_test['age']:
    if (i >= 18 and i <= 25):
        age_group_test.append('18-25')
    elif (i >= 26 and i <= 33):
        age_group_test.append('26-33')
    elif (i >= 34 and i <= 41):
        age_group_test.append('34-41')
    elif (i >= 42 and i <= 49):
        age_group_test.append('42-49')
    elif (i >= 50 and i <= 57):
        age_group_test.append('50-57')
    elif (i >= 58 and i <= 65):
        age_group_test.append('58-65')
    elif (i >= 66 and i <= 73):
        age_group_test.append('66-73')
    elif (i >= 74 and i <= 81):
        age_group_test.append('74-81')
    elif (i >= 82 and i <= 89):
        age_group_test.append('82-89')
    elif (i >= 90 and i <= 97):
        age_group_test.append('90-97')
    else:
        pass
df_test['age_group'] = age_group_test
plt.figure(figsize = (10, 8))
sns.countplot(x ='age_group', hue = 'subscribed', data = df_train)
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Age Group', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
sns.distplot(df_train['age'])
sns.distplot(y_df['age'])
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Age', fontsize = 15)
plt.legend(['Age', 'Subscribers Age'])
plt.show()
plt.figure(figsize = (20, 16))
sns.countplot(x ='job', hue = 'subscribed', data = df_train)
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Job', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
sns.countplot(x ='marital', hue = 'subscribed', data = df_train)
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Marital Status', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
sns.countplot(x ='education', hue = 'subscribed', data = df_train)
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Education', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
sns.countplot(x ='default', hue = 'subscribed', data = df_train)
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Default', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
balance_train = []
for i in df_train['balance']:
    if (i >= -14999 and i <= 0):
        balance_train.append('Negative Balance')
    elif (i >= 1 and i <= 15000):
        balance_train.append('0K - 15K')
    elif (i >= 15001 and i <= 30000):
        balance_train.append('15K - 30K')
    elif (i >= 30001 and i <= 45000):
        balance_train.append('30K - 45K')
    elif (i >= 45001 and i <= 60000):
        balance_train.append('45K - 60K')
    elif (i >= 60001 and i <= 75000):
        balance_train.append('60K - 75K')
    elif (i >= 75001 and i <= 90000):
        balance_train.append('75K - 90K')
    elif (i >= 90001 and i <= 105000):
        balance_train.append('90K - 105K')
    else:
        pass
df_train['balance_group'] = balance_train
balance_test = []
for i in df_test['balance']:
    if (i >= -14999 and i <= 0):
        balance_test.append('Negative Balance')
    elif (i >= 1 and i <= 15000):
        balance_test.append('0K - 15K')
    elif (i >= 15001 and i <= 30000):
        balance_test.append('15K - 30K')
    elif (i >= 30001 and i <= 45000):
        balance_test.append('30K - 45K')
    elif (i >= 45001 and i <= 60000):
        balance_test.append('45K - 60K')
    elif (i >= 60001 and i <= 75000):
        balance_test.append('60K - 75K')
    elif (i >= 75001 and i <= 90000):
        balance_test.append('75K - 90K')
    elif (i >= 90001 and i <= 105000):
        balance_test.append('90K - 105K')
    else:
        pass
df_test['balance_group'] = balance_test
plt.figure(figsize = (20, 16))
sns.countplot(x ='balance_group', hue = 'subscribed', data = df_train)
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Balance Group', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.legend(loc = 'upper right')
sns.distplot(df_train['balance'])
sns.distplot(y_df['balance'])
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Balance', fontsize = 15)
plt.legend(['Balance', 'Subscribers Balance'])
plt.show()
sns.distplot((df_train['balance']) ** (1/5))
sns.distplot((y_df['balance']) ** (1/5))
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Balance', fontsize = 15)
plt.legend(['Balance', 'Subscribers Balance'])
plt.show()
df_train['balance'] = df_train['balance'] ** (1/5)
df_test['balance'] = df_test['balance'] ** (1/5)
df_train['balance'] = df_train['balance'].fillna(df_train['balance'].mean())
df_test['balance'] = df_test['balance'].fillna(df_test['balance'].mean())
sns.countplot(x ='housing', hue = 'subscribed', data = df_train)
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Housing Loan', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
sns.countplot(x ='loan', hue = 'subscribed', data = df_train)
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Personal Loan', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
sns.countplot(x ='contact', hue = 'subscribed', data = df_train)
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Contact', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
week_train = []
for i in df_train['day']:
    if i < 8:
        week_train.append(1)
    elif i >= 8 and i < 16:
        week_train.append(2)
    elif i >=16 and i < 22:
        week_train.append(3)
    else:
        week_train.append(4)
df_train['week'] = week_train
df_train = df_train.drop('day', axis = 1)
week_test = []
for i in df_test['day']:
    if i < 8:
        week_test.append(1)
    elif i >= 8 and i < 16:
        week_test.append(2)
    elif i >=16 and i < 22:
        week_test.append(3)
    else:
        week_test.append(4)
df_test['week'] = week_test
df_test = df_test.drop('day', axis = 1)
sns.countplot(x ='week', hue = 'subscribed', data = df_train)
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Week', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.figure(figsize = (20, 16))
sns.countplot(x ='month', hue = 'subscribed', data = df_train)
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Month', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
duration_train = []
for i in df_train['duration']:
    if (i >= 0 and i <= 500):
        duration_train.append('0-500')
    elif (i >= 501 and i <= 1000):
        duration_train.append('501-1000')
    elif (i >= 1001 and i <= 1500):
        duration_train.append('1001-1500')
    elif (i >= 1501 and i <= 2000):
        duration_train.append('1501-2000')
    elif (i >= 2001 and i <= 2500):
        duration_train.append('2001 - 2500')
    elif (i >= 2501 and i <= 3000):
        duration_train.append('2501-3000')
    elif (i >= 3001 and i <= 3500):
        duration_train.append('3001-3500')
    elif (i >= 3501 and i <= 4000):
        duration_train.append('3501-4000')
    elif (i >= 4001 and i <= 4500):
        duration_train.append('4001-4500')
    elif (i >= 4501 and i <= 5000):
        duration_train.append('4501-5000')
    else:
        pass
df_train['duration_group'] = duration_train
duration_test = []
for i in df_test['duration']:
    if (i >= 0 and i <= 500):
        duration_test.append('0-500')
    elif (i >= 501 and i <= 1000):
        duration_test.append('501-1000')
    elif (i >= 1001 and i <= 1500):
        duration_test.append('1001-1500')
    elif (i >= 1501 and i <= 2000):
        duration_test.append('1501-2000')
    elif (i >= 2001 and i <= 2500):
        duration_test.append('2001 - 2500')
    elif (i >= 2501 and i <= 3000):
        duration_test.append('2501-3000')
    elif (i >= 3001 and i <= 3500):
        duration_test.append('3001-3500')
    elif (i >= 3501 and i <= 4000):
        duration_test.append('3501-4000')
    elif (i >= 4001 and i <= 4500):
        duration_test.append('4001-4500')
    elif (i >= 4501 and i <= 5000):
        duration_test.append('4501-5000')
    else:
        pass
df_test['duration_group'] = duration_test
plt.figure(figsize = (20, 16))
sns.countplot(x ='duration_group', hue = 'subscribed', data = df_train)
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Duration Group', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.legend(loc = 'upper right')
sns.distplot(df_train['duration'])
sns.distplot(y_df['duration'])
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Duration', fontsize = 15)
plt.legend(['Duration', 'Subscribers Duration'])
plt.show()
sns.distplot((df_train['duration']) ** (1/3))
sns.distplot((y_df['duration']) ** (1/3))
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Duration', fontsize = 15)
plt.legend(['Duration', 'Subscribers Duration'])
plt.show()
df_train['duration'] = df_train['duration'] ** (1/3)
df_test['duration'] = df_test['duration'] ** (1/3)
campaign_train = []
for i in df_train['campaign']:
    if (i >= 1 and i <= 7):
        campaign_train.append('1-7')
    elif (i >= 8 and i <= 14):
        campaign_train.append('8-14')
    elif (i >= 15 and i <= 21):
        campaign_train.append('15-21')
    elif (i >= 22 and i <= 28):
        campaign_train.append('22-28')
    elif (i >= 29 and i <= 35):
        campaign_train.append('29-35')
    elif (i >= 36 and i <= 42):
        campaign_train.append('36-42')
    elif (i >= 43 and i <= 49):
        campaign_train.append('43-49')
    elif (i >= 50 and i <= 56):
        campaign_train.append('50-56')
    elif (i >= 57 and i <= 63):
        campaign_train.append('57-63')
    else:
        pass
df_train['campaign_group'] = campaign_train
campaign_test = []
for i in df_test['campaign']:
    if (i >= 1 and i <= 7):
        campaign_test.append('1-7')
    elif (i >= 8 and i <= 14):
        campaign_test.append('8-14')
    elif (i >= 15 and i <= 21):
        campaign_test.append('15-21')
    elif (i >= 22 and i <= 28):
        campaign_test.append('22-28')
    elif (i >= 29 and i <= 35):
        campaign_test.append('29-35')
    elif (i >= 36 and i <= 42):
        campaign_test.append('36-42')
    elif (i >= 43 and i <= 49):
        campaign_test.append('43-49')
    elif (i >= 50 and i <= 56):
        campaign_test.append('50-56')
    elif (i >= 57 and i <= 63):
        campaign_test.append('57-63')
    else:
        pass
df_test['campaign_group'] = campaign_test
plt.figure(figsize = (10, 8))
sns.countplot(x ='campaign_group', hue= 'subscribed', data = df_train)
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Campaign Group', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.legend(loc = 'upper right')
sns.distplot(df_train['campaign'])
sns.distplot(y_df['campaign'])
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Campaign', fontsize = 15)
plt.legend(['Campaign', 'Subscribers Campaign'])
plt.show()
sns.distplot((df_train['campaign']) ** (1/3))
sns.distplot((y_df['campaign']) ** (1/3))
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Campaign', fontsize = 15)
plt.legend(['Campaign', 'Subscribers Campaign'])
plt.show()
df_train['campaign'] = df_train['campaign'] ** (1/3)
df_test['campaign'] = df_test['campaign'] ** (1/3)
pdays_train = []
for i in df_train['pdays']:
    if i < 0:
        pdays_train.append(0)
    else:
        pdays_train.append(i)
df_train['pdays'] = pdays_train
pdays_test = []
for i in df_test['pdays']:
    if i < 0:
        pdays_test.append(0)
    else:
        pdays_test.append(i)
df_test['pdays'] = pdays_test
pdays_train = []
for i in df_train['pdays']:
    if (i >= 0 and i <= 100):
        pdays_train.append('1-100')
    elif (i >= 101 and i <= 200):
        pdays_train.append('101-200')
    elif (i >= 201 and i <= 300):
        pdays_train.append('201-300')
    elif (i >= 301 and i <= 400):
        pdays_train.append('301-400')
    elif (i >= 401 and i <= 500):
        pdays_train.append('401-500')
    elif (i >= 501 and i <= 600):
        pdays_train.append('501-600')
    elif (i >= 601 and i <= 700):
        pdays_train.append('601-700')
    elif (i >= 701 and i <= 800):
        pdays_train.append('701-800')
    elif (i >= 801 and i <= 900):
        pdays_train.append('801-900')
    else:
        pass
df_train['pdays_group'] = pdays_train
pdays_test = []
for i in df_test['pdays']:
    if (i >= 0 and i <= 100):
        pdays_test.append('1-100')
    elif (i >= 101 and i <= 200):
        pdays_test.append('101-200')
    elif (i >= 201 and i <= 300):
        pdays_test.append('201-300')
    elif (i >= 301 and i <= 400):
        pdays_test.append('301-400')
    elif (i >= 401 and i <= 500):
        pdays_test.append('401-500')
    elif (i >= 501 and i <= 600):
        pdays_test.append('501-600')
    elif (i >= 601 and i <= 700):
        pdays_test.append('601-700')
    elif (i >= 701 and i <= 800):
        pdays_test.append('701-800')
    elif (i >= 801 and i <= 900):
        pdays_test.append('801-900')
    else:
        pass
df_test['pdays_group'] = pdays_test
plt.figure(figsize = (10, 8))
sns.countplot(x ='pdays_group', hue= 'subscribed', data = df_train)
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Pdays Group', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
sns.kdeplot(df_train['pdays'], bw = 10)
sns.kdeplot(y_df['pdays'], bw = 10)
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Pdays', fontsize = 15)
plt.legend(['Pdays', 'Subscribers Pdays'])
plt.show()
sns.distplot(df_train['pdays'], kde = False)
sns.distplot(y_df['pdays'], kde = False)
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Pdays', fontsize = 15)
plt.legend(['Pdays', 'Subscribers Pdays'])
plt.show()
sns.kdeplot((df_train['pdays']) ** (1/3), bw = 10)
sns.kdeplot((y_df['pdays']) ** (1/3), bw = 10)
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Pdays', fontsize = 15)
plt.legend(['Pdays', 'Subscribers Pdays'])
plt.show()
sns.distplot((df_train['pdays']) ** (1/3), kde = False)
sns.distplot((y_df['pdays']) ** (1/3), kde = False)
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Pdays', fontsize = 15)
plt.legend(['Pdays', 'Subscribers Pdays'])
plt.show()
df_train['pdays'] = df_train['pdays'] ** (1/3)
df_test['pdays'] = df_test['pdays'] ** (1/3)
previous_train = []
for i in df_train['previous']:
    if (i >= 0 and i <= 50):
        previous_train.append('1-50')
    elif (i >= 51 and i <= 100):
        previous_train.append('51-100')
    elif (i >= 101 and i <= 150):
        previous_train.append('101-150')
    elif (i >= 151 and i <= 200):
        previous_train.append('151-200')
    elif (i >= 201 and i <= 250):
        previous_train.append('201-250')
    elif (i >= 251 and i <= 300):
        previous_train.append('251-300')
    else:
        pass
df_train['previous_groups'] = previous_train
previous_test = []
for i in df_test['previous']:
    if (i >= 0 and i <= 50):
        previous_test.append('1-50')
    elif (i >= 51 and i <= 100):
        previous_test.append('51-100')
    elif (i >= 101 and i <= 150):
        previous_test.append('101-150')
    elif (i >= 151 and i <= 200):
        previous_test.append('151-200')
    elif (i >= 201 and i <= 250):
        previous_test.append('201-250')
    elif (i >= 251 and i <= 300):
        previous_test.append('251-300')
    else:
        pass
df_test['previous_groups'] = previous_test
plt.figure(figsize = (10, 8))
sns.countplot(x ='previous_groups', hue= 'subscribed', data = df_train)
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Previous Group', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
sns.kdeplot(df_train['previous'], bw = 10)
sns.kdeplot(y_df['previous'], bw = 10)
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Previous', fontsize = 15)
plt.legend(['Previous', 'Subscribers Previous'])
plt.show()
sns.distplot(df_train['previous'], kde = False)
sns.distplot(y_df['previous'], kde = False)
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Previous', fontsize = 15)
plt.legend(['Previous', 'Subscribers Previous'])
plt.show()
sns.kdeplot((df_train['previous']) ** (1/2), bw = 10)
sns.kdeplot((y_df['previous']) ** (1/2), bw = 10)
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Previous', fontsize = 15)
plt.legend(['Previous', 'Subscribers Previous'])
plt.show()
sns.distplot((df_train['previous']) ** (1/2), kde = False)
sns.distplot((y_df['previous']) ** (1/2), kde = False)
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Previous', fontsize = 15)
plt.legend(['Previous', 'Subscribers Previous'])
plt.show()
df_train['previous'] = df_train['previous'] ** (1/2)
df_test['previous'] = df_test['previous'] ** (1/2)
sns.countplot(x ='poutcome', hue = 'subscribed', data = df_train)
plt.xticks(fontsize = 10, ha = "right")
plt.yticks(fontsize = 10)
plt.xlabel('Previous Outcome', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
df_train_dummies = pd.get_dummies(df_train[['job', 'marital', 'default', 'housing', 'loan', 'contact', 'poutcome', 'week']], drop_first = True)
df_train_label = df_train[['education', 'month']].apply(LabelEncoder().fit_transform)
df_test_dummies = pd.get_dummies(df_test[['job', 'marital', 'default', 'housing', 'loan', 'contact', 'poutcome', 'week']], drop_first = True)
df_test_label = df_test[['education', 'month']].apply(LabelEncoder().fit_transform)
df_train = pd.concat([df_train.drop(['job', 'marital', 'default', 'housing', 'loan', 'contact', 'poutcome', 'education', 'month', 'week'], axis = 1), df_train_dummies, df_train_label], axis = 1)
df_test = pd.concat([df_test.drop(['job', 'marital', 'default', 'housing', 'loan', 'contact', 'poutcome', 'education', 'month', 'week'], axis = 1), df_test_dummies, df_test_label], axis = 1)
df_train = df_train.drop(['ID', 'age_group', 'balance_group', 'duration_group', 'campaign_group', 'pdays_group', 'previous_groups'], axis = 1)
df_test = df_test.drop(['ID', 'age_group', 'balance_group', 'duration_group', 'campaign_group', 'pdays_group', 'previous_groups'], axis = 1)
df_train['subscribed'] = df_train['subscribed'].map({'yes': 1, 'no': 0})
df_train_scaled = pd.DataFrame(StandardScaler().fit_transform(df_train.drop('subscribed', axis = 1)), columns = df_test.columns)
df_test_scaled = pd.DataFrame(StandardScaler().fit_transform(df_test), columns = df_test.columns)
pca_columns = []
for i in range(df_train_scaled.shape[1]):
    pca_columns.append('PC' + str(i+1))
pca_model = PCA()
pca_model.fit(df_train_scaled)
df_pca_train = pd.DataFrame(pca_model.transform(df_train_scaled), columns = pca_columns)
explained_info_train = pd.DataFrame(pca_model.explained_variance_ratio_, columns=['Explained Info']).sort_values(by = 'Explained Info', ascending = False)
imp = []
for i in range(explained_info_train.shape[0]):
    imp.append(explained_info_train.head(i).sum())
explained_info_train_sum = pd.DataFrame()
explained_info_train_sum['Variable'] = pca_columns
explained_info_train_sum['Importance'] = imp
explained_info_train_sum
pca_columns = []
for i in range(19):
    pca_columns.append('PC' + str(i+1))
pca_model = PCA(n_components = 19)
pca_model.fit(df_train_scaled)
df_pca_train = pd.DataFrame(pca_model.transform(df_train_scaled), columns = pca_columns)
df_pca_train.head()
pca_model = PCA(n_components = 19)
pca_model.fit(df_test_scaled)
df_pca_test = pd.DataFrame(pca_model.transform(df_test_scaled), columns = pca_columns)
X = df_pca_train
y = df_train['subscribed']
df_train['subscribed'].value_counts()
smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X, y)
y_smote.value_counts()
X_train, X_val, y_train, y_val = train_test_split(X_smote, y_smote, test_size = 0.3, random_state = 17)
X_test = df_pca_test
X_train.shape, X_val.shape, y_train.shape, y_val.shape, X_test.shape
X_train_sm = sm.add_constant(X_train)
lg = sm.Logit(y_train,X_train_sm)
lg = lg.fit()
print(lg.summary())
models = [LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier(), AdaBoostClassifier(), GradientBoostingClassifier(), KNeighborsClassifier(), SVC(), XGBClassifier()]
model_names = ['LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', 'AdaBoostClassifier', 'GradientBoostingClassifier', 'KNeighborsClassifier', 'SVC', 'XGBClassifier']
accuracy_train = []
accuracy_val = []
for model in models:
    mod = model
    mod.fit(X_train, y_train)
    y_pred_train = mod.predict(X_train)
    y_pred_val = mod.predict(X_val)
    accuracy_train.append(accuracy_score(y_train, y_pred_train))
    accuracy_val.append(accuracy_score(y_val, y_pred_val))
data = {'Modelling Algorithm' : model_names, 'Train Accuracy' : accuracy_train, 'Validation Accuracy' : accuracy_val}
data = pd.DataFrame(data)
data['Difference'] = ((np.abs(data['Train Accuracy'] - data['Validation Accuracy'])) * 100)/(data['Train Accuracy'])
data.sort_values(by = 'Validation Accuracy', ascending = False)
knc = KNeighborsClassifier()
possible_parameter_values = {'n_neighbors' : range(1, 100)}
knc_rs_cv = RandomizedSearchCV(estimator = knc, param_distributions = possible_parameter_values, cv = 10, scoring = 'accuracy')
knc_rs_cv.fit(X_train, y_train)
knc_rs_cv.best_params_
knc_rs_cv.best_score_
knc = KNeighborsClassifier(n_neighbors = 8)
knc.fit(X_train, y_train)
y_pred_val = knc.predict(X_val)
accuracy_score(y_val, y_pred_val)
knc = KNeighborsClassifier(n_neighbors = 8)
knc.fit(X_train, y_train)
y_pred_test = knc.predict(X_test)
y_pred_test = pd.DataFrame(y_pred_test, columns = ['Prediction'])
y_pred_test.head()
y_pred_test.to_csv('Prediction.csv')
pca_columns = []
for i in range(19):
    pca_columns.append('PC' + str(i+1))
org_var = pd.DataFrame(pca_model.components_, index = pca_columns, columns = df_train_scaled.columns)
values = []
for i in org_var.columns:
    values.append(org_var[i].sum())
dep_var = pd.DataFrame()
dep_var['Variables'] = df_train_scaled.columns
dep_var['Values'] = values
dep_var.sort_values(by = 'Values', ascending = False)
