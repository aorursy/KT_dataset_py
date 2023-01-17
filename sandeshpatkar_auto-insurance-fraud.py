# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import datetime as dt



#data viz

#for better viz

import plotly.express as px

import plotly.graph_objects as go



#for quick viz

import seaborn as sns



#ml

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import classification_report

from imblearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC

df = pd.read_csv('/kaggle/input/auto-insurance-claims-data/insurance_claims.csv')

df.head()
df.isnull().sum()
# removing column named _c39 as it contains only null values



df = df.drop(['_c39'], axis = 1)
df.info()
df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'])
df.describe().T
for i in df.columns:

    if df[i].dtype == 'object':

        print(i, ":", df[i].nunique())
drop_columns = ['policy_state', 'policy_csl', 'incident_date', 'incident_state', 'incident_city', 'incident_location']

df = df.drop(drop_columns, axis = 1)

df.head()
for i in df.columns:

    if df[i].dtype == 'object':

        print(i, ":", df[i].nunique())
df['fraud_reported'] = df['fraud_reported'].str.replace('Y', '1')

df['fraud_reported'] = df['fraud_reported'].str.replace('N', '0')

df['fraud_reported'] = df['fraud_reported'].astype(int)
df['fraud_reported'].unique()
sns.countplot(df['fraud_reported'])
def vis_data(df, x, y = 'fraud_reported', graph = 'countplot'):

    if graph == 'hist':

        fig = px.histogram(df, x = x)

        fig.update_layout(title = 'Distribution of {x}'.format(x = x))

        fig.show()

    elif graph == 'bar':

      fig = px.bar(df, x = x, y = y)

      fig.update_layout(title = '{x} vs. {y}'.format(x = x, y = y))

      fig.show()

    elif graph == 'countplot':

      a = df.groupby([x,y]).count()

      a.reset_index(inplace = True)

      no_fraud = a[a['fraud_reported'] == 0]

      yes_fraud = a[a['fraud_reported'] == 1]

      trace1 = go.Bar(x = no_fraud[x], y = no_fraud['policy_number'], name = 'No Fraud')

      trace2 = go.Bar(x = yes_fraud[x], y = yes_fraud['policy_number'], name = 'Fraud')

      fig = go.Figure(data = [trace1, trace2])

      fig.update_layout(title = '{x} vs. {y}'.format(x=x, y = y))

      fig.update_layout(barmode = 'group')

      fig.show()
vis_data(df, 'insured_sex')
vis_data(df, 'insured_education_level')
vis_data(df, 'insured_occupation')
vis_data(df, 'insured_relationship')
vis_data(df, 'incident_type')
vis_data(df, 'collision_type')
vis_data(df, 'incident_severity')
vis_data(df, 'authorities_contacted')
vis_data(df, 'insured_hobbies')
hobbies = df['insured_hobbies'].unique()

for hobby in hobbies:

  if (hobby != 'chess') & (hobby != 'cross-fit'):

    df['insured_hobbies'] = df['insured_hobbies'].str.replace(hobby, 'other')



df['insured_hobbies'].unique()
df.head()
vis_data(df, 'age', 'anything', 'hist')
df['age'].describe()
bin_labels = ['15-20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51-55', '56-60', '61-65']

bins = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]



df['age_group'] = pd.cut(df['age'], bins = bins, labels = bin_labels, include_lowest = True)
vis_data(df, 'age_group')
vis_data(df, 'months_as_customer', 'not', 'hist')
df['months_as_customer'].describe()
bins = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

bin_labels = ['0-50','51-100','101-150','151-200','201-250','251-300','301-350','351-400','401-450','451-500']



df['months_as_customer_groups'] = pd.cut(df['months_as_customer'], bins = 10, labels = bin_labels, include_lowest= True)
vis_data(df, 'months_as_customer_groups')
vis_data(df, 'auto_make')
vis_data(df, 'number_of_vehicles_involved')
vis_data(df, 'witnesses', 'fraud_reported')
vis_data(df, 'bodily_injuries')
vis_data(df, 'total_claim_amount', 'y', 'hist')
vis_data(df, 'incident_hour_of_the_day')
vis_data(df, 'number_of_vehicles_involved')
vis_data(df, 'witnesses')
vis_data(df, 'auto_year')
df['policy_annual_premium'].describe()
bins = list(np.linspace(0,2500, 6, dtype = int))

bin_labels = ['very low', 'low', 'medium', 'high', 'very high']



df['policy_annual_premium_groups'] = pd.cut(df['policy_annual_premium'], bins = bins, labels=bin_labels)
vis_data(df, 'policy_annual_premium_groups')

df['policy_deductable'].describe()
bins = list(np.linspace(0,2000, 5, dtype = int))

bin_labels = ['0-500', '501-1000', '1001-1500', '1501-2000']



df['policy_deductable_group'] = pd.cut(df['policy_deductable'], bins = bins, labels = bin_labels)



vis_data(df, 'policy_deductable_group')
vis_data(df, 'property_damage')
vis_data(df, 'police_report_available')
#removing columns for which we created groups

df = df.drop(['age', 'months_as_customer', 'policy_deductable', 'policy_annual_premium'], axis = 1)

df.columns
required_columns = ['policy_number', 'insured_sex', 'insured_education_level', 'insured_occupation',

       'insured_hobbies', 'capital-gains', 'capital-loss', 'incident_type', 'collision_type', 'incident_severity',

       'authorities_contacted', 'incident_hour_of_the_day', 'number_of_vehicles_involved',

       'witnesses', 'total_claim_amount',

       'injury_claim', 'property_claim', 'vehicle_claim',

       'fraud_reported', 'age_group',

       'months_as_customer_groups', 'policy_annual_premium_groups']



print(len(required_columns))
df1 = df[required_columns]



corr_matrix = df1.corr()



fig = go.Figure(data = go.Heatmap(

                                z = corr_matrix.values,

                                x = list(corr_matrix.columns),

                                y = list(corr_matrix.index)))



fig.update_layout(title = 'Correlation')



fig.show()
t = df['total_claim_amount'].iloc[1]

a = df['vehicle_claim'].iloc[1]

b = df['property_claim'].iloc[1]

c = df['injury_claim'].iloc[1]



print(t)

a+b+c
required_columns = ['insured_sex', 'insured_occupation',

       'insured_hobbies', 'capital-gains', 'capital-loss', 'incident_type', 'collision_type', 'incident_severity',

       'authorities_contacted', 'incident_hour_of_the_day', 'number_of_vehicles_involved',

       'witnesses', 'total_claim_amount', 'fraud_reported', 'age_group',

       'months_as_customer_groups', 'policy_annual_premium_groups']



print(len(required_columns))
df1 = df1[required_columns]

df1.head()
cat_cols = ['age_group', 'months_as_customer_groups', 'policy_annual_premium_groups']

for col in cat_cols:

  df1[col] = df1[col].astype('object')



columns_to_encode = []

for col in df1.columns:

  if df1[col].dtype == 'object':

    columns_to_encode.append(col)



columns_to_encode
df1.info()
df1.head()
df2 = pd.get_dummies(df1, columns = columns_to_encode)



df2.head()
features = []

for col in df2.columns:

  if col != 'fraud_reported':

    features.append(col)



target = 'fraud_reported'



X = df2[features]

y = df2[target]
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

X = sc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)
lr = LogisticRegression()



lr.fit(X_train, y_train)

preds = lr.predict(X_test)



score = lr.score(X_test, y_test)

print(score)
print(classification_report(y_test, preds))
oversample = SMOTE(random_state=9)
X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, random_state = 1)
X_over, y_over = oversample.fit_resample(X_train, y_train)
lr.fit(X_train, y_train)

preds = lr.predict(X_test)

score = lr.score(X_test, y_test)

print(score)

print()

print(classification_report(y_test, preds))
dtc = DecisionTreeClassifier()



dtc.fit(X_train, y_train)

preds = dtc.predict(X_test)



score = dtc.score(X_test, y_test)

print(score)

print()

print(classification_report(y_test, preds))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state = 1)

rfc.fit(X_train, y_train)
preds = rfc.predict(X_test)



score = rfc.score(X_test, y_test)

print(score*100)

print()

print(classification_report(y_test, preds))
#implementing 



svc = SVC(kernel='linear')

svc.fit(X_train, y_train)



preds = svc.predict(X_test)



print('Score:' , svc.score(X_test, y_test))

print('Classification report:', classification_report(y_test, preds))
degrees = [2,3,4,5,6,7,8]

kernels = ['poly', 'rbf', 'sigmoid']

c_value = [1,2,3]
scores = {}

for degree in degrees:

    for kernel in kernels:

        for c in c_value:

            svc_t = SVC(kernel = kernel, degree = degree, C = c)

            svc_t.fit(X_train, y_train)

            

            preds = svc_t.predict(X_test)

            score = svc_t.score(X_test,y_test)

#             print('Score with degree as {d}, kernel as {k}, C as {c} is:'.format(d = degree, k = kernel, c = c), score)

            scores['Score with degree as {d}, kernel as {k}, C as {c} is best'.format(d = degree, k = kernel, c = c)] = score



print(max(scores, key=scores.get))
svc_tuned = SVC(kernel='sigmoid', degree = 2, C = 3)

svc_tuned.fit(X_train, y_train)



preds = svc_tuned.predict(X_test)



print('Score:' , svc_tuned.score(X_test, y_test))

print('Classification report:', classification_report(y_test, preds))
rfc_tuned = RandomForestClassifier(n_estimators = 1000, random_state = 1, min_samples_split = 2)

rfc_tuned.fit(X_train, y_train)

preds_tuned = rfc_tuned.predict(X_test)

score = rfc_tuned.score(X_test, y_test)

print(score)
n_estimators = [100, 300, 500, 800, 1200]

max_depth = [5, 8, 15, 25, 30]

min_samples_split = [2, 5, 10, 15, 100]

min_samples_leaf = [1, 2, 5, 10] 



hyper = dict(n_estimators = n_estimators, max_depth = max_depth,  

              min_samples_split = min_samples_split, 

             min_samples_leaf = min_samples_leaf)



grid = GridSearchCV(rfc, hyper, cv = 3, verbose = 1, 

                      n_jobs = -1)

best = grid.fit(X_train, y_train)
print(best)
rfc_tuned = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,

                                              class_weight=None,

                                              criterion='gini', max_depth=None,

                                              max_features='auto',

                                              max_leaf_nodes=None,

                                              max_samples=None,

                                              min_impurity_decrease=0.0,

                                              min_impurity_split=None,

                                              min_samples_leaf=1,

                                              min_samples_split=2,

                                              min_weight_fraction_leaf=0.0,

                                              n_estimators=100, n_jobs=None,

                                              oob_score=False, random_state=1,

                                              verbose=0, warm_start=False)



rfc_tuned.fit(X_train, y_train)

preds_tuned = rfc_tuned.predict(X_test)



score = rfc_tuned.score(X_test, y_test)



print(score)