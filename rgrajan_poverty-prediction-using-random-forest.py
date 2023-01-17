import pandas as pd
df = pd.read_csv('../input/train.csv')
df.info()
import numpy as np

# Calculate ECDF for a series
def ecdf(data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1/n) / n
    return x, y
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
df.v2a1.isnull().sum()
df[(df.v2a1.isnull()) & (df['tipovivi3'] == 1)]['v2a1'].sum()
df.v2a1.fillna(0, inplace=True)
x_ep, y_ep = ecdf(df[df['Target']==1].v2a1)
x_mp, y_mp = ecdf(df[df['Target']==2].v2a1)
x_vh, y_vh = ecdf(df[df['Target']==3].v2a1)
x_nh, y_nh = ecdf(df[df['Target']==4].v2a1)

plt.figure(figsize=(15,8))
plt.plot(x_ep, y_ep, marker = '.', linestyle='none')
plt.plot(x_mp, y_mp, marker = '.', linestyle='none')
plt.plot(x_vh, y_vh, marker = '.', linestyle='none')
plt.plot(x_nh, y_nh, marker = '.', linestyle='none', color='y')

plt.legend(('Extreme Poverty', 'Moderate Poverty', 'Vulnerable Household', 'Non-vulnerable Household'))


plt.margins(0.02)
plt.xlabel('Rent')
plt.ylabel('ECDF')
plt.show()
df[df['v2a1'] > 1000000].head()
df = df[df['v2a1'] < 1000000]
df.v18q1.fillna(0, inplace=True)
df.meaneduc.fillna(df.SQBmeaned, inplace=True)
df.meaneduc.fillna(0, inplace=True)
df['meaneduc'] =  pd.to_numeric(df['meaneduc'])
df.rez_esc.fillna(0, inplace=True)
df.dependency.fillna(df.SQBdependency, inplace=True)
for item in df['idhogar'].unique():
    df_household = df[df['idhogar'] == item]
    head_target = df_household[df_household['parentesco1'] == 1]['Target'].values
    
    for index, row in df_household.iterrows():
        if (row['Target'] != head_target):
            df.loc[df['Id']==row['Id'], 'Target'] = head_target
def pearson_r(x, y):
    corr_mat = np.corrcoef(x, y)
    return corr_mat[0,1]
for col in df.columns:
    if ((df[col].dtype != 'str') & (df[col].dtype != 'object')) :
        print('Column : {0}, Corr : {1}'.format(col, pearson_r(df[col], df.Target)))
from sklearn.model_selection import train_test_split

X = df[['v2a1','rooms','refrig','v18q','v18q1','r4h2', 'escolari', 'paredblolad','pisomoscer','cielorazo','energcocinar2',
         'elimbasu1', 'epared3', 'etecho3','eviv3','estadocivil3','hogar_adul','meaneduc','instlevel8','bedrooms','tipovivi2',
              'computer','television','qmobilephone','lugar1','age']]
y= df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=0, oob_score=True, n_jobs=-1)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start=100, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_sample_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators' : n_estimators,
               'max_features' : max_features,
               'max_depth' : max_depth,
               'min_samples_split' : min_samples_split,
               'min_samples_leaf': min_sample_leaf,
               'bootstrap' : bootstrap
              }

rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv=3, 
                               verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(X_train, y_train)
rf_random.best_params_
from sklearn.ensemble import RandomForestClassifier
model_random = rf_random.best_estimator_
model_random.fit(X_train, y_train)
predictions_random = model_random.predict(X_test)
print(classification_report(y_test, predictions_random))
from sklearn.model_selection import GridSearchCV

param_grid = {
    'bootstrap' : [False],
    'max_depth' : [30, 40, 50, 55],
    'max_features' : ['auto'],
    'min_samples_leaf' : [1, 2, 3],
    'min_samples_split' : [2, 3, 4],
    'n_estimators' : [1200, 1300, 1350, 1375]
    
}

rf = RandomForestClassifier()
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv=3, n_jobs = -1, verbose = 2)

grid_search.fit(X_train, y_train)
grid_search.best_params_
model_grid = grid_search.best_estimator_
model_grid.fit(X_train, y_train)
predictions_grid = model_grid.predict(X_test)
print(classification_report(y_test, predictions_grid))
df_test = pd.read_csv('../input/test.csv')
df_test.v2a1.fillna(0, inplace=True)
df_test.v18q1.fillna(0, inplace=True)
df_test.meaneduc.fillna(df.SQBmeaned, inplace=True)
df_test.meaneduc.fillna(0, inplace=True)
df_test['meaneduc'] =  pd.to_numeric(df_test['meaneduc'])
df_test.rez_esc.fillna(0, inplace=True)
df_test.dependency.fillna(df.SQBdependency, inplace=True)
ids = df_test['Id']
test_features = df_test[['v2a1','rooms','refrig','v18q','v18q1','r4h2', 'escolari', 'paredblolad','pisomoscer','cielorazo','energcocinar2',
         'elimbasu1', 'epared3', 'etecho3','eviv3','estadocivil3','hogar_adul','meaneduc','instlevel8','bedrooms','tipovivi2',
              'computer','television','qmobilephone','lugar1','age']]
test_pred = model_grid.predict(test_features)
submit = pd.DataFrame({'Id' : ids, 'Target' : test_pred})
submit.to_csv('submit.csv', index=False)
