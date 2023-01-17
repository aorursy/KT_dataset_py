# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from pandas.plotting import scatter_matrix

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import MinMaxScaler, FunctionTransformer, OneHotEncoder, KBinsDiscretizer, MaxAbsScaler, LabelEncoder

from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split as split

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.mixture import GaussianMixture

from time import time



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/employee-compensation.csv')
df.info()
df.head()
#Checking for nulls

print(df.isnull().sum())

df[df.Union.isnull() == True].head()
#Checking what are the jobs that have nulls

df[df.Union.isnull() == True]['Organization Group'].value_counts()

df[df.Union.isnull() == True][df['Organization Group'] == 'Community Health'].Job.value_counts()

df[df.Job == 'Technology Expert II'].shape
sns.heatmap(df.corr())
salaries_sm = scatter_matrix(df[['Salaries', 'Total Salary', 'Total Compensation']])

#['Salaries', 'Overtime', 'Other Salaries', 'Total Salary', 'Retirement', 'Health/Dental', 'Other Benefits', 'Total Benefits', 'Total Compensation']
benefits_sm = scatter_matrix(df[['Retirement', 'Health/Dental', 'Other Benefits', 'Salaries']])
ax = sns.kdeplot(df['Salaries'])

ax#.set_xlim(20000,50000)
df['Salaries'].describe()
salary = ['Salaries', 'Total Salary', 'Total Compensation']



for col in salary:

    ax_salary = sns.kdeplot(df[col])

    ax_salary#.set_xlim(-25000,75000)
benefits = ['Retirement', 'Health/Dental', 'Other Benefits', 'Total Benefits'] 



for col in benefits:

    ax_benefits = sns.kdeplot(df[col])

    ax_benefits#.set_xlim(-25000,75000)
# Removing Salaries lower than 35000 

df[df['Salaries'] < 35000].count()

#ax.set_xlim(20000,50000)

#df['Job'][df.Salaries < 0].value_counts()
df2 = df[df['Salaries'] > 35000]
# What and how many organizations we are losing by reducing the data to salaries < 35,000$

org_x = df[df.Salaries<35000]['Organization Group'].value_counts()

org_y = df['Organization Group'].value_counts()

org_z = pd.concat([org_x, (org_x/org_y)], axis=1, join='inner', sort=False)

org_z.columns = ['Organization Count', 'Organization %']

org_z
org_ax = org_z['Organization Count'].plot('bar')

for p in org_ax.patches:

    org_ax.annotate(int(p.get_height()), (p.get_x(), p.get_height() * 1.01))
# What and how many departments we are losing by reducing the data to salaries < 35,000$

dep_x = df[df.Salaries<35000].Department.value_counts()

dep_y = df.Department.value_counts()

dep_z = pd.concat([dep_x, (dep_x/dep_y)], axis=1, join='inner', sort=False)

dep_z.columns = ['Department Count', 'Department %']

dep_z.head(10)
# What and how many jobs we are losing by reducing the data to salaries < 35,000$

job_x = df[df.Salaries<35000].Job.value_counts()

job_y = df.Job.value_counts()

job_z = pd.concat([job_x, (job_x/job_y)], axis=1, join='inner', sort=False)

job_z.columns = ['Job Count', 'Job %']

job_z.head(10)
plt.figure(figsize=(15,8))

for col in list(df2['Organization Group'].unique()):

    ax = sns.kdeplot(df2['Salaries'][df2['Organization Group'] == col], label = col)

#ax.set_ylim(0,0.0001)
plt.figure(figsize=(15,8))

for col in list(df2['Year'].unique()):

    sns.kdeplot(df2['Salaries'][df2['Year'] == col], label = col)
plt.figure(figsize=(15,8))

for col in list(df2['Organization Group'].unique()):

    sns.kdeplot(df2['Overtime'][df2['Organization Group'] == col], label = col).set_xlim(-25000,25000)
df_groupby_Job = df.groupby('Job').mean()

df_groupby_Job['Total Compensation'].sort_values(ascending = False).head(5).plot('bar')
df_groupby_Job['Total Compensation No Extras'] = df_groupby_Job['Total Compensation'] - df_groupby_Job['Overtime'] - df_groupby_Job['Other Salaries']

df_groupby_Job['Total Compensation No Extras'].sort_values(ascending = False).head(5).plot('bar')
df2['Overtime Amount'] = pd.Series(df2.Overtime / df2.Salaries)

#df2['Overtime Amount'].sort_values(ascending = False)#.hist()



df2['Overtime Amount'] = pd.cut(df2['Overtime Amount'], 5, labels = ['few overtime', 'less than average overtime', 'average overtime',

                                                                     'more than average overtime', 'many overtime'])

#df2['Overtime Amount']
df2['Retirement Amount'] = pd.Series(df2.Retirement / df2.Salaries)

#df2['Retirement Amount'].sort_values(ascending = False)#.hist()



df2['Retirement Amount'] = pd.cut(df2['Retirement Amount'], 5, labels = ['few retirement', 'less than average retirement', 'average retirement',

                                                                     'more than average retirement', 'many retirement'])

#df2['Retirement Amount']
df2['Health/Dental Amount'] = pd.Series(df2['Health/Dental'] / df2.Salaries)

#df2['Health/Dental Amount'].sort_values(ascending = False)#.hist()



df2['Health/Dental Amount'] = pd.cut(df2['Health/Dental Amount'], 5, labels = ['few Health/Dental', 'less than average Health/Dental', 

                                                                               'average Health/Dental', 'more than average Health/Dental', 

                                                                               'many Health/Dental'])

#df2['Health/Dental Amount']
df2['Other Benefits Amount'] = pd.Series(df2['Other Benefits'] / df2.Salaries)

#df2['Other Benefits Amount'].sort_values(ascending = False)#.hist()



df2['Other Benefits Amount'] = pd.cut(df2['Other Benefits Amount'], 5, labels = ['few Other Benefits', 'less than average Other Benefits', 

                                                                                 'average Other Benefits', 'more than average Other Benefits', 

                                                                                 'many Other Benefits'])

#df2['Other Benefits Amount']
df2.head()
X = df2.drop(['Year Type', 'Organization Group', 'Department', 'Union', 'Job Family', 'Job', 'Employee Identifier', 'Salaries', 'Overtime', 

              'Other Salaries', 'Total Salary', 'Retirement', 'Health/Dental', 'Other Benefits', 'Total Benefits', 'Total Compensation'

             ], axis = 1)

y = df2.Salaries

print(X.shape,

      y.shape)
X.head()
#Set a pipeline for filling the nulls

def get_Job_Family_Code(df):

    return df[['Job Family Code']]



FullTransformerJobFamilyCode  = Pipeline([("Select_Columns",  FunctionTransformer(func=get_Job_Family_Code, validate=False)),

                                          ("Fill_Null", SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='No Value')),

                                          ("One_Hot_Encoder", OneHotEncoder(sparse = False, handle_unknown='ignore'))

                                         ])

#pd.DataFrame(FullTransformerJobFamilyCode.fit_transform(X[:6950], y[:6950]))
#Set a pipeline for filling the nulls

def get_features_Union_Code(df):

    return df[['Union Code']]



FullTransformerUnionCode  = Pipeline([("Select_Columns",  FunctionTransformer(func=get_features_Union_Code, validate=False)),

                                      ("Fill_Null", SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)),

                                      ("One_Hot_Encoder", OneHotEncoder(sparse = False, handle_unknown='ignore'))

                                     ])

#pd.DataFrame(FullTransformerUnionCode.fit_transform(X[:6950], y[:6950]))
#Set a pipeline for one hot encoding

def get_features_onehot(df):

    return df.drop(['Union Code', 'Job Family Code'], axis=1)



FullTransformerOneHotEncoding  = Pipeline([("Select_Columns",  FunctionTransformer(func=get_features_onehot, validate=False)),

                                           ("One_Hot_Encoder", OneHotEncoder(sparse = False, handle_unknown='ignore'))])



#pd.DataFrame(FullTransformerOneHotEncoding.fit_transform(X[:7000], y[:7000]))
FeatureUnionTransformer = FeatureUnion([("FTJobFamilyCode",  FullTransformerJobFamilyCode),

                                        ("FTUnionCode",      FullTransformerUnionCode),

                                        ("FTOneHotEncoding", FullTransformerOneHotEncoding)

                                       ])



#pd.DataFrame(FeatureUnionTransformer.fit_transform(X[:7000], y[:7000]))
X_train, X_test, y_train, y_test = split(X, y)

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
lr = LinearRegression()
FeatureUnionTransformer.fit_transform(X_train)

X_train_transformed = FeatureUnionTransformer.transform(X_train)

X_test_transformed = FeatureUnionTransformer.transform(X_test)
scores = cross_val_score(lr, X_train_transformed, y_train, 

                         scoring='neg_mean_squared_error', cv=10)
def show_results(scores):

    scores_ = (-scores)**0.5

    print(scores_)

    print("Mean:", scores_.mean())

    print("Std:", scores_.std()) 
show_results(scores)
y_train_pred = lr.fit(X_train_transformed, y_train).predict(X_train_transformed)
plt.plot(y_train, y_train_pred, '.', label='Data')

plt.plot([0, 500000], [0, 500000], label='Ideal')

plt.axes().set_aspect('equal')

plt.legend()
index = (((y_train.values) - y_train_pred)**2).argmax()

index
y_train_pred[index]
X_train.iloc[index]
y_pred = lr.fit(X_train_transformed, y_train).predict(X_test_transformed)
plt.plot(y_test, y_pred, '.', label='Data')

plt.plot([0, 500000], [0, 500000], label='Ideal')

plt.axes().set_aspect('equal')

plt.legend()

plt.xlim(0,500000)

plt.ylim(0,500000)
dt = DecisionTreeRegressor(max_depth=5)
scores = cross_val_score(dt, X_train_transformed, y_train, 

                         scoring='neg_mean_squared_error', cv=10)
show_results(scores)
y_train_pred = dt.fit(X_train_transformed, y_train).predict(X_train_transformed)
plt.plot(y_train, y_train_pred, '.', label='Data')

plt.plot([0, 500000], [0, 500000], label='Ideal')

plt.axes().set_aspect('equal')

plt.legend()
y_pred = dt.fit(X_train_transformed, y_train).predict(X_test_transformed)
plt.plot(y_test, y_pred, '.', label='Data')

plt.plot([0, 500000], [0, 500000], label='Ideal')

plt.axes().set_aspect('equal')

plt.legend()
rf = RandomForestRegressor(max_depth=20, n_estimators=10, n_jobs=-1)
t1 = time()



scores = cross_val_score(rf, X_train_transformed, y_train, 

                         scoring='neg_mean_squared_error', cv=10)



print ("Calculation time: {:.2f} [sec]".format(time()-t1))
show_results(scores)
y_train_pred = rf.fit(X_train_transformed, y_train).predict(X_train_transformed)
plt.plot(y_train, y_train_pred, '.', label='Data')

plt.plot([0, 500000], [0, 500000], label='Ideal')

plt.axes().set_aspect('equal')

plt.legend()
y_pred = rf.fit(X_train_transformed, y_train).predict(X_test_transformed)
plt.plot(y_test, y_pred, '.', label='Data')

plt.plot([0, 500000], [0, 500000], label='Ideal')

plt.axes().set_aspect('equal')

plt.legend()
df3 = df2.copy()
df3.isnull().sum()
# Filling the nulls

df3['Union Code'] = df3['Union Code'].fillna(0)

df3[['Union', 'Job Family Code', 'Job Family']] = df3[['Union', 'Job Family Code', 'Job Family']].fillna('No Value')

df3.isnull().sum()
#Get dummies

df3 = pd.get_dummies(df3, columns=['Year', 'Organization Group Code', 'Department Code', 'Union Code', 'Job Family Code', 'Job Code'])
df3.head()
X_og = df3.drop(['Year Type', 'Organization Group', 'Department', 'Union', 'Job Family', 'Job', 'Employee Identifier', 'Salaries', 'Total Salary',

                 'Total Benefits', 'Total Compensation', 'Overtime Amount', 'Retirement Amount', 'Health/Dental Amount', 'Other Benefits Amount'

                ], axis = 1)

y_og = df3.Salaries

print(X_og.shape,

      y_og.shape)
X_og.head()
X_train_og, X_test_og, y_train_og, y_test_og = split(X_og, y_og)

print(X_train_og.shape, y_train_og.shape)

print(X_test_og.shape, y_test_og.shape)
scores = cross_val_score(lr, X_train_og, y_train_og, 

                         scoring='neg_mean_squared_error', cv=10)
show_results(scores)
y_train_pred_og = lr.fit(X_train_og, y_train_og).predict(X_train_og)
plt.plot(y_train_og, y_train_pred_og, '.', label='Data')

plt.plot([0, 400000], [0, 400000], label='Ideal')

plt.axes().set_aspect('equal')

plt.legend()
y_pred_og = lr.fit(X_train_og, y_train_og).predict(X_test_og)
plt.plot(y_test_og, y_pred_og, '.', label='Data')

plt.plot([0, 400000], [0, 400000], label='Ideal')

plt.axes().set_aspect('equal')

plt.legend()
dt = DecisionTreeRegressor(max_depth=10)
t1 = time()



scores = cross_val_score(dt, X_train_og, y_train_og, 

                         scoring='neg_mean_squared_error', cv=10)



print ("Calculation time: {:.2f} [sec]".format(time()-t1))
show_results(scores)
y_train_pred_og = dt.fit(X_train_og, y_train_og).predict(X_train_og)
plt.plot(y_train_og, y_train_pred_og, '.', label='Data')

plt.plot([0, 400000], [0, 400000], label='Ideal')

plt.axes().set_aspect('equal')

plt.legend()
y_pred_og = dt.fit(X_train_og, y_train_og).predict(X_test_og)
plt.plot(y_test_og, y_pred_og, '.', label='Data')

plt.plot([0, 400000], [0, 400000], label='Ideal')

plt.axes().set_aspect('equal')

plt.legend()
rf = RandomForestRegressor(max_depth=20, n_estimators=10, n_jobs=-1)
t1 = time()



scores = cross_val_score(rf, X_train_og, y_train_og, 

                         scoring='neg_mean_squared_error', cv=10)



print ("Calculation time: {:.2f} [sec]".format(time()-t1))
show_results(scores)
y_train_pred_og = rf.fit(X_train_og, y_train_og).predict(X_train_og)
plt.plot(y_train_og, y_train_pred_og, '.', label='Data')

plt.plot([0, 500000], [0, 500000], label='Ideal')

plt.axes().set_aspect('equal')

plt.legend()
y_pred_og = rf.fit(X_train_og, y_train_og).predict(X_test_og)
plt.plot(y_test_og, y_pred_og, '.', label='Data')

plt.plot([0, 500000], [0, 500000], label='Ideal')

plt.axes().set_aspect('equal')

plt.legend()
#def show_results_scores(scores):

#    scores_ = (-scores)**0.5

#    return scores_.mean()

#

#t1 = time()

#

#max_depths = np.linspace(1, 25, 25, endpoint=True)

#train_results = []

#for max_depth in max_depths:

#    dt = DecisionTreeRegressor(max_depth=max_depth)

#    # Add score to previous train results

#    scores = cross_val_score(dt, X_train_og, y_train_og, scoring='neg_mean_squared_error', cv=10)

#    train_results.append(show_results_scores(scores))

#print ("Calculation time: {:.2f} [sec]".format(time()-t1))

#    

#from matplotlib.legend_handler import HandlerLine2D

#line1, = plt.plot(max_depths, train_results, 'b', label="Train Score")

#plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

#plt.ylabel('Score')

#plt.xlabel('Tree depth')

#plt.show()
#t1 = time()

#

#min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)

#train_results = []

#for min_samples_split in min_samples_splits:

#    dt = DecisionTreeRegressor(min_samples_split=min_samples_split)

#    # Add score to previous train results

#    scores = cross_val_score(dt, X_train_og, y_train_og, scoring='neg_mean_squared_error', cv=10)

#    train_results.append(show_results_scores(scores))

#print ("Calculation time: {:.2f} [sec]".format(time()-t1))

#

#from matplotlib.legend_handler import HandlerLine2D

#line1, = plt.plot(min_samples_splits, train_results, 'b', label="Train Score")

#plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

#plt.ylabel('Score')

#plt.xlabel('min samples split')

#plt.show()
#t1 = time()

#dt_gs = GridSearchCV(dt, 

#                      param_grid={'max_depth': range(1, 21),

#                                  'min_samples_split': range(5, 31),                                  

#                                  'min_samples_leaf': range(5, 31)},

#                      cv=10)

#dt_gs.fit(X_train_og, y_train_og)

#print ("Calculation time: {:.2f} [sec]".format(time()-t1))

#print ("Best parameters:", dt_gs.best_params_)
gmm = GaussianMixture(n_components=4)
clust = np.array(df[['Retirement', 'Health/Dental']])
clust.shape
gmm.fit(clust)
gmm.means_

gmm.covariances_

#gmm.predict(X)
plt.scatter(clust[:,0], clust[:,1], s=2, c= gmm.predict(clust))

plt.scatter(gmm.means_[:, 0], gmm.means_[:,1])

plt.show()
gmm = GaussianMixture(n_components=8)
gmm.fit(clust)
plt.scatter(clust[:,0], clust[:,1], s=2, c= gmm.predict(clust))

plt.scatter(gmm.means_[:, 0], gmm.means_[:,1])

plt.show()
clust_compensation = np.array(X_og[['Overtime', 'Other Salaries', 'Retirement', 'Health/Dental', 'Other Benefits']])
gmm = GaussianMixture(n_components=4)
clust_compensation_minmax = MinMaxScaler().fit_transform(clust_compensation)

#pd.DataFrame(clust_compensation_minmax)
gmm.fit(clust_compensation_minmax)

#gmm.predict(clust_compensation)
X_og['Clust Compensation'] = gmm.predict(clust_compensation_minmax)
X_og = pd.get_dummies(X_og, columns=['Clust Compensation'])
X_og.head()
X_train_og, X_test_og, y_train_og, y_test_og = split(X_og, y_og)

print(X_train_og.shape, y_train_og.shape)

print(X_test_og.shape, y_test_og.shape)
scores = cross_val_score(lr, X_train_og, y_train_og, 

                         scoring='neg_mean_squared_error', cv=10)



show_results(scores)
y_train_pred_og = lr.fit(X_train_og, y_train_og).predict(X_train_og)
plt.plot(y_train_og, y_train_pred_og, '.', label='Data')

plt.plot([0, 400000], [0, 400000], label='Ideal')

plt.axes().set_aspect('equal')

plt.legend()