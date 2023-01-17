import collections
import numpy as np
import os
import pandas as pd

from decimal import Decimal
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from matplotlib import pyplot as plt
# from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
import seaborn as sns

# function file to data frame
def file_to_df(file):
    filename, file_extension = os.path.splitext(file)
    if file_extension=='.csv':
        df = pd.read_csv(file, sep=',', header=0)
    elif file_extension=='.tsv':
        df = pd.read_csv(file, sep='\t', header=0)
    else:
        print('Please provide csv or tsv file format.')
    return df

df_features_init = file_to_df('../input/data-science-for-good/2016 School Explorer.csv')
df_target_init = file_to_df('../input/augmented-d5-shsat-2/augmented_D5_SHSAT_Registrations_and_Testers.csv')
# subset to numeric only plus school ID
df_features_init.rename(columns={'Location Code':'DBN'}, inplace=True)
colnames = list(df_features_init)

df_features = pd.DataFrame()
df_features['DBN'] = df_features_init['DBN']
for i in range(0,len(colnames)):
    if is_numeric_dtype(df_features_init[colnames[i]]):
        df_features[colnames[i]] = df_features_init[colnames[i]]
# match features and targets
DBN_list = df_target_init['DBN'].unique()
df_features = df_features[df_features['DBN'].isin(DBN_list)]

DBN_list = df_features['DBN'].unique()
df_target = df_target_init.copy()
df_target = df_target[df_target['DBN'].isin(DBN_list)]

# still not fitting, due to year inconsistencies
print(df_features.shape)
print(df_target.shape)
print(df_target.shape[0]/df_features.shape[0]*1.)
df_var = df_target.sort_values(by=['DBN'],ascending=True)
# for i in range(0,df_var.shape[0]):
#     print(df_var['year'].iloc[i])

# check if constant school performance over years
years = ['2013','2014','2015','2016']
x = range(0,len(DBN_list))
y = []
z = []

for i in range(0,len(DBN_list)):
    df_var = (df_target[df_target['DBN'] == DBN_list[i]])
    y.append(df_var['register_percentile'].max() - df_var['register_percentile'].min())
    z.append(df_var['took_test_percentile'].max() - df_var['took_test_percentile'].min())
    # print(str(round(var,3))+'    '+DBN_list[i])
y.sort()
z.sort()

plt.scatter(x, y, label='Q(Registered)')
plt.scatter(x, z, label='Q(Participated)')
plt.xlabel('Arbitraty school number ', fontsize=16)
plt.ylabel('Maximal percentile change ', fontsize=16)
plt.legend(loc = "lower right", ncol=1, prop={'size':12})
plt.show()

# match features and targets for 2016
DBN_list = df_target_init['DBN'].unique()
df_features = df_features[df_features['DBN'].isin(DBN_list)]

DBN_list = df_features['DBN'].unique()
df_target = df_target_init.copy()
df_target = df_target[df_target['year'] == 2016]
df_target = df_target[df_target['DBN'].isin(DBN_list)]

df_var = df_target.sort_values(by=['DBN'],ascending=True)
# for i in range(0,df_var.shape[0]):
#     print(str(df_var['DBN'].iloc[i])+'   '+str(df_var['grade'].iloc[i])+'   '+str(df_var['register_percentile'].iloc[i])+'   '+str(df_var['took_test_percentile'].iloc[i]))
# still not fitting, due to grade inconsistencies
print(df_features.shape)
print(df_target.shape)
# print(df_target.shape[0]/df_features.shape[0]*1.)

# check if constant school performance over grades
x = range(0,len(DBN_list))
y = []
z = []

for i in range(0,len(DBN_list)):
    df_var = (df_target[df_target['DBN'] == DBN_list[i]])
    y.append(df_var['register_percentile'].max() - df_var['register_percentile'].min())
    z.append(df_var['took_test_percentile'].max() - df_var['took_test_percentile'].min())

y.sort()
z.sort()

plt.scatter(x, y, label='Q(Registered)')
plt.scatter(x, z, label='Q(Participated)')
plt.xlabel('Arbitraty school number ', fontsize=16)
plt.ylabel('Maximal percentile change ', fontsize=16)
plt.legend(loc = "lower right", ncol=1, prop={'size':12})
plt.show()

# Do all the schools have grade 8?
print(list(df_target))
df_target = df_target[df_target['grade'] == 8]
# print(df_target['grade'])
print(df_features.shape)
print(df_target.shape)
print(df_target.shape[0]/df_features.shape[0]*1.)

# yes!

# reload the data and select 8th graders
df_features_init = file_to_df('../input/data-science-for-good/2016 School Explorer.csv')
df_target_init = file_to_df('../input/augmented-d5-shsat-2/augmented_D5_SHSAT_Registrations_and_Testers.csv')

# subset to numeric only plus school ID
df_features_init.rename(columns={'Location Code':'DBN'}, inplace=True)
colnames = list(df_features_init)

df_features = pd.DataFrame()
df_features['DBN'] = df_features_init['DBN']
for i in range(0,len(colnames)):
    if is_numeric_dtype(df_features_init[colnames[i]]):
        df_features[colnames[i]] = df_features_init[colnames[i]]

# match features and targets
DBN_list = df_target_init['DBN'].unique()
df_features = df_features[df_features['DBN'].isin(DBN_list)]

DBN_list = df_features['DBN'].unique()
df_target = df_target_init.copy()
df_target = df_target[df_target['DBN'].isin(DBN_list)]

# Select 8th graders
df_target = df_target[df_target['grade'] == 8]

# check if constant school performance over years
years = ['2013','2014','2015','2016']
x = range(0,len(DBN_list))
y = []
z = []

for i in range(0,len(DBN_list)):
    df_var = (df_target[df_target['DBN'] == DBN_list[i]])
    y.append(df_var['register_percentile'].max() - df_var['register_percentile'].min())
    z.append(df_var['took_test_percentile'].max() - df_var['took_test_percentile'].min())
    # print(str(round(var,3))+'    '+DBN_list[i])
y.sort()
z.sort()

plt.scatter(x, y, label='Q(Registered)')
plt.scatter(x, z, label='Q(Participated)')
plt.xlabel('Arbitraty school number ', fontsize=16)
plt.ylabel('Maximal percentile change (8th)', fontsize=16)
plt.legend(loc = "lower right", ncol=1, prop={'size':12})
plt.show()

print(df_features.shape)
print(df_target.shape)

# match features and targets for 2016 and 8th graders
DBN_list = df_target_init['DBN'].unique()
df_features = df_features[df_features['DBN'].isin(DBN_list)]

DBN_list = df_features['DBN'].unique()
df_target = df_target_init.copy()
df_target = df_target[df_target['year'] == 2016]
df_target = df_target[df_target['grade'] == 8]
df_target = df_target[df_target['DBN'].isin(DBN_list)]

df_var = df_target.sort_values(by=['DBN'],ascending=True)

print(df_features.shape)
print(df_target.shape)
print(df_target.shape[0]/df_features.shape[0]*1.)
# match school id and sort row numbers
df_target = df_target.sort_values(by=['DBN'],ascending=True)
df_target = df_target.reset_index(drop=True)
df_features = df_features.sort_values(by=['DBN'],ascending=True)
df_features = df_features.reset_index(drop=True)

# print(df_features.shape)
# print(list(df_features))


# Step 1: Normalize test takers etc. to the number of students in the school
df_var = df_features.copy()
df_ft = df_features.copy()

grades = [3,4,5,6,7,8]
test_types = ['ELA','Math|math']
for i in range(0,len(grades)):
    for j in range(0,len(test_types)):
        df_var = df_features.copy()
        # grade
        colnames = list(df_var.filter(regex=str(grades[i])+' ').columns)
        df_var = df_var[colnames]
    
        # test type
        colnames = list(df_var.filter(regex=test_types[j]).columns)
        df_var = df_var[colnames]
    
        # number of pupils tested
        all_colname = list(df_var.filter(regex='Tested|tested').columns)
        df_all = df_var[all_colname]
        df_var = df_var.drop(all_colname, axis=1)
        colnames = list(df_var)
    
        # normalize
        for col in range(0,len(colnames)):
            df_var[colnames[col]] = df_var[colnames[col]]/df_all[all_colname[0]]*1.0
        df_var = df_var.fillna(0)
    
        # substitute normalized results into original df
        df_ft[colnames] = df_var[colnames]
        

# Step 2: Now, standardize all features:
df_var = df_ft.copy()
df_var = df_var.drop(df_var[['DBN']], axis=1)
feature_names = list(df_var.columns)
scaler = preprocessing.StandardScaler()
df_var = pd.DataFrame(scaler.fit_transform(df_var))
df_var.columns = feature_names

df_features = df_var
# print(df_var.head(4))

# Step 3: Remove features with constant values
def variance_threshold_select(df, thresh=0.0, na_replacement=-999):
    df1 = df.copy(deep=True) # Make a deep copy of the dataframe
    selector = VarianceThreshold(thresh)
    selector.fit(df1.fillna(na_replacement)) # Fill NA values as VarianceThreshold cannot deal with those
    df2 = df.loc[:,selector.get_support(indices=False)] # Get new dataframe with columns deleted that have NA values
    return df2

df_features = variance_threshold_select(df_features)


# Pearson correlation of coefficients
corr = df_features.corr(method='pearson')**2
corr.columns = corr.columns.str.replace('_', ' ')
corr.index = corr.index.str.replace('_', ' ')
corr = corr.abs()
fig = plt.figure(figsize=(20, 20))
sns.heatmap(corr, cmap="Blues", square=True)
plt.title(r'Pearson correlation')
plt.tight_layout()
plt.show()

# Target: register_percentile
X = df_features.iloc[:, 6:].astype(float)
y = df_target.iloc[:, -2:].astype(float)
y_idx = 0
y = y.iloc[:,y_idx]
print('Target: '+str(y.name))

# Loop through regularization strength on a log scale and run LASSO.
print('>>> Testing regularization strength')
strength = 10 ** np.linspace(-3, 1., 7)
strength.sort()
for alpha in strength:
    clf = linear_model.Lasso(alpha=alpha,max_iter=100000000)
    clf.fit(X, y)
    print(str('%.0E' % Decimal(alpha)), round(clf.score(X, y),3))
    
# Fit and predict with reasonable alpha
alpha = 5E-01
print('>>> LASSO with alpha = '+str(alpha))
clf = linear_model.Lasso(alpha=alpha,max_iter=100000000)
clf.fit(X, y)
r_value = clf.score(X, y)
y_hat = clf.predict(X)
res = y_hat - y
mae = np.mean(np.abs(res))

# parity plot
axmin = 0
axmax = 100
msg = "$MAE$ = " + str(round(mae,3)) + '\n $R^2$ = ' + str(round(r_value, 3))
plt.plot([-1, 130], [-1,130], color='black', lw=1.)
plt.scatter(y_hat, y)
plt.text(axmin*0.8+0.1*axmax, axmax*0.7, msg)
plt.title('LASSO on registration percentiles', fontsize = 14)
plt.xlabel(r'Predicted registration percentile ($\hat{y}$)', fontsize=16)
plt.ylabel(r'Actual registration percentile', fontsize=16)

plt.xlim((0,100))
plt.ylim((0,100))
plt.show()

# Descriptor relevance
features = list(X)
print("R2 = " + str(round(clf.score(X, y),3)))
coefficients = np.round(clf.coef_,3)

coeff_dict = dict(zip(features, coefficients))
coeff_rank = sorted(coeff_dict, key=lambda dict_key: abs(coeff_dict[dict_key]),reverse=True)
best_features = []
# for i in range(0,len(coeff_rank)):
for i in range(0,20):
    print(str(coeff_rank[i]+'    '+str(coeff_dict[coeff_rank[i]])))
    best_features.append(str(coeff_rank[i]))

best_num = 8
best_features = best_features[0:best_num]

# Manual feature selection
X_best =  X[best_features]
print(list(X_best))

# Pearson correlation of coefficients
corr = X_best.corr(method='pearson')**2
corr.columns = corr.columns.str.replace('_', ' ')
corr.index = corr.index.str.replace('_', ' ')
corr = corr.abs()
fig = plt.figure(figsize=(8, 8))
sns.heatmap(corr, cmap="Blues", square=True)
plt.title(r'Pearson correlation')
plt.tight_layout()
plt.show()

# Manual feature selection
best_features = ['Grade 7 Math 4s - Economically Disadvantaged', 'Grade 8 ELA - All Students Tested', 'Grade 6 Math 4s - Limited English Proficient',
                 'Grade 3 Math 4s - Black or African American', 'Grade 7 ELA 4s - Black or African American',  'Grade 7 Math - All Students Tested']
X_best =  X[best_features]
print(list(X_best))

# Linear regression
clf = linear_model.LinearRegression()
clf.fit(X_best, y)
r_value = clf.score(X_best, y)
y_hat = clf.predict(X_best)
res = y_hat - y
mae = np.mean(np.abs(res))

# Parity plot
msg = "$MAE$ = " + str(round(mae,3)) + '\n $R^2$ = ' + str(round(r_value, 3))
axmin = 0
axmax = 100
plt.plot([axmin,axmax], [axmin,axmax], color='black', lw=1.)
plt.scatter(y_hat, y)
plt.xlabel(r'Predicted value ($\hat{y}$)')
plt.ylabel(r'True value')
plt.text(axmin*0.8+0.1*axmax, axmax*0.7, msg)
plt.title('Plain least squares fitting with top '+str(best_num)+' descriptors', fontsize = 14)
plt.xlim((axmin, axmax))
plt.ylim((axmin, axmax))
plt.show()

# sensitivity analysis
features = best_features
print("R2 = " + str(round(clf.score(X_best, y),3)))
coefficients = np.round(clf.coef_,3)

coeff_dict = dict(zip(features, coefficients))
coeff_rank = sorted(coeff_dict, key=lambda dict_key: abs(coeff_dict[dict_key]),reverse=True)
print('Coefficients: ')
for i in range(0,len(coeff_rank)):
    print(str(coeff_rank[i]+'    '+str(coeff_dict[coeff_rank[i]])))





# Manual feature selection
X_best =  X[best_features]
print(list(X_best))

# Pearson correlation of coefficients
corr = X_best.corr(method='pearson')**2
corr.columns = corr.columns.str.replace('_', ' ')
corr.index = corr.index.str.replace('_', ' ')
corr = corr.abs()
fig = plt.figure(figsize=(8, 8))
sns.heatmap(corr, cmap="Blues", square=True)
plt.title(r'Pearson correlation')
plt.tight_layout()
plt.show()

