# Pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# Matplotlib and seaborn for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Scipy for statistics
from scipy import stats

# os to manipulate files
import os

from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

colors = [ "#3498db", "#e74c3c", "#2ecc71","#9b59b6", "#34495e", "#95a5a6"]
def convert_to_number(col,convert_type=int,changes = ['-']):
    
    # string will be considered as object
    if col.dtype.name == 'object':
        col_temp = col.copy()
        
        # Change any occurence in changes to ''
        for change in changes:
                col_temp = col_temp.str.replace(change,'')
                
        # Changes empty string elements for NaN
        col_temp.loc[(col_temp == '')] = np.nan
        
        # Convert to number the not nan elements
        col_temp[col_temp.notna()] = col_temp[col_temp.notna()].astype(convert_type)
        
        # Fill nan elements with the mean
        col_temp = col_temp.fillna(int(col_temp.mean()))
        
        return col_temp
    else:
        return col
def plot_predictions(X_new,y_new,descr = '',cols = ['area', 'hoa','rent amount','property tax']):
    y_col = 'total'

    #cols = ['area', 'hoa','rent amount','property tax','fire insurance']
    
    k = 0
    for x_col in cols:
        plt.close()
        plt.figure(figsize=(8, 5))
        plt.scatter(X_trn[x_col],y_trn,c='lightgray',label = 'Training Dataset',marker='o',zorder=1)
        plt.scatter(X_new[x_col],y_new, label = 'Predictions on Test Dataset',marker='.', c=colors[k], lw = 0.5,zorder=2,alpha = 0.8)
        #plt.scatter(X_tst[x_col],y_pr_tst, label = 'Predictions',marker='.', c='tab:blue', lw = 0.5,zorder=2)


        plt.xlabel(x_col, size = 18)
        plt.ylabel(y_col, size = 18); 
        plt.legend(prop={'size': 12});
        plt.title(descr+y_col+' vs '+x_col, size = 20);
        #plt.savefig('results/'+descr+y_col+' vs '+x_col+'.png')
        plt.show()
        k += 1
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Import Dataset
df1 = pd.read_csv(os.path.join(dirname,'houses_to_rent.csv')).drop('Unnamed: 0',axis=1)
#df2 = pd.read_csv(os.path.join(dirname,'houses_to_rent_v2.csv'))
#df2.columns = df1.columns
df1.info()
# elements to remove from the dataset
remove = ['R','$',',','-','Sem info','Incluso']

# columns of numerical data
cols = ['hoa','rent amount','property tax','fire insurance','total','floor']

# Making the substitutions
for col in cols:
    df1[col]  = convert_to_number(df1[col],changes=remove)
    
# converting floor to int 
df1['floor'] = df1['floor'].astype('int')

# Getting dummies
df1[['animal','furniture']] = pd.get_dummies(df1[['animal','furniture']], prefix_sep='_', drop_first=True)

# dealing with outliers
cols = ['area','hoa','property tax','rent amount']
for col in cols:
    df1 = df1[np.abs(stats.zscore(df1[col])) < 4]
df1.head(10)
correlations = df1.corr()['total'].abs().sort_values(ascending=False).drop('total',axis=0).to_frame()
correlations.plot(kind='bar');
#totalprice correlation matrix
k = 10 #number of variables for heatmap
plt.figure(figsize=(16,8))
corr = df1.corr()

hm = sns.heatmap(corr, 
                 cbar=True, 
                 annot=True, 
                 square=True, fmt='.2f', 
                 annot_kws={'size': 10}, 
                 yticklabels=corr.columns.values,
                 xticklabels=corr.columns.values,
                 cmap="YlGnBu")
plt.show()
# Selecting features and target
x_col = 'rent amount'
y_col = 'total'

X = df1[[x_col]]
y = df1[y_col]

# splitting
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.333)

# Create regression object
MLR = linear_model.LinearRegression()

poly = PolynomialFeatures(degree=1)
X_trn_pl = poly.fit_transform(X_trn)
X_tst_pl = poly.fit_transform(X_tst)


# Train the model using the training sets
MLR.fit(X_trn_pl,y_trn)

y_pr_tst = MLR.predict(X_tst_pl)
y_pr_trn = MLR.predict(X_trn_pl)

mae = mean_absolute_error(y_tst,y_pr_tst)
r2 = r2_score(y_tst,y_pr_tst)

print('MAE:{:7.2f},{:7.2f}% of mean'.format(mae,100*mae/y_pr_tst.mean()))
print('R2:{:6.3f}'.format(r2))

# Plotting
plt.figure(figsize=(8, 5))
plt.scatter(X_tst,y_tst,c='lightgray',label = 'observations',alpha = 0.6,marker='.',zorder=1)
plt.plot(X_tst,y_pr_tst, label = 'Predictions', c='tab:blue', lw = 3,zorder=2)
plt.xlabel(x_col, size = 18)
plt.ylabel(y_col, size = 18); 
plt.legend(prop={'size': 16});
plt.title(y_col+' vs '+x_col, size = 20);
# Selecting features and target
X = df1[['rent amount','fire insurance']]
y = df1['total']

# splitting
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.333)

# Create regression object
MLR = linear_model.LinearRegression()

poly = PolynomialFeatures(degree=1)
X_trn_pl = poly.fit_transform(X_trn)
X_tst_pl = poly.fit_transform(X_tst)

# Train the model using the training sets
MLR.fit(X_trn_pl,y_trn)

y_pr_tst = MLR.predict(X_tst_pl)
y_pr_trn = MLR.predict(X_trn_pl)

mae = mean_absolute_error(y_tst,y_pr_tst)
r2 = r2_score(y_tst,y_pr_tst)


print('MAE:{:7.2f},{:7.2f}% of mean'.format(mae,100*mae/y_pr_tst.mean()))
print('R2:{:6.3f}'.format(r2))
# Selecting features and target
X = df1[['rent amount','bathroom']]
y = df1['total']

# splitting
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.333)

# Create regression object
MLR = linear_model.LinearRegression()

poly = PolynomialFeatures(degree=1)
X_trn_pl = poly.fit_transform(X_trn)
X_tst_pl = poly.fit_transform(X_tst)

# Train the model using the training sets
MLR.fit(X_trn_pl,y_trn)

y_pr_tst = MLR.predict(X_tst_pl)
y_pr_trn = MLR.predict(X_trn_pl)

mae = mean_absolute_error(y_tst,y_pr_tst)
r2 = r2_score(y_tst,y_pr_tst)


print('MAE:{:7.2f},{:7.2f}% of mean'.format(mae,100*mae/y_pr_tst.mean()))
print('R2:{:6.3f}'.format(r2))
# Selecting features and target
X = df1[['rent amount','property tax']]
y = df1['total']

# splitting
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.333)

# Create regression object
MLR = linear_model.LinearRegression()

poly = PolynomialFeatures(degree=1)
X_trn_pl = poly.fit_transform(X_trn)
X_tst_pl = poly.fit_transform(X_tst)

# Train the model using the training sets
MLR.fit(X_trn_pl,y_trn)

y_pr_tst = MLR.predict(X_tst_pl)
y_pr_trn = MLR.predict(X_trn_pl)

mae = mean_absolute_error(y_tst,y_pr_tst)
r2 = r2_score(y_tst,y_pr_tst)


print('MAE:{:7.2f},{:7.2f}% of mean'.format(mae,100*mae/y_pr_tst.mean()))
print('R2:{:6.3f}'.format(r2))
# Selecting features and target
X = df1[['rent amount','hoa']]
y = df1['total']

# splitting
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.333)

# Create regression object
MLR = linear_model.LinearRegression()

poly = PolynomialFeatures(degree=1)
X_trn_pl = poly.fit_transform(X_trn)
X_tst_pl = poly.fit_transform(X_tst)

# Train the model using the training sets
MLR.fit(X_trn_pl,y_trn)

y_pr_tst = MLR.predict(X_tst_pl)
y_pr_trn = MLR.predict(X_trn_pl)

mae = mean_absolute_error(y_tst,y_pr_tst)
r2 = r2_score(y_tst,y_pr_tst)


print('MAE:{:7.2f},{:7.2f}% of mean'.format(mae,100*mae/y_pr_tst.mean()))
print('R2:{:6.3f}'.format(r2))
# Selecting features and target
X = df1[['rent amount','hoa']]
y = df1['total']

# splitting
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.333)

# Create regression object
MLR = linear_model.LinearRegression()

poly = PolynomialFeatures(degree=2)
X_trn_pl = poly.fit_transform(X_trn)
X_tst_pl = poly.fit_transform(X_tst)

# Train the model using the training sets
MLR.fit(X_trn_pl,y_trn)

y_pr_tst = MLR.predict(X_tst_pl)
y_pr_trn = MLR.predict(X_trn_pl)

mae = mean_absolute_error(y_tst,y_pr_tst)
r2 = r2_score(y_tst,y_pr_tst)


print('MAE:{:7.2f},{:7.2f}% of mean'.format(mae,100*mae/y_pr_tst.mean()))
print('R2:{:6.3f}'.format(r2))
# Selecting features and target
#X = df1.drop('total',axis=1).copy()
X = df1.drop(['total','fire insurance'],axis=1).copy()
y = df1['total'].copy()

# splitting
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.333)

# dictionary
model_mae = {}
model_r2 = {}
# Create regression object
MLR = linear_model.LinearRegression()

poly = PolynomialFeatures(degree=1)
X_trn_pl = poly.fit_transform(X_trn)
X_tst_pl = poly.fit_transform(X_tst)
MLR.fit(X_trn_pl,y_trn)

y_pr_tst = MLR.predict(X_tst_pl)
mae = mean_absolute_error(y_tst,y_pr_tst)
r2 = r2_score(y_tst,y_pr_tst)

model_mae['linear'] = mae
model_r2['linear'] = r2

print('MAE:{:7.2f},{:7.2f}% of mean'.format(mae,100*mae/y_pr_tst.mean()))
print('R2:{:6.3f}'.format(r2))
# Getting a reduced sample to test
size = len(X_tst)
indexes = np.random.choice(len(X_tst), size, replace=False)
X_new = X_tst.iloc[indexes]
y_new = MLR.predict(poly.fit_transform(X_new))

plot_predictions(X_new,y_new,descr = 'Linear Regression: ')
from sklearn.tree import DecisionTreeRegressor

d_tree = DecisionTreeRegressor()
d_tree.fit(X_trn,y_trn)

y_pr_tst = d_tree.predict(X_tst)

mae = mean_absolute_error(y_tst,y_pr_tst)
r2 = r2_score(y_tst,y_pr_tst)


model_mae['tree'] = mae
model_r2['tree'] = r2

print('MAE:{:7.2f},{:7.2f}% of mean'.format(mae,100*mae/y_pr_tst.mean()))
print('R2:{:6.3f}'.format(r2))
# Getting a reduced sample to test
size = len(X_tst)
indexes = np.random.choice(len(X_tst), size, replace=False)
X_new = X_tst.iloc[indexes]
y_new = d_tree.predict(X_new)

plot_predictions(X_new,y_new,descr = 'Decision Tree: ')
from sklearn.ensemble import RandomForestRegressor

rnd_frst = RandomForestRegressor()
rnd_frst.fit(X_trn,y_trn)

y_pr_tst = rnd_frst.predict(X_tst)

mae = mean_absolute_error(y_tst,y_pr_tst)
r2 = r2_score(y_tst,y_pr_tst)


model_mae['rnd_forest'] = mae
model_r2['rnd_forest'] = r2

print('MAE:{:7.2f},{:7.2f}% of mean'.format(mae,100*mae/y_pr_tst.mean()))
print('R2:{:6.3f}'.format(r2))
# Getting a reduced sample to test
size = len(X_tst)
indexes = np.random.choice(len(X_tst), size, replace=False)
X_new = X_tst.iloc[indexes]
y_new = rnd_frst.predict(X_new)

plot_predictions(X_new,y_new,descr = 'Random Forest: ')
print('Minor Error: ',min(model_mae, key=model_mae.get),', (',min(list(model_mae.values())),')')
print('Best R2    : ',max(model_r2, key=model_r2.get),', (',max(list(model_r2.values())),')')
# Selecting features and target
#X = df1.drop('total',axis=1).copy()
X = df1.drop(['total','fire insurance','property tax','rent amount','hoa'],axis=1).copy()
y = df1['total'].copy()

# splitting
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.333)

# dictionary
model_mae = {}
model_r2 = {}
# Create regression object
MLR = linear_model.LinearRegression()

poly = PolynomialFeatures(degree=1)
X_trn_pl = poly.fit_transform(X_trn)
X_tst_pl = poly.fit_transform(X_tst)
MLR.fit(X_trn_pl,y_trn)

y_pr_tst = MLR.predict(X_tst_pl)
mae = mean_absolute_error(y_tst,y_pr_tst)
r2 = r2_score(y_tst,y_pr_tst)

model_mae['linear'] = mae
model_r2['linear'] = r2

print('MAE:{:7.2f},{:7.2f}% of mean'.format(mae,100*mae/y_pr_tst.mean()))
print('R2:{:6.3f}'.format(r2))


# Getting a reduced sample to test
size = len(X_tst)
indexes = np.random.choice(len(X_tst), size, replace=False)
X_new = X_tst.iloc[indexes]
y_new = MLR.predict(poly.fit_transform(X_new))

plot_predictions(X_new,y_new,descr = 'REAL - Linear Regression: ',cols = ['area'])
MLR = linear_model.LinearRegression()

poly = PolynomialFeatures(degree=2)
X_trn_pl = poly.fit_transform(X_trn)
X_tst_pl = poly.fit_transform(X_tst)
MLR.fit(X_trn_pl,y_trn)

y_pr_tst = MLR.predict(X_tst_pl)
mae = mean_absolute_error(y_tst,y_pr_tst)
r2 = r2_score(y_tst,y_pr_tst)

model_mae['quadratic'] = mae
model_r2['quadratic'] = r2

print('MAE:{:7.2f},{:7.2f}% of mean'.format(mae,100*mae/y_pr_tst.mean()))
print('R2:{:6.3f}'.format(r2))

# Getting a reduced sample to test
size = len(X_tst)
indexes = np.random.choice(len(X_tst), size, replace=False)
X_new = X_tst.iloc[indexes]
y_new = MLR.predict(poly.fit_transform(X_new))

plot_predictions(X_new,y_new,descr = 'REAL - Quadratic Regression: ',cols = ['area'])
from sklearn.tree import DecisionTreeRegressor

d_tree = DecisionTreeRegressor()
d_tree.fit(X_trn,y_trn)

y_pr_tst = d_tree.predict(X_tst)

mae = mean_absolute_error(y_tst,y_pr_tst)
r2 = r2_score(y_tst,y_pr_tst)


model_mae['tree'] = mae
model_r2['tree'] = r2## Quadratic Regression

print('MAE:{:7.2f},{:7.2f}% of mean'.format(mae,100*mae/y_pr_tst.mean()))
print('R2:{:6.3f}'.format(r2))

# Getting a reduced sample to test
size = len(X_tst)
indexes = np.random.choice(len(X_tst), size, replace=False)
X_new = X_tst.iloc[indexes]
y_new = d_tree.predict(X_new)

plot_predictions(X_new,y_new,descr = 'REAL - Decision Tree: ',cols = ['area'])
from sklearn.ensemble import RandomForestRegressor

rnd_frst = RandomForestRegressor()
rnd_frst.fit(X_trn,y_trn)

y_pr_tst = rnd_frst.predict(X_tst)

mae = mean_absolute_error(y_tst,y_pr_tst)
r2 = r2_score(y_tst,y_pr_tst)

model_mae['rnd_forest'] = mae
model_r2['rnd_forest'] = r2

print('MAE:{:7.2f},{:7.2f}% of mean'.format(mae,100*mae/y_pr_tst.mean()))
print('R2:{:6.3f}'.format(r2))

# Getting a reduced sample to test
size = len(X_tst)
indexes = np.random.choice(len(X_tst), size, replace=False)
X_new = X_tst.iloc[indexes]
y_new = rnd_frst.predict(X_new)

plot_predictions(X_new,y_new,descr = 'REAL - Random Forest: ',cols = ['area'])
print('Minor Error: ',min(model_mae, key=model_mae.get),', (',min(list(model_mae.values())),')')
print('Best R2    : ',max(model_r2, key=model_r2.get),', (',max(list(model_r2.values())),')')