import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df=pd.read_csv('../input/Financial Distress.csv')
df.head()
df.describe()
cdf=df.drop(labels=['Company','Time'], axis=1)
cdf.head()
cdf = cdf[cdf['Financial Distress']>-2.5]
cdf = cdf[cdf['Financial Distress']<4]
cdf.describe()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(cdf.drop('Financial Distress', axis=1)))  
y = cdf['Financial Distress']
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

MSE = metrics.make_scorer(metrics.mean_squared_error)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lrCV=cross_val_score(lr,X,y,scoring=MSE,cv=10)
print('Mean RMSE is:',np.sqrt(lrCV).mean() )
print('Std is:', np.sqrt(lrCV).std() )
from sklearn.linear_model import LassoCV

las = LassoCV(n_alphas=500, tol=0.001)
lasCV=cross_val_score(las,X,y,scoring=MSE,cv=10)
print('Mean RMSE is:',np.sqrt(lasCV).mean() )
print('Std is:', np.sqrt(lasCV).std() )
from sklearn.linear_model import RidgeCV

ri = RidgeCV(alphas=(0.1,1,10,100,1000,10000))
riCV=cross_val_score(ri,X,y,scoring=MSE,cv=10)
print('Mean RMSE is:',np.sqrt(riCV).mean() )
print('Std is:', np.sqrt(riCV).std() )
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rfCV=cross_val_score(rf,X,y,scoring=MSE,cv=10)
print('Mean RMSE is:',np.sqrt(rfCV).mean() )
print('Std is:', np.sqrt(rfCV).std() )
from sklearn.decomposition import PCA

# This will be our PCA calculation function.
def calc_pca_models(_X,_y,_cv=10):
    
    # This will be our results DataFrame:
    RMSE_PCA = pd.DataFrame(0, dtype=float, index=range(len(_X.columns)), 
                                columns=['lr','lr_std', 'las', 'las_std', 'ri', 'ri_std',
                                         'rf', 'rf_std','PCA'])
    
    # PCA of the data.
    pca = PCA(n_components=len(_X.columns))
    pca.fit(_X)
    X_pca=pd.DataFrame(pca.transform(_X))

    #Start crunching the numbers!

    for i in range (len(_X.columns)):
        
        X_pca_c = X_pca[X_pca.columns[:i+1]] # Choose how many components to consider.
        
        RMSE_PCA['PCA'][i] = i+1     # Write down how many components we're discussing.

        # Regression
        lrCV=cross_val_score(lr,X_pca_c,y,scoring=MSE,cv=_cv)
        RMSE_PCA['lr'][i] = np.sqrt(lrCV).mean() # Write mean RSME.
        RMSE_PCA['lr_std'][i] = np.sqrt(lrCV).std() # Write down RSME std. 
        
        #Lasso
        lasCV=cross_val_score(las,X_pca_c,y,scoring=MSE,cv=_cv)
        RMSE_PCA['las'][i] = np.sqrt(lasCV).mean() # Write mean RSME.
        RMSE_PCA['las_std'][i] = np.sqrt(lasCV).std() # Write down RSME std.
    
        #Ridge
        riCV=cross_val_score(ri,X_pca_c,y,scoring=MSE,cv=_cv)
        RMSE_PCA['ri'][i] = np.sqrt(riCV).mean() # Write mean RSME.
        RMSE_PCA['ri_std'][i] = np.sqrt(riCV).std() # Write down RSME std.
    
        # Random Forest!
        rfCV=cross_val_score(rf,X_pca_c,y,scoring=MSE,cv=_cv)
        RMSE_PCA['rf'][i] = np.sqrt(rfCV).mean() # Write mean RSME.
        RMSE_PCA['rf_std'][i] = np.sqrt(rfCV).std() # Write down RSME std.
        
    return RMSE_PCA
df_precol = calc_pca_models(X,y,)  # Get the numbers.

o_las = np.sqrt(lasCV).mean()*np.ones(df_precol.shape[0]) 
o_ri = np.sqrt(riCV).mean()*np.ones(df_precol.shape[0]) 
o_rf = np.sqrt(rfCV).mean()*np.ones(df_precol.shape[0]) 

# Plot the numbers.
fig = plt.figure()
ax = fig.add_axes([0,0,2,2])
x=df_precol['PCA']
plt.ylim(0.5,1.5)
plt.xlim(0,df_precol.shape[0]+1)

#ax.plot(x, df_precol['lr']-df_precol['lr_std'], '--', 'b')
ax.plot(x, df_precol['lr'], 'b', label='Regression',)  
ax.plot(x, df_precol['lr']+df_precol['lr_std'], 'bv')

#ax.plot(x, df_precol['las']-df_precol['las_std'], '--', color='orange')
ax.plot(x, df_precol['las'],'orange', label='Lasso')
ax.plot(x, df_precol['las']+df_precol['las_std'], 'v', color='orange', label = 'Upper Lasso std')
ax.plot(x, o_las, '.', color='orange', label = 'Original Lasso')


ax.plot(x, df_precol['ri']-df_precol['ri_std'], 'g^')
ax.plot(x, df_precol['ri'], 'g', label='Ridge')
#ax.plot(x, df_precol['ri']+df_precol['ri_std'], 'g--')
ax.plot(x, o_ri, '.', color='g', label = 'Original Ridge')

ax.plot(x, df_precol['rf']-df_precol['rf_std'], 'r^', label = 'Lower RF std')
ax.plot(x, df_precol['rf'], 'r', label='RandomForest')
#ax.plot(x, df_precol['rf']+df_precol['rf_std'], 'r--')
ax.plot(x, o_rf, '.', color='r', label = 'Orginal Random Forest')


ax.set_xlabel('Number of PCA Vectors')
ax.set_ylabel('RMSE')
ax.plot()

ax.legend()

o_las = np.sqrt(lasCV).mean()*np.ones(df_precol.shape[0]) 
o_las_std = np.sqrt(lasCV).std()*np.ones(df_precol.shape[0]) 
o_ri = np.sqrt(riCV).mean()*np.ones(df_precol.shape[0]) 
o_ri_std = np.sqrt(riCV).std()*np.ones(df_precol.shape[0])

# Plot the numbers.
fig = plt.figure()
ax = fig.add_axes([0,0,2,2])
x=df_precol['PCA']
plt.ylim(0.5,1.5)
plt.xlim(0.8,11)

ax.plot(x, df_precol['lr']-df_precol['lr_std'], 'b^')
ax.plot(x, df_precol['lr'], 'b', label='Regression',)  
ax.plot(x, df_precol['lr']+df_precol['lr_std'], 'bv')

ax.plot(x, df_precol['las']-df_precol['las_std'], '^', color='orange')
ax.plot(x, df_precol['las'],'orange', label='Lasso')
ax.plot(x, df_precol['las']+df_precol['las_std'], 'v', color='orange', label = 'Lasso std')

ax.plot(x, o_las+o_las_std, '--', color='orange', label = 'Original Lasso std')
ax.plot(x, o_las, '.', color='orange', label = 'Original Lasso')
ax.plot(x, o_las-o_las_std, '--', color='orange')

ax.plot(x, df_precol['ri']-df_precol['ri_std'], 'g^', label = 'Ridge std')
ax.plot(x, df_precol['ri'], 'g', label='Ridge')
ax.plot(x, df_precol['ri']+df_precol['ri_std'], 'gv')

ax.plot(x, o_ri+o_ri_std, '--', color='g', label = 'Original Ridge std')
ax.plot(x, o_ri, '.', color='g', label = 'Original Ridge')
ax.plot(x, o_ri-o_ri_std, '--', color='g')


ax.set_xlabel('Number of PCA Vectors')
ax.set_ylabel('RMSE')
ax.plot()

ax.legend()

corrmat = cdf.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
from statsmodels.stats.outliers_influence import variance_inflation_factor    

def remove_multicol(X, thresh=5.0):
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped=False
        vif = [variance_inflation_factor(X[variables].values, ix) for ix in range(X[variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            del variables[maxloc]
            dropped=True

    return X[variables]

X_colli = remove_multicol(X,5)
corrmat = X_colli.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
print('___Before adjusting for multicollinearity___')
lrCV=cross_val_score(lr,X,y,scoring=MSE,cv=10)
print('Regression')
print('Mean RMSE is:',np.sqrt(lrCV).mean() )
print('Std is:', np.sqrt(lrCV).std() )

lasCV=cross_val_score(las,X,y,scoring=MSE,cv=10)
print('Lasso')
print('Mean RMSE is:',np.sqrt(lasCV).mean() )
print('Std is:', np.sqrt(lasCV).std() )

riCV=cross_val_score(ri,X,y,scoring=MSE,cv=10)
print('Ridge')
print('Mean RMSE is:',np.sqrt(riCV).mean() )
print('Std is:', np.sqrt(riCV).std() )

rfCV=cross_val_score(rf,X,y,scoring=MSE,cv=10)
print('Random Forest')
print('Mean RMSE is:',np.sqrt(rfCV).mean() )
print('Std is:', np.sqrt(rfCV).std() )

print('___After adjusting for multicollinearity___')
lrCV=cross_val_score(lr,X_colli,y,scoring=MSE,cv=10)
print('Regression')
print('Mean RMSE is:',np.sqrt(lrCV).mean() )
print('Std is:', np.sqrt(lrCV).std() )

lasCV=cross_val_score(las,X_colli,y,scoring=MSE,cv=10)
print('Lasso')
print('Mean RMSE is:',np.sqrt(lasCV).mean() )
print('Std is:', np.sqrt(lasCV).std() )

riCV=cross_val_score(ri,X_colli,y,scoring=MSE,cv=10)
print('Ridge')
print('Mean RMSE is:',np.sqrt(riCV).mean() )
print('Std is:', np.sqrt(riCV).std() )

rfCV=cross_val_score(rf,X_colli,y,scoring=MSE,cv=10)
print('Random Forest')
print('Mean RMSE is:',np.sqrt(rfCV).mean() )
print('Std is:', np.sqrt(rfCV).std() )
df_col = calc_pca_models(X_colli,y,)  # Get the numbers.

o_las = np.sqrt(lasCV).mean()*np.ones(df_col.shape[0]) 
o_las_std = np.sqrt(lasCV).std()*np.ones(df_col.shape[0]) 
o_ri = np.sqrt(riCV).mean()*np.ones(df_col.shape[0]) 
o_ri_std = np.sqrt(riCV).std()*np.ones(df_col.shape[0])


# Plot the numbers.
fig = plt.figure()
ax = fig.add_axes([0,0,2,2])
x=df_col['PCA']
plt.ylim(0.5,1.5)
plt.xlim(0,df_col.shape[0]+1)

ax.plot(x, df_col['lr'], 'b', label='Regression',)  
ax.plot(x, df_col['lr']+df_col['lr_std'], 'bv')

ax.plot(x, df_col['las']-df_col['las_std'], '^', color='orange')
ax.plot(x, df_col['las'],'orange', label='Lasso')

ax.plot(x, df_col['ri'], 'g', label='Ridge')
ax.plot(x, df_col['ri']+df_col['ri_std'], 'gv', label = 'Ridge std')

ax.plot(x, o_ri+o_ri_std, '--', color='g', label = 'Original Ridge std')
ax.plot(x, o_ri, '.', color='g', label = 'Original Ridge')
ax.plot(x, o_ri-o_ri_std, '--', color='g')

ax.plot(x, o_las+o_las_std, '--', color='orange', label = 'Original Lasso std')
ax.plot(x, o_las, '.', color='orange', label = 'Original Lasso')
ax.plot(x, o_las-o_las_std, '--', color='orange')

ax.set_xlabel('Number of PCA Vectors')
ax.set_ylabel('RMSE')
ax.plot()

ax.legend()

