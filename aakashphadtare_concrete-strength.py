import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/concrete-data/Concrete_Data.csv')
df.head()
for col in df.columns:
    print(col,'\n',df[col].unique(),'\n'*2 )
df.skew()
df.isnull().sum()
from scipy.stats import shapiro
df.columns
X = df.drop('Concrete compressive strength(MPa, megapascals) ',axis=1)
y= df['Concrete compressive strength(MPa, megapascals) ']
from sklearn.model_selection import train_test_split
# train data -70% 
X_train, X_test , y_train, y_test = train_test_split(X,y, test_size = 0.30, random_state = 42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
import warnings 
warnings.filterwarnings('ignore')
import statsmodels.api as sm
import scipy.stats as st
X_constant = sm.add_constant(X)
lin_reg = sm.OLS(y,X_constant).fit()
lin_reg.summary()
## strong multicollinearity
## 
# Assumption 1: Normality Test
st.jarque_bera(lin_reg.resid)

sns.distplot(lin_reg.resid)


import pylab
from statsmodels.graphics.gofplots import ProbPlot
st_residual = lin_reg.get_influence().resid_studentized_internal
st.probplot(st_residual, dist="norm", plot = pylab)
plt.show()
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = [variance_inflation_factor(X_constant.values, i) for i in range(X_constant.shape[1])]
pd.DataFrame({'vif': vif[1:]}, index=X.columns).T

def calculate_vif(x):
    thresh = 5.0
    output = pd.DataFrame()
    k = x.shape[1]
    vif = [variance_inflation_factor(x.values, j) for j in range(x.shape[1])]
    for i in range(1,k):
        print("Iteration no.")
        print(i)
        print(vif)
        a = np.argmax(vif)
        print("Max VIF is for variable no.:")
        print(a)
        if vif[a] <= thresh :
            break
        if i == 1 :          
            output = x.drop(x.columns[a], axis = 1)
            vif = [variance_inflation_factor(output.values, j) for j in range(output.shape[1])]
        elif i > 1 :
            output = output.drop(output.columns[a],axis = 1)
            vif = [variance_inflation_factor(output.values, j) for j in range(output.shape[1])]
    return(output)
## passing X to the function so that the multicollinearity gets removed.
train_out = calculate_vif(X)
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
lasso = linear_model.Lasso()
print(cross_val_score(lasso, X, y, cv=10))

l=[] 
for i in df.columns: 
    if((df[i].skew()<0.1) or (df[i].skew()>0.2) and (i!='Outcome')): 
        l.append(i)
for i in df.columns:
    if i in l: 
        df[i]=list(st.boxcox(df[i]+1)[0]) 
df.skew()
#Backward Elimination
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)
X = df.drop(['Concrete compressive strength(MPa, megapascals) ','Coarse Aggregate  (component 6)(kg in a m^3 mixture)',
             'Fine Aggregate (component 7)(kg in a m^3 mixture)'],axis=1)
y= df['Concrete compressive strength(MPa, megapascals) ']
X_constant = sm.add_constant(X)
lin_reg = sm.OLS(y,X_constant).fit()
lin_reg.summary()
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_trains=ss.fit_transform(X_train)
X_tests=ss.transform(X_test)

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

lr=LinearRegression()
knn=KNeighborsRegressor()
rf=RandomForestRegressor()
svr=SVR()

models=[]
models.append(('MVLR',lr))
models.append(('KNN',knn))
models.append(('RF',rf))
models.append(('SVR',svr))

results=[]
names=[]
for name,model in models:
    kfold=KFold(shuffle=True,n_splits=3,random_state=42)
    cv_results=cross_val_score(model,X,y,cv=kfold,scoring='neg_mean_squared_error')
    results.append(np.sqrt(np.abs(cv_results)))
    names.append(name)
    print("%s: %f (%f)"%(name,np.mean(np.sqrt(np.abs(cv_results))),np.var(np.sqrt(np.abs(cv_results)),ddof=1)))
    

