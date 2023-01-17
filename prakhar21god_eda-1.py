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
import matplotlib.pyplot as plt
import seaborn as sln
%matplotlib inline
data=pd.read_csv('/kaggle/input/pump-sensor-data/sensor.csv')
data.head()
pd.set_option('display.max.columns',999)
data.head()
data['machine_status'].value_counts()
## confused about what does recovering mean? is broken means completely broken?
## lets check what possible states can be reached
### Normal,recovering,broken
# check the idx before the machine broke
data[data['machine_status']=='BROKEN']
# each observation is taken a 1 min of gap
# first failure occured at 17155 index after approx 13 days
#lets check status of 17154
data.iloc[17154]['machine_status']
data.iloc[17156]['machine_status']
data.iloc[17154]
## this is weird everytime time machine breaks from normal within one min????
## first recovering state 
data[data['machine_status']=='RECOVERING']
# okay and then machine starts recovering
## so what they do to to change it state?

## it took 17155/60 hours to break
17155/60
# days
286/24

# second broken
24510/60
408.5/24
### lets average it for days 
## lets pick up three most important sensor from artgor kernel
## sensor_00,sensor_04,sensor_01
## avg about 1440 rows
idx=0
idx2=1440
mean_sensor0=[]
mean_sensor4=[]
mean_sensor1=[]
mean_sensor47=[]
for i in range(0,20):
    mean_sensor0.append(data['sensor_00'].iloc[idx:idx2].mean(axis=0))
    mean_sensor4.append(data['sensor_04'].iloc[idx:idx2].mean(axis=0))
    mean_sensor1.append(data['sensor_01'].iloc[idx:idx2].mean(axis=0))
    mean_sensor47.append(data['sensor_47'].iloc[idx:idx2].mean(axis=0))
    idx+=1440
    idx2+=1440
    
import plotly.express as px
len(range(1,21))
avg_days=pd.DataFrame({'day':range(1,21),'sensor_00':mean_sensor0,'sensor__01':mean_sensor1,'sensor__04':mean_sensor4,'sensor__47':mean_sensor47})
fig = px.line(avg_days, x="day", y="sensor_00", 
        line_shape="spline", render_mode="svg")

fig.show()
fig=px.line(data[data['machine_status']=='BROKEN'],x='timestamp',y='sensor_00',line_shape='spline',render_mode='svg')
fig.show()
fig = px.line(avg_days, x="day", y="sensor__01", 
        line_shape="spline", render_mode="svg")

fig.show()
fig = px.line(avg_days, x="day", y="sensor__04", 
        line_shape="spline", render_mode="svg")

fig.show()
data.head()
## high correlation between sensor__00 to sensor__12
## sensor_12 to sensor_36
## sesnor 15 to sensor__15 to sensor 36 seems uncorrelated lets check?
data_copy=data.copy(True)
data_copy=data_copy.drop(['Unnamed: 0','timestamp','sensor_15'],axis=1)
def correlation(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] <= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
    return col_corr
no_corr_cols=correlation(data_copy,0.05)
no_corr_cols
data_copy=data_copy.drop('machine_status',axis=1)
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(data_copy, 20))
corr = data_copy.corr()
# 'RdBu_r' & 'BrBG' are other good diverging colormaps
corr.style.background_gradient(cmap='coolwarm')
## sensor_47,sensor_48
fig = px.line(avg_days, x="day", y="sensor__47", 
        line_shape="spline", render_mode="svg")

fig.show()
for col in data_copy.columns:
    data_copy[col] = data_copy[col].fillna(data_copy[col].mean())
data["machine_status"]=data['machine_status'].astype('category')
data['status']=data['machine_status'].cat.codes
# 0 IS BROKEN,1 IS NORMAL 2 IS RECOVERING
data["machine_status"].cat.categories
data["machine_status"].cat.codes.unique()
plt.figure(figsize=(10,10))
plt.plot(data['status'],label='state')
data_copy['status']=data['status']
data_copy.plot(figsize=(15,120), subplots=True)
def dist(col):
    try:
        
        sln.distplot(data_copy[data_copy['status']==1][col],label=col+'_normal')
        sln.distplot(data_copy[data_copy['status']==0][col],label=col+'_broken')
        sln.distplot(data_copy[data_copy['status']==2][col],label=col+'_recovering')
        plt.legend()
        plt.show()
    except:
        pass
for col in data_copy.columns:dist(col)
! pip install factor-analyzer==0.3.2
from factor_analyzer import FactorAnalyzer
# need to perform Bartlett's test to know if factors actually are present
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square_value,p_value=calculate_bartlett_sphericity(data_copy)
print(chi_square_value, p_value)
## KMO test
from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all,kmo_model=calculate_kmo(data_copy)
print(kmo_model)
# choosing no of factors
# Create factor analysis object and perform factor analysis
fa = FactorAnalyzer()
fa.fit(data_copy, 25)
# Check Eigenvalues
ev, v = fa.get_eigenvalues()
# Create scree plot using matplotlib
plt.scatter(range(1,data_copy.shape[1]+1),ev)
plt.plot(range(1,data_copy.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()
ev>1
# 9 factors ev>1 are considered as factors
fa = FactorAnalyzer(rotation="varimax",n_factors=9)
fa.fit(data_copy)
fa.loadings_.shape
loadings=pd.DataFrame(fa.loadings_)
### Now we may pick top most similar vales
for col in loadings.columns:
    print(loadings.nlargest(4, col).index)
## so 20,18,19,23 are similar
## 4,10,11,2
#41,42,38,40
#34,35,13,33
#7,9,8,6
#44,45,43,48
#5,0,50,6
#40,37,46,43
#26,48,47,29
## lets take one sensor out of nine categories and check for spearman correlation
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr(method='spearman').abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(data_copy, 20))

sln.scatterplot(data_copy['sensor_17'],data_copy['sensor_18'])
data_copy['sensor_18'].plot()

data_copy['sensor_17'].plot()
df=data_copy[['sensor_20','sensor_18','sensor_19','sensor_23']]
fig, axes = plt.subplots(nrows=2, ncols=2, dpi=120, figsize=(10,6))
for i, ax in enumerate(axes.flatten()):
    print(i)
    data = df[df.columns[i]]
    ax.plot(data, color='red', linewidth=1)
    # Decorations
    ax.set_title(df.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)
## SCALING for plotting purpose
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
scaled=pd.DataFrame(sc.fit_transform(data_copy),columns=data_copy.columns)
scaled.head()
plt.plot(scaled['sensor_00'],label='00')
plt.plot(scaled['sensor_01'],label='01')
plt.plot(scaled['sensor_04'],label='04')
plt.legend()
plt.figure(figsize=(10,7))
#plt.plot(scaled['sensor_00'],label='00')
plt.plot(scaled['sensor_00'].rolling(100).mean(),label='mean')
plt.legend()
## check stationarity
scaled['sensor_00'].hist()
plt.show()
from statsmodels.tsa.stattools import adfuller
for col in scaled.columns:
    result = adfuller(X)



from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest , chi2
from sklearn.impute import SimpleImputer
data.drop(['Unnamed: 0','sensor_15'],axis=1,inplace=True)
train = scaled[:int(0.8*(len(scaled)))]
valid = scaled[int(0.8*(len(scaled))):]
from statsmodels.tsa.vector_ar.var_model import VAR
data_copy.head()
### granger causality test
from statsmodels.tsa.stattools import grangercausalitytests
maxlag=12
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):
    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df
grangers_causation_matrix(scaled, variables = scaled.columns)       
model = VAR(endog=train)

for i in [1,2,3,4,5,6,7,8,9,10,11,12,13]:
    result = model.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic)
    print('BIC : ', result.bic)
    print('FPE : ', result.fpe)
    print('HQIC: ', result.hqic, '\n')
x = model.select_order(maxlags=15)
x.summary()
#lag=7
model_fit = model.fit(7)

cols=train.columns
prediction = model_fit.forecast(model_fit.y, steps=len(valid))
#converting predictions to dataframe
from sklearn.metrics import mean_squared_error
from math import sqrt
pred = pd.DataFrame(index=range(0,len(prediction)),columns=[cols])
for j in range(0,51):
    for i in range(0, len(prediction)):
        pred.iloc[i][j] = prediction[i][j]
pred.columns=cols
valid=valid.reset_index()
valid=valid.drop('index',axis=1)
for col in cols.tolist():
    print(f'rmse value for', {col}, 'is : ', mean_squared_error(valid[col].values,pred[col].values))
def diff(col):
    plt.plot(pred[col],label='predicted')
    plt.plot(valid[col],label='valid')
    plt.legend()
    plt.show()
    
for col in pred.columns:diff(col)
