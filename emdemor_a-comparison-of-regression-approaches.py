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



# Matplotlib and seaborn for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Scipy for statistics
#import scipy


from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model
# Dataframe to models
df_model_properties = pd.DataFrame({
    'Model':['linear_t','quad_t','SPPM_t','DPPM_t',
             'multlin_th','multquad_th','power_th',
             'multlin_tht','multquad_tht',
             'multlin_all','multquad_all',
             'multcub_all',
             'multquart_all'],
    
    'Features': [['t'],['t'],['t'],['t'],['t','h'],
                 ['t','h'],['t','h'],['t','h','T'],
                 ['t','h','T'],['t','h','T','A','W','H'],
                 ['t','h','T','A','W','G','H'],
                 ['t','h','T','A','W','G','H'],
                 ['t','h','T','A','W','G','H']],
    'MAE_test':['','','','','','','','','','','','',''],
    'R2_test':['','','','','','','','','','','','',''],
    'MAE_train':['','','','','','','','','','','','',''],
    'R2_train':['','','','','','','','','','','','','']
}).set_index('Model')
# Import Calories Dataset
df_cal = pd.read_csv(os.path.join(dirname,'calories.csv'))

# Import Exercises Dataset
df_ex  = pd.read_csv(os.path.join(dirname,'exercise.csv'))

# Merging Datasets
df = pd.merge(df_ex, df_cal, on = 'User_ID')
df.head()

# Get dummies
df['Gender'] = pd.get_dummies(df['Gender'], prefix_sep='_', drop_first=True)
df.head()
correlations = df.drop(columns=['User_ID'],axis=1).copy().corr()['Calories']
correlations = correlations.sort_values(ascending=False).drop('Calories',axis=0)
print(correlations)
correlations.to_frame().plot.bar();
df_3f = df[['Duration','Heart_Rate','Body_Temp','Calories']].copy()
df_3f.rename(columns={'Duration':'t',
                      'Heart_Rate':'h',
                      'Body_Temp':'T',
                      'Calories':'C'},inplace=True)


df_6f = df[['Duration','Heart_Rate','Body_Temp','Age','Weight','Height','Calories']].copy()
df_6f.rename(columns={'Duration':'t',
                      'Heart_Rate':'h',
                      'Body_Temp':'T',
                      'Age':'A',
                      'Weight':'W',
                      'Height':'H',
                      'Calories':'C'},inplace=True)


df_7f = df[['Duration','Heart_Rate','Body_Temp','Age','Weight','Gender','Height','Calories']].copy()
df_7f.rename(columns={'Duration':'t',
                      'Heart_Rate':'h',
                      'Body_Temp':'T',
                      'Age':'A',
                      'Weight':'W',
                      'Gender':'G',
                      'Height':'H',
                      'Calories':'C'},inplace=True)

sns.pairplot(df_3f)
conditions = True
conditions &= df_3f['C'] > 0
conditions &= df_3f['t'] > 0

df_3f_log = np.log10(df_3f[conditions]).copy()

df_3f_log.rename(columns={'t':'log_t',
                      'h':'log_h',
                      'T':'log_T',
                      'C':'log_C'},inplace=True)

sns.pairplot(df_3f_log)
# Splitting into test and train

C_i = df_3f['C'].to_numpy()
t_i = df_3f['t'].to_numpy()
t_train, t_test, C_train, C_test = train_test_split(t_i, C_i, test_size=0.333, random_state=42)

logC_i = df_3f_log['log_C'].to_numpy()
logt_i = df_3f_log['log_t'].to_numpy()
logt_train, logt_test, logC_train, logC_test = train_test_split( logt_i, logC_i, test_size=0.333, random_state=42)
plt.figure(figsize=(8,8))
plt.scatter(df_3f['t'],df_3f['C'])
plt.xlabel('$t$ (min)', size = 18)
plt.ylabel('$C$ (kcal)', size = 18)
plt.title('Calories burned vs Duration of Exercise', size = 20)
plt.show()
# Create a lineat regression object
lin_reg = linear_model.LinearRegression()

# Train the model using the training sets
lin_reg.fit(t_train.reshape(-1,1)   ,C_train.reshape(-1,1))

# Selecting a interval for duration
t_val = np.linspace(t_i.min(),t_i.max(),100)

# Predicting
C_lin_model = lin_reg.predict(t_val.reshape(-1,1))

# Plotting and comparing
plt.figure(figsize=(8, 8))
plt.scatter(df['Duration'],df['Calories'],c='lightgray',label = 'observations',alpha = 0.6,marker='.',zorder=1)
plt.plot(t_val,C_lin_model, c='tab:red',ls='-.', label = 'Linear Model', lw = 3,zorder=2)
plt.xlabel('Duration (min)', size = 18)
plt.ylabel('Calories', size = 18); 
plt.legend(prop={'size': 16})
plt.title('Calories burned vs Duration of Exercise', size = 20);
# Predictions for the test set
C_lin_test = lin_reg.predict(t_test.reshape(-1,1))

# Predictions for the train set
C_lin_train = lin_reg.predict(t_train.reshape(-1,1))

# Filling dataframe
df_model_properties.loc['linear_t']['MAE_test'] = mean_absolute_error(C_lin_test,C_test)
df_model_properties.loc['linear_t']['R2_test'] = r2_score(C_lin_test,C_test)
df_model_properties.loc['linear_t']['MAE_train'] = mean_absolute_error(C_lin_train,C_train)
df_model_properties.loc['linear_t']['R2_train'] = r2_score(C_lin_train,C_train)

# Printing  results
print('Mean error (test): ',df_model_properties.loc['linear_t']['MAE_test'])
print('R2 (test):    ',df_model_properties.loc['linear_t']['R2_test'])


print('\nMean error (train): ',df_model_properties.loc['linear_t']['MAE_train'])
print('R2 (train):  ',df_model_properties.loc['linear_t']['R2_train'])

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

# Create a linear regression object
quad_reg = linear_model.LinearRegression()

# Reshaping feature
T_train = t_train.reshape(-1,1)

# Transfomr features to a polynomial regression
quad = PolynomialFeatures(degree=2)
T_train_quad = quad.fit_transform(T_train)

# Training
quad_reg.fit(T_train_quad ,C_train.reshape(-1,1))

# Selecting a interval for duration an adapting shape
T_val = np.linspace(t_i.min(),t_i.max(),100).reshape(-1,1)
T_val_quad = quad.fit_transform(T_val)

# Predicting values
C_quad_model = quad_reg.predict(T_val_quad)

# Plotting
plt.figure(figsize=(8, 8))
plt.scatter(df['Duration'],df['Calories'],c='lightgray',label = 'observations',alpha = 0.6,marker='.',zorder=1)
plt.plot(t_val,C_quad_model, label = 'Quadratic Model', c='tab:blue', lw = 3,zorder=2)
plt.xlabel('Duration (min)', size = 18)
plt.ylabel('Calories', size = 18); 
plt.legend(prop={'size': 16})
plt.title('Calories burned vs Duration of Exercise', size = 20);
C_quad_test = quad_reg.predict(quad.fit_transform(t_test.reshape(-1,1)))
C_quad_train = quad_reg.predict(quad.fit_transform(t_train.reshape(-1,1)))

df_model_properties.loc['quad_t']['MAE_test'] = mean_absolute_error(C_quad_test,C_test)
df_model_properties.loc['quad_t']['R2_test'] = r2_score(C_quad_test,C_test)
df_model_properties.loc['quad_t']['MAE_train'] = mean_absolute_error(C_quad_train,C_train)
df_model_properties.loc['quad_t']['R2_train'] = r2_score(C_quad_train,C_train)


print('Mean error (test): ',df_model_properties.loc['quad_t']['MAE_test'])
print('R2 (test):    ',df_model_properties.loc['quad_t']['R2_test'])


print('\nMean error (train): ',df_model_properties.loc['quad_t']['MAE_train'])
print('R2 (train):  ',df_model_properties.loc['quad_t']['R2_train'])

plt.figure(figsize=(8, 8))
plt.scatter(df['Duration'],df['Calories'],c='lightgray',label = 'observations',alpha = 0.6,marker='.',zorder=1)
plt.plot(t_val,C_lin_model, c='tab:red',ls='-.', label = 'Linear Model', lw = 3,zorder=2)
plt.plot(t_val,C_quad_model, label = 'Quadratic Model', c='tab:blue', lw = 3,zorder=2)
plt.xlabel('Duration (min)', size = 18)
plt.ylabel('Calories', size = 18); 
plt.legend(prop={'size': 16})
plt.title('Calories burned vs Duration of Exercise', size = 20);
plt.figure(figsize=(8,8))
plt.scatter(df_3f_log['log_t'],df_3f_log['log_C'])
plt.xlabel('$t$ (min)', size = 18)
plt.ylabel('$C$ (kcal)', size = 18)
plt.title('Calories burned vs Duration of Exercise', size = 20)
plt.show()
# In fact, this is the only calculation for this method
c_1 = np.dot(logC_train,logt_train)/np.dot(logt_train,logt_train)

print('The value of c_1 is: ',c_1)

# With the value of c1, its possible to define a predict function
def sing_par_predict(t,c1 = c_1):
    return np.power(t,c1) 
t_val = np.linspace(t_i.min(),t_i.max(),100)
C_sing = sing_par_predict(t_val)

plt.figure(figsize=(8, 8))
plt.scatter(df['Duration'],df['Calories'],c='lightgray',label = 'Observations',alpha = 0.6,marker='.',zorder=1)
plt.plot(t_val,C_sing, c='tab:green',ls='-', label = 'SPPM', linewidth = 3)
plt.xlabel('Duration (min)', size = 18)
plt.ylabel('Calories', size = 18); 
plt.legend(prop={'size': 16})
plt.title('Calories burned vs Duration of Exercise', size = 20);
C_sppm_test = sing_par_predict(t_test)
C_sppm_train = sing_par_predict(t_train)

df_model_properties.loc['SPPM_t']['MAE_test'] = mean_absolute_error(C_sppm_test,C_test)
df_model_properties.loc['SPPM_t']['R2_test'] = r2_score(C_sppm_test,C_test)
df_model_properties.loc['SPPM_t']['MAE_train'] = mean_absolute_error(C_sppm_train,C_train)
df_model_properties.loc['SPPM_t']['R2_train'] = r2_score(C_sppm_train,C_train)

print('Mean error (test): ',df_model_properties.loc['SPPM_t']['MAE_test'])
print('R2 (test):    ',df_model_properties.loc['SPPM_t']['R2_test'])


print('\nMean error (train): ',df_model_properties.loc['SPPM_t']['MAE_train'])
print('R2 (train):  ',df_model_properties.loc['SPPM_t']['R2_train'])

# Create a linear regression object
log_reg = linear_model.LinearRegression()

# Train the model using the training sets
log_reg.fit(logt_train.reshape(-1,1),logC_train.reshape(-1,1))

logC_pred = log_reg.predict(logt_test.reshape(-1,1))

def dbl_par_predict(t):
    logt = np.log10(t)
    return  np.power(10,log_reg.predict(logt)) 
t_val = np.linspace(t_i.min(),t_i.max(),100)
C_log_model = dbl_par_predict(t_val.reshape(-1,1))
plt.figure(figsize=(8, 8))
plt.scatter(df['Duration'],df['Calories'],c='lightgray',label = 'observations',alpha = 0.6,marker='.',zorder=1)
plt.plot(t_val,C_log_model, c='tab:blue',ls='--', label = 'DPPM', lw = 3,zorder=2)
plt.xlabel('Duration (min)', size = 18)
plt.ylabel('Calories', size = 18); 
plt.legend(prop={'size': 16})
plt.title('Calories burned vs Duration of Exercise', size = 20);
C_dppm_test = dbl_par_predict(t_test.reshape(-1,1))
C_dppm_train = dbl_par_predict(t_train.reshape(-1,1))


df_model_properties.loc['DPPM_t']['MAE_test'] = mean_absolute_error(C_dppm_test,C_test)
df_model_properties.loc['DPPM_t']['R2_test'] = r2_score(C_dppm_test,C_test)
df_model_properties.loc['DPPM_t']['MAE_train'] = mean_absolute_error(C_dppm_train,C_train)
df_model_properties.loc['DPPM_t']['R2_train'] = r2_score(C_dppm_train,C_train)


print('Mean error (test): ',df_model_properties.loc['DPPM_t']['MAE_test'])
print('R2 (test):    ',df_model_properties.loc['DPPM_t']['R2_test'])


print('\nMean error (train): ',df_model_properties.loc['DPPM_t']['MAE_train'])
print('R2 (train):  ',df_model_properties.loc['DPPM_t']['R2_train'])

t_val = np.linspace(t_i.min(),t_i.max(),100)
C_lin_model = lin_reg.predict(t_val.reshape(-1,1))
C_log_model = dbl_par_predict(t_val.reshape(-1,1))
plt.figure(figsize=(8, 8))
plt.scatter(df['Duration'],df['Calories'],c='lightgray',label = 'observations',alpha = 0.6,marker='.',zorder=1)
plt.plot(t_val,C_sing, c='tab:green',ls='-', label = 'SPPM', lw = 3,zorder=2)
plt.plot(t_val,C_log_model, c='tab:blue',ls='--', label = 'DPPM', lw = 3,zorder=2)
plt.plot(t_val,C_lin_model, c='tab:red',ls='-.', label = 'Linear Model', lw = 3,zorder=2)
plt.plot(t_val,C_quad_model, label = 'Quadratic Model', c='tab:purple', lw = 3,zorder=2)
plt.xlabel('Duration (min)', size = 18)
plt.ylabel('Calories', size = 18); 
plt.legend(prop={'size': 16})
plt.title('Calories burned vs Duration of Exercise', size = 20);
# Organization of dataset
TH_i = df_3f[['t','h']].to_numpy()
C_i = df_3f['C'].to_numpy().reshape(-1,1)
TH_trn, TH_tst, C_trn, C_tst = train_test_split( TH_i,C_i, test_size=0.333, random_state=42)

# Organization of log dataset
logTH_i = df_3f_log[['log_t','log_h']].to_numpy()
logC_i = df_3f_log['log_C'].to_numpy().reshape(-1,1)
logTH_trn, logTH_tst, logC_trn, logC_tst = train_test_split(logTH_i,logC_i,test_size=0.333, random_state=42)
# Create regression object
ML2 = linear_model.LinearRegression()

# Train the model using the training sets
ML2.fit(TH_trn,C_trn)

# Predicting for test and train
C_ml2_tst = ML2.predict(TH_tst)
C_ml2_trn = ML2.predict(TH_trn)

# Updating results dataframe
df_model_properties.loc['multlin_th']['MAE_test'] = mean_absolute_error(C_tst,C_ml2_tst)
df_model_properties.loc['multlin_th']['R2_test'] = r2_score(C_tst,C_ml2_tst)
df_model_properties.loc['multlin_th']['MAE_train'] = mean_absolute_error(C_trn,C_ml2_trn)
df_model_properties.loc['multlin_th']['R2_train'] = r2_score(C_trn,C_ml2_trn)

# Showing
print('Mean error (test): ',df_model_properties.loc['multlin_th']['MAE_test'])
print('R2 (test):    ',df_model_properties.loc['multlin_th']['R2_test'])
print('\nMean error (train): ',df_model_properties.loc['multlin_th']['MAE_train'])
print('R2 (train):  ',df_model_properties.loc['multlin_th']['R2_train'])

MQ2 = linear_model.LinearRegression()

MQ2_poly = PolynomialFeatures(degree=2)

TH_trn_pl = MQ2_poly.fit_transform(TH_trn)

TH_tst_pl = MQ2_poly.fit_transform(TH_tst)

MQ2.fit(TH_trn_pl,C_trn)

C_mq2_tst = MQ2.predict(TH_tst_pl)

C_mq2_trn = MQ2.predict(TH_trn_pl)

df_model_properties.loc['multquad_th']['MAE_test'] = mean_absolute_error(C_tst,C_mq2_tst)
df_model_properties.loc['multquad_th']['R2_test'] = r2_score(C_tst,C_mq2_tst)
df_model_properties.loc['multquad_th']['MAE_train'] = mean_absolute_error(C_trn,C_mq2_trn)
df_model_properties.loc['multquad_th']['R2_train'] = r2_score(C_trn,C_mq2_trn)


print('Mean error (test): ',df_model_properties.loc['multquad_th']['MAE_test'])
print('R2 (test):    ',df_model_properties.loc['multquad_th']['R2_test'])

print('\nMean error (train): ',df_model_properties.loc['multquad_th']['MAE_train'])
print('R2 (train):  ',df_model_properties.loc['multquad_th']['R2_train'])

MP1 = linear_model.LinearRegression()

MP1.fit(logTH_trn,logC_trn)

def power_pth_predict(TH):
    logTH = np.log10(TH)
    return  np.power(10,MP1.predict(logTH))
C_mp1_tst = power_pth_predict(TH_tst)
C_mp1_trn = power_pth_predict(TH_trn)

df_model_properties.loc['power_th']['MAE_test'] = mean_absolute_error(C_tst,C_mp1_tst)
df_model_properties.loc['power_th']['R2_test'] = r2_score(C_tst,C_mp1_tst)

df_model_properties.loc['power_th']['MAE_train'] = mean_absolute_error(C_trn,C_mp1_trn)
df_model_properties.loc['power_th']['R2_train'] = r2_score(C_trn,C_mp1_trn)


print('Mean error (test): ',df_model_properties.loc['power_th']['MAE_test'])
print('R2 (test):    ',df_model_properties.loc['power_th']['R2_test'])

print('\nMean error (train): ',df_model_properties.loc['power_th']['MAE_train'])
print('R2 (train):  ',df_model_properties.loc['power_th']['R2_train'])

THT_i = df_3f[['t','h','T']].to_numpy()
C_i = df_3f['C'].to_numpy().reshape(-1,1)
THT_trn, THT_tst, C_trn, C_tst = train_test_split( THT_i,C_i, test_size=0.333, random_state=42)

logTHT_i = df_3f_log[['log_t','log_h','log_T']].to_numpy()
logC_i = df_3f_log['log_C'].to_numpy().reshape(-1,1)
logTHT_trn, logTHT_tst, logC_trn, logC_tst = train_test_split(logTHT_i,logC_i,test_size=0.333, random_state=42)
# Create regression object
ML3 = linear_model.LinearRegression()

# Train the model using the training sets
ML3.fit(THT_trn,C_trn)

C_ml3_tst = ML3.predict(THT_tst)
C_ml3_trn = ML3.predict(THT_trn)


df_model_properties.loc['multlin_tht']['MAE_test'] = mean_absolute_error(C_tst,C_ml3_tst)
df_model_properties.loc['multlin_tht']['R2_test'] = r2_score(C_tst,C_ml3_tst)

df_model_properties.loc['multlin_tht']['MAE_train'] = mean_absolute_error(C_trn,C_ml3_trn)
df_model_properties.loc['multlin_tht']['R2_train'] = r2_score(C_trn,C_ml3_trn)


print('Mean error (test): ',df_model_properties.loc['multlin_tht']['MAE_test'])
print('R2 (test):    ',df_model_properties.loc['multlin_tht']['R2_test'])

print('\nMean error (train): ',df_model_properties.loc['multlin_tht']['MAE_train'])
print('R2 (train):  ',df_model_properties.loc['multlin_tht']['R2_train'])
# Create regression object
MQ3 = linear_model.LinearRegression()

MQ3_poly = PolynomialFeatures(degree=2)
THT_trn_pl = MQ3_poly.fit_transform(THT_trn)
THT_tst_pl = MQ3_poly.fit_transform(THT_tst)


# Train the model using the training sets
MQ3.fit(THT_trn_pl,C_trn)

C_mq3_tst = MQ3.predict(THT_tst_pl)
C_mq3_trn = MQ3.predict(THT_trn_pl)

df_model_properties.loc['multquad_tht']['MAE_test'] = mean_absolute_error(C_tst,C_mq3_tst)
df_model_properties.loc['multquad_tht']['R2_test'] = r2_score(C_tst,C_mq3_tst)

df_model_properties.loc['multquad_tht']['MAE_train'] = mean_absolute_error(C_trn,C_mq3_trn)
df_model_properties.loc['multquad_tht']['R2_train'] = r2_score(C_trn,C_mq3_trn)


print('Mean error (test): ',df_model_properties.loc['multquad_tht']['MAE_test'])
print('R2 (test):    ',df_model_properties.loc['multquad_tht']['R2_test'])

print('\nMean error (train): ',df_model_properties.loc['multquad_tht']['MAE_train'])
print('R2 (train):  ',df_model_properties.loc['multquad_tht']['R2_train'])

X_i = df_7f[['t','h','T','A','W','G','H']].to_numpy()
C_i = df_7f['C'].to_numpy().reshape(-1,1)
X_trn, X_tst, C_trn, C_tst = train_test_split( X_i,C_i, test_size=0.333, random_state=42)
# Create regression object
ML6 = linear_model.LinearRegression()

# Train the model using the training sets
ML6.fit(X_trn,C_trn)

C_ml6_tst = ML6.predict(X_tst)
C_ml6_trn = ML6.predict(X_trn)


df_model_properties.loc['multlin_all']['MAE_test'] = mean_absolute_error(C_tst,C_ml6_tst)
df_model_properties.loc['multlin_all']['R2_test'] = r2_score(C_tst,C_ml6_tst)

df_model_properties.loc['multlin_all']['MAE_train'] = mean_absolute_error(C_trn,C_ml6_trn)
df_model_properties.loc['multlin_all']['R2_train'] = r2_score(C_trn,C_ml6_trn)


print('Mean error (test): ',df_model_properties.loc['multlin_all']['MAE_test'])
print('R2 (test):    ',df_model_properties.loc['multlin_all']['R2_test'])

print('\nMean error (train): ',df_model_properties.loc['multlin_all']['MAE_train'])
print('R2 (train):  ',df_model_properties.loc['multlin_all']['R2_train'])
# Create regression object
MQ6 = linear_model.LinearRegression()

MQ6_poly = PolynomialFeatures(degree=2)
X_trn_pl = MQ6_poly.fit_transform(X_trn)
X_tst_pl = MQ6_poly.fit_transform(X_tst)


# Train the model using the training sets
MQ6.fit(X_trn_pl,C_trn)

C_mq6_tst = MQ6.predict(X_tst_pl)
C_mq6_trn = MQ6.predict(X_trn_pl)

df_model_properties.loc['multquad_all']['MAE_test'] = mean_absolute_error(C_tst,C_mq6_tst)
df_model_properties.loc['multquad_all']['R2_test'] = r2_score(C_tst,C_mq6_tst)

df_model_properties.loc['multquad_all']['MAE_train'] = mean_absolute_error(C_trn,C_mq6_trn)
df_model_properties.loc['multquad_all']['R2_train'] = r2_score(C_trn,C_mq6_trn)


print('Mean error (test): ',df_model_properties.loc['multquad_all']['MAE_test'])
print('R2 (test):    ',df_model_properties.loc['multquad_all']['R2_test'])

print('\nMean error (train): ',df_model_properties.loc['multquad_all']['MAE_train'])
print('R2 (train):  ',df_model_properties.loc['multquad_all']['R2_train'])

# Create regression object
MQ6 = linear_model.LinearRegression()

MQ6_poly = PolynomialFeatures(degree=3)
X_trn_pl = MQ6_poly.fit_transform(X_trn)
X_tst_pl = MQ6_poly.fit_transform(X_tst)


# Train the model using the training sets
MQ6.fit(X_trn_pl,C_trn)

C_mq6_tst = MQ6.predict(X_tst_pl)
C_mq6_trn = MQ6.predict(X_trn_pl)

df_model_properties.loc['multcub_all']['MAE_test'] = mean_absolute_error(C_tst,C_mq6_tst)
df_model_properties.loc['multcub_all']['R2_test'] = r2_score(C_tst,C_mq6_tst)

df_model_properties.loc['multcub_all']['MAE_train'] = mean_absolute_error(C_trn,C_mq6_trn)
df_model_properties.loc['multcub_all']['R2_train'] = r2_score(C_trn,C_mq6_trn)


print('Mean error (test): ',df_model_properties.loc['multcub_all']['MAE_test'])
print('R2 (test):    ',df_model_properties.loc['multcub_all']['R2_test'])

print('\nMean error (train): ',df_model_properties.loc['multcub_all']['MAE_train'])
print('R2 (train):  ',df_model_properties.loc['multcub_all']['R2_train'])

# Create regression object
MQ6 = linear_model.LinearRegression()

MQ6_poly = PolynomialFeatures(degree=4)
X_trn_pl = MQ6_poly.fit_transform(X_trn)
X_tst_pl = MQ6_poly.fit_transform(X_tst)


# Train the model using the training sets
MQ6.fit(X_trn_pl,C_trn)

C_mq6_tst = MQ6.predict(X_tst_pl)
C_mq6_trn = MQ6.predict(X_trn_pl)

df_model_properties.loc['multquart_all']['MAE_test'] = mean_absolute_error(C_tst,C_mq6_tst)
df_model_properties.loc['multquart_all']['R2_test'] = r2_score(C_tst,C_mq6_tst)

df_model_properties.loc['multquart_all']['MAE_train'] = mean_absolute_error(C_trn,C_mq6_trn)
df_model_properties.loc['multquart_all']['R2_train'] = r2_score(C_trn,C_mq6_trn)


print('Mean error (test): ',df_model_properties.loc['multquart_all']['MAE_test'])
print('R2 (test):    ',df_model_properties.loc['multquart_all']['R2_test'])

print('\nMean error (train): ',df_model_properties.loc['multquart_all']['MAE_train'])
print('R2 (train):  ',df_model_properties.loc['multquart_all']['R2_train'])

df_model_properties
