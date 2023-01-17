import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
pwd
df_train = pd.read_csv('/kaggle/input/bits-f464-l1/train.csv')
df_train
df_train.isna().any().any()
df_train = df_train.drop(labels=['id'],axis=1)
df_train.nunique().sort_values().to_csv("df_train_nunique.txt")
uniq = df_train.nunique().sort_values()

cols_with_1 = uniq[uniq == 1].index

cols_with_1
df_train.head()
df_tr_states = df_train.iloc[:,0:95]

df_tr_states
df_tr_agents = df_train.iloc[:,95:101]

df_tr_agents
df_tr_lables = df_train['label']

df_tr_lables
df_tr_states.nunique().sort_values().to_csv("df_train_nunique.txt")
uniq = df_tr_states.nunique().sort_values()

cols_with_1 = uniq[uniq == 1].index

cols_with_1
df_tr_states.describe().to_csv("df_tr_states_describe.csv")
uniq2 = df_tr_states.nunique().sort_values()

poten_cat_vars = uniq2[uniq2 <10].index

poten_cat_vars
for idx,val in enumerate(range(len(poten_cat_vars))):

    sns.boxplot(x = df_tr_states[poten_cat_vars[val]],y=df_tr_lables)

    #plt.savefig('pics/boxplots/'+poten_cat_vars[val]+'.png')
sns.boxplot(x = df_tr_states[poten_cat_vars[0]],y=df_tr_lables)
sns.boxplot(x = df_tr_states[poten_cat_vars[1]],y=df_tr_lables)
sns.boxplot(x = df_tr_states[poten_cat_vars[2]],y=df_tr_lables)
sns.boxplot(x = df_tr_states[poten_cat_vars[3]],y=df_tr_lables)
sns.boxplot(x = df_tr_states[poten_cat_vars[4]],y=df_tr_lables)
sns.boxplot(x = df_tr_states[poten_cat_vars[5]],y=df_tr_lables)
sns.boxplot(x = df_tr_states[poten_cat_vars[6]],y=df_tr_lables)
sns.boxplot(x = df_tr_states[poten_cat_vars[7]],y=df_tr_lables) #has variation
sns.boxplot(x = df_tr_states[poten_cat_vars[8]],y=df_tr_lables) #has variation for some values
sns.boxplot(x = df_tr_states[poten_cat_vars[9]],y=df_tr_lables)
sns.boxplot(x = df_tr_states[poten_cat_vars[10]],y=df_tr_lables)
sns.boxplot(x = df_tr_states[poten_cat_vars[11]],y=df_tr_lables)
sns.boxplot(x = df_tr_states[poten_cat_vars[12]],y=df_tr_lables)
sns.boxplot(x = df_tr_states[poten_cat_vars[13]],y=df_tr_lables) #has variation
sns.regplot(x = df_tr_states[some_states[0]],y=df_tr_lables)
# Compute the correlation matrix

corr = pd.concat([df_tr_states[df_tr_states.columns[1:30]],df_tr_lables],axis=1).corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(12, 9))



# Generate a custom diverging colormap

# cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
cols_to_drop1 = ['b16','b18','b14','b20','b32','b7']
# Compute the correlation matrix

corr = pd.concat([df_tr_states[df_tr_states.columns[30:60]],df_tr_lables],axis=1).corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(12, 9))



# Generate a custom diverging colormap

# cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
cols_to_drop2 = ['b70','b52','b67','b51','b66','b48']
# Compute the correlation matrix

corr = pd.concat([df_tr_states[df_tr_states.columns[60:90]],df_tr_lables],axis=1).corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(12, 9))



# Generate a custom diverging colormap

# cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
cols_to_drop3 = ['b92','b73','b85']
df_tr_states.shape
df_tr_states =df_tr_states.drop(['time'],axis=1)
some_states = df_tr_states.columns[0:20]

for val in some_states:

    plt.figure()

    ax = plt.subplot()

    sns.distplot(a= df_tr_states[val],ax=ax,kde=False)

    #plt.savefig('pics/distplots/'+val+'.png')
df_tr_states.nunique().sort_values()
df_tr_states.shape
some_states = df_tr_states.columns[20:40]

for val in some_states:

    plt.figure()

    ax = plt.subplot()

    sns.distplot(a= df_tr_states[val],ax=ax,kde=False)

    #plt.savefig('pics/distplots/'+val+'.png')
some_states = df_tr_states.columns[40:]

for val in some_states:

    plt.figure()

    ax = plt.subplot()

    sns.distplot(a= df_tr_states[val],ax=ax,kde=False)

    #plt.savefig('pics/distplots/'+val+'.png')
df_tr_states.replace(np.nan,0,inplace=True)
df_tr_states.isna().sum()
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

selected_columns=['b16','b18','b14','b20','b32','b7','b70','b52','b67','b51','b66','b48','b92','b73','b85']

df_tr_states = df_tr_states.drop(selected_columns,axis=1)
X_scaled = scaler.fit_transform(df_tr_states)
X_scaled_new = np.concatenate([df_tr_states.values,df_tr_agents.values],axis=1)
from sklearn.model_selection import train_test_split



X_train,X_val,y_train,y_val = train_test_split(X_scaled_new,df_tr_lables,test_size=0.25,random_state=42)  #Checkout what does random_state do
from sklearn.linear_model import LinearRegression



reg = LinearRegression()

reg.fit(X_train,y_train)
from sklearn.metrics import mean_squared_error



y_pred_lr = reg.predict(X_val)



mse_lr = mean_squared_error(y_pred_lr,y_val)



print("Mean Squared Error of Linear Regression: {}".format(mse_lr))
from sklearn.ensemble import RandomForestRegressor



reg_rf = RandomForestRegressor(n_estimators = 150,max_depth = 50,n_jobs=-1)

reg_rf.fit(X_train,y_train)

y_pred_rf = reg_rf.predict(X_val)



mse_rf = mean_squared_error(y_pred_rf,y_val)



print("Mean Squared Error of RandomForestRegressor: {}".format(mse_rf))
df_test = pd.read_csv('/kaggle/input/bits-f464-l1/test.csv')
df_test
df_test1 = df_test[df_tr_states.columns]
from sklearn.preprocessing import StandardScaler



scaler2 = StandardScaler()

X_test_scaled = scaler2.fit_transform(df_test1)
X_test_scaled_new = np.concatenate([df_test1.values,df_test[df_tr_agents.columns].values],axis=1)
y_pred_rf = reg_rf.predict(X_test_scaled_new)
y_pred_rf
df_sub = pd.DataFrame({'id':df_test['id'],'label':y_pred_rf})
df_sub.to_csv('sub.csv',index=False)
df_test[df_tr_states.columns].nunique().sort_values()
df_tr_states.nunique().sort_values()