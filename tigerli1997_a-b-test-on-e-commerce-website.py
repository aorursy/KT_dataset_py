import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
%matplotlib inline

#set the seed to assure you get the same answers, otherwise, you could remove the following line
random.seed(100)

file = '../input/ab_data1.csv'
df = pd.read_csv(file)
df.head()
df.shape[0]
df.user_id.nunique()
"%.4f" % (df.converted.sum() / df.converted.count())
df.groupby(['group','landing_page']).count()['converted']
df.isnull().sum()
#df2 = df[(df.group == 'treatment') & (df.landing_page == 'new_page')]
df2 = df.query("group == 'treatment' & landing_page == 'new_page'")
df2.head()
# Double Check all of the corresponding rows were removed - this should be 0
df2[((df2['group'] == 'treatment') & (df2['landing_page'] == 'new_page')) == False].shape[0]
df2.user_id.nunique()
df2[df2.user_id.duplicated() == True]
# Row 2893, user_id 773192
df2.drop_duplicates(subset='user_id', keep='first', inplace=True)
"%.4f"% (df.converted.sum() / df.converted.count())
obs_old = df.query('group == "control"')['converted'].sum() / \
        df.query('group == "control"')['converted'].count()
'%.4f' % obs_old
obs_new = df.query('group == "treatment"')['converted'].sum() / \
df.query('group == "treatment"')['converted'].count()
'%.4f' % obs_new
null_rate = df['converted'].sum() / df.shape[0]
null_rate
df.drop_duplicates(subset='user_id', keep='first', inplace=True)
# Use a sample size for each page equal to the ones in ab_data1.csv. 
df.groupby('landing_page').count()['user_id']
n_new , n_old = 145319 , 145261
new_page_converted = np.random.choice(2,n_new,p=[1-null_rate,null_rate])
new_page_converted
old_page_converted = np.random.choice(2,n_old,p=[1-null_rate,null_rate])
old_page_converted
new_page_converted.sum() / n_new - old_page_converted.sum() / n_old
p_diffs = []
# bootstrap sampling with python, several minites may cost
for _ in range (10000):
    bootstrap_new = np.random.choice(2,n_new,p=[1-null_rate,null_rate])
    bootstrap_old = np.random.choice(2,n_old,p=[1-null_rate,null_rate])
    p_new = bootstrap_new.sum() /n_new
    p_old = bootstrap_old.sum() / n_old
    p_diffs.append(p_new - p_old)
    
# convert list to array
p_diffs = np.array(p_diffs)
plt.hist(p_diffs)
obs_diff = obs_new-obs_old
plt.axvline(x = obs_diff,color='red')
(p_diffs > obs_diff).mean()
import statsmodels.api as sm

convert_old = df.query('landing_page == "old_page"')['converted'].sum()
convert_new = df.query('landing_page == "new_page"')['converted'].sum()
z_score, p_value = sm.stats.proportions_ztest\
([convert_new,convert_old],[n_new, n_old],alternative='larger')
(z_score, p_value)
# check if NaN contains in column 'converted'
df[df.converted.isna() ==True]
df.dropna(inplace=True)
df['interpret']=1 
df = df.join(pd.get_dummies(df['group']))
#rename treatment row
df.rename(index = str,columns = {'treatment':'ab_page'},inplace = True)
# drop column :control
df.drop('control',axis = 1,inplace=True)
# change datatype into integer
df.converted = df.converted.astype(int)
df.head()
log_mod = sm.Logit(df.converted,df[['ab_page','interpret']])
result = log_mod.fit()
result.summary()
file1 = '../input/countries.csv'
countries_df = pd.read_csv(file1)
df_new = countries_df.set_index('user_id').join(df.set_index('user_id'), how='inner')
### Create the necessary dummy variables
df_new[['CA','UK','US']] = pd.get_dummies(df_new['country'])
# drop CA row to keep two columns
df_new.drop('CA',axis=1,inplace=True)
df_new.head()
df_new['interpret'] = 1 # add column for interpret

mod1 = sm.Logit(df_new.converted,df_new[['interpret','US','UK']])
mod1.fit().summary()
df_new['page_UK'] = df_new['ab_page'] * df_new['UK']
df_new['page_US'] = df_new['ab_page'] * df_new['US']
### Fit Your Linear Model And Obtain the Results
mod2 = sm.Logit(df_new.converted,df_new[['interpret','page_US','page_UK','US','UK','ab_page']])
mod2.fit().summary()
