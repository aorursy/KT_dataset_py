import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from scipy.stats import norm
# Read in the dataset and take a look at the top few rows
df = pd.read_csv('../input/ab_data.csv')
df.head()
df.info()
#The total of users is 294,478
# check the number of unique users in the dataset.
df.user_id.nunique()
#Check the proportion of users converted
p= df.query('converted == 1').user_id.nunique()/df.shape[0]

print("The proportion of users converted is {0:.2%}".format(p))
# Check the number of times the new_page and treatment don't line up.
l = df.query('(group == "treatment" and landing_page != "new_page" ) \
         or (group != "treatment" and landing_page == "new_page")').count()[0]
print("The number of times the new_page and treatment don't line up is {}".format(l))
#Check missing values
df.isnull().sum()
df2 =df.drop(df.query('(group == "treatment" and landing_page != "new_page" ) \
                      or (group != "treatment" and landing_page == "new_page") or (group == "control" and landing_page != "old_page") or (group != "control" and landing_page == "old_page")').index)
# Double Check all of the correct rows were removed 
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]
#Check the number of unique user_ids are in df2
df2.user_id.nunique()
df2.head()
df['landing_page'].value_counts().plot(kind='bar', figsize=(8,8));
df['landing_page'].value_counts().plot(kind='pie', figsize=(8,8));

lan_rev =df.groupby('landing_page').sum()['converted']

ind = np.arange(len(lan_rev))  # the x locations for the groups
width = 0.35  

plt.subplots(figsize=(18,10))
gen_bars =plt.bar(ind, lan_rev, width, color='g', alpha=.7)
#adv_bars =plt.bar(ind, adv, width, color='b', alpha=.7, label="Adventure")
plt.ylabel('converted',size=14) # title and labels
plt.xlabel('landing_page',size=14)
plt.title('Conversion by landing_page',size=18)
locations = ind + width / 2  # xtick locations
labels = ['old_page', 'new_page']  # xtick labels
plt.xticks(locations, labels)
#Check duplicates rows
df2.user_id.duplicated().sum()
#Check the repeated user_id
df2[df2.duplicated(['user_id'],keep=False)]['user_id']
print("The user_id repeated is 773192")
# Check the row information for the repeat user_id
df2.query('user_id == 773192')
#Remove the duplicated rows
df2 = df2.drop(df2.query('user_id == 773192 and timestamp == "2017-01-09 05:37:58.781806"').index)
#Check if there is any repeated user_id 
df2.user_id.duplicated().sum()
# Calculate the probability of an individual converting regardless of the page they receive
df_prob =df2.query('converted == 1').user_id.nunique()/df2.user_id.nunique()
df_prob

print("The probability of an individual converting regardless of the page they receive is {0:.2%}".format(df_prob))
# Calculate the probabilty the individual was in the control group to convert
p_cont = df2.query('converted == 1 and group == "control"').user_id.nunique() \
/df2.query('group == "control"').user_id.nunique()

print("The probability they converted based on control group is {0:.2%}".format(p_cont))
# Calculate the probabilty the individual was in the treatment group to convert
p_treat = df2.query('converted == 1 and group == "treatment"').user_id.nunique() \
/df2.query('group == "treatment"').user_id.nunique()

print("The probability they converted based on treatment group is {0:.2%}".format(p_treat))
# Calculate the probabilty that an individual received the new page
p_n = df2.query('landing_page == "new_page"').user_id.nunique()/df2.user_id.nunique()
#The probability that an individual received the new page is 50.00%
print("The probability that an individual received the new page is {0:.2%}".format(p_n))
# Since P_new and P_old both have "true" success rates equally, their converted rate 
#will have the same result.
p_new = df2.converted.mean()

print("The convert rate for p_new under the null is {0:.4}".format(p_new))
# Since P_new and P_old both have "true" success rates equally, their converted rate 
#will have the same result.
p_old = df2.converted.mean()

print("The convert rate for p_old under the null is {0:.4}".format(p_old))
# Count the total unique users with new page
n_new = df2.query('landing_page == "new_page" ').count()[0]
n_new
# Count the total unique users with old page
n_old = df2.query('landing_page == "old_page" ').count()[0]
n_old
#Simulate n_new transactions with a convertion rate of  p_new under the null. 
#Store these n_new 1's and 0's in new_page_converted

new_page_converted = np.random.choice([0,1],n_new, p=(p_new,1-p_new))
new_page_converted
#Simulate n_new transactions with a convert rate of  p_old under the null. 
#Store these  n_new 1's and 0's in old_page_converted

old_page_converted = np.random.choice([0,1],n_old, p=(p_old,1-p_old))
old_page_converted
# Find the difference between p_new and p_old
#For discovering the difference between p_new and p_old, it is necessary to find out the mean 
#of new_page_converted and old_page_converted.
new_page_converted.mean()
old_page_converted.mean()
#diff_conv is the difference between p_new and p_old.
diff_conv = new_page_converted.mean() - old_page_converted.mean()
diff_conv
# Simulate 10,000 p_new - p_old values with random binomial

new_converted_simulation = np.random.binomial(n_new, p_new,  10000)/n_new
old_converted_simulation = np.random.binomial(n_old, p_old,  10000)/n_old
p_diffs = new_converted_simulation - old_converted_simulation
p_diffs = np.array(p_diffs)
plt.hist(p_diffs);
# Calculate actual difference observed
new_convert = df2.query('converted == 1 and landing_page == "new_page"').count()[0]/n_new
old_convert = df2.query('converted == 1 and landing_page == "old_page"').count()[0]/n_old
obs_diff = new_convert - old_convert
obs_diff
#Check the proportion of the p_diffs are greater than the actual difference observed in ab_data.
null_vals = np.random.normal(0, p_diffs.std(), p_diffs.size)
plt.hist(null_vals);
plt.axvline(x=obs_diff, color='red')
(null_vals > obs_diff).mean()
convert_old = df2.query('converted == 1 and landing_page == "old_page"').count()[0]
convert_new = df2.query('converted == 1 and landing_page == "new_page"').count()[0]
n_old = df2.query('landing_page == "old_page" ').count()[0]
n_new = df2.query('landing_page == "new_page" ').count()[0]
convert_old,convert_new,n_old,n_new
z_score, p_value = sm.stats.proportions_ztest(np.array([convert_new,convert_old]),\
                                              np.array([n_new,n_old]), alternative = 'larger')
z_score, p_value
norm.cdf(z_score)
#0.09494168724097551 # Tells us how significant our z-score is
norm.ppf(1-(0.05/2))
# 1.959963984540054 # Tells us what our critical value at 96% confidence is
df2.head()
#Create intercept and dummies columns
df2['intercept'] = 1
df2[['ab_page','old_page']] = pd.get_dummies(df2['landing_page'])
df2 = df2.drop('old_page', axis = 1)
df2.head()
#Create a model
log = sm.Logit(df2['converted'], df2[['intercept', 'ab_page']])
results = log.fit()
results.summary()
countries_df = pd.read_csv('../input/countries.csv')

#Merge the countries data frame with df2
df3 = df2.merge(countries_df, on='user_id', how='inner')
df3.head()
df3.country.unique()
### Create the necessary dummy variables
df3[['US', 'CA', 'UK']] = pd.get_dummies(df3['country'])
df3 = df3.drop(['country', 'US'], axis = 1)
df3.head()
#Print Summary
log2 = sm.Logit(df3['converted'], df3[['intercept','ab_page','CA','UK']])
results2 = log2.fit()
results2.summary()
# For better visualizing the coef, we exponentiated them with numpy.
1/np.exp(-0.0149),np.exp(0.0408), np.exp(0.0506)
df3.head()
#For understanding the interaction between page and country we need two create 
#two columns that multiple ab_page to the country.

df3['CA_new_page']=df3['ab_page']*df3['CA']
df3['UK_new_page']=df3['ab_page']*df3['UK']
df3.head()
### Print Summary
log3 = sm.Logit(df3['converted'], df3[['intercept', 'ab_page', 'CA', 'UK','CA_new_page','UK_new_page']])
results3 = log3.fit()
results3.summary()
# For better visualizing the coef, we exponentiated them with numpy.
1/np.exp(-0.0674),np.exp(0.0118), np.exp(0.0175),np.exp(0.0783),np.exp(0.0469)
X = df3[['CA','UK','CA_new_page','UK_new_page']]
y = df3['converted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
log_mod = LogisticRegression()
log_mod.fit(X_train, y_train)
preds = log_mod.predict(X_test)
confusion_matrix(y_test, preds)
accuracy_score(y_test, preds)