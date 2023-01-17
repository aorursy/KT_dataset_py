# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random

import matplotlib.pyplot as plt

%matplotlib inline

#We are setting the seed to assure you get the same answers on quizzes as we set up

random.seed(42)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/abtesting/ab_data.csv")

df.head()
print(df.shape)

print(len(df.index))
df.user_id.nunique()
len(df.query('converted==1'))/len(df.index)
df.info()
df2=df
# dataframe where where treatment is not aligned with new_page or control is not aligned with old_page 

df2 = df[((df.group=='treatment') & (df.landing_page=='new_page')) | ((df.group=='control') & (df.landing_page=='old_page'))]
# Double Check all of the correct rows were removed - this should be 0

df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]
# Fine the unique user_ids 

df2.user_id.nunique()
# There is user_id repeated in df2

df2.user_id[df2.user_id.duplicated()]
# The row information for the repeat user_id

df2.loc[df2.user_id.duplicated()]
# Now we remove duplicate rows

df2 = df2.drop_duplicates()

# Check agin if duplicated values are deleted or not

sum(df2.duplicated())


# Probability of an individual converting regardless of the page they receive

df2['converted'].mean()
# The probability of an individual converting given that an individual was in the control group

control_group = len(df2.query('group=="control" and converted==1'))/len(df2.query('group=="control"'))

control_group
# The probability of an individual converting given that an individual was in the treatment group

treatment_group = len(df2.query('group=="treatment" and converted==1'))/len(df2.query('group=="treatment"'))

treatment_group
# The probability of individual received new page

len(df2.query('landing_page=="new_page"'))/len(df2.index)
p_new = len(df2.query( 'converted==1'))/len(df2.index)

p_new
p_old = len(df2.query('converted==1'))/len(df2.index)

p_old
# probablity under null

p=np.mean([p_old,p_new])

print(p)

# difference of p_new and p_old

p_diff=p_new-p_old
#calculate number of queries when landing_page is equal to new_page

n_new = len(df2.query('landing_page=="new_page"'))

#print n_new

n_new
#calculate number of queries when landing_page is equal to old_page

n_old = len(df2.query('landing_page=="old_page"'))

#print n_old

n_old
## simulate n_old transactions with a convert rate of p_new under the null

new_page_converted = np.random.choice([0, 1], n_new, p = [p_new, 1-p_new])
# simulate n_old transactions with a convert rate of p_old under the null

old_page_converted = np.random.choice([0, 1], n_old, p = [p_old, 1-p_old])
# differences computed in from p_new and p_old

obs_diff= new_page_converted.mean() - old_page_converted.mean()# differences computed in from p_new and p_old

obs_diff
# Create sampling distribution for difference in p_new-p_old simulated values

# with boostrapping

p_diffs = []

for i in range(10000):

    

    # 1st parameter dictates the choices you want.  In this case [1, 0]

    p_new1 = np.random.choice([1, 0],n_new,replace = True,p = [p_new, 1-p_new])

    p_old1 = np.random.choice([1, 0],n_old,replace = True,p = [p_old, 1-p_old])

    p_new2 = p_new1.mean()

    p_old2 = p_old1.mean()

    p_diffs.append(p_new2-p_old2)

#_p_diffs = np.array(_p_diffs)
p_diffs=np.array(p_diffs)

#histogram of p_diff

plt.hist(p_diffs)

plt.title('Graph of p_diffs')#title of graphs

plt.xlabel('Page difference') # x-label of graphs

plt.ylabel('Count') # y-label of graphs
#histogram of p_diff

plt.hist(p_diffs);



plt.title('Graph of p_diffs') #title of graphs

plt.xlabel('Page difference') # x-label of graphs

plt.ylabel('Count') # y-label of graphs



plt.axvline(x= obs_diff, color='r');


var1 = df2[df2['landing_page'] == 'new_page']

var1=var1['converted'].mean()

var2 = df2[df2['landing_page'] == 'old_page']

var2 = var2['converted'].mean()

actual_diff = var1-var2

count = 0

for i in p_diffs:

    if i> actual_diff:

        count = count+1

        

print (count/(len(p_diffs)))
import statsmodels.api as sm



convert_old = len(df2.query('converted==1 and landing_page=="old_page"')) #rows converted with old_page

convert_new = len(df2.query('converted==1 and landing_page=="new_page"')) #rows converted with new_page

n_old = len(df2.query('landing_page=="old_page"')) #rows_associated with old_page

n_new = len(df2.query('landing_page=="new_page"')) #rows associated with new_page

n_new
#Computing z_score and p_value

z_score, p_value = sm.stats.proportions_ztest([convert_old,convert_new], [n_old, n_new],alternative='smaller') 



#display z_score and p_value

print(z_score,p_value)
from scipy.stats import norm

norm.cdf(z_score) #how significant our z_score is
norm.ppf(1-(0.05)) #critical value of 95% confidence
#adding an intercept column

df2['intercept'] = 1



#Create dummy variable column

df2['ab_page'] = pd.get_dummies(df2['group'])['treatment']



df2.head()
import statsmodels.api as sm

model=sm.Logit(df2['converted'],df2[['intercept','ab_page']])

results=model.fit()
results.summary()
# Store Countries.csv data in dataframe

countries = pd.read_csv('/kaggle/input/countriy/countries.csv')

countries.head()
#Inner join two datas

new = countries.set_index('user_id').join(df2.set_index('user_id'), how = 'inner')

new.head()
#adding dummy variables with 'CA' as the baseline

new[['US', 'UK']] = pd.get_dummies(new['country'])[['US', "UK"]]

new.head()
new['US_ab_page'] = new['US']*new['ab_page']

new.head()
new['UK_ab_page'] = new['UK']*new['ab_page']

new.head()
logit3 = sm.Logit(new['converted'], new[['intercept', 'ab_page', 'US', 'UK', 'US_ab_page', 'US_ab_page']])

logit3
#Check the result

result3 = logit3.fit()
result3.summary()