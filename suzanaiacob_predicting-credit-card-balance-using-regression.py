import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import statsmodels.api as sm

import statsmodels.formula.api as smf

%matplotlib inline



col_list = ['#005f9a', '#00CDCD', '#f1bdbf']

sns.set_palette(col_list)

credit_df = pd.read_csv("../input/Credit.csv", index_col=0)
credit_df.Gender = credit_df.Gender.astype('category')

credit_df.Student = credit_df.Student.astype('category')

credit_df.Married = credit_df.Married.astype('category')

credit_df.Ethnicity = credit_df.Ethnicity.astype('category')
credit_df.describe()
credit_df.head()
credit_df.describe(include=['category'])
sns.distplot(credit_df.Balance)
active_credit_df = credit_df.loc[credit_df.Balance>0,].copy()

active_credit_df.Balance.describe() 
sns.distplot(active_credit_df.Balance)
credit_df['Active'] = np.where(credit_df['Balance']>0, 'Yes', 'No')  

credit_df.Active.describe()
numeric_credit_df = credit_df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(8,8))

plt.matshow(credit_df.corr(), cmap=plt.cm.Blues, fignum=1)

plt.colorbar()

tick_marks = [i for i in range(len(numeric_credit_df.columns))]

plt.xticks(tick_marks, numeric_credit_df.columns)

plt.yticks(tick_marks, numeric_credit_df.columns)
from scipy.stats import pearsonr

r1, p1 = pearsonr(credit_df.Balance, credit_df.Limit)

msg = "Correlation coefficient Balance-Limit: {}\n p-value: {}\n"

print(msg.format(r1, p1))

r2, p2 = pearsonr(credit_df.Balance, credit_df.Rating)

msg = "Correlation coefficient Balance-Rating: {}\n p-value: {}\n"

print(msg.format(r2, p2))

r3, p3 = pearsonr(credit_df.Balance, credit_df.Income)

msg = "Correlation coefficient Balance-Income: {}\n p-value: {}\n"

print(msg.format(r3, p3))

r4, p4 = pearsonr(credit_df.Limit, credit_df.Rating)

msg = "Correlation coefficient Limit-Rating: {}\n p-value: {}\n"

print(msg.format(r4, p4))

r5, p5 = pearsonr(credit_df.Limit, credit_df.Income)

msg = "Correlation coefficient Limit-Income: {}\n p-value: {}\n"

print(msg.format(r5, p5))

r6, p6 = pearsonr(credit_df.Rating, credit_df.Income)

msg = "Correlation coefficient Rating-Income: {}\n p-value: {}\n"

print(msg.format(r6, p6))

sns.regplot(x='Limit',

           y='Rating',

           data=credit_df,

           scatter_kws={'alpha':0.2},

           line_kws={'color':'black'})
f, axes = plt.subplots(2, 2, figsize=(15, 6))

f.subplots_adjust(hspace=.3, wspace=.25)

credit_df.groupby('Gender').Balance.plot(kind='kde', ax=axes[0][0], legend=True, title='Balance by Gender')

credit_df.groupby('Student').Balance.plot(kind='kde', ax=axes[0][1], legend=True, title='Balance by Student')

credit_df.groupby('Married').Balance.plot(kind='kde', ax=axes[1][0], legend=True, title='Balance by Married')

credit_df.groupby('Ethnicity').Balance.plot(kind='kde', ax=axes[1][1], legend=True, title='Balance by Ethnicity')
f, axes = plt.subplots(2, 2, figsize=(15, 6))

f.subplots_adjust(hspace=.3, wspace=.25)

active_credit_df.groupby('Gender').Balance.plot(kind='kde', ax=axes[0][0], legend=True, title='Balance by Gender')

active_credit_df.groupby('Student').Balance.plot(kind='kde', ax=axes[0][1], legend=True, title='Balance by Student')

active_credit_df.groupby('Married').Balance.plot(kind='kde', ax=axes[1][0], legend=True, title='Balance by Married')

active_credit_df.groupby('Ethnicity').Balance.plot(kind='kde', ax=axes[1][1], legend=True, title='Balance by Ethnicity')
sns.boxplot(x='Student', y='Balance', data = credit_df)
mod0 = smf.ols('Balance ~ Income + Rating + Cards + Age + Education + Gender + Student + Married + Ethnicity', data = credit_df).fit()

mod0.summary()
active_mod0 = smf.ols('Balance ~ Income + Rating + Cards + Age + Education + Gender + Student + Married + Ethnicity', data = active_credit_df).fit()

active_mod0.summary()
mod1 = smf.ols('Balance ~ Income + Rating + Age + Student', data = credit_df).fit()

mod1.summary()
f, axes = plt.subplots(3, 2, figsize=(12, 10))

f.subplots_adjust(hspace=.5, wspace=.25)

credit_df.groupby('Student').Income.plot(kind='kde', ax=axes[0][0], title='Income by Student')

credit_df.groupby('Student').Rating.plot(kind='kde', ax=axes[0][1], title='Rating by Student')

credit_df.plot(kind='scatter', x='Age' , y='Income' , ax=axes[1][0], title='Income and Age')

credit_df.plot(kind='scatter', x='Age' , y='Rating' , ax=axes[1][1], color='orange', title='Rating and Age')

credit_df.plot(kind='scatter', x='Rating' , y='Income' , ax=axes[2][0], color='orange', title='Income and Rating')

credit_df.groupby('Student').Age.plot(kind='kde', ax=axes[2][1], legend=True, title='Age by Student')
sns.lmplot(x='Income',

          y='Balance',

          data=active_credit_df,

          line_kws={'color':'black'},

          lowess=True,

          col='Student')
mod2 = smf.ols('Balance ~ Income + I(Income**2) + Age + Student + Rating', data = credit_df).fit()

mod2.summary()
sns.regplot('Balance', 'Income',

           data = active_credit_df,

           ci=None,

           order=2,

           line_kws={'color':'black'})
mod3 = smf.ols('Balance ~ Income + I(Income**2) + Age + Student + Income*Rating', data = credit_df).fit()

mod3.summary()
mod4 = smf.ols('Balance ~ Income + I(Income**2) + Rating + Age + Student + Education*Income', data = credit_df).fit()

mod4.summary()
mod5 = smf.ols('Balance ~ Income + I(Income**2) + Rating + Age + Student + Married*Age', data = credit_df).fit()

mod5.summary()
sns.lmplot(x="Age", 

           y="Balance", 

           hue="Married", 

           ci=None,

           data=active_credit_df);
mod6 = smf.ols('Balance ~ Income + I(Income**2) + Rating + Age + Student + Gender*Cards', data = credit_df).fit()

mod6.summary()
sns.lmplot(x="Cards", 

           y="Balance", 

           hue="Gender", 

           ci=None,

           data = credit_df);
active_mod7 = smf.ols('Balance ~ Income + I(Income**2) + Rating + Age + Student + Income*Rating', 

                      data = active_credit_df).fit()

active_mod7.summary()
active_mod8 = smf.ols('Balance ~ Limit + Rating + Income + Age + Student + Cards', data = active_credit_df).fit()

active_mod8.summary()
sns.regplot(x='Limit',

          y='Balance',

          data=active_credit_df,

          line_kws={'color':'black'},

          lowess=True)
mod9 = smf.ols('Balance ~ Rating', data = credit_df).fit()

mod9.summary()
log_mod = smf.glm('Active ~ Limit + Rating + Income + Age + Cards + Education', 

                   data = credit_df,

                   family=sm.families.Binomial()).fit()

log_mod.summary()
df_new=pd.DataFrame({'Income':np.random.normal(45, 20, 40),

                    'Rating':np.random.normal(355, 55, 40),

                    'Limit':np.random.normal(4735, 200, 40),

                    'Age':np.random.normal(56, 17, 40),

                    'Cards':list(range(0,10))*4,

                    'Student':['Yes']*20+['No']*20})

df_new.Cards[df_new.Cards == 0] = 3

df_new.Income[df_new.Income <= 0] = df_new.Income.mean()

df_new.Rating[df_new.Rating <= 0] = df_new.Rating.mean()

df_new.Limit[df_new.Limit <= 0] = df_new.Limit.mean()

df_new['Balance']= active_mod8.predict(df_new)

df_new.describe()
mod8 = smf.ols('Balance ~ Income + I(Income**2) + Age + Student + Income*Rating + Limit + Cards', data = credit_df).fit()

mod8.summary()