# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import pandas as pd, numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

df1 = pd.read_csv("/kaggle/input/predicting-poverty/train_values_wJZrCmI.csv")

df2 = pd.read_csv("/kaggle/input/predicting-poverty/train_labels.csv")



df = df1.merge(df2, on='row_id')
df.head()
df.shape
df.dtypes
df.isnull().sum()
df.drop(['bank_interest_rate', 'mm_interest_rate', 'mfi_interest_rate', 'other_fsp_interest_rate'], axis = 1) 
def age_grouping(data):

    age_condition = [

    (data['age'] < 30 ),

    (data['age'] >= 30) & (data['age'] < 45),

    (data['age'] >= 45) & (data['age'] < 60),

    (data['age'] >= 60)

    ]

    age_bins = ['< 30', '30 to 44', '45 to 60', '> 60']

    data['age_group'] = np.select(age_condition, age_bins)



age_grouping(df)
def count_unique(df, cols):

    for col in cols:

        print(df[col].value_counts())



categ_cols = ['age_group','country','is_urban','female','married','religion','relationship_to_hh_head',

 'education_level','literacy','can_add','can_divide','can_calc_percents','can_calc_compounding',

 'employed_last_year','employment_category_last_year','employment_type_last_year',

 'income_ag_livestock_last_year','income_friends_family_last_year','income_government_last_year',

 'income_own_business_last_year','income_private_sector_last_year','income_public_sector_last_year',

 'borrowing_recency','formal_savings','informal_savings','cash_property_savings',

 'has_insurance','has_investment','borrowed_for_emergency_last_year','borrowed_for_daily_expenses_last_year',

 'borrowed_for_home_or_biz_last_year','phone_technology','can_call','can_text','can_use_internet',

 'can_make_transaction','phone_ownership','advanced_phone_use','reg_bank_acct',

 'reg_mm_acct','reg_formal_nbfi_account','financially_included','active_bank_user',

 'active_mm_user','active_formal_nbfi_user','active_informal_nbfi_user','nonreg_active_mm_user', 'share_hh_income_provided', 

'num_times_borrowed_last_year','num_shocks_last_year','num_formal_institutions_last_year',

            'num_informal_institutions_last_year']



count_unique(df, categ_cols)
for column in df[['education_level', 'share_hh_income_provided']]:

    mode = df[column].mode()

    df[column] = df[column].fillna(mode)
def df_converted(df, convert_from, convert_to):

    cols = df.select_dtypes(include=[convert_from]).columns

    for col in cols:

        df[col] = df[col].values.astype(convert_to)

    return df



df = df_converted(df, np.bool, np.int64)
df['Poverty_conditional'] = [1 if poverty_probability>=0.5 

                             else 0 for poverty_probability in df['poverty_probability']] 
df['is_urban'].value_counts().plot(kind='bar')
ax = sns.countplot(x="is_urban", hue="female", data=df)
g = sns.catplot(x="is_urban", hue="female", col="Poverty_conditional",

                data = df, kind="count",

                height=4, aspect=.7)
ax = sns.countplot(x="is_urban", hue="literacy", data=df)
g = sns.catplot(x="is_urban", hue="literacy", col="Poverty_conditional",

                data = df, kind="count",

                height=8, aspect=1)
g = sns.catplot(x="is_urban", hue="employment_category_last_year", col="Poverty_conditional",

                data = df, kind="count",

                height=10, aspect=1)
def f(row):

    if row['formal_savings'] == 1:

        sav = 1

    elif row['informal_savings'] == 1:

        sav = 1

    else:

        sav = 0

    return sav

    

df['sav'] = df.apply(f, axis=1)
g = sns.catplot(x="is_urban", hue="employment_category_last_year", col="sav",

                data = df, kind="count",

                height=10, aspect=1)
g = sns.catplot(x="is_urban", hue="education_level", col="sav",

                data = df, kind="count",

                height=10, aspect=1)
g = sns.catplot(x="is_urban", hue="literacy", col="sav",

                data = df, kind="count",

                height=10, aspect=1)
g = sns.catplot(x="is_urban", hue="literacy", col="Poverty_conditional",

                data = df, kind="count",

                height=10, aspect=1)
def d(row):

    if row['borrowed_for_emergency_last_year'] == 1:

        borrowed = 1

    elif row['borrowed_for_daily_expenses_last_year'] == 1:

        borrowed = 1

    elif row['borrowed_for_home_or_biz_last_year'] == 1:

        borrowed = 1

    else:

        borrowed = 0

    return borrowed



df['borrowed'] = df.apply(d, axis=1)

df['borrowed'].value_counts()
g = sns.catplot(x="is_urban", hue="borrowed", col="Poverty_conditional",

                data = df, kind="count",

                height=10, aspect=1)
g = sns.catplot(x="is_urban", hue="religion", col="sav",

                data = df, kind="count",

                height=10, aspect=1)
g = sns.catplot(x="is_urban", hue="age_group", col="Poverty_conditional",

                data = df, kind="count",

                height=10, aspect=1)
from statsmodels.formula.api import ols



model = ols('poverty_probability ~ age + C(female)', data=df)

fitted_model = model.fit()

fitted_model.summary()
from numpy import mean

g = sns.barplot(x="female", y="poverty_probability", data = df, estimator=mean)
num_cols = ['age', 'num_financial_activities_last_year', 'poverty_probability'] 



sns.set(style="darkgrid")

sns.pairplot(df[num_cols])



corrs = df[num_cols].corr()
plt.figure(figsize=(5,2))

sns.heatmap(df[num_cols].corr(),annot=True,cmap='RdBu_r')

plt.title("Correlation")

plt.show()
export_csv = df.to_csv('export_df.csv', index = None, header=True)