%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from bokeh.plotting import output_notebook, figure, show

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

#new_column_dtype = {'Exclusions': np.object, 'Explanation': np.object, 'IsStateMandate': np.object, 'IsSubjToDedTier2': np.object, 'IsSubjToDedTier1': np.object, 'CoinsInnTier2': np.object, 'CopayInnTier2': np.object}
#benefits_df = pd.read_csv('../input/BenefitsCostSharing.csv', dtype=new_column_dtype)

# Any results you write to the current directory are saved as output.
output_notebook()
### RATE ###

df_init = pd.read_csv('../input/Rate.csv', nrows=1, low_memory=False)
#######
#columns_list = df_init.columns.values.tolist()
#['BusinessYear', 'StateCode', 'IssuerId', 'SourceName', 'VersionNum', 'ImportDate', 
#'IssuerId2', 'FederalTIN', 'RateEffectiveDate', 'RateExpirationDate', 'PlanId', 
#'RatingAreaId', 'Tobacco', 'Age', 'IndividualRate', 'IndividualTobaccoRate', 'Couple', 
#'PrimarySubscriberAndOneDependent', 'PrimarySubscriberAndTwoDependents', 
#'PrimarySubscriberAndThreeOrMoreDependents', 'CoupleAndOneDependent', 
#'CoupleAndTwoDependents', 'CoupleAndThreeOrMoreDependents', 'RowNumber']

df = pd.read_csv('../input/Rate.csv')
#df.describe(include='all')
df = df[df['IndividualRate']<200000]
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df = df[df['IndividualRate']<2000]
model = pd.ols(x=df['Age'],y=df['IndividualRate'],intercept=True)
model
df['pred'] = df['Age']*8.0846-8.4609
df.describe()
plot = figure()
plot.scatter(x=df['Age'],y=df['IndividualRate'])
show(plot)

df['IssuerId'].describe()
rate_under_twenty = rate_per_age[rate_per_age.Age == '0-20']
ages_and_rates['0-20'] = rate_under_twenty.IndividualRate.mean()

rate_over_sixtyfive = rate_per_age[rate_per_age.Age == '65 and over']
ages_and_rates['65 and over'] = rate_over_sixtyfive.IndividualRate.mean()

rate_per_age = pd.DataFrame.from_dict(data=ages_and_rates,orient='index')

rate_per_age = pd.to_numeric(rate_per_age, errors='coerce')