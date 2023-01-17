import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/dirty_deputies_v2.csv')

df.shape
df.head()
## TOTAL SPENDING

total = sum(df['refund_value'])

print('Total spending of all deputies: {}.'.format(round(total, 0)))
spend_area = np.unique(df['refund_description'])

print('There are {} areas of spending.'.format(len(spend_area)))
# TOP 10 COSTLY AREAs OF SPENDING

df.groupby('refund_description')['refund_value'].sum().sort_values(ascending=False)[:10]
## HOW MUCH DO TOP AREAS OF SPENDING ACCOUNT FOR?



by_area = df.groupby('refund_description')['refund_value'].sum().sort_values(ascending=False)

top5_area = by_area[:5]

percent5 = round(100*sum(top5_area)/total, 1)

print('Top 5 areas of spending already account for {} percent of total spending'.format(percent5))
sub_df = df[df['refund_date'].notnull()]

sub_df.shape
type(sub_df['refund_date'].iloc[0])
from datetime import *

def toDate(x):

    date_format = '%Y-%m-%dT%H:%M:%S'

    return datetime.strptime(str(x), date_format)
# test

toDate(df['refund_date'].loc[0])
tmp = sub_df['refund_date'].apply(toDate)
weird_dates = [s for s in sub_df['refund_date'] if s.find('.943') >= 0]

weird_dates[0]
import re

def valid_date(datestring):

    try:

        mat=re.match('(\d{4}-\d{2}-\d{2})', datestring)

        if mat is not None:

            date_format = '%Y-%m-%d'

            return datetime.strptime(mat.group(1), date_format)        

    except ValueError:

        print('Invalid date time format in string {}'.format(datestring))

        pass        

    return None
valid_date(weird_dates[0])
sub_df['valid_date'] = sub_df['refund_date'].apply(valid_date)
def spendInDateRange(first, last, area, df):

    res = df[df['refund_description']==area][df['valid_date'] >= first][df['valid_date'] <= last]

    spend = res['refund_value'].sum()

#     print('Spend in area {} in duration from {} to {}: {}'.format(area, first, last, spend))

    return spend
first, last = valid_date('2016-01-01'), valid_date('2016-01-31')

first_area = top5_area.keys()[0]

spend = spendInDateRange(first, last, df=sub_df, area=first_area)

print('Spend of {} in duration from {} to {} is: {}'.format(first_area, first, last, spend))
from calendar import *

# pad 0 to 1-digit months

def pad0(m):

    return '0' + str(m) if m < 10 else str(m)



def lastDate(m, yr=2016):

    n_day = monthrange(yr, m)[1]

    date_str = '-'.join([str(yr), pad0(m), str(n_day)])

    return valid_date(date_str)



def firstDate(mth, yr):

    date_str = '-'.join([str(yr), pad0(mth), '01'])

    return valid_date(date_str)
months = range(1, 13)

# get first and last date of each month

first_dates = {m:firstDate(mth=m, yr=2016) for m in months}

last_dates = {m:lastDate(m, yr=2016) for m in months}
def spendInMonth(m, area):

    return spendInDateRange(first_dates[m], last_dates[m], df=sub_df, area=area)



def monthlySpend(area):

    spend_by_month = [spendInMonth(m, area) for m in months]

    res = pd.DataFrame({'month': months, 'spend': spend_by_month})

    res.rename(columns={'spend':area}, inplace=True)

    return res
first_area = top5_area.keys()[0]

res1 = monthlySpend(first_area)

res1
second_area = top5_area.keys()[1]

res2 = monthlySpend(second_area)

res2
df = pd.merge(res1, res2)

df.rename(columns={'DISSEMINATION OF PARLIAMENTARY ACTIVITY': 'PARLIAMENT ACTIVITY'}, inplace=True)



del df['month']



import matplotlib.pyplot as plt



plt.figure(); df.plot(); plt.legend(loc='best')

plt.xlabel('Month')

plt.ylabel('Spend (in R)')

plt.xticks(range(1, 13))



plt.show()