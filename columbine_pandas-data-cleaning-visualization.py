# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import altair as alt

import seaborn as sns



import os, warnings

warnings.filterwarnings("ignore")

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

alt.renderers.enable('kaggle')
labor_market_statistics = pd.read_csv('/kaggle/input/british-labor-market-states/labor_market_statistics.csv', 

                                      na_values=['.', '??','?', '', ' ', 'NA', 'na', 'NaN', 'N/A', 'N/a', 'n/a'])



labor_market_statistics
labor_market_statistics.shape
description = labor_market_statistics.describe().T



description.to_csv('/kaggle/working/description.csv', encoding='utf-8')
labor_market_statistics.dtypes
labor_market_statistics['YBZY'].value_counts()
labor_market_statistics.nunique()
for column in labor_market_statistics.columns:

    print(f'{column}')

    print(f'{labor_market_statistics[column].unique()}', '\n')
%pylab inline

fig = plt.figure(figsize=(20,10))



sns.heatmap(labor_market_statistics.isnull())
# 由于表中最后两行几乎全部都是空数据，我们首先删除这两行的值



labor_market_statistics_without_lastest = labor_market_statistics.drop(index=[320, 321])

# 获取含有空值的属性



cols_with_null = [col for col in labor_market_statistics_without_lastest.columns

                 if labor_market_statistics_without_lastest[col].isnull().any()]



print(cols_with_null)

labor_market_statistics_without_lastest_filled = labor_market_statistics_without_lastest.fillna(labor_market_statistics_without_lastest.mean())



labor_market_statistics_without_lastest_filled.index = range(labor_market_statistics_without_lastest_filled.shape[0])
labor_market_statistics_recent = labor_market_statistics_without_lastest_filled.loc[labor_market_statistics['Year']>=2005]



labor_market_statistics_recent.index = range(labor_market_statistics_recent.shape[0])



labor_market_statistics_recent_LFS = labor_market_statistics_recent[['Year', 'Month', 'YBZY', 'YBZR', 'YBZQ', 'YBZT', 'YBZS', 'YBZV','YBZU', 'YBZW']]



# labor_market_statistics_recent

labor_market_statistics_recent_LFS.head(3)
List = []



for i in labor_market_statistics_recent_LFS.index:

    Year, Month, YBZY, YBZR, YBZQ, YBZT, YBZS, YBZV, YBZU, YBZW = labor_market_statistics_recent_LFS.iloc[i]

    if Month in ['Q1', 'Q2', 'Q3', 'Q4']:

        if(YBZY != 'NaN' and YBZR != 'NaN' and YBZQ != 'NaN' and YBZT != 'NaN' and YBZS != 'NaN'and YBZV != 'NaN'and YBZU != 'NaN'and YBZW != 'NaN'):

            time = str(Year) + ' ' + str(Month)

            List.append([time, 'YBZY', YBZY])

            List.append([time, 'YBZR', YBZR])

            List.append([time, 'YBZQ', YBZQ])

            List.append([time, 'YBZT', YBZT])

            List.append([time, 'YBZS', YBZS])

            List.append([time, 'YBZV', YBZV])

            List.append([time, 'YBZU', YBZU])

            List.append([time, 'YBZW', YBZW])

            

labor_market_statistics_recent_trans = pd.DataFrame(List, columns=['Time', 'Type', 'Values'])
bar_graph = alt.Chart(labor_market_statistics_recent_trans).mark_line(

    color='lightblue'

).encode(

    x=alt.X('Time'),

    y=alt.Y('Values'),

    color='Type'

)



bar_graph.properties(width=800, height=400)
labor_market_statistics_recent_JPC = labor_market_statistics_without_lastest_filled[['Year', 'Month', 'JPC2', 'JPC3', 'JPC4', 'JPC5']]



labor_market_statistics_recent_JPC.head(10)
JPC_list = []

trans = {'JAN':'01', 'FEB':'02', 'MAR':'03', 'APR':'04','MAY':'05','JUN':'06', 'JUL':'07', 'AUG':'08','SEP':'09','OCT':'10', 'NOV':'11', 'DEC':'12'}

for i in labor_market_statistics_recent_JPC.index:

    Year, Month, JPC2, JPC3, JPC4, JPC5 = labor_market_statistics_recent_JPC.iloc[i]

    if Month not in ['Q1', 'Q2', 'Q3', 'Q4', 'YEAR']:

        time = str(Year) + ' ' + trans[str(Month)]

        JPC_list.append([time, 'JPC2', JPC2])

        JPC_list.append([time, 'JPC3', JPC3])

        JPC_list.append([time, 'JPC4', JPC4])

        JPC_list.append([time, 'JPC5', JPC5])

            

labor_market_statistics_recent_trans_JPC = pd.DataFrame(JPC_list, columns=['Time', 'Type', 'Ratio'])
labor_market_statistics_recent_trans_JPC.head(10)
bar_graph = alt.Chart(labor_market_statistics_recent_trans_JPC).mark_line(

    color='lightblue',

#     interpolate='step-after'

).encode(

    x='Time',

    y='Ratio',

    color='Type'

)

bar_graph.properties(width=1000, height=400)