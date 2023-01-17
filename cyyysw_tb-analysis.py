# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import os
print(os.listdir("../input"))
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
tb=pd.read_csv('../input/TB_Burden_Country.csv')
tb.describe()
tb.shape
tb.isnull().sum()
tb.info()
tb.sample(20)
tb_rename=tb.rename(columns={'Country or territory name': 'country_name', 'Estimated total population number': 'total_population', 'Estimated prevalence of TB (all forms)': 'total_TB', 'Estimated number of deaths from TB (all forms, excluding HIV)': 'death_TB'})
tb_select=tb_rename.loc[:, ['country_name', 'total_population', 'total_TB', 'death_TB', 'Year']]
tb_select.sample(3)
tb_update=tb_select.insert(loc=4, column='death_rate', value=tb_select['death_TB']/tb_select['total_TB'])
tb_select.head(3)
tb_CU=tb_select.loc[tb_select['country_name'].isin(['China', 'United States of America'])]
tb_CU.insert(loc=2, column='occur_rate', value=tb_CU['total_TB']/tb_CU['total_population'])
tb_CU.head(3)
import numpy as np
xco=tb_CU.loc[tb_CU['country_name']=='China', ['occur_rate', 'Year']]
xcd=tb_CU.loc[tb_CU['country_name']=='China', ['death_rate', 'Year']]
xuo=tb_CU.loc[tb_CU['country_name']=='United States of America', ['occur_rate', 'Year']]
xud=tb_CU.loc[tb_CU['country_name']=='United States of America', ['death_rate', 'Year']]
plt.figure(1)
plt.subplot(211)
plt.plot(xco['Year'], xco['occur_rate'], 'bo-', label='China OR')
plt.plot(xuo['Year'], xuo['occur_rate'], 'go-', label='US OR')
plt.xticks([])
plt.yticks(np.arange(0.00, 0.003, 0.001))
plt.ylabel('Occur rate')
plt.legend(loc=1)
plt.title("China vs US TB")

plt.subplot(212)
plt.plot(xcd['Year'], xcd['death_rate'], 'rs--', label='China DR')
plt.plot(xud['Year'], xud['death_rate'], 'ms--', label='US DR')
plt.xticks(np.arange(1989, 2014, 2), rotation=45)
plt.yticks(np.arange(0.00, 0.12, 0.02))
plt.legend(loc=1)
plt.xlabel('Year')
plt.ylabel('Death rate')

plt.show()
fig, (ax1, ax2)=plt.subplots(2,1, figsize=(10,6))
ax1.plot(xco['Year'], xco['occur_rate'], 'bo-', label='China OR')
ax1.plot(xuo['Year'], xuo['occur_rate'], 'go-', label='US OR')
ax1.set_xticks([])
ax1.set_yticks(np.arange(0.00, 0.003, 0.001))
ax1.set_ylabel('Occur rate', size=18)
ax1.legend(loc=1)
ax1.set_title("China vs US TB", size=18)

ax2.plot(xcd['Year'], xcd['death_rate'], 'rs--', label='China DR')
ax2.plot(xud['Year'], xud['death_rate'], 'ms--', label='US DR')
plt.xticks(np.arange(1989, 2014, 2), rotation=45)
ax2.set_yticks(np.arange(0.00, 0.12, 0.02))
ax2.legend(loc=1)
plt.xlabel('Year', size=18)
ax2.set_ylabel('Death rate', size=18)
