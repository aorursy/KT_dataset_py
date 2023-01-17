# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sn

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
gpe = pd.read_csv('../input/data.csv')
gpe.head(5)
popul_male_series_code = ['SP.POP.0004.MA','SP.POP.0509.MA','SP.POP.1014.MA','SP.POP.1519.MA','SP.POP.2024.MA','SP.POP.2529.MA','SP.POP.3034.MA','SP.POP.3539.MA','SP.POP.4044.MA','SP.POP.4549.MA','SP.POP.5054.MA','SP.POP.5559.MA','SP.POP.6064.MA','SP.POP.6569.MA','SP.POP.7074.MA','SP.POP.7579.MA','SP.POP.80UP.MA']
popul_female_series_code = ['SP.POP.0004.FE','SP.POP.0509.FE','SP.POP.1014.FE','SP.POP.1519.FE','SP.POP.2024.FE','SP.POP.2529.FE','SP.POP.3034.FE','SP.POP.3539.FE','SP.POP.4044.FE','SP.POP.4549.FE','SP.POP.5054.FE','SP.POP.5559.FE','SP.POP.6064.FE','SP.POP.6569.FE','SP.POP.7074.FE','SP.POP.7579.FE','SP.POP.80UP.FE']
popul_male_age_wise = gpe.loc[gpe['Series Code'].isin(popul_male_series_code)]
popul_male_age_wise[['Country Name']].describe()
popul_female_age_wise = gpe.loc[gpe['Series Code'].isin(popul_female_series_code)]
popul_female_age_wise[['Country Name']].describe()
popul_total_age_wise = pd.concat([popul_male_age_wise, popul_female_age_wise])
popul_total_age_wise[['Country Name']].describe()
popul_country_wise = popul_total_age_wise.groupby('Country Name').sum()
popul_country_wise.head(5)
popul_total_country_wise_1 = gpe.loc[gpe['Series Code'] == 'SP.POP.TOTL']
popul_total_country_wise_1.head(5)
popul_total = popul_total_country_wise_1.sum()
popul_total.head(7)
popul_total_bn = popul_total[4:]/10 ** 9
popul_total_bn = popul_total_bn.reset_index()
popul_total_bn.head(5)
popul_total_bn['Year'] = popul_total_bn['index'].str[:4].astype(int)
popul_total_bn['Population'] = popul_total_bn[0].astype(float)
popul_total_bn.drop(['index',0],axis=1,inplace=True)
popul_total_bn.head(5)
popul_total_bn.info()
yp_lp = sn.lineplot(x='Year', y='Population', data=popul_total_bn)
yp_lp.set_title('Yearly Population estimates')
yp_lp.set_yticklabels([str(i) + 'B' for i in range(0,int(max(popul_total_bn['Population'])),10)])
plt.show()