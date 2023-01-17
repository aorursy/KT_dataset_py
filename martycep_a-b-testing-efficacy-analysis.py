# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from pandasql import sqldf 

import seaborn as sns, numpy as np

from scipy.stats import binom



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/ab-testing-results/ab_data.csv')



data['misaligned']=((data.group=='treatment') & (data.landing_page=='old_page')) | ((data.group=='control') & (data.landing_page=='new_page'))



ab_data =data.query('misaligned==False')

ab_data.shape
ab_data.head()
##Check to see if there are any misaligned rows where treatment and new landing page dont align

((ab_data.group=='treatment') & (ab_data.landing_page=='old_page')).sum()+ ((ab_data.group=='control') & (ab_data.landing_page=='new_page')).sum()
# Check the conversion rates between the old and new pages 

ab_data.pivot_table(['converted'],['group'])
import statsmodels.api as sm

#number of conversions for each page, and the number of individuals who received each page

convert_old = (ab_data.query('landing_page=="old_page"')['converted']==1).sum()

convert_new = (ab_data.query('landing_page=="new_page"')['converted']==1).sum()

n_old = (ab_data['landing_page']=='old_page').sum()

n_new=(ab_data['landing_page']=='new_page').sum()



convert_old, convert_new, n_old, n_new         
z_score, p_value = sm.stats.proportions_ztest([convert_new, convert_old], [n_new, n_old], alternative='larger')

z_score, p_value