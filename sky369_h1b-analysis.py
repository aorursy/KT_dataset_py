# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
from subprocess import check_output
from matplotlib.ticker import FuncFormatter
from scipy.stats import gaussian_kde
import math
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import re
# Any results you write to the current directory are saved as output.

df=pd.read_csv('..//input//h1b_kaggle.csv')


def get_normalized_string(x):
    try:
        string=re.sub(r"[^a-zA-Z0-9]+", ' ', x)
        return(string)
    except:
        return('')

    
    
df['CASE_STATUS']=df['CASE_STATUS'].map(lambda x: get_normalized_string(x))
df['EMPLOYER_NAME']=df['EMPLOYER_NAME'].map(lambda x: get_normalized_string(x))
df['JOB_TITLE']=df['JOB_TITLE'].map(lambda x: get_normalized_string(x))
df['WORKSITE']=df['WORKSITE'].map(lambda x: get_normalized_string(x))

df['SOC_NAME']=df['SOC_NAME'].map(lambda x: get_normalized_string(x))

#Is the number of petitions with Informatica developer job title increasing over time?
plt.plot(df[df['JOB_TITLE']=="INFORMATICA DEVELOPER"][['JOB_TITLE','YEAR']].groupby(by=['YEAR']).count())

plt.xlabel('Year')
plt.ylabel('Number of applicantions')
plt.title('H1B applications with Job title Informatica developer ')


