# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import matplotlib 

import matplotlib.pyplot as plt

import sklearn

%matplotlib inline

import seaborn as sns

sns.set(style="whitegrid", color_codes=True)

import matplotlib.pyplot as plt 

plt.rcParams["figure.figsize"] = [16, 12]

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

filenames = check_output(["ls", "../input"]).decode("utf8").strip()

df = pd.read_csv("../input/train.csv")

dg = pd.read_csv("../input/test.csv") 

aId = dg['ID'] 
def str2age(st): 

    if isinstance(st, str):



        num, unit = (st.split())

        num = int(num)

        if unit[:4] == 'year':

            pass

        elif unit[:5] == 'month':

            num /= 12

        elif unit[:4] =='week':

            num /= 52



        else:

            pass



        return num

    else:

        return np.NaN

     

        

        

df['AgeuponOutcome'] = df['AgeuponOutcome'].apply(str2age)

dg['AgeuponOutcome'] = dg['AgeuponOutcome'].apply(str2age)

df['DateTime'] = pd.to_datetime(df['DateTime'], format = '%Y-%m-%d %H:%M:%S')

dg['DateTime'] = pd.to_datetime(dg['DateTime'], format = '%Y-%m-%d %H:%M:%S')
dg.head()
dg.isnull().sum()
varnames = df.columns.values



for varname in varnames:

    if varname not in ['Name', 'AnimalID', 'DateTime'] and df[varname].dtype == 'object':

        lst = df[varname].unique()

        print(varname + " : " + str(len(lst)) + " values such as " + str(lst))

df['noName'] = df['Name'].isnull()

dg['noName'] = dg['Name'].isnull()
del df['Name']

del df['AnimalID']

del dg['Name']

del dg['ID']
df.head()