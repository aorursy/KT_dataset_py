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
import numpy as np 

import pandas as pd



train = pd.read_csv("../input/avazu-ctr-train/train.csv", nrows = 20000000)
train.info()
pd.set_option('display.max_columns',None)

train.head()
train.describe()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



params = {'legend.fontsize':'x-large',

         'figure.figsize':(30,30),

         'axes.labelsize':'x-large',

         'axes.titlesize':'x-large',

         'xtick.labelsize':'x-large',

         'ytick.labelsize':'x-large'}



sns.set_style('whitegrid')

sns.set_context('talk')



plt.rcParams.update(params)

pd.options.display.max_colwidth = 600
numerical_features =['hour','C1','C14','C15','C16','C17','C18','C19','C20','C21','click']

train[numerical_features].hist()
corrMatt = train[['C1','C14','C15','C16','C17','C18','C19','C20','C21','click','banner_pos','hour']].corr()

mask = np.array(corrMatt)

mask[np.tril_indices_from(mask)] = False

sns.heatmap(corrMatt,mask=mask,vmax=.8,square=True,annot=True)
train_data = pd.DataFrame(columns = ["id","click","hour", "banner_pos", "site_id", "site_domain", "site_category","app_id", "app_domain", "app_category",

                             "device_id","device_ip","device_model","device_type","device_conn_type","C14","C15","C16","C17","C18","C19","C20","C21"])



#train_data

hour_index = 14102200



while(hour_index < 14102224):

    data = train.loc[train['hour'] == hour_index]

    train_data = pd.concat([train_data,data])

    hour_index = hour_index + 1

    

del hour_index
sns.stripplot(x="banner_pos",y="click",data=train_data)

plt.title("2014年10月22日24个时段散点图")

plt.show()
item = train_data['hour'].value_counts()

item.index
for hour_index in item.index:

    data = train_data.loc[train_data['hour'] == hour_index]

    sns.stripplot(x="banner_pos",y="click",data=data)

    plt.title(hour_index)

    plt.show()