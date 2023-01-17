# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

pd.options.mode.chained_assignment = None
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pd.set_option('max_columns',None)
soccer=pd.read_csv('/kaggle/input/soccer-players-statistics/FullData.csv')
soccer
salaries=pd.read_csv('/kaggle/input/us-major-league-soccer-salaries/mls-salaries-2017.csv')
salaries['name']=salaries['first_name']+' '+salaries['last_name']
salaries
mls=soccer[soccer['Name'].isin(salaries['name'].tolist())]
mls
mls['base_salary']=[salaries[salaries['name'].isin([player])].reset_index()['base_salary'][0]
                   for player in mls['Name']]
mls['guaranteed_compensation']=[salaries[salaries['name'].isin([player])].reset_index()['guaranteed_compensation'][0]
                   for player in mls['Name']]
mls
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(30,20))
sns.heatmap(mls.corr(),annot=True,linewidth=0.5)
df=pd.DataFrame(mls.corr()['guaranteed_compensation']).reset_index()
df['Beat Threshold']=abs(df['guaranteed_compensation'])>0.35

sns.lmplot(x='index', y="guaranteed_compensation", data=df,hue='Beat Threshold',fit_reg=False,height=4,
           aspect=4).set_xticklabels(rotation=90)
features=['Rating', 'Reactions','Vision','Composure','Freekick_Accuracy']
y=mls['guaranteed_compensation']
X=mls[features]

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state=1, test_size=0.4)
basic_model = DecisionTreeRegressor(random_state=1)
basic_model.fit(train_X, train_y)
predictions=basic_model.predict(val_X)

df=pd.DataFrame(val_X)
df['prediction']=predictions
df['Name']=[mls['Name'][index] for index in df.reset_index()['index']]
df['guaranteed_compensation']=[mls['guaranteed_compensation'][index] for index in df.reset_index()['index']]
#df=df[['name','ID','r','h','double','rbi','tb','pos','salary','prediction']]
df['excess']=df['prediction']-df['guaranteed_compensation']
df['absolute error']=abs(df['excess'])
df
df.mean()['absolute error']
df.mean()['guaranteed_compensation']
df['percent error']=100*(df['absolute error']/df['guaranteed_compensation'])
df.sort_values(by='percent error')
sns.lmplot(data=df,x='guaranteed_compensation',y='percent error')
sns.lmplot(data=df,x='guaranteed_compensation',y='percent error').set(ylim=(0, 400),xlim=(1000000,0))