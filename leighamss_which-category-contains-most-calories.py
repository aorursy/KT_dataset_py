import numpy as np 

import seaborn as sns 

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams['figure.figsize'] = 11,8
mc= pd.read_csv('../input/menu.csv')

#Clean Data

mc.isnull().any()
Most_Calories = mc.groupby('Category').max().sort_values('Calories',ascending=False)

sns.swarmplot(data =Most_Calories, x= Most_Calories.index,y = 'Calories',hue ='Item',size =10 )

plt.tight_layout()
Least_Calories = mc.groupby('Category').min().sort_values('Calories',ascending=False)

sns.swarmplot(data =Least_Calories, x= Least_Calories.index,y = 'Calories',hue ='Item',size =10 )

plt.tight_layout()
amount = mc.groupby('Category').count()

amount_count = amount[['Item']].sort_values('Item',ascending= False)

amount_count
sns.boxplot(data= mc, x = 'Category',y = 'Dietary Fiber')

plt.tight_layout

plt.show()