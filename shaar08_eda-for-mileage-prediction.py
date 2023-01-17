import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
df=pd.read_csv("../input/autompg/auto-mpg.data",na_values='?',comment='\t',sep=' ',skipinitialspace=True)
data=df.copy()
data.head()
cols=['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
data.columns=cols
data.head()
print(data.info())  # 8 columns, 398 observations
print("-"*60)
print(data.isna().sum()) # Horsepower 6 missing values

data.describe()
#Study Horsepower column 
sns.boxplot(data['Horsepower'])
plt.show()
median=data['Horsepower'].median()
data['Horsepower']=data['Horsepower'].fillna(median)
data.info()
print(data['Cylinders'].value_counts()/len(data)) # Show Proportion of data in Cylinder
print("-"*40)
print(data['Origin'].value_counts())              # Frequency Count of values in Origin
origin_dict={
    1:'India',
    2:'USA',
    3:'Germany'
}
data['Origin']=data.Origin.replace(origin_dict)
data.head()
# Encoding the categorical columns
data=pd.get_dummies(data,prefix='',prefix_sep='')
data.tail()
#Coorelation 
sns.pairplot(data,diag_kind='kde')
plt.show()
corr_matrix=data.corr()
corr_matrix['MPG'].sort_values(ascending=False)
data['displacement_on_power'] = data['Displacement'] / data['Horsepower']
data['weight_cylinder'] = data['Weight'] / data['Cylinders']
data['acc_power'] = data['Acceleration'] / data['Horsepower']
data['acc_cylinder'] = data['Acceleration'] / data['Cylinders']

corr_matrix= data.corr()
corr_matrix['MPG'].sort_values(ascending=False)