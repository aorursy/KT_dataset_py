import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



#%matplotliib inline
data = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

data = data.fillna(0)



# Creating new column for placement with binary values

Placed = {'Placed': 1,'Not Placed': 0}  

data['Placed_num'] = [ Placed[item] for item in data.status] 

data.head()
data.describe()
sns.pairplot(data)
data.columns
df_marks = data[['ssc_p','hsc_p','degree_p','etest_p','mba_p','status']].groupby('status').mean().add_prefix('mean_')

                

df_marks.plot(kind='bar')
df_gender_placed = pd.pivot_table(data[['gender','status','Placed_num']], index=['gender','status'], aggfunc='count')

plot = df_gender_placed.loc['F'].plot.pie(y='Placed_num',labels=['Female Not Placed','Female Placed'],figsize=(6, 6),autopct='%1.1f%%')
plot = df_gender_placed.loc['M'].plot.pie(y='Placed_num',labels=['Male Not Placed','Male Placed'],figsize=(6, 6),autopct='%1.1f%%')
sns.countplot(data['status'],hue=data['degree_t'],palette='bright')
sns.countplot(data['status'],hue=data['workex'],palette='bright')
sns.countplot(data['status'],hue=data['specialisation'],palette='bright')
x = sns.countplot(data['status'],hue=data['ssc_b'],palette='bright')
gender = {'M': 1,'F': 2}  

hsc_b = {'Others': 1,'Central': 2}

hsc_s = {'Commerce': 1,'Science': 2, 'Arts': 3} 

ssc_b = {'Others': 1,'Central': 2}

degree_t = {'Sci&Tech':1,'Comm&Mgmt':2,'Others':1}

workex = {'No':1,'Yes':2}

specialisation = {'Mkt&HR':1,'Mkt&Fin':2}

status = {'Placed': 1,'Not Placed': 0}



data['gender'] = [ gender[item] for item in data.gender]

data['hsc_b'] = [ hsc_b[item] for item in data.hsc_b] 

data['hsc_s'] = [ hsc_s[item] for item in data.hsc_s]

data['ssc_b'] = [ ssc_b[item] for item in data.ssc_b] 

data['degree_t'] = [ degree_t[item] for item in data.degree_t] 

data['workex'] = [ workex[item] for item in data.workex] 

data['specialisation'] = [ specialisation[item] for item in data.specialisation] 

data['status'] = [ Placed[item] for item in data.status] 
data.head(2)
data.drop(['sl_no','salary','Placed_num'],axis=1,inplace=True)

data.head(2)
X=data.drop('status',axis=1)

y=data['status']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

from sklearn.linear_model import LogisticRegression



logmodel = LogisticRegression(solver='liblinear',dual=False)

logmodel.fit(X_train,y_train)
predictions  = logmodel.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,predictions)

cm
from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))