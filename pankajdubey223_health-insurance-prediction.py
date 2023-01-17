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



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

plt.style.use('dark_background')

data = pd.read_csv('/kaggle/input/insurance/insurance.csv')

data.head()
data.shape
data.describe(include = 'all')
import seaborn as sns

sns.pairplot(data)
data.hist(by = 'sex', column = 'charges')
data.hist(by = 'smoker', column = 'charges')
data.groupby(data['children']).count()
data.groupby(data['region']).count()
data.groupby(data['smoker']).count()
sns.boxplot(x = 'sex', y = 'age', data = data)
sns.boxplot(x= 'sex' , y = 'charges', data = data)
sns.boxplot(x= 'sex' , y = 'bmi', data = data)
data.corr()
import matplotlib.pyplot as plt

corr_new_train=data.corr()

plt.figure(figsize=(5,15))

sns.heatmap(corr_new_train[['charges']].sort_values(by=['charges'],ascending=False).head(30),annot_kws={"size": 16},vmin=-1, cmap='PiYG', annot=True)

sns.set(font_scale=2)
data['sex'] = pd.get_dummies(data['sex'],drop_first = True )

data['smoker'] = pd.get_dummies(data['smoker'],drop_first = True )

#data['region'] = pd.get_dummies(data['region'],drop_first = True )



#new_data = pd.get_dummies(data['region'],drop_first = True )

#data = pd.concat([new_data,data], axis = 1)

data.head()
data['region'] = pd.get_dummies(data['region'])
plt.style.use('dark_background')

fig, axes = plt.subplots(4, 2,figsize=(20,80))

fig.subplots_adjust(hspace=0.2)

colors=[plt.cm.prism_r(each) for each in np.linspace(0, 1, len(data.columns))]

for i,ax,color in zip(data.columns,axes.flatten(),colors):

    sns.regplot(x=data[i], y=data["charges"], fit_reg=True,marker='o',scatter_kws={'s':50,'alpha':0.8},color=color,ax=ax)

    plt.xlabel(i,fontsize=12)

    plt.ylabel('charges',fontsize=12)

    ax.set_yticks(np.arange(100,90001,10000))

    ax.set_title('charges'+' - '+str(i),color=color,fontweight='bold',size=20)
X = data.iloc[:,:-1].values

y = data.iloc[:,-1].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.tree import DecisionTreeRegressor

clf1 = DecisionTreeRegressor()

clf1.fit(X_train,y_train)
pred = clf1.predict(X_test)

clf1.score(X_test,y_test)
from sklearn import metrics

metrics.scorer.r2_score(pred,y_test)
sns.set(style = 'darkgrid', color_codes = True)





with sns.axes_style('white'):

    sns.jointplot(x = y_test, y = pred, kind = 'reg', color = 'k')
from xgboost import XGBRegressor



my_model = XGBRegressor(n_estimators=10000, learning_rate=0.12, n_job= 2)

my_model.fit(X_train, y_train,

             early_stopping_rounds=10, 

             eval_set=[(X_train, y_train)], 

             verbose=1)

y_head=my_model.predict(X_test)

print('-'*10+'XGBRegressor'+'-'*10)

print('R square Accuracy: ',metrics.scorer.r2_score(y_test,y_head))

print('Mean Absolute Error Accuracy: ',metrics.scorer.mean_absolute_error(y_test,y_head))

print('Mean Squared Error Accuracy: ',metrics.scorer.mean_squared_error(y_test,y_head))
sns.set(style = 'darkgrid', color_codes = True)





with sns.axes_style('white'):

    sns.jointplot(x = y_test, y = y_head, kind = 'reg', color = 'k')