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
print('Hello World')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.style.use('ggplot')
med_insur = pd.read_csv('../input/insurance/insurance.csv')
med_insur.head(10)
med_insur.info()
a = med_insur.isnull().sum()
for index, value in a.items():
    print(f" In column '{index}', there is {value} null objects")
mean_age = med_insur['age'].mean()
print(f'The avarage of ages is {mean_age}')
mean_bmi = med_insur['bmi'].mean()
print(f'The avarage of BMI is {mean_bmi}')
mean_children = med_insur['children'].mean()
print(f"The avarage number of children is {mean_children}")
mean_charge = med_insur['charges'].mean()
print(f'The avarage of charges is {mean_charge}')
fig, axs = plt.subplots(2,2,figsize=(15,15))
fig.subplots_adjust(hspace=0.2, wspace=0.2)

ax1 = sns.countplot(x='region', data=med_insur, ax=axs[0][0])
ax2 = sns.countplot(x='children', data=med_insur, ax=axs[0][1])
ax3 = sns.countplot(x='sex', data=med_insur, ax=axs[1][0])
ax4 = sns.countplot(x='smoker', data=med_insur, ax=axs[1][1])
#How bmi changes with smoking practice??
ax1 = sns.violinplot(x='smoker', y='bmi', hue='sex', data=med_insur)
#Now we can transform our categorical data in dummy variables!
dummy_med_insur = pd.DataFrame()
dummy_med_insur['age'] = med_insur['age']
dummy_med_insur['sex'] = med_insur['sex'].map({'female':0, 'male':1})
dummy_med_insur['bmi'] = med_insur['bmi']
dummy_med_insur['children'] = med_insur['children']
dummy_med_insur['smoker'] = med_insur['smoker'].map({'yes':1, 'no':0})
dummy_med_insur['region'] = med_insur['region'].map({'southeast':0,'southwest':1,'northwest':2,'northeast':3})
dummy_med_insur['charges'] = med_insur['charges']
 
dummy_med_insur.head()
ax = plt.figure(figsize=(8,5))
ax = sns.heatmap(dummy_med_insur.corr(), annot=True)
ax = sns.pairplot(dummy_med_insur)
ax1 = sns.lmplot(x="bmi", y="charges", hue="smoker", data=dummy_med_insur)
ax2 = sns.lmplot(x="age", y="charges", hue="smoker", data=dummy_med_insur)
bmi_sep = dummy_med_insur
bins = [0,30.66,100]
slots = [0,1]
bmi_sep['bmi'] = pd.cut(bmi_sep['bmi'], bins=bins, labels=slots)
bmi_sep.head()
#Let's take only the smokers!!!

df_mask1=bmi_sep['smoker']==1
filtered_df1 = bmi_sep[df_mask1]
filtered_df1.head()
ax1 = sns.lmplot(x="age", y="charges", hue="bmi", data=filtered_df1)
#Let's take only the NO smokers now...!!!

df_mask2=bmi_sep['smoker']==0
filtered_df2 = bmi_sep[df_mask2]
filtered_df2.head()
ax1 = sns.lmplot(x="age", y="charges", hue="bmi", data=filtered_df2)
import statsmodels.api as sm
from statsmodels.formula.api import ols

print()
print('Smokers')
print()

mod = ols('charges~bmi+age+bmi*age', data=filtered_df1).fit()
aov = sm.stats.anova_lm(mod, type=2)
print(aov)

print()
print('No Smokers')
print()

mod2 = ols('charges~bmi+age+bmi*age', data=filtered_df2).fit()
aov2 = sm.stats.anova_lm(mod2, type=2)
print(aov2)
y_data =filtered_df2['charges'] 
x_data = filtered_df2.drop('charges', axis=1)
x_data.head(100)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=1)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                  ('linear', LinearRegression(fit_intercept=False))])
# fit to an order-2 polynomial data
x = X_train['age']
y = y_train
model = model.fit(x[:, np.newaxis], y)
coefs = model.named_steps['linear'].coef_
print(coefs)
ax = plt.scatter(x_data['age'],y_data)
m = np.linspace(15,80,1064)
ax2 = plt.plot(m,coefs[2]*m**2+coefs[1]*m+coefs[0], color = 'blue')
n= np.linspace(15,80,320)
a = coefs[2]*n**2+coefs[1]*n+coefs[0]
b = y_test
sns.distplot(a-b, kde=False)
#Metrics
from sklearn import metrics
print('MAE', metrics.mean_absolute_error(b,a))
med_insur = pd.read_csv('../input/insurance/insurance.csv')
dummy_med_insur = pd.DataFrame()
dummy_med_insur['age'] = med_insur['age']
dummy_med_insur['sex'] = med_insur['sex'].map({'female':0, 'male':1})
dummy_med_insur['bmi'] = med_insur['bmi']
dummy_med_insur['children'] = med_insur['children']
dummy_med_insur['smoker'] = med_insur['smoker'].map({'yes':1, 'no':0})
dummy_med_insur['region'] = med_insur['region'].map({'southeast':0,'southwest':1,'northwest':2,'northeast':3})
dummy_med_insur['charges'] = med_insur['charges']
df_mask12=dummy_med_insur['smoker']==1
filtered_df12 = dummy_med_insur[df_mask12]
filtered_df12.head()
filtered_df12.describe().T
y_data_nfum = filtered_df12['charges'] 
x_data_nfum = filtered_df12.drop('charges', axis=1)
x_data_nfum.head(100)

X_train, X_test, y_train, y_test = train_test_split(x_data_nfum, y_data_nfum, test_size=0.3, random_state=1)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
predict = lm.predict(X_test)
plt.scatter(y_test,predict)
print(lm.coef_)
sns.distplot(y_test-predict, kde=False)
df = dummy_med_insur


bins = [17,35,55,1000]
slots = ['Young adult','Senior Adult','Elder']

bins2 = [0,18.5,25,30,100]
slots2 = ['Subpeso', 'Normal', 'Sobrepeso', 'Obeso' ]

df['bmi_range']=pd.cut(dummy_med_insur['bmi'],bins=bins2,labels=slots2)

df['Age_range']=pd.cut(dummy_med_insur['age'],bins=bins,labels=slots)
plt.figure(figsize=(25, 16))
plt.subplot(2,3,1)
sns.violinplot(x = 'smoker', y = 'charges', data = df)
plt.title('Smoker vs Charges',fontweight="bold", size=20)
plt.subplot(2,3,2)
sns.violinplot(x = 'children', y = 'charges', data = df,palette="husl")
plt.title('Children vs Charges',fontweight="bold", size=20)
plt.subplot(2,3,3)
sns.violinplot(x = 'sex', y = 'charges', data = df, palette= 'husl')
plt.title('Sex vs Charges',fontweight="bold", size=20)
plt.subplot(2,3,4)
sns.violinplot(x = 'region', y = 'charges', data = df,palette="bright")
plt.title('Region vs Charges',fontweight="bold", size=20)
plt.subplot(2,3,5)
sns.violinplot(x = 'Age_range', y = 'charges', data = df, palette= 'husl')
plt.title('Age vs Charges',fontweight="bold", size=20)
plt.subplot(2,3,6)
sns.violinplot(x = 'bmi_range', y = 'charges', data = df, palette= 'husl')
plt.title('Bmi vs Charges',fontweight="bold", size=20)



plt.show()
