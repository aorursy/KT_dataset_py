#import warnings

import warnings

warnings.filterwarnings('ignore')
#import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import r2_score
#read the file

fish_data = pd.read_csv('../input/fish-market/Fish.csv')

fish_data.head()
#shape of the df:

fish_data.shape
#info of the df

fish_data.info()
fish_data.describe(percentiles = [0.05,0.10,0.25,0.50,0.75,0.90,0.99])
#let us check for null

fish_data.isnull().sum()
fish_data.Species.value_counts()
sns.countplot(data = fish_data, x = 'Species')
sns.pairplot(data= fish_data, x_vars = ['Length1','Length2','Length3','Height','Width'], y_vars = 'Weight', hue = 'Species')
#let us check the correlation

sns.heatmap(fish_data.corr(), annot = True)
#Variable Weight

sns.boxplot(fish_data['Weight'])
#checking outlier rows

fish_weight = fish_data['Weight']

Q3 = fish_weight.quantile(0.75)

Q1 = fish_weight.quantile(0.25)

IQR = Q3-Q1

lower_limit = Q1 -(1.5*IQR)

upper_limit = Q3 +(1.5*IQR)
weight_outliers = fish_weight[(fish_weight <lower_limit) | (fish_weight >upper_limit)]

weight_outliers
sns.boxplot(fish_data['Length1'])
#checking outlier rows

fish_Length1 = fish_data['Length1']

Q3 = fish_Length1.quantile(0.75)

Q1 = fish_Length1.quantile(0.25)

IQR = Q3-Q1

lower_limit = Q1 -(1.5*IQR)

upper_limit = Q3 +(1.5*IQR)

length1_outliers = fish_Length1[(fish_Length1 <lower_limit) | (fish_Length1 >upper_limit)]

length1_outliers
sns.boxplot(fish_data['Length2'])
#checking outlier rows

fish_Length2 = fish_data['Length2']

Q3 = fish_Length2.quantile(0.75)

Q1 = fish_Length2.quantile(0.25)

IQR = Q3-Q1

lower_limit = Q1 -(1.5*IQR)

upper_limit = Q3 +(1.5*IQR)

length2_outliers = fish_Length2[(fish_Length2 <lower_limit) | (fish_Length2 >upper_limit)]

length2_outliers
sns.boxplot(fish_data['Length3'])
#checking outlier rows

fish_Length3 = fish_data['Length3']

Q3 = fish_Length3.quantile(0.75)

Q1 = fish_Length3.quantile(0.25)

IQR = Q3-Q1

lower_limit = Q1 -(1.5*IQR)

upper_limit = Q3 +(1.5*IQR)

length3_outliers = fish_Length3[(fish_Length3 <lower_limit) | (fish_Length3 >upper_limit)]

length3_outliers
sns.boxplot(fish_data['Height'])
sns.boxplot(fish_data['Width'])
fish_data[142:145]
#let us drop these rows:

df = fish_data.drop([142,143,144])
# let us check our df after removal of outliers

df.describe(percentiles = [0.05,0.10,0.25,0.50,0.75,0.90,0.99])
#creating dummies - to handle categorical variable.

#species_dummies = pd.get_dummies(df['Species'], prefix = 'Species' , drop_first = True)
# final_df = pd.concat([df,species_dummies], axis =1)

# final_df.head()
#dropping the original column as we have created dummies

#final_df = final_df.drop(['Species'], axis =1)
#final_df.shape
df_train, df_test = train_test_split(df, train_size = 0.7, test_size = 0.3, random_state =100)
df_train.shape
df_test.shape
scaler = StandardScaler()
scaling_columns = ['Weight', 'Length1','Length2','Length3','Height','Width']

df_train[scaling_columns] = scaler.fit_transform(df_train[scaling_columns])

df_train.describe()
y_train = df_train['Weight']

X_train = df_train.iloc[:,2:7]
X_train.head()
X_train_sm = sm.add_constant(X_train)

model1 = sm.OLS(y_train,X_train_sm).fit()
print(model1.summary())
VIF = pd.DataFrame()

VIF['Features'] = X_train.columns

VIF['vif'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

VIF['vif'] = round(VIF['vif'] ,2)

VIF = VIF.sort_values(by='vif',ascending = False)

VIF
X2 = X_train.drop(['Length2'], axis =1)

X2_sm = sm.add_constant(X2)



model2 = sm.OLS(y_train,X2_sm).fit()
print(model2.summary())
#vif

VIF = pd.DataFrame()

VIF['Features'] = X2.columns

VIF['vif'] = [variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])]

VIF['vif'] = round(VIF['vif'] ,2)

VIF = VIF.sort_values(by='vif',ascending = False)

VIF
X3 = X2.drop(['Length3'], axis =1)

X3_sm = sm.add_constant(X3)



model3 = sm.OLS(y_train,X3_sm).fit()
print(model3.summary())
#vif

VIF = pd.DataFrame()

VIF['Features'] = X3.columns

VIF['vif'] = [variance_inflation_factor(X3.values, i) for i in range(X3.shape[1])]

VIF['vif'] = round(VIF['vif'] ,2)

VIF = VIF.sort_values(by='vif',ascending = False)

VIF
y_train_pred = model3.predict(X3_sm)

y_train_pred.head()
residual = y_train - y_train_pred

sns.distplot(residual)
#plotting y_train and y_train_pred

c = [i for i in range(1,110,1)]

plt.plot(c, y_train,color = 'Blue')

plt.plot(c, y_train_pred,color = 'red')

plt.title('Test(Blue) vs pred(Red)')
# treating test columns same way as train dataset

df_test[scaling_columns] = scaler.transform(df_test[scaling_columns])

df_test.describe()
y_test = df_test['Weight']

X_test = df_test.iloc[:,2:7]
cols = X3.columns

cols
# considering only those columns which was part of our model 3.

X_test = X_test[cols]

X_test.columns
#predicting

X_test_sm = sm.add_constant(X_test)

y_pred = model3.predict(X_test_sm)
y_pred.head()
r_square = r2_score(y_test,y_pred)

r_square
#plotting y_test and y_pred

c = [i for i in range(1,48,1)]

plt.plot(c, y_test,color = 'Blue')

plt.plot(c, y_pred,color = 'red')

plt.title('Test(Blue) vs pred(Red)')