import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import scipy.stats as st

import os

%matplotlib inline
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/cancer-dataset-aggregated/cancer_reg.csv',encoding='latin-1')
df.shape
df.head()
df.info()
df['binnedInc']=df['binnedInc'].str.replace('(','')

df['binnedInc']=df['binnedInc'].str.replace('[','')

df['binnedInc']=df['binnedInc'].str.replace(']','')
x=df['binnedInc'].str.split(',',expand=True).astype(float)
x=df['binnedInc'].str.split(',',expand=True).astype(float)

y=(x[0]+x[1])/2

df['binnedInc']=y

df.head()
df.describe()
for i in df:

    if (i=='Geography'):

        continue

    else:

        plt.figure()

        df.boxplot(column=[i])
print('count of outliers below lower whisker is :',(df['TARGET_deathRate']<df['TARGET_deathRate'].quantile(0.25)-(1.5*(st.iqr(df['TARGET_deathRate'])))).sum())
print('count of outliers above upper whisker is :',(df['TARGET_deathRate']>df['TARGET_deathRate'].quantile(0.75)+(1.5*(st.iqr(df['TARGET_deathRate'])))).sum())
# since target variable has outliers less then 10% of the data, drop the outliers

df1=df[(df['TARGET_deathRate']>df['TARGET_deathRate'].quantile(0.25)-(1.5*(st.iqr(df['TARGET_deathRate']))))&(df['TARGET_deathRate']<df['TARGET_deathRate'].quantile(0.75)+(1.5*(st.iqr(df['TARGET_deathRate']))))]
df1.shape
df1.isna().sum()
#since 'PctEmployed16_Over' have missing value less then 10% it is imputed with median

df1['PctEmployed16_Over']=df1['PctEmployed16_Over'].fillna(df1['PctEmployed16_Over'].median())

for i in df1.columns:

    if (i=='Geography' or i=='PctSomeCol18_24' or i=='PctPrivateCoverageAlone'):

        continue

    else:

        plt.figure()

        sns.distplot(df1[i],kde=False,color='g',bins=10,rug=True)
# corr matrix

df1.corr()
plt.figure(figsize=(30,15))

sns.heatmap(df1.corr(),annot=True)
import warnings 

warnings.filterwarnings('ignore')

import statsmodels.api as sm

#drop the 'geography' ,'pctsomecol18_24' ,'pctprivatecoveragealone' as they have the most missing values.

# drop the dependent variable 

x=df1.drop(['Geography' ,'PctSomeCol18_24','PctPrivateCoverageAlone','TARGET_deathRate'],axis=1)

x_constant = sm.add_constant(x)

y=df1['TARGET_deathRate']

y1=list(y)
model = sm.OLS(y1,x_constant).fit()

model.summary()
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = [variance_inflation_factor(x_constant.values, i) for i in range(x_constant.shape[1])]

pd.DataFrame({'vif': vif[1:]}, index=x.columns)
# features which has highly multicollinarity

vif_=pd.DataFrame({'vif': vif[1:]}, index=x.columns)

vif_[vif_['vif']>10]
features=[]

a=model.pvalues

for i in range(a.shape[0]):

    if  a[i]<0.05:

        features.append(a.index[i])

    else:

        continue

print(features)
X=df1[[ 'avgAnnCount', 'avgDeathsPerYear', 'incidenceRate', 'popEst2015', 'MedianAgeMale', 'PercentMarried', 'PctHS18_24', 'PctHS25_Over', 'PctBachDeg25_Over', 'PctEmployed16_Over', 'PctPrivateCoverage', 'PctEmpPrivCoverage', 'PctOtherRace', 'PctMarriedHouseholds', 'BirthRate']]
for i in X.columns:

    plt.figure()

    sns.scatterplot(x=df1['TARGET_deathRate'],y=i,data=df1)
for i in X.columns:

        plt.figure()

        sns.kdeplot(X[i])
nor=[]

for i in  X.columns:

    if st.shapiro(X[i])[1]<0.05:

        nor.append(i)

    else:

        continue

print(nor)
to_t=X[['avgAnnCount', 'avgDeathsPerYear', 'incidenceRate', 'popEst2015', 'MedianAgeMale', 'PercentMarried', 'PctHS18_24', 'PctHS25_Over', 'PctBachDeg25_Over', 'PctEmployed16_Over', 'PctPrivateCoverage', 'PctEmpPrivCoverage', 'PctOtherRace', 'PctMarriedHouseholds', 'BirthRate']]
right=[]

left=[]

for i in  to_t.columns:

    if st.skew(to_t[i])>0.5:

        right.append(i)

    elif st.skew(to_t[i])<-0.5:

            left.append(i)

    else:

        continue

print('right skwed :\n ', right,'\n\nleft skwed :\n ',left)
to_t['avgDeathsPerYear']=np.log((to_t['avgDeathsPerYear']))

to_t['avgAnnCount']=np.log((to_t['avgAnnCount']))

to_t['popEst2015']=np.log((to_t['popEst2015']))

to_t['PctBachDeg25_Over']=np.log((to_t['PctBachDeg25_Over']))

to_t['PctOtherRace']=(np.log((to_t['PctOtherRace'])+1))

to_t['BirthRate']=np.sqrt((to_t['BirthRate']))

to_t['PercentMarried']=((to_t['PercentMarried'])**2)

to_t['PctMarriedHouseholds']=((to_t['PctMarriedHouseholds'])**2)
for i in to_t:

    a=st.skew(to_t[i])

    print(i,':  ',a)
for i in to_t.columns:

        plt.figure()

        sns.kdeplot(to_t[i])
# data split into train and test

from sklearn.model_selection import train_test_split

X_train, X_test , y_train, y_test = train_test_split(to_t,y1, test_size = 0.30, random_state = 20)
from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)
# r2 for the train data

print('r2 score for train data :',lin_reg.score(X_train, y_train))
# r2 for the test data

print('r2 score for train data :',lin_reg.score(X_test, y_test))
#  y predict for test data

y_predict=lin_reg.predict(X_test)
#rmse score

from sklearn.metrics import mean_squared_error as ms

print('rmse score :',np.sqrt(ms(y_predict,y_test)))
sns.scatterplot(y_predict,y_test)

plt.xlabel('y_predict')

plt.ylabel('y_test')

plt.show()
#distribution of 'y_predict-y_test'

a=y_predict-y_test

sns.kdeplot(np.array(a))

plt.xlabel('y_predict-y_test')

plt.show()
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree = 2) 

X_poly = poly.fit_transform(to_t) 

  

X_train, X_test , y_train, y_test = train_test_split(X_poly,y1, test_size = 0.30, random_state = 207)
lin2 = LinearRegression() 

lin2.fit(X_train,y_train) 
#r2 for the train data

print('r2 score for train data :',lin2.score(X_train, y_train))
#r2 for the test data

print('r2 score for test data :',lin2.score(X_test, y_test))
# predict y

y_predict=lin2.predict(X_test)
#  rmse score

print('rmse score :',np.sqrt(ms(y_predict,y_test)))
sns.scatterplot(y_predict,y_test)

plt.xlabel('y_predict')

plt.ylabel('y_test')

plt.show()
a=y_predict-y_test

sns.kdeplot(np.array(a))

plt.xlabel('y_predict-y_test')

plt.show()
from sklearn.model_selection import cross_val_score

scores1 = cross_val_score(lin2,X=X_train,y=y_train, cv=10)

print ('Cross-validated scores:', scores1)
print('score_mean',scores1.mean(),': score_std',scores1.std())