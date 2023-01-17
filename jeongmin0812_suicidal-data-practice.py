# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
total_suicidal=pd.read_csv("../input/master.csv")
total_suicidal.head()
total_suicidal["country"].value_counts()
total_suicidal.info()
total_suicidal.describe()
import matplotlib.pyplot as plt

total_suicidal.hist(bins=50, figsize=(20,15))

plt.show()
year_grouped=total_suicidal.groupby("year")
year_grouped_sum=year_grouped.sum()
import seaborn as sns
plt.rcParams["axes.labelsize"] = 20

ax = sns.lineplot(x='year', y="suicides/100k pop", data=year_grouped_sum.reset_index())

ax.set(xlabel='Year', ylabel='Suicides / 100k population')
gender_grouped=total_suicidal.groupby("sex")
gender_grouped_sum=gender_grouped.sum()
gender_grouped_sum=gender_grouped_sum.reset_index()
plt.rcParams["axes.labelsize"] = 20

ax=sns.catplot(x='sex', y='suicides/100k pop', 

            data=gender_grouped_sum, kind="bar", height=10, aspect=0.6)

ax.set(xlabel='Gender', ylabel='Suicides / 100k population')

plt.show()
country_grouped=total_suicidal.groupby("country")
country_grouped_sum=country_grouped.sum()
country_grouped_sum=country_grouped_sum.reset_index()
country_grouped_sum_sorted=country_grouped_sum.sort_values('suicides/100k pop')
plt.rcParams["axes.labelsize"] = 30

ax=sns.catplot(x='suicides/100k pop', y='country', 

            data=country_grouped_sum_sorted, kind="bar", height=20, aspect=0.6)

ax.set(xlabel='Total Suicides', ylabel='Country')

plt.show()
gen_grouped=total_suicidal.groupby("generation")
gen_grouped_sum=gen_grouped.sum()
gen_grouped_sum=gen_grouped_sum.reset_index()
plt.rcParams["axes.labelsize"] = 40

sns.set(font_scale=2)

ax=sns.catplot(x='suicides/100k pop', y='generation', 

            data=gen_grouped_sum, kind="bar", height=15, aspect=0.8)

ax.set(xlabel='Total suicides/100K population', ylabel='Generation')

plt.show()
se=total_suicidal.groupby(['country','year'])['suicides/100k pop'].mean().reset_index().groupby('country').mean()
se=se.reset_index()
se2=total_suicidal.groupby(['country','year'])['gdp_per_capita ($)'].mean().reset_index().groupby('country').mean()
se2=se2.reset_index()
se['100k_avg']=se["suicides/100k pop"]
se['gdp_per_capita_average']=se2["gdp_per_capita ($)"]
se.head()
sns.set(rc={'figure.figsize':(15,10)})

plt.rcParams["axes.labelsize"] = 20

ax=sns.scatterplot(x='gdp_per_capita_average', y='100k_avg', s=100, hue="100k_avg",palette="Set1",data=se)

ax.set(xlabel='GDP per capita ($)', ylabel='Suicides / 100K')

plt.show()
se_num_X=se["gdp_per_capita_average"]
se_num_Y=se["100k_avg"]
se_num=pd.concat([se_num_X,se_num_Y],axis=1)
se_num_Xarray=se_num_X.values.reshape(-1,1)
se_num_Yarray=se_num_Y.values.reshape(-1,1)
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(se_num_Xarray,se_num_Yarray)
print("Y intercept value is", lin_reg.intercept_)
print("Slope is", lin_reg.coef_)
plt.scatter(se_num_X, se_num_Y, label='Data (WITH outliers)', color='green', marker='^', alpha=.5)

ax=sns.regplot(x='gdp_per_capita_average', y='100k_avg', data=se_num, scatter=None, color="g",label="Linear_outlier")

ax.set(xlabel='GDP per capita ($)', ylabel='Suicides / 100K')

plt.legend(loc="best")

plt.show()
from scipy.stats import zscore
se_Y_zscore=zscore(se_num_Y)
se_bool_Y=np.absolute(se_Y_zscore)<3
se_bool_series=pd.Series(se_bool_Y)
new_se=pd.concat([se_num_X,se_num_Y,se_bool_series],axis=1)
new_se=new_se[new_se[0]]
new_se_num_Xarray=new_se["gdp_per_capita_average"].values.reshape(-1,1)
new_se_num_Yarray=new_se["100k_avg"].values.reshape(-1,1)
lin_reg.fit(new_se_num_Xarray,new_se_num_Yarray)
print("Y intercept value is", lin_reg.intercept_)
print("Slope is", lin_reg.coef_)
plt.scatter(se_num_X, se_num_Y, label='Data (WITH outliers)', color='green', marker='^', alpha=.5)

sns.regplot(x='gdp_per_capita_average', y='100k_avg', data=se_num, scatter=None, color="g",label="Linear_outlier")

plt.scatter(new_se["gdp_per_capita_average"], new_se["100k_avg"], label='Data (WITHOUT outliers)', color='red', marker='o', alpha=.5)

ax=sns.regplot(x='gdp_per_capita_average', y='100k_avg', data=new_se, scatter=None, marker="^",color="r",label="Linear_no_outlier")

ax.set(xlabel='GDP per capita ($)', ylabel='Suicides / 100K')

plt.legend(loc="best")

plt.show()
se3=total_suicidal.groupby(['country','year'])['suicides/100k pop'].sum().unstack()
se3=se3.reset_index()
se3.head()
se3=se3.T.reset_index()
se3.columns = se3.iloc[0]
se3=se3.rename(columns = {'country':'year'})
se3=se3.drop([0])
se3=se3.replace('NaN', np.NaN)
se3.head()
column_length=len(se3.columns)
from sklearn.metrics import r2_score
index_append=[]

r2_append=[]

coeff_append=[]

for x in range(1,column_length):

    new_series=pd.concat([se3['year'],se3.iloc[:,x]], axis=1)

    new_series2=new_series.dropna()

    new_data=pd.DataFrame(new_series2).reset_index()

    year_array=new_data['year'].values.reshape(-1,1)

    sui_array=new_data.iloc[:,2].values.reshape(-1,1)

    lin_reg.fit(year_array,sui_array)

    coeff=lin_reg.coef_

    y_pred = lin_reg.predict(year_array)

    r2_sklearn = r2_score(sui_array,y_pred) 

    if 0.8<r2_sklearn<1:

        index_append.append(x)

        r2_append.append(r2_sklearn)

        coeff_append.append(coeff)
scal_coeff=[]

for x in coeff_append:

    co=np.asscalar(x)

    scal_coeff.append(co)

print(scal_coeff)
cols=se3.columns
L=[]

L2=[]

for x in range(0,len(index_append)):

    new_country=cols[index_append[x]]

    L.append(new_country)

se4=pd.DataFrame(L, columns=['country'])

se5=pd.DataFrame(r2_append, columns=['R^2'])

se_co=pd.DataFrame(scal_coeff, columns=['coeff'])

se6=pd.concat([se4,se5,se_co], axis=1)

#se_sign=se6['coeff'].apply(lambda x: x<0)

#se6['negative']=se_sign

se6.loc[se6.coeff<0, 'sign']='Negative trend'

se6.loc[se6.coeff>0, 'sign']='Positive trend'
ax = sns.catplot(x="R^2", y="country", data=se6, saturation=0.5, kind="bar", hue="sign",ci=None, height=10,aspect=1,palette="Set2",legend=False)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

ax.set(xlabel='R^2 value for linear regression', ylabel='Country')
L=['Finland','France','Hungary','year']
se_N=se3[L]
se_N=se_N.reset_index()
plt.scatter(se_N['year'], se_N['Finland'], label='Finland', color='blue', marker='^', alpha=.5)

ax=sns.regplot(x='year',y='Finland',data=se_N, scatter=None, color='blue',label="Finland")

plt.scatter(se_N['year'], se_N['France'], label='France', color='red', marker='^', alpha=.5)

ax2=sns.regplot(x='year', y='France', data=se_N, marker="^",color="red",scatter=None,label="France")

plt.scatter(se_N['year'], se_N['Hungary'], label='Hungary', color='green',marker='^', alpha=.5)

ax3=sns.regplot(x='year', y='Hungary', data=se_N, marker="x",color="green", scatter=None, label="Hungary")

ax3.set(xlabel='Year', ylabel='Suicides / 100K')

plt.legend(loc="best")

plt.show()
L=['Republic of Korea','Mexico','Philippines','year']
se_P=se3[L]
se_P=se_P.reset_index()
plt.scatter(se_P['year'], se_P['Republic of Korea'], label='Republic of Korea', color='blue', marker='^', alpha=.5)

ax=sns.regplot(x='year',y='Republic of Korea',data=se_P, scatter=None, color='blue',label="Republic of Korea")

plt.scatter(se_P['year'], se_P['Mexico'], label='Mexico', color='red', marker='^', alpha=.5)

ax2=sns.regplot(x='year', y='Mexico', data=se_P, marker="^",color="red",scatter=None,label="Mexico")

plt.scatter(se_P['year'], se_P['Philippines'], label='Hungary', color='green',marker='^', alpha=.5)

ax3=sns.regplot(x='year', y='Philippines', data=se_P, marker="x",color="green", scatter=None, label="Philippines")

ax3.set(xlabel='Year', ylabel='Suicides / 100K')

plt.legend(loc="best")

plt.show()