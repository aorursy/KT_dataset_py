# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

housing_data=pd.read_csv('../input/brooklyn_sales_map.csv')



#scatterplot visualisation

plt.scatter(x=housing_data['year_of_sale'],y=housing_data['sale_price'])

ax =plt.gca()

ax.get_yaxis().get_major_formatter().set_scientific(False)

plt.draw()



#highest selling property

housing_data.sort_values('sale_price').tail(1)

housing_data['sale_price'].describe().apply(lambda x: format(x, 'f'))

housing_data = housing_data[housing_data.sale_price > 0]



#more visualisations

bins=[-100000000,20000,40000,60000,80000,100000,1000000,10000000,500000000]

choices =['$0-$200k','$200k-$400k','$400k-$600k','$600k-$800k','$800k-$1mlln','$1mlln-$10mlln','$10mlln-$100mlln','$100mlln-$500mlln']

housing_data['price_range']=pd.cut(housing_data['sale_price'],bins=bins,labels=choices)

def conv(year):

  return housing_data[housing_data['year_of_sale']==year].groupby('price_range').size()

perc_total = [x/sum(x)*100 for x in [conv(2003),conv(2004),conv(2005),conv(2006),conv(2007),conv(2008),conv(2009),conv(2010),conv(2011),conv(2012),conv(2013),conv(2014),conv(2015),conv(2016),conv(2017)]]

year_names = list(range(2003,2018))

housing_df = pd.DataFrame(perc_total, index= year_names)

ax_two = housing_df.plot(kind='barh', stacked=True, width=0.80)

horiz_offset = 1

vert_offset = 1

ax_two.set_xlabel('Percentages')

ax_two.set_ylabel('Years')

ax_two.legend(bbox_to_anchor=(horiz_offset, vert_offset))

housing_data.groupby(['neighborhood','price_range']).size().unstack().plot.bar(stacked=True)

horiz_offset = 1

vert_offset = 1

plt.rcParams["figure.figsize"] = [40,20]



#removing outliers

def remove_outlier(df, col):

  q1 = df[col].quantile(0.25)

  q3 = df[col].quantile(0.75)

  iqr = q3 - q1

  lower_bound  = q1 - (1.5  * iqr)

  upper_bound = q3 + (1.5 * iqr)

  out_df = df.loc[(df[col] > lower_bound) & (df[col] < upper_bound)]

  return out_df

housing_data = remove_outlier(housing_data,"sale_price")



#cleaning up columns with too many NAs

threshold = len(housing_data) * .75

housing_data.dropna(thresh = threshold, axis = 1, inplace = True)



#more clean up

housing_data = housing_data.drop(['APPBBL','BoroCode','Borough','BBL','price_range','PLUTOMapID','YearBuilt','CondoNo','BuiltFAR','FireComp','MAPPLUTO_F','Sanborn','SanitBoro','Unnamed: 0','Version', 'block','borough','Address','OwnerName','zip_code'],axis=1)



#if basement data is missing it might be safer to assume that whether or not the apartment/building is unknown which is represented by the number 5

housing_data['BsmtCode'] = housing_data['BsmtCode'].fillna(5)

#Community Area- not applicable or available if Na

housing_data[['ComArea','CommFAR','FacilFAR','FactryArea','RetailArea','ProxCode','YearAlter1','YearAlter2']] = housing_data[['ComArea','CommFAR','FacilFAR','FactryArea','RetailArea','ProxCode','YearAlter1','YearAlter2']].fillna(0)

housing_data[['XCoord','YCoord','ZipCode','LotType','SanitDistr','HealthArea','HealthCent','PolicePrct','SchoolDist','tax_class_at_sale','CD','Council']] = housing_data[['XCoord','YCoord','ZipCode','LotType','SanitDistr','HealthArea','HealthCent','PolicePrct','SchoolDist','tax_class_at_sale','CD','Council']].apply(lambda x: x.fillna(x.mode()[0]))

#soft impute

from fancyimpute import SoftImpute

feature_arr = housing_data.drop(['OfficeArea','commercial_units','residential_units','ResArea'],axis=1).select_dtypes(include=[np.number])

softimpute = SoftImpute()

housing_data2 = pd.DataFrame(softimpute.fit_transform(feature_arr.values),columns=feature_arr.columns,index=feature_arr.index)

#fill in missing values with imputation values

housing_data = housing_data.fillna(housing_data2)

housing_data = housing_data.apply(lambda x: x.fillna(x.median()) if x.dtype.kind in 'iufc' else x.fillna(x.mode()[0]))

housing_data['Age of House at Sale'] = housing_data['year_of_sale'] - housing_data['year_built']

housing_data = housing_data.drop(['year_of_sale','year_built'],axis=1)



#change strings to ints to preprocess for ML algo

def strnums(cols):

  return dict(zip(set(housing_data[cols]),list(range(0,len(set(housing_data[cols]))))))

for columns in set(housing_data.select_dtypes(exclude='number')):

  housing_data[columns] = housing_data[columns].map(strnums(columns))

from sklearn.dummy import DummyRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split

features =  list(housing_data.drop(['sale_price'],axis=1)) 

y = housing_data.sale_price

X = housing_data[features]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=100)

dummy_median = DummyRegressor(strategy='mean')

dummy_regressor = dummy_median.fit(X_train,y_train)

dummy_predicts = dummy_regressor.predict(X_test)

print("Model Accuracy:", dummy_regressor.score(X_test,y_test)*100)

print('$',mean_absolute_error(y_test,dummy_predicts))



#multiple models

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

models = [RandomForestRegressor(n_estimators=200,criterion='mse',max_depth=20,random_state=100),DecisionTreeRegressor(criterion='mse',max_depth=11,random_state=100),GradientBoostingRegressor(n_estimators=200,max_depth=12)]

learning_mods = pd.DataFrame()

temp = {}

#run through models

for model in models:

    print(model)

    m = str(model)

    temp['Model'] = m[:m.index('(')]

    model.fit(X_train, y_train)

    temp['R2_Price'] = r2_score(y_test, model.predict(X_test))

    print('score on training',model.score(X_train, y_train))

    print('r2 score',r2_score(y_test, model.predict(X_test)))

    learning_mods = learning_mods.append([temp])

learning_mods.set_index('Model', inplace=True)

 

fig, axes = plt.subplots(ncols=1, figsize=(10, 4))

learning_mods.R2_Price.plot(ax=axes, kind='bar', title='R2_Price')

plt.show()



#feature importance

regressionTree_imp = model.feature_importances_

plt.figure(figsize=(16,6))

plt.yscale('log',nonposy='clip')

plt.bar(range(len(regressionTree_imp)),regressionTree_imp,align='center')

plt.xticks(range(len(regressionTree_imp)),features,rotation='vertical')

plt.title('Feature Importance')

plt.ylabel('Importance')

plt.show()


