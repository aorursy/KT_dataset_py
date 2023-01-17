#Run OLS on selected variables to study the effect of various variables on calories using open food facts data set.



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

foodfacts=pd.read_csv('../input/en.openfoodfacts.org.products.tsv',  sep='\t')
#clean missing data

variables=foodfacts[['fiber_100g','proteins_100g',

'fat_100g',

'sugars_100g',

'salt_100g',

'sodium_100g',

'additives_n',

'carbohydrates_100g']]

locations=np.where(variables.fiber_100g.notnull() & variables.proteins_100g.notnull() &

                   variables.fat_100g.notnull() & variables.sugars_100g.notnull()  

                    & variables.salt_100g.notnull() &

                   variables.sodium_100g.notnull() & variables.additives_n.notnull() &

                   variables.carbohydrates_100g.notnull()

                   )[0]



variablesnew=variables.loc[locations,:]
#append energy_100g, the subject of interest to variablesnew

variablesnew['energy_100g']=foodfacts.energy_100g

#run linear regression

import statsmodels.formula.api as sm

result=sm.ols(formula="energy_100g ~ fiber_100g+proteins_100g+fat_100g+sugars_100g+salt_100g+sodium_100g+additives_n+carbohydrates_100g",

             data=variablesnew).fit()

result.summary()

#based on this table, salt and sodium are not good candidates because of their high standard error. 

#fiber is not correlated based on its t value.

#plot regression line of 'additives_n','fat_100g','sugars_100g','carbohydrates_100g','proteins_100g'

import matplotlib.pyplot as plt

xgrid=np.linspace(0,100,1000)

plt.plot(xgrid,xgrid*result.params['additives_n']+result.params['Intercept'],'-')

plt.plot(xgrid,xgrid*result.params['fat_100g']+result.params['Intercept'],'-')

plt.plot(xgrid,xgrid*result.params['sugars_100g']+result.params['Intercept'],'-')

plt.plot(xgrid,xgrid*result.params['carbohydrates_100g']+result.params['Intercept'],'-')

plt.plot(xgrid,xgrid*result.params['proteins_100g']+result.params['Intercept'],'-')

plt.legend(labels=['additives_n','fat_100g','sugars_100g','carbohydrates_100g','proteins_100g'])

plt.show

plt.ylabel('calories')

#we can see that fat, sugar, proteins and carbo hydrates both influence positively on calories. Interestingly,

#although the slope is small, number of additives has a positive influence on calories of food.

#It means that high calories food tend to have more additives added.

#Interestingly, the slope on sugar is also very small, although positive. It means that sugar is 

#not a major source that contributes to calories compared with fat,carbohydrates and proteins

#run regression again on only these four variables and see if the result differs.

resultnew=sm.ols(formula="energy_100g ~ proteins_100g+fat_100g+sugars_100g+additives_n+carbohydrates_100g",

             data=variablesnew).fit()

resultnew.summary()
#plot again

plt.plot(xgrid,xgrid*result.params['additives_n']+resultnew.params['Intercept'],'-')

plt.plot(xgrid,xgrid*result.params['fat_100g']+resultnew.params['Intercept'],'-')

plt.plot(xgrid,xgrid*result.params['sugars_100g']+resultnew.params['Intercept'],'-')

plt.plot(xgrid,xgrid*result.params['carbohydrates_100g']+resultnew.params['Intercept'],'-')

plt.plot(xgrid,xgrid*result.params['proteins_100g']+resultnew.params['Intercept'],'-')

plt.legend(labels=['additives_n','fat_100g','sugars_100g','carbohydrates_100g','proteins_100g'])

plt.show

plt.ylabel('calories')