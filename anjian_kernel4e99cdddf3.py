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
import pandas as pd

from statsmodels.formula.api import ols

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

import seaborn as sns

pd.set_option('display.max_columns',11)

df=pd.read_csv('../input/world-happiness-report-2019.csv')  

df=df.rename(columns={'Country (region)':'Country','SD of Ladder':'SD',

                      'Positive affect':'Positive','Negative affect':'Negative',

                      'Social support':'Social','Healthy life\nexpectancy':'Healthy_life_expectancy',

                      'Log of GDP\nper capita':'Log_of_GDP_per_capita'})

df.describe()
df=df.fillna(df.mean())

df.describe()
import seaborn as sns

import matplotlib.pyplot as plt



figure=plt.figure(figsize=(10,10))

sns.heatmap(df.corr(),annot=True)

data=df[['Ladder','Healthy_life_expectancy','Log_of_GDP_per_capita','Generosity','Social','Freedom','Negative','Positive','SD']]

train,test=train_test_split(data,test_size=0.2,random_state=1234)

model=ols('Ladder~Generosity+Social+Freedom+Negative+Positive+Healthy_life_expectancy+SD+Log_of_GDP_per_capita',data=train).fit()

print(model.summary())
def forward_select(data, response):

    remaining = set(data.columns)

    remaining.remove(response)

    selected = []

    current_score, best_new_score = float('inf'), float('inf')

    while remaining:

        aic_with_candidates=[]

        for candidate in remaining:

            formula = "{} ~ {}".format(response,' + '.join(selected + [candidate]))

            aic = ols(formula=formula, data=data).fit().aic

            aic_with_candidates.append((aic, candidate))

        aic_with_candidates.sort(reverse=True)

        best_new_score, best_candidate=aic_with_candidates.pop()

        if current_score > best_new_score:

            remaining.remove(best_candidate)

            selected.append(best_candidate)

            current_score = best_new_score

            print ('aic is {},continuing!'.format(current_score))

        else:

            print ('forward selection over!')

            break

    formula = "{} ~ {} ".format(response,' + '.join(selected))

    print('final formula is {}'.format(formula))

    model = ols(formula=formula, data=data).fit()

    return(model)



data_for_select=train

var_select=forward_select(data=data_for_select,response='Ladder')

print('R-squared:',var_select.rsquared) 
def vif(df,col_i):

    from statsmodels.formula.api import ols



    cols=list(df.columns)

    cols.remove(col_i)

    cols_noti=cols

    formula=col_i+'~'+'+'.join(cols_noti)

    r2=ols(formula,df).fit().rsquared

    return 1./(1.-r2)



exog=train.drop(['Ladder','Negative','SD','Generosity'],axis=1)

for i in exog.columns:

    print(i,'\t',vif(df=exog,col_i=i))
continuous_xcols=data[['Social','Healthy_life_expectancy','Freedom','Log_of_GDP_per_capita','Positive']]

Y=data['Ladder']

x_train,x_test,y_train,y_test=train_test_split(continuous_xcols,Y,test_size=0.2,random_state=1234)



model=LinearRegression()

model.fit(x_train,y_train)

predict_y=model.predict(x_test)

print('r2 score:',r2_score(y_test,predict_y))



plt.scatter(y_test,predict_y)

plt.show()