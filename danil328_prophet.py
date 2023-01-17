# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

from fbprophet import Prophet

import matplotlib.pyplot as plt

plt.style.use('ggplot')
train_1 = pd.read_csv('../input/train_1.csv')

train_1.head()
train_1.fillna(0, inplace=True)

page=pd.DataFrame(train_1[train_1.columns[0]])

train_1=train_1.drop(train_1.columns[0], axis=1)

train_1.columns=range(len(train_1.columns))

train_1=train_1.transpose()  

train_1.columns=[page[page.columns[0]]]    

train_1=train_1.convert_objects(convert_numeric=True)

train_1.head()
df=pd.DataFrame(train_1[train_1.columns[0]])

df.insert(loc=1, column='visits', value=train_1[train_1.columns[1]])  #number columns 

y=df[-60:] 

y=y.visits.values

df=df[:-60]



count=0

for i in df.visits:

    if i == 0:

        count=count+1

        

df=df[count:]

df.visits.replace(to_replace=0,value=df.visits.mean(),inplace=True)

df['visits'] = np.log(df['visits'])

df.columns = ["ds", "y"]
articles = pd.DataFrame({

  'holiday': 'publish',

  'ds': pd.to_datetime(['2014-09-27', '2014-10-05', '2014-10-14', '2014-10-26', '2014-11-9',

                        '2014-11-18', '2014-11-30', '2014-12-17', '2014-12-29', '2015-01-06',

                        '2015-01-20', '2015-02-02', '2015-02-16', '2015-03-23', '2015-04-08',

                        '2015-05-04', '2015-05-17', '2015-06-09', '2015-07-02', '2015-07-13',

                        '2015-08-17', '2015-09-14', '2015-10-26', '2015-12-07', '2015-12-30',

                        '2016-01-26', '2016-04-06', '2016-05-16', '2016-06-15', '2016-08-23',

                        '2016-08-29', '2016-09-06', '2016-11-21', '2016-12-19', '2016-12-31',

                        '2017-01-01', '2017-01-17', '2017-02-06', '2017-02-21']),

  'lower_window': 0,

  'upper_window': 3,

})
m = Prophet(holidays=articles,changepoint_prior_scale=0.01,weekly_seasonality=True,yearly_seasonality=True).fit(df)

future = m.make_future_dataframe(periods=60)

forecast = m.predict(future)



forecast["Sessions"] = np.exp(forecast.yhat).round()

forecast["Sessions_lower"] = np.exp(forecast.yhat_lower).round()

forecast["Sessions_upper"] = np.exp(forecast.yhat_upper).round()

forecast[(forecast.ds > "3-5-2017") &(forecast.ds < "4-1-2017")][["ds", "yhat", "Sessions_lower","Sessions", "Sessions_upper"]]



forecast["Projected_Sessions"] = np.exp(forecast.yhat).round()

forecast["Projected_Sessions_lower"] = np.exp(forecast.yhat_lower).round()

forecast["Projected_Sessions_upper"] = np.exp(forecast.yhat_upper).round()
final_proj = forecast[(forecast.ds > "2016-11-01") &(forecast.ds < "2017-03-02")][["ds", "Projected_Sessions_lower","Projected_Sessions", "Projected_Sessions_upper"]]

m.plot(forecast);



def getsmape(pred,target):

    smape=0

    for i in range(len(pred)):

        smape=smape+(abs(target[i]-pred[i])/((abs(target[i])+abs(pred[i]))/2))

    smape=smape*100/len(pred)

    return smape



x=final_proj.Projected_Sessions

x=x.values

x=x*0.8  #0.8 it's well

smape=getsmape(x,y)

print (smape)