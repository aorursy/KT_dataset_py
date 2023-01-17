# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataset=pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")
dataset.head()
dataset['Country/Region'].unique()
dataset['ObservationDate']=dataset['ObservationDate'].astype('datetime64[ns]')
start_date=dataset['ObservationDate'][0]

print(start_date)
dataset['Date difference'] = dataset['ObservationDate']-start_date
dataset['Date difference']
dataset.head()
dataset['Location'] = [0 if value in ['Mainland China','Taiwan','Macau','Hong Kong'] else 1 for value in dataset['Country/Region']]
dataset.head()
valid_columns = ['Confirmed','Deaths','Recovered','Date difference', 'Location']
X = dataset[valid_columns]
X.head()
newX = X.loc[X['Location']==0].groupby(['Date difference'])['Confirmed'].sum()

newX_df = pd.DataFrame(newX)

newX_df.reset_index()
chinaX=X.loc[X['Location']==0]

values_chinaX=pd.DataFrame(chinaX).groupby(['Date difference']).sum().reset_index()

x_plot = values_chinaX['Date difference'].dt.days

y_plot = values_chinaX['Confirmed']



outside_chinaX=X.loc[X['Location']==1]

values_outside_chinaX = pd.DataFrame(outside_chinaX).groupby(['Date difference']).sum().reset_index()

x_plot_outside = values_outside_chinaX['Date difference'].dt.days

y_plot_outside = values_outside_chinaX['Confirmed']
y_plot
import matplotlib.pyplot as plt
plt.plot(x_plot_outside,y_plot_outside)

plt.plot(x_plot,y_plot)

plt.legend(['Outside China','Inside China'])

plt.show()
# Predictions in China

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures



predict_dates = list(range(46,46+31))

poly_features=PolynomialFeatures(degree=3)



X_poly=poly_features.fit_transform(pd.DataFrame(x_plot))

pred_poly = poly_features.fit_transform(pd.DataFrame(predict_dates))



model=LinearRegression()

model.fit(X_poly,y_plot)



predictions=model.predict(pred_poly)
from sklearn.ensemble import RandomForestRegressor



rf_reg=RandomForestRegressor(n_estimators=25)

rf_reg.fit(pd.DataFrame(x_plot),pd.DataFrame(y_plot))

pred_rf=rf_reg.predict(pd.DataFrame(predict_dates))



rf_reg_outside=RandomForestRegressor(n_estimators=25)

rf_reg_outside.fit(pd.DataFrame(x_plot_outside),pd.DataFrame(y_plot_outside))

pred_rf_outside=rf_reg_outside.predict(pd.DataFrame(predict_dates))
# Predictions outside China



X_poly_outside=poly_features.fit_transform(pd.DataFrame(x_plot_outside))

#pred_poly = poly_features.fit_transform(pd.DataFrame(predict_dates))



model=LinearRegression()

model.fit(X_poly_outside,y_plot_outside)



predictions_outside=model.predict(pred_poly)
plt.plot(predict_dates,predictions)

plt.plot(predict_dates,predictions_outside)

plt.plot(predict_dates,pred_rf)

plt.plot(predict_dates,pred_rf_outside)

plt.legend(['Inside China','Outside China','RF inside','RF outside'])

plt.show()
import datetime
new_preds = pd.concat([pd.DataFrame(predictions),pd.DataFrame(predictions_outside)],axis=1)

new_preds.columns=['Inside China','Outside China']

new_preds['Dates'] = pd.date_range(start=dataset['ObservationDate'][3991]+datetime.timedelta(days=1),periods=len(new_preds),freq='D')
new_preds
total_preds = pd.DataFrame()

total_preds['Count'] = new_preds['Inside China']+new_preds['Outside China']

total_preds['Count']=total_preds['Count'].astype(int)

total_preds['Date'] = new_preds['Dates']
total_preds
total_preds.to_csv("/kaggle/working/output.csv")
new_preds.to_csv("/kaggle/working/output_separate.csv")