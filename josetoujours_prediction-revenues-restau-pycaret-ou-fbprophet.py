import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
train= pd.read_csv("../input/restaurant-revenue-prediction/train.csv.zip")
train.head()
train.info()
train.describe()
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
train_1= train[["Open Date", "revenue"]]
train_1['ds']= train_1['Open Date']
train_1['y']= train_1['revenue']
train_1.drop(columns=["Open Date", "revenue"], inplace= True)
train_1.head()
m= Prophet()
m.fit(train_1)
future= m.make_future_dataframe(periods=365)
future.tail()

forecast= m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1= m.plot(forecast)
fig2 = m.plot_components(forecast)

m = Prophet(changepoint_prior_scale=2.5)
m.fit(train_1)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
# Calculate root mean squared error.
print('RMSE: %f' % np.sqrt(np.mean((forecast['yhat']-train_1['y'])**2)) )

!pip install pycaret
from pycaret.regression import *
prr= setup(train, target= 'revenue')
compare_models()
hub = create_model('huber') 
plot_model(hub)
pred= predict_model(hub)

# finalize a model
hub_final = finalize_model(hub)

# importing unseen data 
test = pd.read_csv('../input/restaurant-revenue-prediction/test.csv.zip')
test.head()
# generate predictions on unseen data
pred_test = predict_model(hub_final, data = test)
pred_test.head()
# Save predictions in format used for competition scoring
output = pd.DataFrame({'Id': test.Id,
                       'Label': hub_final})
output.to_csv('submission_revenue_4.csv', index=False)
#sub= pd.read_csv("../input/restaurant-revenue-prediction/sampleSubmission.csv")

#sub.head()
#sub["Prediction"]= pred_test["Label"]

#sub.head()
#sub.to_csv("submission_revenues_restau_3.csv", index= False)