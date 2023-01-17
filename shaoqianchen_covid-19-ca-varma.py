import pandas as pd

import matplotlib.pyplot as plt

from statsmodels.tsa.arima_model import ARIMA
df = pd.read_csv('../input/covid19-local-us-ca-forecasting-week-1/ca_train.csv')

df.head()
data_confirm = df['ConfirmedCases'].to_list()

print("Length of initial data_confim = ",len(data_confirm)) 

data_Fat = df['Fatalities'].to_list()

print("Length of initial data_Fat = ",len(data_Fat)) 
def ARIMA_FIT(data,num_prediction):

    temp_list = data

    for i in range(num_prediction):

        model = ARIMA(temp_list, order=(0, 2, 2))

        model_fit = model.fit(disp=False)

        yhat = model_fit.predict(len(temp_list), len(temp_list), typ='levels')

        temp_list.append(round(yhat[0]))

    return temp_list
ConfirmedCases = ARIMA_FIT(data_confirm,36)

print("Length of predict data_confim = ",len(ConfirmedCases))



Fatalities = ARIMA_FIT(data_Fat,36)

print("Length of predict data_Fat = ",len(Fatalities))
plt.plot(ConfirmedCases)

plt.plot(Fatalities)
df_submission = pd.read_csv('../input/covid19-local-us-ca-forecasting-week-1/ca_submission.csv')

submission = pd.DataFrame({"ForecastId": df_submission['ForecastId'], "ConfirmedCases": ConfirmedCases[50:],"Fatalities":Fatalities[50:]})

submission.to_csv('submission.csv', index=False)