import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import geopandas as gpd



confirmed_cases = pd.read_csv('/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_confirmed_global.csv')

death_cases = pd.read_csv('/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_deaths_global.csv')

recovered_cases = pd.read_csv('/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_recovered_global.csv')
from IPython.display import display

display("Confirmed Covid-19 Cases (Global)")

display(confirmed_cases.head())

display("Deaths resulting from Covid-19 Cases(Global)")

display(death_cases.head())

display("Recovered Covid-19 Cases (Global)")

display(recovered_cases.head())
date_keys = confirmed_cases.keys()[4:]

dates = confirmed_cases.keys()[4:].to_numpy().reshape(1, -1)[0]

dates_plot = list(range(len(dates)))

pd.DataFrame(dates)
global_cases = np.sum(confirmed_cases[dates]).astype(int)

global_recovered_cases = np.sum(recovered_cases[dates]).astype(int)

global_deaths = np.sum(death_cases[dates]).astype(int)
plt.figure(figsize=(16, 9))

plt.grid()

plt.plot(dates_plot, global_cases, 'r-', linewidth=3, label="Confirmed Cases")

plt.plot(dates_plot, global_deaths, color=(46/255, 49/255, 49/255, 1), linewidth=3, label="Deaths")

plt.plot(dates_plot, global_recovered_cases, 'g-', linewidth=3, label="Recovered Cases")

plt.legend()

plt.ticklabel_format(style='plain', axis='y')

plt.title('Number of Coronavirus Cases Over Time(Global)')

plt.xlabel('Days Since 1/22/2020', size=20)

plt.ylabel('Number of Cases', size=20)

plt.xticks(size=15)

plt.yticks(size=15)

plt.show()
colors = ['#fee0d2','#fc9272','#de2d26']



total_cases_distributed = confirmed_cases.groupby('Country/Region')[dates[-1]].sum().values.tolist()

country_list = list(set(confirmed_cases["Country/Region"].to_list()))

country_list.sort()



new_df = pd.DataFrame(list(zip(country_list, total_cases_distributed)), 

               columns =['Country', 'Total_Confirmed_Cases'])



fig = go.Figure(data=go.Choropleth(

    locationmode = "country names",

    locations = new_df['Country'],

    z = new_df['Total_Confirmed_Cases'],

    text = new_df['Total_Confirmed_Cases'],

    colorscale = colors,

    autocolorscale=False,

    reversescale=False,

    colorbar_title = 'Covid-19 Cases',

))



fig.update_layout(

    title_text='Global - Covid-19 Cases as of 2020.04.17',

    geo=dict(

        showcoastlines=False,

    ),

)



fig.show()
confirmed_cases_ca = confirmed_cases.loc[confirmed_cases['Country/Region'] == 'Canada'].loc[confirmed_cases['Province/State'] != 'Recovered'].loc[confirmed_cases['Province/State'] != 'Diamond Princess']

death_cases_ca = death_cases.loc[death_cases['Country/Region'] == 'Canada'].loc[death_cases['Province/State'] != 'Recovered'].loc[death_cases['Province/State'] != 'Diamond Princess']

recovered_cases_ca = recovered_cases.loc[recovered_cases['Country/Region'] == 'Canada'].loc[recovered_cases['Province/State'] != 'Recovered'].loc[recovered_cases['Province/State'] != 'Diamond Princess']

display(confirmed_cases_ca)

display(death_cases_ca)

display(recovered_cases_ca)
plt.figure(figsize=(16, 9))

plt.grid()

plt.plot(dates_plot, np.sum(confirmed_cases_ca[dates]).astype(int), 'r-', linewidth=3, label="Confirmed Cases")

plt.plot(dates_plot, np.sum(death_cases_ca[dates]).astype(int), color=(46/255, 49/255, 49/255, 1), linewidth=3, label="Deaths")

plt.plot(dates_plot, np.sum(recovered_cases_ca[dates]).astype(int), 'g-', linewidth=3, label="Recovered Cases")

plt.legend()

plt.ticklabel_format(style='plain', axis='y')

plt.ticklabel_format(style='plain', axis='y')

plt.title('Number of Coronavirus Cases Over Time(Canada)')

plt.xlabel('Days Since 1/22/2020', size=20)

plt.ylabel('Number of Cases', size=20)

plt.xticks(size=15)

plt.yticks(size=15)

plt.show()
provinceData = confirmed_cases.loc[confirmed_cases['Country/Region'] == 'Canada'].drop(['Lat','Long', 'Country/Region'],axis =1)

provinceData
## Plot the data



# Get the list of provicenes

provinces = col_one_list = provinceData['Province/State'].tolist()



# Drop the silly cruise lines

provinces.remove("Diamond Princess")

provinces.remove("Recovered")

provinces.remove("Grand Princess")



# Control the number of days to plot

days = 40

figure = plt.figure(figsize=(15,10))

ax = figure.add_subplot(111)



# Go through and plot each province

for province in provinces:

    

    # Get the data for the province

    tempData = provinceData.loc[provinceData['Province/State'] == province].drop(['Province/State'],axis =1).values[0]

    

    # Scale it to find the first

    tempData2 = tempData[tempData > 0]

    

    # Generate the coresponding x scale

    xValues = np.linspace(0, len(tempData2)-1, len(tempData2))

    

    # Add it to the chart

    plt.plot(xValues,tempData2,label = province,linewidth =2)



# Plot Setup

plt.title("Confirmed COVID Case Count Per Province",fontsize=20)

plt.xlabel("Days Since First Case",fontsize=15)

plt.ylabel("Case Count",fontsize=15)



# Add the legend

plt.legend(loc = "upper left")

plt.yscale("log")

plt.grid(which="both")



# Show

plt.show()    
confirmed_cases_on = confirmed_cases.loc[confirmed_cases['Province/State'] == 'Ontario'].loc[confirmed_cases['Province/State'] != 'Recovered'].loc[confirmed_cases['Province/State'] != 'Diamond Princess']

death_cases_on = death_cases.loc[death_cases['Province/State'] == 'Ontario'].loc[death_cases['Province/State'] != 'Recovered'].loc[death_cases['Province/State'] != 'Diamond Princess']

recovered_cases_on = recovered_cases.loc[recovered_cases['Province/State'] == 'Ontario'].loc[recovered_cases['Province/State'] != 'Recovered'].loc[recovered_cases['Province/State'] != 'Diamond Princess']

display(confirmed_cases_on.head())

display(death_cases_on.head())

#display(recovered_cases_on.head())

cases_on = np.sum(confirmed_cases_on[dates]).astype(int)

deaths_on = np.sum(death_cases_on[dates]).astype(int)

#global_recovered_cases = np.sum(recovered_cases[dates]).astype(int)
plt.figure(figsize=(16, 9))

plt.grid()

plt.plot(dates_plot, np.sum(confirmed_cases_on[dates]).astype(int), 'r-', linewidth=3, label="Confirmed Cases")

plt.plot(dates_plot, np.sum(death_cases_on[dates]).astype(int), color=(46/255, 49/255, 49/255, 1), linewidth=3, label="Deaths")

# plt.plot(dates_plot, np.sum(recovered_cases_on[dates]).astype(int), 'g-', linewidth=3, label="Recovered Cases") Ontario recovery cases are included in Canada

plt.legend()

plt.ticklabel_format(style='plain', axis='y')

plt.ticklabel_format(style='plain', axis='y')

plt.title('Number of Coronavirus Cases Over Time(Ontario)')

plt.xlabel('Days Since 1/22/2020', size=20)

plt.ylabel('Number of Cases', size=20)

plt.xticks(size=15)

plt.yticks(size=15)

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_log_error

import tensorflow as tf

from keras.models import Sequential

from keras.layers.core import Dense, Activation

from keras.layers.recurrent import LSTM

from keras.optimizers import Adagrad

from keras.callbacks import EarlyStopping
info_cases = pd.DataFrame(zip(cases_on, deaths_on), columns=['Cases', 'Deaths'])
info_cases
case_scaler = StandardScaler()

info_cases["Cases"] = case_scaler.fit_transform(info_cases["Cases"].values.reshape(len(info_cases["Cases"].values),1))





death_scaler = StandardScaler()

info_cases["Deaths"] = death_scaler.fit_transform(info_cases["Deaths"].values.reshape(len(info_cases["Deaths"].values),1))
X = []

Y_cases = []

Y_deaths = []

for i, row in info_cases.iterrows():

    for j in range(len(info_cases) - 15):

        if info_cases.iloc[j+14]['Cases'] != 0 or info_cases.iloc[j+14]['Deaths'] != 0: # Predict 0 as 0, we cannot predict the outbreak

            X.append(info_cases[['Cases','Deaths']].iloc[j:(j+15)].values)

            Y_cases.append(info_cases[['Cases']].iloc[j+15].values)

            Y_deaths.append(info_cases[['Deaths']].iloc[j+15].values)

            

X=np.array(X)

Y_cases=np.array(Y_cases)

Y_deaths=np.array(Y_deaths)
X_train, X_test, Y_cases_train, Y_cases_test = train_test_split(X, Y_cases, test_size=0.1, random_state = 42)

X_train, X_test, Y_deaths_train, Y_deaths_test = train_test_split(X, Y_deaths, test_size=0.1, random_state = 42)
def huber_loss(y_true, y_pred, clip_delta=1.0):

  error = y_true - y_pred

  cond  = tf.keras.backend.abs(error) < clip_delta



  squared_loss = 0.5 * tf.keras.backend.square(error)

  linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)



  return tf.where(cond, squared_loss, linear_loss)



def huber_loss_mean(y_true, y_pred, clip_delta=1.0):

  return tf.keras.backend.mean(huber_loss(y_true, y_pred, clip_delta))
epochs_num = 20

batch_size_num = 10

n_hidden = 300

n_in = 2



model_cases = Sequential()

model_cases.add(LSTM(n_hidden,

               batch_input_shape=(None, 15, n_in),

               kernel_initializer='random_uniform',

               return_sequences=False))

model_cases.add(Dense(1, kernel_initializer='random_uniform'))

model_cases.add(Activation("linear"))

opt = Adagrad(lr=0.01, epsilon=1e-08, decay=1e-4)

model_cases.compile(loss = huber_loss_mean, optimizer=opt)



model_deaths = Sequential()

model_deaths.add(LSTM(n_hidden,

               batch_input_shape=(None, 15, n_in),

               kernel_initializer='random_uniform',

               return_sequences=False))

model_deaths.add(Dense(1, kernel_initializer='random_uniform'))

model_deaths.add(Activation("linear"))

opt = Adagrad(lr=0.01, epsilon=1e-08, decay=1e-4)

model_deaths.compile(loss = huber_loss_mean, optimizer=opt)
early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=1)



hist_cases = model_cases.fit(X_train, Y_cases_train, batch_size=batch_size_num, epochs=epochs_num,

                 callbacks=[early_stopping],shuffle=False)
hist_deaths = model_deaths.fit(X_train, Y_deaths_train, batch_size=batch_size_num, epochs=epochs_num,

                 callbacks=[early_stopping],shuffle=False)
predicted_cases_std = model_cases.predict(X_test)

result_cases_std= pd.DataFrame(predicted_cases_std)

result_cases_std.columns = ['predict']

result_cases_std['actual'] = Y_cases_test



predicted_deaths_std = model_deaths.predict(X_test)

result_deaths_std= pd.DataFrame(predicted_deaths_std)

result_deaths_std.columns = ['predict']

result_deaths_std['actual'] = Y_deaths_test
loss_c = hist_cases.history["loss"]

epochs = len(loss_c)

plt.figure()

plt.title("loss(Confirmed Cases)")

plt.plot(range(epochs), loss_c, marker=".")

plt.grid()

plt.show()





loss_f = hist_deaths.history["loss"]

epochs = len(loss_f)

plt.figure()

plt.title("loss(Deaths)")

plt.plot(range(epochs), loss_f, marker=".")

plt.grid()

plt.show()
predicted_cases = case_scaler.inverse_transform(predicted_cases_std)

Y_cases_inv_test = case_scaler.inverse_transform(Y_cases_test)



predicted_deaths = death_scaler.inverse_transform(predicted_deaths_std)

Y_deaths_inv_test = death_scaler.inverse_transform(Y_deaths_test)

print("Reverse scale done...")
result= pd.DataFrame(predicted_cases)

result.columns = ['predict']

result['actual'] = Y_cases_inv_test

result[:30].plot.bar(title = "Confirmed Cases")

plt.grid()

plt.show()





result_deaths = pd.DataFrame(predicted_deaths)

result_deaths.columns = ['predict']

result_deaths['actual'] = Y_deaths_inv_test

result_deaths[:30].plot.bar(title = "Deaths")

plt.grid()

plt.show()
X_predict = X

predicted_cases = []

predicted_deaths = []

for i in range(0, 14):

    predict_case = model_cases.predict(np.array(list([X_predict[-1]])))

    predict_deaths = model_deaths.predict(np.array(list([X_predict[-1]])))

    temp_array = X_predict[-1][1:]

    tmp_pred = temp_array[0]

    tmp_pred[0] = predict_case[0][0]

    tmp_pred[1] = predict_deaths[0][0]

    temp_array = np.concatenate((temp_array, [tmp_pred]))

    X_predict = np.concatenate((X_predict, [temp_array]))

    predicted_cases.append(predict_case)

    predicted_deaths.append(predict_deaths)
predictions_c = case_scaler.inverse_transform(predicted_cases)

predictions_d = death_scaler.inverse_transform(predicted_deaths)



predictions_rc = [] 

predictions_rd = []

for i in predictions_c:

    predictions_rc.append(i[0][0])

    

for i in predictions_d:

    predictions_rd.append(i[0][0])
plt.plot(list(range(0,14)), predictions_rc, 'ro-')

plt.title('New Cases')

plt.xlabel("Days",fontsize=15)

plt.ylabel("Case Count",fontsize=15)

plt.grid()

plt.show()



plt.plot(list(range(0,14)), predictions_rd, 'o-')

plt.title('New Deaths')

plt.xlabel("Days",fontsize=15)

plt.ylabel("Death Count",fontsize=15)

plt.grid()

plt.show()