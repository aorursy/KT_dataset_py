from datetime import date

import pandas as pd

from statsmodels.tsa.holtwinters import *



print(date.today())

print("\n")



df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

df = df.drop(['Province/State'], axis=1)



# aggregate the data because each country may have multiple rows

country = df[df['Country/Region']=='Bulgaria'].T[3:]

country["num"] = country.sum(axis=1) 



# time series (only larger then 100 cases)

y = country[country['num']>100].loc[:,"num"].to_numpy()



print("Actual values..............................................")

print(y)



fitted_model = ExponentialSmoothing(y, trend='mul').fit(

    smoothing_level=0.8, 

    optimized=True, 

    use_brute=True, 

    use_basinhopping=True)





# round forecast values

vrounder = np.vectorize(lambda t: round(t))

forecast5 = vrounder(fitted_model.forecast(5))



# print(fitted_model.fittedvalues)

# print("\n")

print("forecast....................................................")

print(forecast5)

print("\n")

print("summary.....................................................")

print(fitted_model.summary())

# print("\n")

# print("mle retvals...............................................")

# print(fitted_model.mle_retvals)