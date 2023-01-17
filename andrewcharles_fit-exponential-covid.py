import pandas as pd

import numpy as np

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import FunctionTransformer

import matplotlib.pyplot as plt

pd.set_option("display.precision", 3)

pd.set_option("display.expand_frame_repr", False)

pd.set_option("display.max_rows", 25)

covid_19_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

by_date = covid_19_data.groupby(['ObservationDate','Country/Region'])[['Confirmed']].agg("sum")

country_cols = by_date['Confirmed'].unstack()
countries = set(list(covid_19_data['Country/Region']))

print(countries)
COL = 'Italy'

#COL = 'Australia'

#countries = ['Australia']

doubling_time = {}

doubling_ts = {}

for COL in countries:

    a = country_cols[COL].copy()[:]

    a_reset = a.reset_index()

    ydf = a_reset[COL].fillna(value=0)

    x = ydf.index.values

    y = ydf.values

    transformer = FunctionTransformer(np.log, validate=True)

    y_trans = transformer.fit_transform(y.reshape(-1,1) + 1) #y[:,np.newaxis]

    y_trans = np.nan_to_num(y_trans)

    x_in = x.reshape(-1,1)

    weights = np.exp(x_in)

    model = LinearRegression().fit(x_in, y_trans, sample_weight=weights.flatten())

    y_fit = model.predict(x_in)

    #y = A * exp(B*x) + C | log(y)  | log(y-C) = log(A) + B * x

    A,B = np.log(model.coef_[0]),(model.intercept_[0])

    A,B = model.coef_[0],model.intercept_[0]

    #print('A=',model.coef_,'B=',model.intercept_)

    #print('A * exp{Bx} + C')

    x = x_in.flatten()

    yfit = np.exp(A*x + B)

    yfit2 = B * np.exp(A*x)

    # The gradient of the line is the exponential power

    ygrad = np.gradient(np.log(y))

    ydouble = np.log(2)/ygrad

    #plt.plot(x, y, "k--", label="Test")

    #plt.plot(x, y, "r--", label="Test2") 

    #plt.legend()

    # y = ab^x = ae^{lnb * x} = 

    # y = B * exp{A}^{x}

    # for y = x_o * b^{t}

    # Tdouble = log(2)/log(b) - in this one Tdoube = log(2)/log(exp(A))

    print('Doubling:', np.log(2)/np.log(np.exp(A)), ',Cases:', country_cols[COL].max())

    doubling_time[COL] = {'Doubling':np.log(2)/np.log(np.exp(A)), 'Cases':country_cols[COL].max()}

    doubling_ts[COL] = ydouble

    
DFD = pd.DataFrame(doubling_ts)

DFD
DF = pd.DataFrame(doubling_time).T

print(DF.columns)
DF
S = DF.loc[DF['Cases'] > 5000]

S = DF.loc[['Australia','Italy','South Korea','US','UK']]
S
#https://stackoverflow.com/questions/3433486/how-to-do-exponential-and-logarithmic-curve-fitting-in-python-i-found-only-poly

#Relationship |  Example   |     General Eqn.     |  Altered Var.  |        Linearized Eqn.  

#-------------|------------|----------------------|----------------|------------------------------------------

#Linear       | x          | y =     B * x    + C | -              |        y =   C    + B * x

#Logarithmic  | log(x)     | y = A * log(B*x) + C | log(x)         |        y =   C    + A * (log(B) + log(x))

#Exponential  | 2**x, e**x | y = A * exp(B*x) + C | log(y)         | log(y-C) = log(A) + B * x

#Power        | x**2       | y =     B * x**N + C | log(x), log(y) | log(y-C) = log(B) + N * 