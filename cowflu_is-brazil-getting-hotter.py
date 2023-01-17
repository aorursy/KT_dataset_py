import os

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





# Some plotting variables

axis_font_size = 14

title_font_size = 16

line_width = 2.2

alp_temp = 0.5

alp_reg = 0.7

color_temperature = "#20948B"

color_regression = "#DE7A22"
path = r"/kaggle/input/temperature-timeseries-for-some-brazilian-cities/"

rio = r"station_rio.csv"

salvador = 'station_salvador.csv'

recife = r'station_recife.csv'

sao_paulo = r'station_sao_paulo.csv'



file_rio = path + rio

file_salvador = path + salvador

file_recife = path + recife

file_sao_paulo = path + sao_paulo



rio_df = pd.read_csv(file_rio)

salvador_df = pd.read_csv(file_salvador)

recife_df = pd.read_csv(file_recife)

sao_paulo_df = pd.read_csv(file_sao_paulo)
salvador_df.head(20)
def clean_df(city_df):

    

    to_drop = ['D-J-F', 'M-A-M', 'J-J-A', 'S-O-N', 'metANN']

    city_df.drop(to_drop, axis=1, inplace=True)

    

    # Change values 999

    city_df.replace(999.9, np.NaN, inplace=True)

    [city_df[col].fillna(city_df[col].mean(), inplace=True) for col in city_df.columns]

    
clean_df(salvador_df)

clean_df(rio_df)

clean_df(recife_df)

clean_df(sao_paulo_df)
salvador_df.head(20)
def find_regression(df, lr):

    

    df_t = df.T

    df_t.drop('YEAR', axis=0, inplace=True)

    df_series = pd.Series(df_t.values.ravel('F'))

    X = np.arange(0, len(df_series)).reshape(-1,1)

    Y = df_series.values.reshape(-1, 1)

    lr.fit(X, Y)  

    Y_pred = linear_regressor.predict(X)  

    

    return df_series, X,Y_pred
cities_names = ['Salvador', 'Rio', 'Recife', 'Sao Paulo']

list_cities = [salvador_df, rio_df, recife_df, sao_paulo_df]

linear_regressor = LinearRegression()  



min_year_list = [x['YEAR'].iloc[0] for x in list_cities]



# Trovo il maggiore così ho gli stessi hanni

max_year = np.max(min_year_list)

print(f"Each cities will be analyzed from {max_year}")



fig, ax = plt.subplots(2, 2, 

                       figsize=(10, 10),

                       sharey=True)

plt.subplots_adjust(hspace=0.5)



ax_flat = ax.flatten()

sns.despine(offset=20)

for index, city in enumerate(list_cities):

    

    city = city.loc[city['YEAR']>=max_year]

    city_series, X_city, Y_city = find_regression(city, linear_regressor)

    city_name = cities_names[index]

    

    # Main plot

    ax_flat[index].plot(city_series.values,

                        color_temperature,

                        alpha=alp_temp,

                        lw=line_width)

    ax_flat[index].plot(X_city, 

                        Y_city, 

                        color=color_regression,

                        alpha=alp_reg,

                        lw=line_width)

    

    # Putting onlty 6 ticks on the axis

    x_axis_spacing = len(X_city)//5

    

    # Setting the y-limits

    ax_flat[index].set_ylim(15, 35)

    

    if index ==0:

        y_ticks = ax_flat[index].get_yticks()[::2]

    

    # Decorations

    years = city['YEAR'].tolist() # to properly name the x-axis

    

    ax_flat[index].set_xticks(X_city[::x_axis_spacing])

    ax_flat[index].set_xticklabels(years[::x_axis_spacing//12], 

                                   fontsize=axis_font_size)

    

    ax_flat[index].set_yticks(y_ticks)

    ax_flat[index].set_yticklabels(ax_flat[index].get_yticks(),

                                   fontsize=axis_font_size)

    

    ax_flat[index].text(x=-0.1,

                        y=1.1,

                        s=city_name,

                        fontsize=title_font_size,

                        color='#414141', 

                        transform=ax_flat[index].transAxes)

    

    variation = Y_city.flat[-1]-Y_city.flat[0]



    print(f"In {city_name}, the temperature increased of {variation:.2f} degrees")