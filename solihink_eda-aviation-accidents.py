import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



aviation_incidents = '../input/aviation-accident-database-synopses/AviationData.csv'



df = pd.read_csv(aviation_incidents, engine ='python')
df.info()
df['Investigation.Type'].value_counts()
df = df[df['Investigation.Type']=='Accident']
import missingno as msno
msno.matrix(df)
df = df.drop(['Airport.Code', 'Airport.Name', 'Aircraft.Category', 'Registration.Number', 

              'Number.of.Engines', 'Engine.Type', 'FAR.Description', 'Schedule', 

              'Air.Carrier', 'Report.Status', 'Publication.Date'], axis=1)
df.shape
# replacing null values with 0 for numerical columns



df['Total.Fatal.Injuries'].replace({np.nan: 0}, inplace=True)

df['Total.Serious.Injuries'].replace({np.nan: 0}, inplace=True)

df['Total.Minor.Injuries'].replace({np.nan: 0}, inplace=True)

df['Total.Uninjured'].replace({np.nan: 0}, inplace=True)
# replacing null values with 'Unknown' or 'UNK' for categorical columns



df['Investigation.Type'].replace({np.nan: 'Unknown'}, inplace=True)

df['Location'].replace({np.nan: 'Unknown'}, inplace=True)

df['Country'].replace({np.nan: 'Unknown'}, inplace=True)

df['Latitude'].replace({np.nan: 'Unknown'}, inplace=True)

df['Longitude'].replace({np.nan: 'Unknown'}, inplace=True)

df['Aircraft.Damage'].replace({np.nan: 'Unknown'}, inplace=True)

df['Make'].replace({np.nan: 'Unknown'}, inplace=True)

df['Model'].replace({np.nan: 'Unknown'}, inplace=True)

df['Amateur.Built'].replace({np.nan: 'Unknown'}, inplace=True)

df['Purpose.of.Flight'].replace({np.nan: 'Unknown'}, inplace=True)

df['Weather.Condition'].replace({np.nan: 'UNK'}, inplace=True)

df['Broad.Phase.of.Flight'].replace({np.nan: 'UNKNOWN'}, inplace=True)
msno.matrix(df)
total_purpose = df['Purpose.of.Flight'].value_counts()

print(total_purpose)
def personal(purpose):

    if purpose == 'Personal':

        return 'Personal'

    

    else:

        return 'Non-personal'
df['Purpose.of.Flight_simplified'] = df['Purpose.of.Flight'].apply(personal)
plt.figure(figsize=(14, 6))

sns.countplot(df['Purpose.of.Flight_simplified'])
flight_purpose_damage = df.groupby(['Aircraft.Damage', 

                                    'Purpose.of.Flight_simplified']).size().reset_index().pivot(columns='Aircraft.Damage', 

                                                                                  index='Purpose.of.Flight_simplified', 

                                                                                  values=0)

flight_purpose_damage.plot(kind='bar', stacked=True)
df['Event.Date'] = pd.to_datetime(df['Event.Date'])

df['Month'] = df['Event.Date'].dt.month

df['Year'] = df['Event.Date'].dt.year

df = df[df['Year']>=1982]
plt.figure(figsize=(20,8))

sns.countplot(df['Year'],palette = 'coolwarm')
plt.figure(figsize=(20,8))

sns.countplot(df['Month'],palette='coolwarm')
df.head()
injury_count = pd.pivot_table(df, index=['Year'],values=['Total.Fatal.Injuries',

                                                         'Total.Serious.Injuries',

                                                         'Total.Minor.Injuries',

                                                         'Total.Uninjured'],aggfunc=np.sum)

injury_count.head()
fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(24, 14))

fig.subplots_adjust(hspace=.3)



fatal_plot = injury_count['Total.Fatal.Injuries'].plot(ax=axes[0,0], 

                                                       kind='bar', 

                                                       title = 'Total fatal injuries per year', 

                                                       color='r')



serious_plot = injury_count['Total.Serious.Injuries'].plot(ax=axes[0,1], 

                                                           kind='bar', 

                                                           title = 'Total serious injuries per year', 

                                                           color='y')



minor_plot = injury_count['Total.Minor.Injuries'].plot(ax=axes[1,0], 

                                                       kind='bar', 

                                                       title = 'Total minor injuries per year', 

                                                       color='g')



uninjured_plot = injury_count['Total.Uninjured'].plot(ax=axes[1,1], 

                                                      kind='bar', 

                                                      title = 'Total uninjured per year', 

                                                      color='b')
percent_phase = df['Broad.Phase.of.Flight'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'

percent_phase
def phase_of_flight(phase):

    if phase in (['UNKNOWN','TAXI','DESCENT','CLIMB','GO-AROUND','STANDING', 'OTHER']):

        return 'OTHERS'

    

    else:

        return phase
df['Phase.of.Flight_simplified'] = df['Broad.Phase.of.Flight'].apply(phase_of_flight)



df['Phase.of.Flight_simplified'].value_counts()
df['Total.Injuries'] = df['Total.Fatal.Injuries'] + df['Total.Serious.Injuries'] + df['Total.Minor.Injuries']



flight_phase_injury = df.groupby(['Phase.of.Flight_simplified'])['Total.Injuries'].agg('sum').reset_index()

print(flight_phase_injury)
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(20, 12))

fig.subplots_adjust(hspace=.3)





accident_plot = sns.countplot(df['Phase.of.Flight_simplified'], 

              ax=axes[0], 

              palette='coolwarm',

              order=['LANDING', 'TAKEOFF', 'CRUISE', 'MANEUVERING', 'APPROACH', 'OTHERS'])



accident_plot.set(xlabel='', ylabel = 'Number of Accidents')



injury_plot = sns.barplot(x = 'Phase.of.Flight_simplified', 

            y = 'Total.Injuries', 

            data = flight_phase_injury, 

            ax=axes[1], 

            palette='coolwarm',

            order=['LANDING', 'TAKEOFF', 'CRUISE', 'MANEUVERING', 'APPROACH', 'OTHERS'])



injury_plot.set(xlabel = '', ylabel = 'Total Injuries Caused')
flight_phase_damage = df.groupby(['Aircraft.Damage', 

                      'Phase.of.Flight_simplified']).size().reset_index().pivot(columns='Aircraft.Damage', 

                                                                                    index='Phase.of.Flight_simplified', 

                                                                                    values=0)



print(flight_phase_damage)



flight_phase_damage.plot(kind='bar', stacked=True)
accident_count = df.groupby(['Year', 

                             'Phase.of.Flight_simplified']).size().reset_index().pivot(columns='Phase.of.Flight_simplified', 

                                                                                       index='Year', 

                                                                                       values=0)

accident_count.head()
plt.figure(figsize = (20,10))

sns.heatmap(accident_count, cmap = 'Blues')

plt.xlabel('')

plt.title('Breakdown of accidents per category')
percent_weather = df['Weather.Condition'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'

percent_weather
weather_explode = (0.2, 0, 0)



weather_chart = df['Weather.Condition'].value_counts().plot(kind = 'pie',

                                            explode = weather_explode,

                                            autopct = '%.1f', 

                                            title = 'Weather Conditon',

                                            textprops = {'fontsize': 12}

                                            )



plt.ylabel('')
df['Make'].value_counts().head(10)
def get_make(make):

    

    if make in ['Cessna','Piper','Beech', 'Bell']:

        return make.upper()

        

    elif make in ['CESSNA','PIPER','BEECH', 'BELL']:

        return make

        

    else:

        return 'OTHER'
df['Make.simplified'] = df['Make'].apply(get_make)
plt.figure(figsize=(14, 6))

sns.countplot(df['Make.simplified'])
df_make = df[df['Make.simplified'] != 'OTHER']
df_make.shape
make_damage = df_make.groupby(['Aircraft.Damage', 

                      'Make.simplified']).size().reset_index().pivot(columns='Aircraft.Damage', 

                                                                                    index='Make.simplified', 

                                                                                    values=0)

make_damage.plot(kind='bar', stacked=True)
accidents_year = df.groupby(['Year'])["Event.Id"].count().reset_index(name="Number of Accidents")

accidents_year.head()
x = accidents_year['Year'].to_numpy()

y = accidents_year['Number of Accidents'].to_numpy()



print(x.shape)

print(y.shape)
x = x.reshape((-1, 1))



print(x.shape)
from sklearn.linear_model import LinearRegression

regr = LinearRegression()



regr.fit(x,y)
extended_x = np.arange(1982, 2025, 1).reshape(-1, 1)

y_pred = regr.predict(extended_x)



f, axes = plt.subplots(1, 1, figsize=(20, 8))

plt.plot(x,y, 'o')

plt.plot(extended_x, y_pred)

plt.plot(extended_x, y_pred, 's')

axes.set_ylim(ymin=0)

axes.set_xlabel('Year')

plt.xticks(np.arange(1982, 2025, 2))



print("Predicting number of accidents for the next 5 years:\n" )

for i in range (0,5):

    year = 2020+i

    n = -5+i

    print('Year %d: %d' % (year, y_pred[n]))

print('')    

print('Slope:', regr.coef_)
from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import mean_squared_error, r2_score
def create_polynomial_regression_model():

    

    for degree in [2,3]:

        poly_features = PolynomialFeatures(degree=degree)



        x_poly = poly_features.fit_transform(x)



        regr_poly = LinearRegression()

        regr_poly.fit(x_poly, y)





        y_poly_pred = regr_poly.predict(x_poly)



        rmse = np.sqrt(mean_squared_error(y,y_poly_pred))

        r2 = r2_score(y,y_poly_pred)

        

        print('Degree: ', degree)

        print('RMSE: ', rmse)

        print('R_squared: ', r2)

        print('')



        poly_2020 = regr_poly.predict(poly_features.fit_transform(np.array(2020).reshape(1,-1)))

        poly_2021 = regr_poly.predict(poly_features.fit_transform(np.array(2021).reshape(1,-1)))

        poly_2022 = regr_poly.predict(poly_features.fit_transform(np.array(2022).reshape(1,-1)))

        poly_2023 = regr_poly.predict(poly_features.fit_transform(np.array(2023).reshape(1,-1)))

        poly_2024 = regr_poly.predict(poly_features.fit_transform(np.array(2024).reshape(1,-1)))



        y_poly_pred_x = np.arange(1982, 2025, 1)



        poly_pred = np.concatenate((poly_2020, poly_2021, poly_2022, poly_2023, poly_2024))



        poly_pred_full = np.concatenate((y_poly_pred, poly_pred))



        data = np.array([[y_poly_pred_x], [poly_pred_full]])



        print("Predicting number of accidents for the next 5 years:\n" )

        for i in range (0,5):

            year = 2020+i

            n = -5+i

            print('Year %d: %d' % (year, poly_pred_full[n]))

        print('')

        

        plt.rcParams["figure.figsize"] = [20,8]

        plt.plot(x, y, 'o')

        

        plt.plot(y_poly_pred_x, poly_pred_full, label="degree %d" % degree +'; $R^2$: %.2f' % r2)

        plt.legend(loc="upper right")

        

        axes.set_ylim(ymin=0)

        axes.set_xlabel('Year')

        plt.xticks(np.arange(1982, 2025, 2))
create_polynomial_regression_model()