import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns; sns.set()

import statsmodels.formula.api as smf
df = pd.read_csv('../input/avocado-prices/avocado.csv',index_col=0)



# Date conversion

df['Date'] = pd.to_datetime(df['Date'])

df.head()
# Proportion of Conventional vs Organic

((df[df.region=='TotalUS']

 .pivot_table(index='type', values='Total Volume',

               aggfunc=np.sum, margins=True)

)/sum(df.loc[df.region=='TotalUS','Total Volume'])

).round(3)
# Focus on conventional type

df = df[df.type == 'conventional']



# Focus on Date, Region, Price, Quantity 

vars_keep = ['Date', 'Region', 'Price', 'Quantity']



df1 = df.rename(columns = {'AveragePrice':'Price',

                          'Total Volume':'Quantity', 

                          'region':'Region'})[vars_keep]



# Add Year, Month, and Year-Month

df1['Yr'] = df1['Date'].dt.year

df1['Mo'] = df1['Date'].dt.month

df1['YrMo'] = df1.Date.dt.strftime('%Y-%m')

df1
# Describe Date

df1.Date.describe()
# Count Region

df1.Region.value_counts()
# Describe Price and Quantity

df1.describe()
# Define National dataset

US = df1[df1.Region == 'TotalUS']

US = US.assign(Q_m = (US.Quantity/pow(10,6)).round(2))



sns.set_palette("RdBu_r")



# Colored by Year

plt.figure(figsize=(8,6))  

fig_US1 = sns.scatterplot('Q_m','Price', data=US, hue = 'Yr', 

                          palette="RdBu_r")

fig_US1.set(xlabel='Quantity, Million');
# Plot by Year

plt.figure(figsize=(12,8))  

fig_US2 = sns.lmplot('Q_m','Price', data=US, col = 'Yr',

                     height=6, aspect=.5)

fig_US2.set(xlabel='Quantity, Million');
# Color by Month

plt.figure(figsize=(8,6))  

sns.scatterplot('Q_m','Price', data=US, hue = 'Mo', 

                palette=sns.color_palette("Paired"));
# Aggregate to Year-Month level

# note: technically a quantity-weighted average should be used for the price. 

US_ym = US.groupby('YrMo').agg({'Price':'mean', 'Q_m':'sum'})

bins_p = np.array([0, 1.2, 1.6, 2])

US_ym = US_ym.assign(labels_p = pd.cut(US_ym.Price, bins_p))

US_ym[:5]
import plotly.express as px



US_ani = px.scatter(US_ym.assign(size = 20).reset_index(), 

                    x="Q_m", y="Price",  animation_frame="YrMo", 

                    color = 'labels_p',  

                    size = 'size',

                    #mode='markers', marker=dict(size=20),

                    color_discrete_sequence=["blue",  "magenta", "red"],

                    range_x=[80,280], range_y=[.7,1.7])



US_ani.update_layout(

    title="Animation: Avocado Price and Quantity, US",

    xaxis_title="Quantity, million/month",

    showlegend=False)

US_ani.show()
# Define a relative quantity variable with 2015-01 as a baseline 

Q_base = US_ym.Q_m["2015-01"]

US_ym['Q_change'] = US_ym.Q_m/Q_base

US_ym[:5]
# Stack Price and Q_change variables

US_ym2 = US_ym.reset_index().melt(id_vars=['YrMo'], 

                                  value_vars=['Price', 'Q_change'])

US_ym2
sns.relplot(x="YrMo", y="value", hue="variable", kind="line",

            palette=["#e74c3c", "#3498db"],

            data=US_ym2, aspect=3);

plt.xticks(rotation=45);
# Exclude non-city areas

large_areas = ['TotalUS','West','California','Midsouth',

               'Northeast','SouthCarolina',

               'SouthCentral','Southeast','GreatLakes',

               'NothernNewEngland','Plains']



cities_ym = (df1

             .assign(Q_m = (df1.Quantity/pow(10,6)).round(3))

             .query('not(Region in @large_areas)')

             .groupby(['YrMo','Region'])

             .agg({'Price':'mean', 'Q_m':'sum'})

            )



cities_ym
# Top 10 cities in 2018-01 

top10 = (cities_ym.loc['2018-01']

         .sort_values('Q_m', ascending=False).iloc[:10])         

top10
top10_city_names = top10.index

top10_city_names
# Keep the data of the top 10 cities

cities_ym = (cities_ym

             .reset_index()

             .set_index("Region")

             .loc[top10_city_names]

            )
cities_ym
# Put the year column back 

cities_ym['Yr'] = (cities_ym.YrMo

                   .str.slice(start=0, stop=4).astype(int))
# Plot by Year and Region

fig_Cities = sns.lmplot('Q_m','Price', 

                        data=cities_ym.reset_index(), 

                     col = 'Yr', row = "Region",

                     height=2, aspect=2.5)

fig_Cities.set(xlabel=''); 

# It is "Quantity, Million" but supressed here for readability
# Define 2015-01 price and quantity as the baseline 

cities_ym_base = (cities_ym[cities_ym.YrMo=="2015-01"]

                  .rename(columns={'Price':"BasePrice",

                                   "Q_m":"BaseQ_m"})

                  .drop(columns=['YrMo', 'Yr'])

                 )

cities_ym_base 
# Join the base price

cities_ym = cities_ym.join(cities_ym_base)           



# Define change in price and quantity as the ratio to the baseline

cities_ym['PriceChange'] = cities_ym.Price/cities_ym.BasePrice

cities_ym['QChange'] = cities_ym['Q_m']/ cities_ym.BaseQ_m

cities_ym
# Check stats on cities_ym 

(cities_ym[['Price','PriceChange', 'Q_m', 'QChange']]

 .groupby('Region')

 .agg(['mean','std','min','max'])

 .round(2).sort_values(('Price','mean'), ascending=False)

)
# Animate the city data! 

cities_ym = cities_ym.reset_index()

fig = px.scatter(cities_ym, x="QChange", y="Price",  

                 animation_frame="YrMo", hover_name="Region",

                 size = "Q_m", color="Region",

                range_x=[0.5,2.25], range_y=[.5,2.2]

                )



fig.show()
# Stack Price and Q_change variables

cities_ym2  = cities_ym.melt(id_vars=['YrMo', 'Region'],

                             value_vars=['Price', 'QChange'])

cities_ym2
# Plot price and quantity along the time axis in two subplots stacked vertically 

sns.relplot(x="YrMo", y="value", hue="Region", kind="line",

            palette=sns.color_palette("Paired", 10),

            data=cities_ym2, aspect=3, row="variable");

plt.xticks(rotation=45);
# OLS estimation

rlt1 = smf.ols('Price ~ 0 + Q_m*Region + YrMo',

               data=cities_ym.reset_index())

rlt1.fit().summary()
# Create residual, take a 1-month lagged variable

cities_ym['res1']  = rlt1.fit().resid

cities_ym['res1_L1'] = cities_ym.groupby('Region').res1.shift(1)

cities_ym 
rlt2 = smf.ols('res1 ~ res1_L1',

               data=cities_ym)

rlt2.fit().summary()
# got this from: http://web.vu.lt/mif/a.buteikis/wp-content/uploads/PE_Book/4-8-Multiple-autocorrelation.html 

from statsmodels.graphics.tsaplots import plot_acf

#

res1   = rlt1.fit().resid

fig = plt.figure(num = 1, figsize = (10, 8))

_ = plot_acf(res1, lags = 10, zero = False, 

             ax = fig.add_subplot(111))

plt.show()
import statsmodels.api as sm

rlt3 = sm.GLSAR(rlt1.endog, rlt1.exog, rho=1)

rlt3_fit = rlt3.fit() # alternatively: .iterative_fit(maxiter = 100)

rlt3_fit.summary()
# Correct the coefficient names in the table

coeff_names = rlt1.fit().summary2().tables[1].index

rlt3_coeff = rlt3_fit.summary2().tables[1].set_index(coeff_names)

rlt3_coeff 
time_trends = rlt3_coeff['Coef.'][rlt3_coeff.index.str.startswith("YrMo")] 

intercepts = rlt3_coeff['Coef.'][rlt3_coeff.index.str.startswith("Region")] 

slopes = rlt3_coeff['Coef.'][rlt3_coeff.index.str.startswith("Q_m")] 

slopes[1:] = slopes[0] + slopes[1:]
# Extract the time dummies and add a base intercept 

time_trends = pd.DataFrame(time_trends).reset_index()

time_trends['YrMo'] =  (pd

                        .to_datetime(

                            time_trends['index']

                            .str.replace("YrMo\\[T.","")

                            .str.replace("\\]","")

                        ).dt.strftime('%Y-%m'))

time_trends['Base_LosAngeles'] = intercepts["Region[LosAngeles]"] + time_trends['Coef.']

time_trends[:5]
figA = sns.relplot(x = "YrMo", y ="Base_LosAngeles", 

                   data = time_trends, kind='line', aspect=2.5);

plt.xticks(rotation=45);

plt.title('Fig A: Estimated Time Trends without Supply Change')

plt.ylabel('Base Price at Los Angeles, $');
# Extract city specific intercept and slope coefficients

Region = (intercepts

          .index.str.replace('Region\\[','')

          .str.replace('\\]',''))

rlt3_cities = (pd.DataFrame({'intercept': intercepts.values,

                             'slope': slopes.values})

               .set_index(Region)

              )

rlt3_cities.index.name = "Region"

rlt3_cities
# Add cities' Quantity range

rlt3_cities = (rlt3_cities

               .join(

                   cities_ym.groupby('Region').Q_m.agg(['min','max']))

              )

rlt3_cities['at_2018_03'] = float(

    time_trends[time_trends.YrMo=="2018-03"]["Coef."]

)

rlt3_cities['low'] = (rlt3_cities.intercept + 

                      rlt3_cities.at_2018_03 + 

                      rlt3_cities.slope * rlt3_cities['min'])

rlt3_cities['high'] = (rlt3_cities.intercept + 

                       rlt3_cities.at_2018_03 + 

                       rlt3_cities.slope * rlt3_cities['max'])

rlt3_cities
# Stack min-low and max-high variables under variable names of Q_m and Price

rlt3_cities_PQ = pd.concat(

    [rlt3_cities[['min','low']].rename(

        columns={'min':"Q_m", "low":"Price"}),

     rlt3_cities[['max','high']].rename(

         columns={'max':"Q_m", "high":"Price"})

    ], axis=0)

rlt3_cities_PQ 
filled_markers = ('o', 'v', '^', '<', '>', 's', '*', 'D', 'P', 'X')

sns.relplot(x = "Q_m", y="Price", hue = "Region", kind="line", 

            style= "Region",

            markers= filled_markers,

            dashes=False, aspect = 2, 

            data = rlt3_cities_PQ.reset_index());

plt.title('Fig B: Estimated Demand Curve by City without Time Trends')

plt.ylabel('Base Price at 2018-03, $');

plt.xlabel('Quantity, million');
# Recall the time trend figure

figA.fig