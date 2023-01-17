#Imports for the notebook

import numpy as np

import pandas as pd

import seaborn as sns

# import plotly.express as px

from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.inspection import permutation_importance

from sklearn import preprocessing

import plotly.graph_objects as go
us_cases_county = pd.read_csv('/kaggle/input/usafacts-updated/covid_confirmed_usafacts_June1.csv')

us_deaths_county = pd.read_csv('/kaggle/input/usafacts-updated/covid_deaths_usafacts_June1.csv')



#Importing data on each county

health_by_county = pd.read_csv('/kaggle/input/county-health-rankings/us-county-health-rankings-2020.csv')

health_by_county.rename(columns={'fips': "countyFIPS"}, inplace=True)

health_by_county.head()
cum_cases_county = us_cases_county.filter(['countyFIPS', 'County Name', 'State', 'stateFIPS', '6/1/20'], axis=1)

cum_deaths_county = us_deaths_county.filter(['countyFIPS', 'County Name', 'State', 'stateFIPS', '6/1/20'], axis=1)
cum_deaths_county[cum_deaths_county['6/1/20'] >= 20].sort_values(by='6/1/20', ascending=False).shape
#Changed names for some columns

cum_cases_county.rename(columns={'6/1/20': "Confirmed Cases"}, inplace=True)

cum_deaths_county.rename(columns={'6/1/20': "Deaths"}, inplace=True)

#Building new Dataframe 

cum_cases_deaths_county = pd.DataFrame(cum_cases_county)

cum_cases_deaths_county['Deaths']= cum_deaths_county['Deaths']

cum_cases_deaths_county['Death Rate'] = cum_cases_deaths_county['Deaths'] / cum_cases_deaths_county['Confirmed Cases']
cum_cases_deaths_county = cum_cases_deaths_county[(cum_cases_deaths_county['countyFIPS'] != 0) 

                                                  & (cum_cases_deaths_county['Death Rate'] <= 1)]

cum_cases_deaths_county[cum_cases_deaths_county['Deaths'] >= 20].sort_values(by='Death Rate', ascending=False).head(10)
death_rate_state = cum_cases_deaths_county.groupby(['State']).agg(

    {'Death Rate': 'mean'}).reset_index().sort_values(by='Death Rate', ascending=False)



death_rate_state.head()
#Geographical map representation death rates

fig = go.Figure(data=go.Choropleth(locations=death_rate_state['State'],

                                   z=death_rate_state['Death Rate'].astype(float),

                                  locationmode='USA-states', 

                                  colorscale='Reds',

                                  colorbar_title='Death Rate of Covid-19'))

fig.update_layout(title_text='Average Death Rate of Covid-19 by State', 

                  geo_scope='usa')

fig.show()
#Getting filtered county data by index

index_lst = []

index_lst.extend(range(7))

index_lst.extend(range(23,31))

index_lst.extend(range(55,71))

index_lst.extend(range(103,112))

index_lst.extend(range(120,122))

index_lst.extend(range(134,141))

index_lst.extend(range(163,167)) #Income

index_lst.extend(range(203,215)) #Housing

index_lst.extend(range(326,330)) #Food

index_lst.extend(range(371,382))

index_lst.extend(range(394,397))

index_lst.append(412) #reduced lunch

index_lst.extend(range(485,507))



#Total 106 features (columns)

simp_health_county = health_by_county.iloc[:, index_lst]

#Outputs our newly filtered dataframe

simp_health_county.head()
#Now do a join on simp_health_county and death rate

county_data = pd.merge(simp_health_county, cum_cases_deaths_county, on='countyFIPS')

county_data.drop(['County Name', 'State'], axis=1, inplace=True)

county_data.head()
#Filter the data for more than 15 deaths (Gives us 448 samples (not much :( )

#10 or more gives us 561 samples 

min_deaths = 20

filtered_county_data = county_data[county_data['Deaths'] >= min_deaths]



#Add random column (used in feature importance)

filtered_county_data['random'] = np.random.random(size=len(filtered_county_data))



#Ouput Data

labels = pd.DataFrame(filtered_county_data['Death Rate'])



#Get input data

x_data = filtered_county_data.drop(['state',

                                    'county', 'Death Rate',

                                    'primary_care_physicians_ratio',

                                    'other_primary_care_provider_ratio',

                                    'Deaths','Confirmed Cases'], axis=1)

#Dealing with the NANS

#average of that column by state 

x_data.fillna(x_data.groupby(['stateFIPS']).transform('mean'), inplace=True)



#Only effective if the state has no value (Put in average for the entire column)

x_data.fillna(x_data.mean(), inplace=True)





#Train Test split 

x_train, x_test, y_train, y_test = train_test_split(x_data, labels, 

                                                    test_size=0.2, random_state=101)



#View our training input data

x_train.head()
#Normalizing the x training data

scaler = preprocessing.MinMaxScaler()

x_train_norm = scaler.fit_transform(x_train)

x_train_norm_df = pd.DataFrame(x_train_norm, columns=x_train.columns)



#From Towards Data Science Article

#Normalizing x test data

scaler = preprocessing.MinMaxScaler()

x_test_norm = scaler.fit_transform(x_test)

x_test_norm_df = pd.DataFrame(x_test_norm, columns=x_test.columns)
#Will use Random Forests

rf = RandomForestRegressor(n_estimators=400, max_features='sqrt', n_jobs=1, oob_score=True,

                           bootstrap=True, random_state=101)

model = rf.fit(x_train_norm_df, y_train.values.ravel())

print('R^2 Training Score: {:.2f}'.format(rf.score(x_train_norm_df, y_train)))

print('OOB Score: {:.2f}'.format(rf.oob_score_))

print('Validation Score: {:.2f}'.format(rf.score(x_test_norm_df, y_test)))

#Using SKLearn feature importance

#Need feature names to be with feature_importances 

feature_importances_init = rf.feature_importances_

feature_importances = []

feat_cols = []

for i in range(feature_importances_init.shape[0]):

    if feature_importances_init[i] >= 0.01:

        feature_importances.append(feature_importances_init[i])

        feat_cols.append(x_train.columns[i])



#Convert lists to numpy arrays 

feature_importances = np.asarray(feature_importances)

feat_cols = np.asarray(feat_cols)



num_features = len(feature_importances)



#We want to sort the importances and in order to plot them

sorted_importances_indices = feature_importances.argsort()
fig, ax = plt.subplots()

fig.set_figheight(15)

ax.barh(range(num_features), feature_importances[sorted_importances_indices], color='b', align='center')

ax.set_yticks(range(num_features))

ax.set_yticklabels(feat_cols)

ax.invert_yaxis()

ax.set_title("Covid-19 Death Rate Feature Importances")



plt.show()
r = permutation_importance(model, x_test, y_test, n_repeats=30, random_state=42)
permutation_df = pd.DataFrame(columns=['Feature', 'Importance Mean', 'Importance'])



for i in r.importances_mean.argsort()[::-1]:

    #Checking if it is within two standard deviations of the mean

    if (r.importances_mean[i] - 2 * r.importances_std[i]) > 0:

        importance_val = str(r.importances_mean[i]) + " +/- " + str(r.importances_std[i])

        permutation_df = permutation_df.append({'Feature': x_train.columns[i], 'Importance Mean': r.importances_mean[i],

                                                'Importance': importance_val}, ignore_index=True)



#Sorts the features in permutation_df from largest to smallest importance

permutation_df.sort_values(by='Importance Mean', ascending=False)
def feature_lin_correlation(x_data, y_data):

    correlation_df = pd.DataFrame(columns=['Feature', 'Correlation'])

    for cols in x_data.columns:

        reg = LinearRegression()

        #Input and output data for linear regression

        x = x_data[cols].to_numpy()

        y = y_data.to_numpy()

        

        #Reshaping data

        x = x.reshape(-1, 1)

        y = y.reshape(-1, 1)

        

        fitReg = reg.fit(x, y) 



        #Adds feature and its correlation value to the correlation dataframe

        correlation_df = correlation_df.append({'Feature': str(cols), 'Correlation': float(fitReg.coef_)}, 

                                               ignore_index=True)

    return correlation_df
correlation_df = feature_lin_correlation(x_data, labels)



#Orders the correlation dataframe from highest postive correlation to highest negative correlation

correlation_df.sort_values(by='Correlation', ascending=False)