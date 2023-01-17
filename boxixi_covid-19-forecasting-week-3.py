import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator

import seaborn as sns

from datetime import datetime

%matplotlib inline



from patsy import dmatrices

import statsmodels.api as sm



from sklearn.compose import TransformedTargetRegressor, make_column_transformer, make_column_selector

from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures

from sklearn.pipeline import make_pipeline

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score, GridSearchCV, TimeSeriesSplit, cross_validate

from sklearn.ensemble import RandomForestRegressor, VotingRegressor

from xgboost import XGBRegressor

from sklearn.metrics import make_scorer



import warnings

warnings.filterwarnings(action='ignore')
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')

submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/submission.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')



elders = pd.read_excel('/kaggle/input/world-stats-from-un/WPP2019_POP_F07_1_POPULATION_BY_AGE_BOTH_SEXES.xlsx')

hospital_beds = pd.read_csv('/kaggle/input/external-datasets/hospital-beds-per-1000-people.csv')

physicians = pd.read_csv('/kaggle/input/external-datasets/physicians-per-1000-people.csv')

median_age = pd.read_excel('/kaggle/input/world-stats-from-un/WPP2019_POP_F05_MEDIAN_AGE.xlsx')

population_density = pd.read_excel('/kaggle/input/world-stats-from-un/WPP2019_POP_F06_POPULATION_DENSITY.xlsx')

sex_ratio = pd.read_excel('/kaggle/input/external-datasets/WPP2019_POP_F04_SEX_RATIO_OF_TOTAL_POPULATION.xlsx')
elder_rate = elders.drop(['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6'], axis = 1)

elder_rate = elder_rate.drop(range(0, 391), axis = 0)

elder_rate['Unnamed: 2'] = elder_rate['Unnamed: 2'].replace({'Bolivia (Plurinational State of)': 'Bolivia', 'Brunei Darussalam': 'Brunei', "Côte d'Ivoire": "Cote d'Ivoire", 'Democratic Republic of the Congo': 'Congo (Kinshasa)', 'Congo': 'Congo (Brazzaville)', 'Iran (Islamic Republic of)': 'Iran', 'Republic of Korea': 'Korea, South', 'Myanmar': 'Burma', "Lao People's Democratic Republic": 'Laos', 'Russian Federation': 'Russia', 'Syrian Arab Republic': 'Syria', 'Republic of Moldova': 'Moldova', 'Viet Nam': 'Vietnam', 'China, Taiwan Province of China': 'Taiwan*', 'United Republic of Tanzania': 'Tanzania', 'United States of America': 'US', 'Venezuela (Bolivarian Republic of)': 'Venezuela'}) 

elder_rate = elder_rate[['Unnamed: 2', 'Unnamed: 7']].join(elder_rate.drop(['Unnamed: 2', 'Unnamed: 7'], axis = 1).div(elder_rate.drop(['Unnamed: 2', 'Unnamed: 7'], axis = 1).sum(axis = 1), axis = 0))

elder_rate = elder_rate[elder_rate['Unnamed: 7'] == 2020][['Unnamed: 2', 'Unnamed: 22', 'Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25', 'Unnamed: 26', 'Unnamed: 27', 'Unnamed: 28']]

elder_rate['Age_Above_70'] = elder_rate.drop(['Unnamed: 2'], axis = 1).sum(axis = 1)

elder_rate = elder_rate[['Unnamed: 2', 'Age_Above_70']]

elder_rate.columns = ['Country_Region', 'Age_Above_70']
year1 = pd.DataFrame({'Entity': hospital_beds.groupby(['Entity'])['Year'].max().index, 'Year': hospital_beds.groupby(['Entity'])['Year'].max().to_list()})

hospital_beds_rate = year1.merge(hospital_beds, how = 'left', left_on = ['Entity', 'Year'], right_on = ['Entity', 'Year'])

hospital_beds_rate.drop(['Year', 'Code'], axis = 1, inplace = True)

hospital_beds_rate.columns = ['Country_Region', 'Hosipital_beds']



hospital_beds_rate['Country_Region'] = hospital_beds_rate['Country_Region'].replace({'Congo': 'Congo (Brazzaville)', 'Democratic Republic of Congo': 'Congo (Kinshasa)', 'Czech Republic': 'Czechia', 'South Korea': 'Korea, South', 'United States': 'US', 'Timor': 'Timor-Leste', 'Macedonia': 'North Macedonia', 'Cape Verde': 'Cabo Verde'}) 
# Obtain the number of physicians per 1,000 people of each country in the recent year

year2 = pd.DataFrame({'Entity':physicians.groupby(['Entity'])['Year'].max().index, 'Year': physicians.groupby(['Entity'])['Year'].max().to_list()})

physician_rate = year2.merge(physicians, how = 'left', left_on = ['Entity', 'Year'], right_on = ['Entity', 'Year'])

physician_rate.drop(['Year', 'Code'], axis = 1, inplace = True)

physician_rate.columns = ['Country_Region', 'Physicians']



physician_rate['Country_Region'] = physician_rate['Country_Region'].replace({'Congo': 'Congo (Brazzaville)', 'Democratic Republic of Congo': 'Congo (Kinshasa)', 'Czech Republic': 'Czechia', 'South Korea': 'Korea, South', 'United States': 'US', 'Timor': 'Timor-Leste', 'Macedonia': 'North Macedonia', 'Cape Verde': 'Cabo Verde', 'Syrian Arab Republic': 'Syria'}) 
# Obtain the median age of each country in 2020

median_age = median_age[['Unnamed: 2', 'Unnamed: 21']]

median_age.columns = ['Country_Region', 'Median_Age_2020']

median_age.drop(range(0, 42), axis = 0, inplace = True)

median_age['Country_Region'] = median_age['Country_Region'].replace({'Bolivia (Plurinational State of)': 'Bolivia', 'Brunei Darussalam': 'Brunei', "Côte d'Ivoire": "Cote d'Ivoire", 'Democratic Republic of the Congo': 'Congo (Kinshasa)', 'Congo': 'Congo (Brazzaville)', 'Iran (Islamic Republic of)': 'Iran', 'Republic of Korea': 'Korea, South', 'Myanmar': 'Burma', "Lao People's Democratic Republic": 'Laos', 'Russian Federation': 'Russia', 'Syrian Arab Republic': 'Syria', 'Republic of Moldova': 'Moldova', 'Viet Nam': 'Vietnam', 'China, Taiwan Province of China': 'Taiwan*', 'United Republic of Tanzania': 'Tanzania', 'United States of America': 'US', 'Venezuela (Bolivarian Republic of)': 'Venezuela'}) 
# Obtain the population density (number of people per square km) of each country in 2020

population_density = population_density[['Unnamed: 2', 'Unnamed: 77']]

population_density.columns = ['Country_Region', 'Population_density_2020']

population_density.drop(range(0, 42), axis = 0, inplace = True)

population_density['Country_Region'] = population_density['Country_Region'].replace({'Bolivia (Plurinational State of)': 'Bolivia', 'Brunei Darussalam': 'Brunei', "Côte d'Ivoire": "Cote d'Ivoire", 'Viet Nam': 'Vietnam', 'Democratic Republic of the Congo': 'Congo (Kinshasa)', 'Congo': 'Congo (Brazzaville)', 'Iran (Islamic Republic of)': 'Iran', 'Republic of Korea': 'Korea, South', 'Myanmar': 'Burma', "Lao People's Democratic Republic": 'Laos', 'Russian Federation': 'Russia', 'Syrian Arab Republic': 'Syria', 'Republic of Moldova': 'Moldova', 'China, Taiwan Province of China': 'Taiwan*', 'United States of America': 'US', 'United Republic of Tanzania': 'Tanzania', 'Venezuela (Bolivarian Republic of)': 'Venezuela'}) 
# Obtain the sex ratio (number of males per 100 females) of each country in 2020

sex_ratio = sex_ratio[['Unnamed: 2', 'Unnamed: 21']]

sex_ratio.columns = ['Country_Region', 'Sex_Ratio_2020']

sex_ratio.drop(range(0, 42), axis = 0, inplace = True)

sex_ratio['Country_Region'] = sex_ratio['Country_Region'].replace({'Bolivia (Plurinational State of)': 'Bolivia', 'Brunei Darussalam': 'Brunei', "Côte d'Ivoire": "Cote d'Ivoire", 'Viet Nam': 'Vietnam', 'Democratic Republic of the Congo': 'Congo (Kinshasa)', 'Congo': 'Congo (Brazzaville)', 'Iran (Islamic Republic of)': 'Iran', 'Republic of Korea': 'Korea, South', 'Myanmar': 'Burma', "Lao People's Democratic Republic": 'Laos', 'Russian Federation': 'Russia', 'Syrian Arab Republic': 'Syria', 'Republic of Moldova': 'Moldova', 'China, Taiwan Province of China': 'Taiwan*', 'United States of America': 'US', 'United Republic of Tanzania': 'Tanzania', 'Venezuela (Bolivarian Republic of)': 'Venezuela'}) 
# Merge all the features into training set

train = train.merge(elder_rate, how = 'left', on = 'Country_Region').merge(hospital_beds_rate, how = 'left', on = 'Country_Region').merge(physician_rate, how = 'left', on = 'Country_Region').merge(median_age, how = 'left', on = 'Country_Region').merge(population_density, how = 'left', on = 'Country_Region').merge(sex_ratio, how = 'left', on = 'Country_Region')
train.drop(['Id'], axis = 1, inplace = True)
train['Date'] = pd.to_datetime(train['Date'])  # Convert feature date into datetime variable

train['Day'] = train['Date'].apply(lambda x: x.day)  # Extract day from date

train['month'] = train['Date'].apply(lambda x: x.month)  # Extract month from date

train['dayofweek'] = train['Date'].apply(lambda x: x.dayofweek)  # Extract day of week from date



# Calculate lags

train['Province_State'] = train['Province_State'].fillna('Missing')

def lags(r, df, col):

    for i in range(r):

        col_name = col[0] + '_' + 'lag' + str(i + 1)

        df[col_name] = df.groupby(['Country_Region', 'Province_State'])[col].shift(i + 1)

    return df



lags(8, train, 'ConfirmedCases')

lags(8, train, 'Fatalities')
# Remove the observations of the first 9 days

train = train[train['Date'] >= datetime.strptime('2020-01-31', '%Y-%m-%d')]
# Calculate the increasing rate

def increase_rate(r, df, col):

    col_t_1 = col[0] + '_' + 'lag' + str(1)

    for i in range(2, r + 1, 1):

        col_t_2 = col[0] + '_' + 'lag' + str(i)

        col_name = col[0] + '_' + 'increase' + str(i - 1)

        div_num = df[col_t_2].apply(lambda x: 0.001 if x == 0 else x)

        df[col_name] = df[col_t_1] - df[col_t_2]

        df[col_name] = df[col_name].div(div_num)

    return df



increase_rate(8, train, 'ConfirmedCases')

increase_rate(8, train, 'Fatalities')
# Convert the feature type from object to float

train['Median_Age_2020'] = train['Median_Age_2020'].apply(lambda x: float(x))

train['Population_density_2020'] = train['Population_density_2020'].apply(lambda x: float(x))

train['Sex_Ratio_2020'] = train['Sex_Ratio_2020'].apply(lambda x: float(x))
train.reset_index(drop = True, inplace = True)

X_train = train.drop(['ConfirmedCases', 'Fatalities'], axis = 1)

y1_train = train['ConfirmedCases']

y2_train = train['Fatalities']



X_train.drop(['Date'], axis = 1, inplace = True)
def neg_rmsle(y, y_pred):

    score = -np.sqrt(np.sum(np.power(np.log(y_pred + 1) - np.log(y + 1), 2)) / len(y))

    return score



my_scorer = make_scorer(neg_rmsle, greater_is_better = True)
categorical = make_column_selector(dtype_include = 'object')

continuous = make_column_selector(dtype_exclude = 'object')

cat_pipe = make_pipeline(SimpleImputer(strategy = 'constant', fill_value = 'Missing'),

                         OneHotEncoder(handle_unknown = 'ignore'))

cont_pipe = make_pipeline(SimpleImputer(strategy = 'median'), StandardScaler())

col_transformer = make_column_transformer((cat_pipe, categorical), (cont_pipe, continuous))



pipe1 = make_pipeline(col_transformer, LinearRegression())



tss = TimeSeriesSplit(n_splits = 5)
pipe9 = make_pipeline(col_transformer, VotingRegressor([('rf', RandomForestRegressor(n_estimators = 50, random_state = 1)), ('XGB', XGBRegressor(learning_rate = 0.7, max_depth = 6, objective = 'count:poisson'))]))

pipe10 = make_pipeline(col_transformer, VotingRegressor([('rf', RandomForestRegressor(n_estimators = 50, random_state = 1)), ('XGB', XGBRegressor(learning_rate = 0.5, max_depth = 6, objective = 'count:poisson'))]))

print('The mean test score for Confirmed Cases Prediction is:', -np.mean(cross_val_score(pipe9, X_train, y1_train, cv = tss, scoring = my_scorer)))

print('The mean test score for Fatalities Prediction is:', -np.mean(cross_val_score(pipe10, X_train, y2_train, cv = tss, scoring = my_scorer)))
# Obtain training set which includes the data between '31/01/2020' and '25/03/2020'

# X_tr = train[train['Date'] < datetime.strptime('2020-03-26', '%Y-%m-%d')].drop(['Date', 'ConfirmedCases', 'Fatalities'], axis = 1)

# y1_tr = train[train['Date'] < datetime.strptime('2020-03-26', '%Y-%m-%d')]['ConfirmedCases']

# y2_tr = train[train['Date'] < datetime.strptime('2020-03-26', '%Y-%m-%d')]['Fatalities']



X_tr = train.drop(['Date', 'ConfirmedCases', 'Fatalities'], axis = 1)

y1_tr = train['ConfirmedCases']

y2_tr = train['Fatalities']



# Obtain the dataset on '25/03/2020' to help predictition

first_rows = train[train['Date'] == datetime.strptime('2020-03-25', '%Y-%m-%d')].drop(['Date', 'ConfirmedCases', 'Fatalities'], axis = 1)

first_rows.reset_index(drop = True, inplace = True)
# Merge all the features into test set

test = test.merge(elder_rate, how = 'left', on = 'Country_Region').merge(hospital_beds_rate, how = 'left', on = 'Country_Region').merge(physician_rate, how = 'left', on = 'Country_Region').merge(median_age, how = 'left', on = 'Country_Region').merge(population_density, how = 'left', on = 'Country_Region').merge(sex_ratio, how = 'left', on = 'Country_Region')



test['Date'] = pd.to_datetime(test['Date'])  # Convert feature date into datetime variable

test['Day'] = test['Date'].apply(lambda x: x.day)  # Extract day from date

test['month'] = test['Date'].apply(lambda x: x.month)  # Extract month from date

test['dayofweek'] = test['Date'].apply(lambda x: x.dayofweek)  # Extract day of week from date

test.drop(['ForecastId', 'Date'], axis = 1, inplace = True)



# Replace NaN with 'Missing'

test['Province_State'] = test['Province_State'].fillna('Missing')
def training(X_training, y1_training, y2_training, model1, model2):

    # Create pipeline

    categorical = make_column_selector(dtype_include = 'object')

    continuous = make_column_selector(dtype_exclude = 'object')

    cat_pipe = make_pipeline(SimpleImputer(strategy = 'constant', fill_value = 'Missing'),

                             OneHotEncoder(handle_unknown = 'ignore'))

    cont_pipe = make_pipeline(SimpleImputer(strategy = 'median'), StandardScaler())

    col_transformer = make_column_transformer((cat_pipe, categorical), (cont_pipe, continuous))

    pipe1 = make_pipeline(col_transformer, model1)

    pipe2 = make_pipeline(col_transformer, model2)

    # Train the model

    pipe1.fit(X_training, y1_training)

    pipe2.fit(X_training, y2_training)

    

    return (pipe1, pipe2)





def predict(model1, model2, data):

    pred1 = model1.predict(data.to_frame().T)

    pred2 = model2.predict(data.to_frame().T)

    return (pred1, pred2)





def prediction(model1, model2, test_set, first):

    df = first

    df.reset_index(drop = True, inplace = True)

    y1_prediction = []

    y2_prediction = []

    for i in range(test_set.shape[0]):

        info = test_set.iloc[i].to_list()

        y1_pred, y2_pred = predict(model1, model2, df.iloc[i].T)

        y1_prediction.append(y1_pred[0])

        y2_prediction.append(y2_pred[0])

        # Update lags

        c_lag = df.iloc[i][11: 19].to_list()

        c_lag.insert(0, y1_pred[0])

        c_lag.pop()

        f_lag = df.iloc[i][19: 27].to_list()

        f_lag.insert(0, y2_pred[0])

        f_lag.pop()

        lag = c_lag + f_lag

        c_increase = []

        f_increase = []

        # Calculate increasing rate

        for i in range(1, 8, 1):

            c_1 = lag[0] - lag[i]

            f_1 = lag[8] - lag[i + 8]

            div_c = 0.001 if lag[i] == 0 else lag[i]

            div_f = 0.001 if lag[i + 8] == 0 else lag[i + 8]

            c_increase.append(c_1 / div_c)

            f_increase.append(f_1 / div_f)

        # Add a new row to df used to predict

        df.loc[len(df)] = info + lag + c_increase + f_increase

    y1_prediction.append(y1_pred[0])

    y2_prediction.append(y2_pred[0])

    return (df, y1_prediction, y2_prediction)
def make_predictions(X_tr, y1_tr, y2_tr, X_te, model_1, model_2, first_rows = first_rows):

    model1, model2 = training(X_tr, y1_tr, y2_tr, model_1, model_2)



    country = X_te.iloc[0][1]

    state = X_te.iloc[0][0]

    start = 0

    column_names = X_tr.columns.to_list()

    df = pd.DataFrame(columns = column_names)

    y1_prediction = []

    y2_prediction = []

    for i in X_te.index:

        idx = first_rows[(first_rows['Country_Region'] == country) & (first_rows['Province_State'] == state)].index

        if X_te.iloc[i][0] == state and X_te.iloc[i][1] == country:

            pass

        else:

            pred = X_te.iloc[start: i].reset_index(drop = True)

            df_pred, y1_p, y2_p = prediction(model1, model2, pred, first_rows.loc[idx])

            df = pd.concat([df, df_pred])

            y1_prediction += y1_p

            y2_prediction += y2_p

            start = i

            country = X_te.iloc[i][1]

            state = X_te.iloc[i][0]



    idx = first_rows[(first_rows['Country_Region'] == country) & (first_rows['Province_State'] == state)].index

    pred = X_te.iloc[start:].reset_index(drop = True)

    df_pred, y1_p, y2_p = prediction(model1, model2, pred, first_rows.loc[idx])

    df = pd.concat([df, df_pred])

    y1_prediction += y1_p

    y2_prediction += y2_p

    df['y1_pred'] = y1_prediction

    df['y2_pred'] = y2_prediction



    df['Date'] = df['month'].apply(lambda x: str(x)) + '-' + df['Day'].apply(lambda x: str(x))

    df = df[df['Date'] != '3-26']

    df.drop(['Date'], axis = 1, inplace = True)

    return df
result_com2 = make_predictions(X_tr, y1_tr, y2_tr, test, VotingRegressor([('rf', RandomForestRegressor(n_estimators = 1000, random_state = 1)), ('XGB', XGBRegressor(learning_rate = 0.7, max_depth = 6, objective = 'count:poisson'))]), VotingRegressor([('rf', RandomForestRegressor(n_estimators = 1000, random_state = 1)), ('XGB', XGBRegressor(learning_rate = 0.5, max_depth = 6, objective = 'count:poisson'))]))
result_com2.reset_index(drop = True, inplace = True)
submission['ConfirmedCases'] = result_com2['y1_pred']

submission['Fatalities'] = result_com2['y2_pred']
# Output

submission.to_csv('submission.csv', index = False)