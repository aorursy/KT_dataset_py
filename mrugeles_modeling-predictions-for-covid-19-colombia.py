import math

import operator

import datetime

import warnings

warnings.filterwarnings('ignore')



import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import plotly.graph_objects as go



from sklearn.metrics import mean_squared_error



from scipy.signal import find_peaks



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import PolynomialFeatures



#/kaggle/input/covid19-daily-reports-by-country/FO.csv
prediction_days =  300



def logistic_funct(x = 1, L = 2, a = 0, k = 3.7):

    return L/(1 + math.e**(-k*(x - a)))     



def get_logistic_df(x_range = [], L = 2, a = 0, k = 3.7):

    logistic_df = pd.DataFrame(columns = ['x', 'y'])

    logistic_df['x'] = x_range

    logistic_df['y'] = [logistic_funct(n, L, a, k) for n in x_range]

    return logistic_df



def get_growth_factors(y):

    return np.array([x / y[i - 1] for i, x in enumerate(y) if i > 0])

        

def get_growth_factor_idx(n, limit):

    x1 = 0

    y1 = 0.01

    growth_factor = 2

    x = 1



    while growth_factor > limit:

        y = (n*x)**2

        growth_factor = y / y1

        x += 1

        y1 = y

    return x        

    



def plot_curve(data, title):

    

    fig = go.Figure()

    for key in data.keys():

        df = data[key]

        fig.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='lines', name=key))



    fig.update_layout(

        title_text=title

    )

    fig.show()

    

    



def get_exp_curve(start, stop, num):

    x_range = np.linspace(start, stop, num)

    df = pd.DataFrame(columns = ['x', 'y'])

    df['x'] = x_range

    df['y'] = [x2_funct(n) for n in x_range]



    # Move to origin

    y_min = df['y'].min()

    x_min = df.loc[df['y'] == df['y'].min()]['x'].min()

    x = -x_min

    y = -y_min

    df['x'] = df['x'].apply(lambda value: value + x)

    df['y'] = df['y'].apply(lambda value: value + y)





    df = df.loc[(df['x'] > 0) & (df['y'] > 0)]



    return df.reset_index(drop = True)





def get_log_exp(L = 2, a = 0, k = 1):

    x_range = np.linspace(0, prediction_days*2, num=2000)



    logistic_df = get_logistic_df(x_range, L = L, a = a, k = k)

    logistic_df = logistic_df.loc[(logistic_df['y'] >= 0.01)].reindex()



    return logistic_df





def get_logistic_func(L = 2, a = 0, k = 1):

    x_range = np.arange(-prediction_days, prediction_days) 



    logistic_df = get_logistic_df(x_range, L = L, a = a, k = k)



    return logistic_df



def move_to_origin(df):

    # Move to origin

    y_min = df['y'].min()

    x_min = df.loc[df['y'] == df['y'].min()]['x'].min()

    x = -x_min

    y = -y_min

    df['x'] = df['x'].apply(lambda value: value + x)

    df['y'] = df['y'].apply(lambda value: value + y)





    df = df.loc[(df['x'] > 0) & (df['y'] > 0)]



    return df.reset_index(drop = True)





from sklearn.model_selection import train_test_split



def to_polinomyal(x):

    polynomial_features= PolynomialFeatures(degree=2)

    x = x[:, np.newaxis]



    return polynomial_features.fit_transform(x)



def build_model(x, y):

    x = to_polinomyal(x)

    y = to_polinomyal(y)



    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=300)



    model = LinearRegression()

    return model.fit(X_train, y_train)



def get_predictions(model, x_predic): 

    x_predic = to_polinomyal(x_predic)

    y_predic = model.predict(x_predic)



    predictions = pd.DataFrame(columns=['x', 'y'])

    predictions['x'] = x_predic[:, 1]

    predictions['y'] = y_predic[:, 1]



    return predictions
dataset = pd.read_csv('/kaggle/input/covid19-daily-reports-by-country/CO.csv')

dataset['Report_Date'] = pd.to_datetime(dataset['Report_Date'])

dataset[['Confirmed', 'Deaths', 'Recovered']] = dataset[['Confirmed', 'Deaths', 'Recovered']].fillna(0)

dataset['x'] = np.arange(0, dataset.shape[0])

dataset.tail()

cases_by_day_df = dataset[['x', 'Confirmed']]

cases_by_day_df.rename(columns = {'Confirmed': 'y'}, inplace=True)

cases_by_day_df.loc[:,'y'] = cases_by_day_df['y'].diff()



data = {

    'Cases by day': cases_by_day_df

}



plot_curve(data, 'Cases by day')
aggregated_cases_df = dataset[['x', 'Confirmed']]

aggregated_cases_df.rename(columns = {'Confirmed': 'y'}, inplace=True)



data = {

    'Aggregated cases': aggregated_cases_df

}



plot_curve(data, 'Aggregated cases')
def get_parabole_segment(n, start, end, num):

    df = pd.DataFrame(columns = ['x', 'y'])

    df['x'] = np.linspace(start, end, num)

    df['y'] = (n*df['x'])**2

    return df



def get_logistic_segment(L, a, k, start, end, num):

    x_range = np.linspace(start, end, num)

    logistic_df = get_logistic_df(x_range, L = L, a = a, k = k)

    #logistic_df = logistic_df.loc[(logistic_df['y'] >= 0.01)].reindex()



    return logistic_df



def diff_area(curve1, curve2):

    x_start_curve1 = curve1['x'].min()

    x_end_curve1 = curve1['x'].max()



    x_start_curve2 = curve2['x'].min()

    x_end_curve2 = curve2['x'].max()

    

    x_start = x_start_curve1 if x_start_curve1 > x_start_curve2 else x_start_curve2

    x_end = x_end_curve1 if x_end_curve1 < x_end_curve2 else x_end_curve2



    curve1 = curve1.loc[(curve1['x'] > x_start) & (curve1['x'] < x_end)]

    curve2 = curve2.loc[(curve2['x'] > x_start) & (curve2['x'] < x_end)]

    

    curve1_area = np.trapz(curve1['y'],curve1['x'])

    curve2_area = np.trapz(curve2['y'],curve2['x'])

    return np.abs(curve2_area - curve1_area)





def fit_curve(data_points):

    area_results = {}

    start = data_points['x'].min()

    end = data_points['x'].max()

    num = end

    n_range = [x for x in np.arange(0, 5, 0.05)]

    

    for n in n_range:

        with warnings.catch_warnings():

            warnings.filterwarnings('error')

            try:       

                data_points_2 = get_parabole_segment(n, start, end, num)

                

                area = diff_area(data_points, data_points_2)

                area_results[str(n)] = str(area)

            except Warning:

                pass      

    area_results_df = pd.DataFrame.from_dict(area_results, orient='index').reset_index()

    area_results_df = area_results_df.astype(float)

    area_results_df.columns = ['n', 'area']



    min_area = area_results_df['area'].min()

    min_area = area_results_df.loc[area_results_df['area'] == min_area]

    return min_area['n'].min()



def calculate_k(n, L, x_inflection):

    parabole_segment = get_parabole_segment(n, 1, x_inflection, 100) / 5

    logistic_curves = []

    area_results = {}

    x_range = [x for x in np.arange(0, 2, 0.01)]

    for k in x_range:

        with warnings.catch_warnings():

            warnings.filterwarnings('error')

            try:      

                log_exp = get_logistic_segment(L = L, a = x_inflection, k = k, start = 1, end = x_inflection, num = 100)

                log_exp = log_exp/5

                

                area = diff_area(log_exp, parabole_segment)

                logistic_curves += [(k, log_exp, area)]

                

                area_results[str(k)] = str(area)

            except Warning:

                pass    

    

    area_results_df = pd.DataFrame.from_dict(area_results, orient='index').reset_index()

    area_results_df = area_results_df.astype(float)

    area_results_df.columns = ['k', 'area']



    min_area = area_results_df['area'].min()

    min_area = area_results_df.loc[area_results_df['area'] == min_area]

    k = float(min_area['k'].min())

    return k

train_df = aggregated_cases_df[:aggregated_cases_df.shape[0]]

model = build_model(train_df['x'], train_df['y'])

predictions = get_predictions(model, np.arange(0, prediction_days))

    

fig = go.Figure()

fig.add_trace(go.Scatter(x=aggregated_cases_df['x'], y=aggregated_cases_df['y'], mode='lines+markers', name='Confirmed cases'))

fig.add_trace(go.Scatter(x=predictions['x'], y=predictions['y'], mode='lines', name='Projection'))



fig.update_layout(

    title_text="Confirmed cases vs Predictions"

)

fig.show()
steps = predictions['y']

growth_factor = np.array([x / steps[i - 1] for i, x in enumerate(steps) if i > 0])

growth_factor = np.concatenate((np.array([0]), growth_factor))

predictions['growth_factor'] = growth_factor

peaks, _ = find_peaks(predictions['growth_factor'], height=0)

'''

init_x = peaks[-1] if len(peaks) > 0 else 0

predictions = predictions[init_x:]

predictions.reset_index(drop = True, inplace = True)

'''

fig = go.Figure()

#fig.add_trace(go.Scatter(x=predictions['x'], y=predictions['y'], mode='lines', name=''))

fig.add_trace(go.Scatter(x=predictions['x'], y=predictions['growth_factor'], mode='lines', name=''))



fig.update_layout(

    title_text="Growth factor"

)

fig.show()

print(peaks)
inflection_row = predictions[predictions['growth_factor'] > 1].tail(1)

display(inflection_row)

inflection_idx = inflection_row.index[0]

inflection_x = inflection_row['x'].max()

inflection_y = inflection_row['y'].max()

L = inflection_y*2



print(f'inflection_idx: {inflection_idx}')

print(f'inflection_x: {inflection_x}')

print(f'inflection_y: {inflection_y}')

print(f'L: {L}')

pred = predictions.iloc[:inflection_idx+1]

fig = go.Figure()

fig.add_trace(go.Scatter(x=pred['x'], y=pred['y'], line=dict(dash='dash'), name = 'Projection'))



fig.update_layout(

   

)



fig.show()
predictions_inv = predictions[:inflection_idx + 1][::-1].copy()

predictions_b = predictions_inv.copy()



predictions_b['x'] = np.arange(inflection_idx, inflection_idx*2+1)

predictions_b['y'] = predictions_b['y'].apply(lambda y: L - y)



full_predictions = predictions[:inflection_idx]

full_predictions = full_predictions.append(predictions_b)



aggregated_cases_df.loc[:,'date'] = pd.date_range(start='3/6/2020', periods=aggregated_cases_df.shape[0])

full_predictions.loc[:,'date'] = pd.date_range(start='3/6/2020', periods=full_predictions.shape[0])



fig = go.Figure()

fig.add_trace(go.Scatter(x=aggregated_cases_df['date'], y=aggregated_cases_df['y'], mode='lines+markers', line=dict(width=1), name='Confirmed cases'))

fig.add_trace(go.Scatter(x=full_predictions['date'], y=full_predictions['y'], line=dict(dash='dash'), name = 'Projection'))



fig.update_layout(

    title_text="Confirmed cases vs Predictions"

)



fig.show()