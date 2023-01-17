%matplotlib inline



import datetime

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly

import plotly.express as px

import plotly.graph_objs as go

import plotly.offline as py

import plotly.graph_objs as go

from plotly.offline import iplot

import seaborn as sns
# Import dataset

covid_df = pd.read_csv('../input/covid19dataexploration/covid19_data.csv')
# Rename columns for ease of use 

covid_df['Date'] = pd.to_datetime(covid_df['Date'])

covid_df = covid_df.rename(columns={'Cumulative tests': 'agg_tests',

                                    'Cumulative tests per million': 'agg_tests_per_mil',

                                    'Total confirmed cases (cases)': 'agg_cases',

                                    'Confirmed cases per million (cases per million)': 'agg_cases_per_mil',

                                    'Total confirmed deaths (deaths)': 'agg_deaths',

                                    'Confirmed deaths per million (deaths per million)': 'agg_deaths_per_mil'})

covid_df[['agg_tests', 'agg_cases', 'agg_deaths']] = covid_df[['agg_tests', 'agg_cases', 'agg_deaths']].fillna(value=0)
# Calculate days since first X (can be cases, tests, death, etc.)

def days_since_first(col):

    first_dates = covid_df[covid_df[col] != 0].groupby('Entity').first()['Date']

    

    def day_diff(row):

        if row['Entity'] not in first_dates:

            return None

        return (row['Date'] - first_dates[row['Entity']]).days

    

    return covid_df.apply(day_diff, axis=1)



covid_df['days_since_1st_death'] = days_since_first('agg_deaths')

covid_df['days_since_1st_case'] = days_since_first('agg_cases')

covid_df['days_since_1st_test'] = days_since_first('agg_tests')
data=covid_df[covid_df['Entity'].isin(['Italy','United States', 'South Korea','New Zealand'])]

fig = px.line(data[data['days_since_1st_death'] >=0], x='days_since_1st_death', y='agg_deaths_per_mil', color='Entity')

fig.update_layout(title='Figure 1. Total Confirmed COVID-19 Deaths per Million Since 1st Confirmed Death',

                   xaxis_title='Days Since 1st Confirmed Death',

                   yaxis_title='Total Confirmed Deaths per Million')

fig.show()

data=covid_df[covid_df['Entity'].isin(['Italy','United States', 'South Korea','New Zealand'])]

fig = px.line(data[data['days_since_1st_case'] >=0], x='days_since_1st_case', y='agg_cases_per_mil', color='Entity')

fig.update_layout(title='Figure 2. Total Confirmed COVID-19 Cases per Million Since 1st Confirmed Case',

                   xaxis_title='Days Since 1st Confirmed Case',

                   yaxis_title='Total Confirmed Cases per Million')

fig.show()

data=covid_df[covid_df['Entity'].isin(['Italy','United States', 'South Korea','New Zealand'])]

fig = px.line(data[data['days_since_1st_test'] >=0], x='days_since_1st_test', y='agg_tests_per_mil', color='Entity')

fig.update_layout(title='Figure 3. Total COVID-19 Tests per Million Since 1st Confirmed Test',

                   xaxis_title='Days Since 1st Reported Test',

                   yaxis_title='Total Confirmed Tests per Million')

fig.show()
data=covid_df[covid_df['Entity'].isin(['Italy','United States', 'South Korea','New Zealand'])]

fig = px.line(data[data['days_since_1st_case'] >= 0], y='agg_tests_per_mil', color='Entity')

fig.update_layout(title='Figure 4. Total COVID-19 Tests per Million since 1st Confirmed Case',

                   xaxis_title='Days Since 1st Confirmed Case',

                   yaxis_title='Total Confirmed Tests per Million')

fig.show()
test_positive_ratio = covid_df['agg_cases_per_mil'].astype(float) / covid_df['agg_tests_per_mil'].astype(float)

covid_df['test_positive_ratio'] = test_positive_ratio
covid_df[covid_df['test_positive_ratio'] > 1]

def generate_max(df, col):

    return df.groupby('Entity')[col].max()



max_positive_ratio = generate_max(covid_df[covid_df['test_positive_ratio'] <= 1], 'test_positive_ratio')

max_death_per_mil = generate_max(covid_df[covid_df['test_positive_ratio'] <= 1], 'agg_deaths_per_mil')

max_cases_per_mil = generate_max(covid_df[covid_df['test_positive_ratio'] <= 1], 'agg_cases_per_mil')

all_entities = covid_df['Entity'].unique()

positive_test_ratio_vs_deaths = pd.DataFrame({'max_positive_ratio': max_positive_ratio[all_entities],

                                              'max_cases_per_mil': max_cases_per_mil[all_entities],

                                              'Entity':all_entities})



fig = px.scatter(positive_test_ratio_vs_deaths,

                 x=positive_test_ratio_vs_deaths['max_positive_ratio'],

                 y=positive_test_ratio_vs_deaths['max_cases_per_mil'],

                 color='Entity')

fig.update_layout(title='Figure 5. Maximum Test Positive Ratio (TPR) vs. Total Deaths per Million',

                   xaxis_title='Maximum Test Positive Ratio (TPR)',

                   yaxis_title='Total Cases per Million')

fig.show()
entity = covid_df[covid_df['days_since_1st_case'] > 0].groupby('Entity')

entity_trr = entity['agg_tests_per_mil'].count() / (entity['days_since_1st_test'].last() - entity['days_since_1st_test'].first())

# Let's drop entities with zero tests.

entity_trr = entity_trr[entity_trr.ne(0)].dropna()



fig = plt.figure(figsize=(15,5))

g = sns.barplot(x=entity_trr.index, y=entity_trr.values)

g.axes.axhline(1, ls='--')

plt.title('Figure 6. Test Reporting Rate by Entity')

plt.xticks(rotation=90)
def plot_against_entity(df, entity_list, x, y, title, xaxis_title, yaxis_title, horiz_line=False):

    entity_df = df[df['Entity'].isin(entity_list)][['Entity', x, y]]

    fig = px.line(entity_df, x=x, y=y, color='Entity')

    fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)

    if horiz_line:

        fig.update_layout(shapes=[dict(type='line',

                                   yref='y', y0=0, y1=0,

                                   xref='paper', x0=0, x1=1)])

    return fig



def generate_rolling_mean(df, days, mean_col):

    return df.reset_index().set_index('days_since_1st_case').groupby('Entity').rolling(days, min_periods=1)[mean_col].mean().values



covid_df.loc[covid_df['test_positive_ratio'] <= 1, 'test_positive_ratio_7_day_rolling'] = generate_rolling_mean(covid_df[covid_df['test_positive_ratio'] <= 1], 7, 'test_positive_ratio')



# covid_df.loc['test_positive_ratio'] <= 1]['test_positive_ratio_5_day_rolling'] = generate_rolling_mean(covid_df[covid_df['test_positive_ratio'] <= 1], 5, 'test_positive_ratio')



plot_against_entity(df=covid_df[(covid_df['days_since_1st_test'] >= 0) & (covid_df['test_positive_ratio'] <= 1)], 

                    entity_list=['Italy','United States', 'South Korea','New Zealand'], 

                    x='days_since_1st_test', y='test_positive_ratio_7_day_rolling',

                    title='Figure 7. 7-Day Rolling Test Positive Ratio (TPR)', 

                    xaxis_title='Days Since 1st Test', 

                    yaxis_title='7-Day Rolling Test Positive Ratio (TPR)')
# covid_df['test_positive_ratio_7_day_rolling_ROC'] = covid_df['test_positive_ratio_7_day_rolling'].pct_change()

covid_df['test_positive_ratio_ROC'] = covid_df[covid_df['test_positive_ratio'] <= 1]['test_positive_ratio'].pct_change()



# Replace infinite values with NaN

covid_df = covid_df.replace([np.inf, -np.inf], np.nan)



covid_df['test_positive_ratio_7_day_rolling_ROC'] = generate_rolling_mean(covid_df, 7, 'test_positive_ratio_ROC')



# max_tpr_5_day_rolling_roc = covid_df.groupby('Entity')['test_positive_ratio_5_day_rolling_ROC'].max()



covid_df_after_7_days = covid_df[covid_df['days_since_1st_test'] >= 7]

fig = plot_against_entity(df=covid_df_after_7_days,

                    entity_list=['Italy', 'New Zealand', 'South Korea', 'United States'],

                    x='days_since_1st_test',

                    y='test_positive_ratio_7_day_rolling_ROC',

                    title='Figure 8. 7-Day Rolling Test Positive Ratio (TPR) Rate of Change (ROC)', 

                    xaxis_title='Days Since 1st Reported Test ', 

                    yaxis_title='7-Day Rolling Test Positive Ratio (TPR) Rate of Change (ROC)',

                    horiz_line=True)

fig.show()
maxidx_tpr_7_day_rolling_roc_per_entity = covid_df_after_7_days.replace([np.inf, -np.inf], np.nan).groupby('Entity')['test_positive_ratio_7_day_rolling_ROC'].idxmax().dropna()



def find_neg_tpr_roc_after_max(row):

    if row['Entity'] not in maxidx_tpr_7_day_rolling_roc_per_entity:

        return

    if row.name < maxidx_tpr_7_day_rolling_roc_per_entity[row['Entity']]:

        return

    if row['test_positive_ratio_7_day_rolling_ROC'] < 0:

        return row['Date']

    

entity_with_completed_peak_list = []

first_neg_tpr_roc_after_max = []

for entity in covid_df_after_7_days['Entity'].unique():

    tmp = covid_df_after_7_days[covid_df_after_7_days['Entity'] == entity].apply(find_neg_tpr_roc_after_max, axis=1)

    if len(tmp.dropna().index) > 0:

        entity_with_completed_peak_list.append(entity)

        first_neg_tpr_roc_after_max.append(tmp.dropna().iloc[0])



entity_first_neg_dict = dict(zip(entity_with_completed_peak_list, first_neg_tpr_roc_after_max))



# Now using this dictionary we can find the length of the peaks for those countries that had a negative TPR ROC after the max.

entities_completed_peak_length = {}

for entity, first_neg_date in entity_first_neg_dict.items():

    tpr_roc_max_idx = maxidx_tpr_7_day_rolling_roc_per_entity[entity]

    tpr_roc_max_date = covid_df.loc[tpr_roc_max_idx, 'Date']

    length = (first_neg_date - tpr_roc_max_date).days

    entities_completed_peak_length.update({entity: length})



print(entities_completed_peak_length)
entities_peak_length_df = pd.DataFrame(entities_completed_peak_length.values(), columns=['Peak Length'], index=entities_completed_peak_length.keys())



fig = px.bar(entities_peak_length_df, x=entities_peak_length_df.index, y=entities_peak_length_df['Peak Length'])

fig.update_layout(title='Figure 9. Completed Peak Lengths by Entity',

                   xaxis_title='Entities',

                   yaxis_title='Peak Length')

fig.show()
average_outbreak_length = np.array([entities_completed_peak_length[entity] for entity in entities_completed_peak_length]).mean()

print('Average Outbreak Length: {} days'.format(average_outbreak_length))



entities_not_yet_through_peak = set(maxidx_tpr_7_day_rolling_roc_per_entity.index).difference(set(entity_with_completed_peak_list))

print('Entities not yet through peak: {}'.format(entities_not_yet_through_peak))
def calc_features(covid_entity):

    # Let's build a DataFrame of maximum aggregate X per million with the outbreak length in days.

    covid_entity['max_agg_deaths_per_mil'] = covid_entity['agg_deaths_per_mil'].max()

    covid_entity['max_agg_cases_per_mil'] = covid_entity['agg_cases_per_mil'].max()

    covid_entity['max_agg_tests_per_mil'] = covid_entity['agg_tests_per_mil'].max()



    # Calculate Case Fatality Rate (CFR) and find the maximum and mean.

    covid_entity['mean_cfr'] = covid_entity['cfr'].mean()

    covid_entity['max_cfr'] = covid_entity['cfr'].max()



    # Calculate mean/max Test Positive Rate

    covid_entity['mean_tpr'] = covid_entity['test_positive_ratio'].mean()

    covid_entity['max_tpr'] = covid_entity['test_positive_ratio'].max()



    # Calculate mean Test Positive Rate Rate of Change

    covid_entity['mean_tpr_roc'] = covid_entity['test_positive_ratio_ROC'].mean()

    covid_entity['max_tpr_roc'] = covid_entity['test_positive_ratio_7_day_rolling_ROC'].max()



    # Calculate Area Under the Curve of Test Positive Rate over Time

    covid_entity_prev_days = covid_entity['days_since_1st_test'].shift(periods=1)

    covid_entity_prev_tpr = covid_entity['test_positive_ratio'].shift(periods=1)

    day_diff = covid_entity['days_since_1st_test'] - covid_entity_prev_days

    tpr_sum = covid_entity['test_positive_ratio'] + covid_entity_prev_tpr

    tpr_auc = 0.5 * day_diff * tpr_sum

    covid_entity['tpr_iauc'] = tpr_auc.sum()



    return covid_entity

    

covid_df['cfr'] = covid_df['agg_deaths'] / covid_df['agg_cases']



calculated_features = covid_df.groupby('Entity').apply(calc_features) # .dropna()



feature_list = ['max_agg_deaths_per_mil', 'max_agg_cases_per_mil', 'max_agg_tests_per_mil',

                'mean_cfr', 'max_cfr',

                'mean_tpr', 'max_tpr', 'max_tpr_roc', 'mean_tpr_roc', 'tpr_iauc']



# Let's see what these features look like.

print(calculated_features[calculated_features['Entity'] == 'South Korea'][feature_list].head())
outbreak_length_df = pd.DataFrame(entities_completed_peak_length.values(), index=entities_completed_peak_length.keys(), columns=['outbreak_length_in_days'])

single_calc_features = calculated_features.groupby('Entity')[feature_list].first()

for feature in feature_list:

    outbreak_length_df[feature] = single_calc_features[feature]



    

outbreak_length_df = outbreak_length_df



# Let's look at this too.

print(outbreak_length_df.head())
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures



X = outbreak_length_df[feature_list].values

y = outbreak_length_df['outbreak_length_in_days'].values



poly_reg = PolynomialFeatures(degree=2)

X_poly = poly_reg.fit_transform(X)

        

poly_model = LinearRegression(normalize=True)

poly_model.fit(X_poly, y)



print('Poly Model Score: {}\n'.format(poly_model.score(X_poly, y)))



model = LinearRegression()

model.fit(X, y)



print('Linear Model Score: {}\n'.format(model.score(X, y)))



def predict_peak(entity):

    values = [single_calc_features.loc[entity, feature_list].values]

    return model.predict(values)[0]



def predict_peak_poly(entity):

    values = poly_reg.fit_transform([single_calc_features.loc[entity, feature_list].values])

    return poly_model.predict(values)[0]
def run_entities_through_model(poly=False):

    predicted_peak = {}

    for entity in entities_not_yet_through_peak:

        if entity not in single_calc_features.index:

            continue

        maxdate = covid_df.loc[maxidx_tpr_7_day_rolling_roc_per_entity[entity], 'Date']

        predicted_peaklength = predict_peak_poly(entity) if poly else predict_peak(entity)

        predicted_peakdate = maxdate + datetime.timedelta(days=predicted_peaklength)

        predicted_peak.update({entity: {'start': maxdate, 'peak': predicted_peakdate, 'length': predicted_peaklength}})

        

    return predicted_peak



predicted_peak_poly = run_entities_through_model(poly=True)



fig = plt.figure(figsize=(15,5))

sns.barplot(x=list(predicted_peak_poly.keys()), y=[val['length'] for val in predicted_peak_poly.values()])

plt.title('Figure 10. Predicted Peak Length in Days (Polynomial Model)')

plt.xlabel('Entity')

plt.ylabel('Peak Length (Days)')

plt.xticks(rotation=45)
predicted_peak = run_entities_through_model(poly=False)



fig = plt.figure(figsize=(15,5))

sns.barplot(x=list(predicted_peak.keys()), y=[val['length'] for val in predicted_peak.values()])

plt.title('Figure 11. Predicted Peak Length in Days (Linear Model)')

plt.xlabel('Entity')

plt.ylabel('Peak Length (Days)')

plt.xticks(rotation=45)
fig = plt.figure(figsize=(20,5))

plt.title('Figure 12. Predicted Peak Timeline')

plt.xlabel('Date')

plt.ylabel('Entity')

plt.grid(True)

for entity, item in predicted_peak.items():

    plt.plot([item['start'], item['peak']], [entity, entity], linewidth=10)
for entity, item in predicted_peak.items():

    print(entity)

    print('Start: {}, End: {}\n'.format(item['start'], item['peak']))
ihme_covid_df = pd.read_csv('/kaggle/input/ihmes-covid19-projections/2020_05_10/Hospitalization_all_locs.csv')

ihme_covid_df['date'] = pd.to_datetime(ihme_covid_df['date'])

ihme_covid_df['tpr'] = ihme_covid_df['confirmed_infections'] / ihme_covid_df['total_tests']



def days_since_first_ihme(col):

    first_dates = ihme_covid_df[ihme_covid_df[col] != 0].groupby('location_name').first()['date']

    

    def day_diff(row):

        if row['location_name'] not in first_dates:

            return None

        return (row['date'] - first_dates[row['location_name']]).days

    

    return ihme_covid_df.apply(day_diff, axis=1)



ihme_covid_df['days_since_1st_case'] = days_since_first_ihme('confirmed_infections')

ihme_covid_df['days_since_1st_test'] = days_since_first_ihme('total_tests')



def generate_rolling_mean_ihme(df, days, mean_col):

    return df.reset_index().set_index('days_since_1st_case').groupby('location_name').rolling(days, min_periods=1)[mean_col].mean().values



ihme_covid_df.loc[ihme_covid_df['tpr'] <= 1, 'tpr_7_day_rolling'] = generate_rolling_mean_ihme(ihme_covid_df[ihme_covid_df['tpr'] <= 1], 7, 'tpr')



ihme_covid_df['tpr_ROC'] = ihme_covid_df[ihme_covid_df['tpr'] <= 1]['tpr'].pct_change()



# Replace infinite values with NaN

ihme_covid_df = ihme_covid_df.replace([np.inf, -np.inf], np.nan)



ihme_covid_df['tpr_7_day_rolling_ROC'] = generate_rolling_mean_ihme(ihme_covid_df, 7, 'tpr_ROC')
ihme_covid_df_after_7_days = ihme_covid_df[ihme_covid_df['days_since_1st_test'] >= 7]

maxidx_tpr_7_day_rolling_roc_per_entity = ihme_covid_df_after_7_days.replace([np.inf, -np.inf], np.nan).groupby('location_name')['tpr_7_day_rolling_ROC'].idxmax().dropna()



def find_neg_tpr_roc_after_max_ihme(row):

    if row['location_name'] not in maxidx_tpr_7_day_rolling_roc_per_entity:

        return

    if row.name < maxidx_tpr_7_day_rolling_roc_per_entity[row['location_name']]:

        return

    if row['tpr_7_day_rolling_ROC'] < 0:

        return row['date']

    

entity_with_completed_peak_list_ihme = []

first_neg_tpr_roc_after_max_ihme = []

for entity in ihme_covid_df_after_7_days['location_name'].unique():

    tmp = ihme_covid_df_after_7_days[ihme_covid_df_after_7_days['location_name'] == entity].apply(find_neg_tpr_roc_after_max_ihme, axis=1)

    if len(tmp.dropna().index) > 0:

        entity_with_completed_peak_list_ihme.append(entity)

        first_neg_tpr_roc_after_max_ihme.append(tmp.dropna().iloc[0])



entity_first_neg_dict_ihme = dict(zip(entity_with_completed_peak_list_ihme, first_neg_tpr_roc_after_max_ihme))



# Now using this dictionary we can find the length of the peaks for those countries that had a negative TPR ROC after the max.

entities_completed_peak_length_ihme = {}

for entity, first_neg_date in entity_first_neg_dict_ihme.items():

    tpr_roc_max_idx = maxidx_tpr_7_day_rolling_roc_per_entity[entity]

    tpr_roc_max_date = ihme_covid_df.loc[tpr_roc_max_idx, 'date']

    length = (first_neg_date - tpr_roc_max_date).days

    entities_completed_peak_length_ihme.update({entity: length})



print(entities_completed_peak_length_ihme)
entity_absolute_relative_error = []

entity_ok = []

for entity, length in entities_completed_peak_length_ihme.items():

    if length <= 0:

        continue

    if entity in predicted_peak:

        print('Entity: {}'.format(entity))

        print('Predicted length of pandemic peak: {}'.format(predicted_peak[entity]['length']))

        print('Actual length of pandemic peak: {}'.format(length))

        score = abs(predicted_peak[entity]['length'] - length) / length

        print('Error: {}\n'.format(score))

        entity_absolute_relative_error.append(score)

        entity_ok.append(entity)

        

        



print('Average relative error: {}'.format(np.array(entity_absolute_relative_error).mean()))




for entity, item in predicted_peak.items():

    if entity not in entity_ok:

        continue

    entity_ihme = ihme_covid_df[ihme_covid_df['location_name'] == entity]

    entity_df = entity_ihme[['date', 'confirmed_infections']].dropna()

    fig = px.line(entity_df, x='date', y='confirmed_infections')

    fig.update_layout(title='{} Infections Per Day with Pandemic Peak Prediction Overlay'.format(entity))

    fig.add_shape(

                type="rect",

                # x-reference is assigned to the x-values

                xref="x",

                # y-reference is assigned to the plot paper [0,1]

                yref="paper",

                x0=item['start'],

                y0=0,

                x1=item['peak'],

                y1=1,

                fillcolor="LightSalmon",

                opacity=0.5,

                layer="below",

                line_width=0,

            )

    fig.show()