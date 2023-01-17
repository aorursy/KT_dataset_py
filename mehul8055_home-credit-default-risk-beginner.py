import numpy as np

import pandas as pd



import plotly.offline as py

import plotly.graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff

from plotly.subplots import make_subplots

import matplotlib.pyplot as plt

import seaborn as sns



import gc

import warnings

import time

warnings.filterwarnings("ignore")



%matplotlib inline
app = pd.read_csv('../input/home-credit-default-risk/application_train.csv')

app_test = pd.read_csv('../input/home-credit-default-risk/application_test.csv')



bureau = pd.read_csv('../input/home-credit-default-risk/bureau.csv')

bureau_bal = pd.read_csv('../input/home-credit-default-risk/bureau_balance.csv')



credit_card_bal = pd.read_csv('../input/home-credit-default-risk/credit_card_balance.csv')

pos_cash_bal = pd.read_csv('../input/home-credit-default-risk/POS_CASH_balance.csv')



previous_app = pd.read_csv('../input/home-credit-default-risk/previous_application.csv')

installment_pay = pd.read_csv('../input/home-credit-default-risk/installments_payments.csv')
name_list = ['application_train.csv', 'application_test.csv', 'bureau.csv', 'bureau_balance.csv',

             'credit_card_balance.csv','POS_CASH_balance.csv','previous_application.csv','installments_payments.csv']

row_list = [app.shape[0], app_test.shape[0], bureau.shape[0], bureau_bal.shape[0],credit_card_bal.shape[0],

            pos_cash_bal.shape[0], previous_app.shape[0], installment_pay.shape[0]]

column_list = [app.shape[1], app_test.shape[1], bureau.shape[1], bureau_bal.shape[1],credit_card_bal.shape[1],

            pos_cash_bal.shape[1], previous_app.shape[1], installment_pay.shape[1]]

summary_df_column_list = ['File Name','Number of Rows', 'Number of Columns']

summary_df = pd.DataFrame(list(zip(name_list, row_list, column_list)),

                          index=[1,2,3,4,5,6,7,8],

                          columns=summary_df_column_list)

summary_df.sort_index()
def feature_tye_split(data, special_list=[]):

    categorical_list = []

    discreat_numerical_list = []

    numerical_list = []

    

    for i in data.columns.tolist():

        if data[i].dtype == 'object':

            categorical_list.append(i)

        elif data[i].nunique() < 25:

            discreat_numerical_list.append(i)

        elif i in special_list:

            discreat_numerical_list.append(i)

        else:

            numerical_list.append(i)

    return categorical_list, discreat_numerical_list, numerical_list



categorical_list,discreat_numerical_list,numerical_list = feature_tye_split(

    app,special_list=['AMT_REQ_CREDIT_BUREAU_YEAR'])
print(len(categorical_list), 'Categorical Features: ', categorical_list)

print('---------------------------------------------------------------')

print(len(discreat_numerical_list), 'Discreate Numerical Features: ', discreat_numerical_list)

print('---------------------------------------------------------------')

print(len(numerical_list), 'Numerical Features: ', numerical_list)
def plot_categorical(data, column, size=[8,4], xlabel_angle=0, title=''):

    plotdata = data[column].value_counts();

    plt.figure(figsize=size)

    sns.barplot(x=plotdata.index, y=plotdata.values)

    plt.title(title)

    plt.xticks(rotation = xlabel_angle)

    plt.show()
def plot_categorical_bar(data, column):    

    plotdata = data[column].value_counts();



    fig = go.Figure(data=[

        go.Bar(x=plotdata.index, y=plotdata.values)

    ])

    fig.show()
def plot_categorical_pie(data, column, title, hole=.3):

    plotdata = data[column].value_counts();

    

    fig = go.Figure(data=[go.Pie(labels=plotdata.index, values=plotdata.values, hole=hole)])

    fig.update_layout(title_text=title)

    fig.show()
def plot_numerical(data, column, size=[8,4], bins=50):

    plt.figure(figsize=size)

    plt.title('Distribution of %s'%column)

    sns.distplot(data[column].dropna(), kde=True, bins=bins)

    plt.show()
def plot_categorical_bylable(data, column, size=[12,6], xlabel_angle=0, title=''):

   

    label1 = data.loc[data.TARGET == 1, column].value_counts()

    label0 = data.loc[data.TARGET == 0, column].value_counts()

    

    plt.figure(figsize=size)

    plt.subplot(1,2,1)

    sns.barplot(x=label1.index, y=label1.values)

    plt.title('Default (TARGET == 1)' + title)

    plt.xticks(rotation = xlabel_angle)

    

    plt.subplot(1,2,2)

    sns.barplot(x=label0.index, y=label0.values)

    plt.title('Non Default (TARGET == 0)' + title)

    plt.xticks(rotation = xlabel_angle)

    

    plt.show()
def plot_categorical_bylabel_bar(data, column):    

    labels=data[column].unique()



    fig = go.Figure(data=[

        go.Bar(name='Non Defaults', x=labels, y=data.loc[data.TARGET == 0, column].value_counts().values),

        go.Bar(name='Defaults', x=labels, y=data.loc[data.TARGET == 1, column].value_counts().values)

    ])

    # Change the bar mode

    fig.update_layout(barmode='group')

    fig.show()
def plot_categorical_bylabel_pie(data, column):

    labels = data[column].unique()



    # Create subplots: use 'domain' type for Pie subplot

    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]],subplot_titles=['Target=0', 'Target=0'])

    fig.add_trace(go.Pie(labels=labels, values=data.loc[app.TARGET == 0, column].value_counts().values, name="Non Defaults"),

                  1, 1)

    fig.add_trace(go.Pie(labels=labels, values=data.loc[app.TARGET == 1,column].value_counts().values, name="Defaults"),

                  1, 2)



    # Use `hole` to create a donut-like pie chart

    fig.update_traces(hole=.4, hoverinfo="label+percent+name")



    fig.update_layout(

        title_text=column +" wise defaulties",

        # Add annotations in the center of the donut pies.

        annotations=[dict(text='Target=0', x=0.18, y=0.55, font_size=20, showarrow=False),

                     dict(text='Target=1', x=0.82, y=0.55, font_size=20, showarrow=False)])

    fig.show()
def plot_numerical_bylabel(data, column, size=[12,6]):

    

    plt.figure(figsize=size)

    corr = data['TARGET'].corr(data[column])

    avg_repaid = data.ix[data.TARGET == 0, column].median()

    avg_not_repaid = data.ix[data.TARGET == 1, column].median()

    

    sns.kdeplot(data.ix[data.TARGET == 0, column], label='TARGET = 0')

    sns.kdeplot(data.ix[data.TARGET == 1, column], label='TARGET = 1')



    #plt.figure(figsize=(12,6))

    plt.xlabel(column);plt.ylabel('Density');plt.title('%s Distributions'%column)

    plt.legend();

    

    print('Correlation between %s and TARGET is %0.4f'%(column, corr))

    print('Average Repaid = %f and Average Not Repaid = %f for Feature = %s'%(avg_repaid, avg_not_repaid, column))
plot_categorical(app, 'TARGET', title='Application Train : TARGET')
plot_categorical_pie(app, 'TARGET', 'Label Target ', .6)
plot_categorical_bar(app,'OCCUPATION_TYPE')
plot_categorical_pie(app, 'OCCUPATION_TYPE', 'Occupation type', .2)
plot_categorical(app,'NAME_INCOME_TYPE', size=[10,6], title='Income Type', xlabel_angle=70)
plot_categorical_pie(app, 'NAME_INCOME_TYPE', 'Income Type', .7)
plot_categorical(app,'NAME_HOUSING_TYPE', size=[10,6], title='House Type', xlabel_angle=70)
plot_numerical(app, 'AMT_CREDIT')
plot_numerical(app, 'AMT_ANNUITY')
plot_numerical(app, 'DAYS_EMPLOYED')
plot_categorical_bylable(app, 'CODE_GENDER', title='Gender' )
plot_categorical_bylabel_bar(app, 'CODE_GENDER')
plot_categorical_bylabel_pie(app, 'CODE_GENDER')
plot_categorical_bylable(app, 'NAME_EDUCATION_TYPE', title='Education Type', xlabel_angle=80 )
plot_categorical_bylabel_bar(app, 'NAME_EDUCATION_TYPE')
plot_categorical_bylabel_pie(app, 'NAME_EDUCATION_TYPE')
plot_categorical_bylable(app, 'OCCUPATION_TYPE', title='Occupation Type', xlabel_angle=80)
plot_categorical_bylabel_bar(app, 'OCCUPATION_TYPE')
plot_categorical_bylabel_pie(app, 'OCCUPATION_TYPE')
plot_numerical_bylabel(app, 'EXT_SOURCE_1')
plot_numerical_bylabel(app, 'EXT_SOURCE_2')
plot_numerical_bylabel(app, 'EXT_SOURCE_3')
corr_matrix = app.corr()

plt.figure(figsize=[15,15])

sns.heatmap(corr_matrix.values, annot=False)

plt.show()
go.Figure(data=go.Heatmap(z=corr_matrix.values))
app_dis_num = app[discreat_numerical_list + ['TARGET']]

app_dis_corr = app_dis_num.corr()

go.Figure(data=go.Heatmap(x=app_dis_corr.index, y=app_dis_corr.index, z=app_dis_corr.values))
app_num = app[numerical_list + ['TARGET']]

app_num_corr = app_num.corr()

app_num_corr.sort_values('TARGET', ascending=False, inplace=True)

go.Figure(data=go.Heatmap(x=app_num_corr.index, y=app_num_corr.index, z=app_num_corr.values))
app['annuity_income_percentage'] = app['AMT_ANNUITY'] / app['AMT_INCOME_TOTAL']

app['car_to_birth_ratio'] = app['OWN_CAR_AGE'] / app['DAYS_BIRTH']

app['car_to_employ_ratio'] = app['OWN_CAR_AGE'] / app['DAYS_EMPLOYED']

app['children_ratio'] = app['CNT_CHILDREN'] / app['CNT_FAM_MEMBERS']

app['credit_to_annuity_ratio'] = app['AMT_CREDIT'] / app['AMT_ANNUITY']

app['credit_to_goods_ratio'] = app['AMT_CREDIT'] / app['AMT_GOODS_PRICE']

app['credit_to_income_ratio'] = app['AMT_CREDIT'] / app['AMT_INCOME_TOTAL']

app['days_employed_percentage'] = app['DAYS_EMPLOYED'] / app['DAYS_BIRTH']

app['income_credit_percentage'] = app['AMT_INCOME_TOTAL'] / app['AMT_CREDIT']

app['income_per_child'] = app['AMT_INCOME_TOTAL'] / (1 + app['CNT_CHILDREN'])

app['income_per_person'] = app['AMT_INCOME_TOTAL'] / app['CNT_FAM_MEMBERS']

app['payment_rate'] = app['AMT_ANNUITY'] / app['AMT_CREDIT']

app['phone_to_birth_ratio'] = app['DAYS_LAST_PHONE_CHANGE'] / app['DAYS_BIRTH']

app['phone_to_employ_ratio'] = app['DAYS_LAST_PHONE_CHANGE'] / app['DAYS_EMPLOYED']
engineered_numerical_columns = ['annuity_income_percentage',

                                'car_to_birth_ratio',

                                'car_to_employ_ratio',

                                'children_ratio',

                                'credit_to_annuity_ratio',

                                'credit_to_goods_ratio',

                                'credit_to_income_ratio',

                                'days_employed_percentage',

                                'income_credit_percentage',

                                'income_per_child',

                                'income_per_person',

                                'payment_rate',

                                'phone_to_birth_ratio',

                                'phone_to_employ_ratio'

                                ]
app_eng = app[engineered_numerical_columns + ['TARGET']]

app_eng_corr = abs(app_eng.corr())

app_eng_corr.sort_values('TARGET', ascending=False)['TARGET']

go.Figure(data=go.Heatmap(x=app_eng_corr.index, y=app_eng_corr.index, z=app_eng_corr.values))
documents = [column for column in app.columns if column[:6] == 'FLAG_D']

app_docs = app[documents + ['TARGET']]

app_docs_corr = app_docs.corr()

app_docs_corr.sort_values('TARGET', ascending=False, inplace=True)

go.Figure(data=go.Heatmap(x=app_docs_corr.index, y=app_docs_corr.index, z=app_docs_corr.values))
app_home_columns = ['APARTMENTS_AVG',

 'BASEMENTAREA_AVG',

 'YEARS_BEGINEXPLUATATION_AVG',

 'YEARS_BUILD_AVG',

 'COMMONAREA_AVG',

 'ELEVATORS_AVG',

 'ENTRANCES_AVG',

 'FLOORSMAX_AVG',

 'FLOORSMIN_AVG',

 'LANDAREA_AVG',

 'LIVINGAPARTMENTS_AVG',

 'LIVINGAREA_AVG',

 'NONLIVINGAPARTMENTS_AVG',

 'NONLIVINGAREA_AVG',

 'APARTMENTS_MODE',

 'BASEMENTAREA_MODE',

 'YEARS_BEGINEXPLUATATION_MODE',

 'YEARS_BUILD_MODE',

 'COMMONAREA_MODE',

 'ELEVATORS_MODE',

 'ENTRANCES_MODE',

 'FLOORSMAX_MODE',

 'FLOORSMIN_MODE',

 'LANDAREA_MODE',

 'LIVINGAPARTMENTS_MODE',

 'LIVINGAREA_MODE',

 'NONLIVINGAPARTMENTS_MODE',

 'NONLIVINGAREA_MODE',

 'APARTMENTS_MEDI',

 'BASEMENTAREA_MEDI',

 'YEARS_BEGINEXPLUATATION_MEDI',

 'YEARS_BUILD_MEDI',

 'COMMONAREA_MEDI',

 'ELEVATORS_MEDI',

 'ENTRANCES_MEDI',

 'FLOORSMAX_MEDI',

 'FLOORSMIN_MEDI',

 'LANDAREA_MEDI',

 'LIVINGAPARTMENTS_MEDI',

 'LIVINGAREA_MEDI',

 'NONLIVINGAPARTMENTS_MEDI',

 'NONLIVINGAREA_MEDI']
app_home_df = app[app_home_columns + ['TARGET']]

app_home_corr = app_home_df.corr()

app_home_corr.sort_values('TARGET', ascending=False, inplace=True)

go.Figure(data=go.Heatmap(x=app_home_corr.index, y=app_home_corr.index, z=app_home_corr.values))
bureau['bureau_credit_active_binary'] = (bureau['CREDIT_ACTIVE'] != 'Closed').astype(int)

bureau['bureau_credit_enddate_binary'] = (bureau['DAYS_CREDIT_ENDDATE'] > 0).astype(int)
groupby_SK_ID_CURR = bureau.groupby(by=['SK_ID_CURR'])

features = pd.DataFrame({'SK_ID_CURR':bureau['SK_ID_CURR'].unique()})

features.head()

group_object = groupby_SK_ID_CURR['DAYS_CREDIT'].agg('count').reset_index()

group_object.rename(index=str, columns={'DAYS_CREDIT': 'bureau_number_of_past_loans'},inplace=True)

features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

features.head()
group_object = groupby_SK_ID_CURR['CREDIT_TYPE'].agg('nunique').reset_index()

group_object.rename(index=str, columns={'CREDIT_TYPE':'bureau_number_of_loan_types'}, inplace=True)

features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

features.head()
features['bureau_average_of_past_loans_per_type'] = features['bureau_number_of_past_loans'] / features['bureau_number_of_loan_types']

features.head()
group_object = groupby_SK_ID_CURR['bureau_credit_active_binary'].agg('mean').reset_index()

features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

features.head()