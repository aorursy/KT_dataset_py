import pandas as pd

import numpy as np

import plotly as py

import plotly.graph_objs as go

from ipywidgets import widgets



import plotly.io as pio

pio.renderers.default = 'kaggle'



import warnings

warnings.filterwarnings('ignore')
input_filder = '../input/ieee-fraud-detection'

train_transaction = pd.read_csv(f'{input_filder}/train_transaction.csv')

train_identity = pd.read_csv(f'{input_filder}/train_identity.csv')

test_transaction = pd.read_csv(f'{input_filder}/test_transaction.csv')

test_identity = pd.read_csv(f'{input_filder}/test_identity.csv')

train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')

test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')



print('train_transaction shape: {}'.format(train_transaction.shape))

print('test_transaction shape: {}'.format(test_transaction.shape))

print('train_identity shape: {}'.format(train_identity.shape))

print('test_identity shape: {}'.format(test_identity.shape))

print('train shape: {}'.format(train.shape))

print('test shape: {}'.format(test.shape))



del train_transaction, train_identity, test_transaction, test_identity
c_cols = [c for c in train.columns if c.startswith('C')]

card_cols = [c for c in train.columns if c.startswith('card')]

id_cols = [c for c in train.columns if c.startswith('id_')]

d_cols = [c for c in train.columns if c.startswith('D') and len(c) < 5]

m_cols = [c for c in train.columns if c.startswith('M')]

v_cols = [c for c in train.columns if c.startswith('V')]

main_cols = [c for c in train.columns if c not in c_cols + id_cols + d_cols + m_cols + v_cols] 



train[main_cols].head(14).T
trace=go.Pie(

    labels=['NoN Fraud', 'Fraud'],

    values=train['isFraud'].value_counts(),

    marker = dict(colors = ['#1499c7',' #f5b041']),

    textinfo='value+percent',

    pull=.03

)



py.offline.iplot([trace])
Train_NaN_percent_cells = int(train.isnull().sum().sum() / (train.shape[0] * train.shape[1]) * 100)

Test_NaN_percent_cells = int(test.isnull().sum().sum() / (test.shape[0] * test.shape[1]) * 100)



data = [

    go.Bar(

        y=['Train', 'Test'],

        x=[Train_NaN_percent_cells, Test_NaN_percent_cells],

        type = 'bar',

        name = 'Null',

        orientation='h'

    ),

    go.Bar(

        y=['Train', 'Test'],

        x=[100 - Train_NaN_percent_cells, 100 - Test_NaN_percent_cells],

        type = 'bar',

        name = 'Not Null',

        orientation='h'

    )

]



layout = {

    'barmode': 'relative',

    'title': 'Percenteg Null cells',

    'xaxis_title_text': 'Percenteg',

    'height': 300

}



fig = go.Figure(

    data=data,

    layout = layout

)



py.offline.iplot(fig)
Train_NaN_cols_count = (train.isna().sum() > 0).sum()

Test_NaN_cols_count = (test.isna().sum() > 0).sum()

Train_cols_count = len(train.columns)

Test_cols_count = len(test.columns)



data = [

    go.Bar(

        y=['Train', 'Test'],

        x=[Train_NaN_cols_count, Test_NaN_cols_count],

        type = 'bar',

        name = 'Null',

        orientation='h'

    ),

    go.Bar(

        y=['Train', 'Test'],

        x=[Train_cols_count - Train_NaN_cols_count - 1, Test_cols_count - Test_NaN_cols_count],

        type = 'bar',

        name = 'Not Null',

        orientation='h'

    )

]



layout = {

    'barmode': 'relative',

    'title': 'Null columns',

    'xaxis_title_text': 'Count',

    'height': 300

}



fig = go.Figure(

    data=data,

    layout = layout

)



py.offline.iplot(fig)
train_null_hist = (train.isnull().sum() / train.shape[0] * 100).astype(int)

test_null_hist = (test.isnull().sum() / test.shape[0] * 100).astype(int)



data = [

    go.Histogram(

        x=train_null_hist, 

        nbinsx=25,

        name = 'train',

        marker_color='#EB89B5'

    ),

    go.Histogram(

        x=test_null_hist, 

        nbinsx=25,

        name = 'test',

        marker_color='#330C73'

    )

]



layout = {

    'title':'Percentage of Null values in columns',

    'xaxis_title_text':'Percenteg NaN values',

    'yaxis_title_text':'Columns count'

}



fig = go.Figure(

    data=data,

    layout = layout    

)



py.offline.iplot(fig)
data = [

    go.Histogram(

        x=train['TransactionDT'], 

        nbinsx=100,

        name = 'train',

        marker_color='#EB89B5'

    ),

    go.Histogram(

        x=test['TransactionDT'], 

        nbinsx=100,

        name = 'test',

        marker_color='#330C73'

    )

]



layout = {

    'title':'Train/Test distribution of TransactionDT',

    'xaxis_title_text':'TransactionDT',

    'yaxis_title_text':'Count'

}



fig = go.Figure(

    data=data,

    layout = layout    

)



py.offline.iplot(fig)
uniq_df = pd.DataFrame(columns=['train','test','max_values','equal_values','test_in_train'])

uniq_df = uniq_df.astype({'train':'int','test':'int','max_values':'int','equal_values':'bool','test_in_train':'bool'})



for c in test.columns:

    train_unique = train[c].unique()

    test_unique = test[c].unique()

    max_values = len(pd.Series(list(train_unique) + list(test_unique)).unique())

    if max_values < 30:

        row = pd.Series({

            'train': len(train_unique),

            'test': len(test_unique),

            'max_values': max_values,

            'equal_values': np.array_equal(np.sort(train[c].dropna().unique()), np.sort(test[c].dropna().unique())),

            'test_in_train': all([i in train[c].dropna().unique() for i in test[c].dropna().unique()])

        }, name = c)

        uniq_df = uniq_df.append(row)



fig = go.Figure(

    data = go.Bar(

        x=uniq_df.query('test_in_train == True').sort_values(by='max_values').index,

        y=uniq_df.query('test_in_train == True').sort_values(by='max_values')['max_values'],

        marker_color='#5b2c6f'

    ),

    layout = {

        'title': 'Categorical features?',

        'xaxis_title_text': 'Columns',

        'yaxis_title_text': 'Unique values count',

        'width': 2300

    }

)



py.offline.iplot(fig)
cols = uniq_df.query('test_in_train == True').sort_values(by='max_values').index

id_cols = [c for c in cols if c.startswith('id_')]

m_cols = [c for c in cols if c.startswith('M')]

v_cols = [c for c in cols if c.startswith('V')]

o_cols = [c for c in cols if c not in id_cols + m_cols + v_cols]



def cat_plot(columns, plot_name):

    mask = [False] * len(columns)

    mask = mask + mask



    fraud_lbl = {0:'Non Fraud',1:'Fraud'}



    traces = []

    buttons = [{

        "args": ["visible", mask], 

        "label": 'Column', 

        "method": "restyle"

    }]



    for i, col in enumerate(columns):

        for f in [0,1]:

            query = train.query(f'isFraud == {f}')[col].fillna('NaN').value_counts(dropna=False)

            trace = go.Bar(

                x=query.index.tolist(),

                y=query.tolist(),

                orientation='v', 

                name=fraud_lbl[f],

                visible=False

            )

            traces.append(trace)



        mask_temp = mask.copy()

        mask_temp[i*2] = True

        mask_temp[i*2+1] = True

        button = {

            "args": ["visible", mask_temp], 

            "label": col, 

            "method": "restyle"

        }

        buttons.append(button)





    layout = {

        "title": f"Fraud by Categorical features ({plot_name})",

        'xaxis_type':'category',

        "updatemenus": [{

            "buttons": buttons,

            "yanchor": "top",

            "y": 1.12,

            "x": 0.085

        }]

    }



    fig = go.Figure(data=traces,layout=layout)

    fig.show()
cat_plot(id_cols, 'id cols')
cat_plot(m_cols, 'M cols')
cat_plot(v_cols, 'V cols')
cat_plot(o_cols, 'other cols')