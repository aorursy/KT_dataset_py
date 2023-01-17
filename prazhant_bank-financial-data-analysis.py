# First, we'll import pandas, a data processing and CSV file I/O library

import pandas as pd



# We'll also import seaborn, a Python graphing library

import warnings # current version of seaborn generates a bunch of warnings that we'll ignore

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)

import plotly.express as px
# importing raw data

# ../input/fabfinancialdata/FAB_financial_data.xls

quar_data = pd.read_excel('../input/FAB_financial_data.xls', sheet_name='FAB Quarterly Financial Data')

annual_data = pd.read_excel('../input/FAB_financial_data.xls', sheet_name='FAB Annual Financial Data')
# visualizing top 5 observations

quar_data.head()
# dropping the unit column

quar_data.drop('Unit', axis=1, inplace=True)

# Transposing the dataframe

quar_data_T = quar_data.T

new_header = quar_data_T.iloc[0] #grab the first row for the header

quar_data_T = quar_data_T[1:] #take the data less the header row

quar_data_T.columns = new_header #set the header row as the df header

quar_data_T.reset_index(inplace=True)

# Renaming Columns

quar_data_T.columns = ['Quarter', 'Net interest Income', 'Non-interest income',

       'Operating income', 'Operating expenses', 'Net impairment charges',

       'Net profit for the period', 'Cash and balances with central banks',

       'Due from banks and financial institutions',

       'Reverse repurchase agreements', 'Loans and advances',

       'Investments (Trading + Non-trading)', 'Investment properties',

       'Intangibles', 'Total Assets',

       'Due to banks and financial institutions', 'Repurchase Agreements',

       'Commercial paper', 'Customers’ deposits', 'Term borrowings',

       'Total liabilities', 'Share capital (net of treasury shares)',

       'Share Premium', 'Tier 1 capital notes', 'Total equity',

       'Cost to income ratio (ex-integration cost)', 'Loans to Assets',

       'Loans to Deposits', 'Capital Adequacy Ratio', 'Tier -I Ratio',

       'Leverage ratio (Assets/Equity)']
quar_data_T['Net Profit Growth Rate'] = quar_data_T['Net profit for the period'].pct_change()

quar_data_T['Operating Income Growth Rate'] = quar_data_T['Operating income'].pct_change()

quar_data_T['Customers’ deposits Growth Rate'] = quar_data_T['Customers’ deposits'].pct_change()

quar_data_T['Loan Growth Rate'] = quar_data_T['Loans and advances'].pct_change()
# Vizualising Transposed dataframe

quar_data_T.head()
import plotly.graph_objects as go



years = quar_data_T['Quarter']



fig = go.Figure()

fig.add_trace(go.Bar(x=years,

                y=quar_data_T['Net interest Income'],

                name='Net interest Income',

                marker_color='rgb(55, 83, 109)'

                ))

fig.add_trace(go.Bar(x=years,

                y=quar_data_T['Non-interest income'],

                name='Non-interest income',

#                 marker_color='rgb(26, 118, 255)'

                ))

fig.add_trace(go.Bar(x=years,

                y=quar_data_T['Operating income'],

                name='Operating income',

                marker_color='rgb(26, 118, 255)'

                ))

fig.add_trace(go.Bar(x=years,

                y=quar_data_T['Operating expenses'],

                name='Operating expenses',

#                 marker_color='rgb(55, 83, 109)'

                ))

fig.add_trace(go.Bar(x=years,

                y=quar_data_T['Net impairment charges'],

                name='Net impairment charges',

#                 marker_color='rgb(26, 118, 255)'

                ))



fig.update_layout(

    title='Income and Expenses',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title="AED '000",

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.15, # gap between bars of adjacent location coordinates.

    bargroupgap=0.1 # gap between bars of the same location coordinate.

)

fig.show()
# Income Breakdown For the latest Quarter Q1/20

labels = ['Net interest Income', 'Non-interest income']

values = [3061,1505]



fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.update_layout(

    title='Income Breakdown For the latest Quarter Q1/20')

fig.show()
years = quar_data_T['Quarter']

# Create traces

fig = go.Figure()

fig.add_trace(go.Scatter(x=years, y=quar_data_T['Net profit for the period'],

                    mode='lines+markers',

                         marker_color='rgb(55, 83, 109)',

                    name='Net profit for the period'))

fig.add_trace(go.Scatter(x=years, y=quar_data_T['Total Assets'],

                    mode='lines+markers',

                    name='Total Assets'))

fig.add_trace(go.Scatter(x=years, y=quar_data_T['Total liabilities'],

                    mode='lines+markers',

                    name='Total liabilities'))



fig.update_layout(

    title='Net Profit, Total Assets & Total liabilities',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title="AED '000",

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    )

)



fig.show()
years = quar_data_T['Quarter']

# Create traces

fig = go.Figure()

fig.add_trace(go.Scatter(x=years, y=quar_data_T['Customers’ deposits'],

                    mode='lines+markers',

                         marker_color='rgb(55, 83, 109)',

                    name='Customers’ deposits'))



fig.update_layout(

    title='Customers’ deposits',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title="Customers’ deposits",

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    )

)



fig.show()
years = quar_data_T['Quarter']

# Create traces

fig = go.Figure()

fig.add_trace(go.Scatter(x=years, y=quar_data_T['Loans and advances'],

                    mode='lines+markers',

                    marker_color='rgb(55, 83, 109)',

                    name='Loans and advances'))



fig.update_layout(

    title='Loans and advances',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title="Loans and advances",

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    )

)



fig.show()
import plotly.graph_objects as go



years = quar_data_T['Quarter']



fig = go.Figure()

fig.add_trace(go.Bar(x=years,

                y=quar_data_T['Cash and balances with central banks'],

                name='Cash and balances with central banks',

#                 marker_color='rgb(55, 83, 109)'

                ))

fig.add_trace(go.Bar(x=years,

                y=quar_data_T['Reverse repurchase agreements'],

                name='Reverse repurchase agreements',

#                 marker_color='rgb(26, 118, 255)'

                ))

fig.add_trace(go.Bar(x=years,

                y=quar_data_T['Loans and advances'],

                name='Loans and advances',

                marker_color='rgb(55, 83, 109)'

                ))

fig.add_trace(go.Bar(x=years,

                y=quar_data_T['Due from banks and financial institutions'],

                name='Due from banks and financial institutions',

#                 marker_color='rgb(55, 83, 109)'

                ))

fig.add_trace(go.Bar(x=years,

                y=quar_data_T['Investments (Trading + Non-trading)'],

                name='Investments (Trading + Non-trading)',

#                 marker_color='rgb(26, 118, 255)'

                ))

fig.add_trace(go.Bar(x=years,

                y=quar_data_T['Investment properties'],

                name='Investment properties',

#                 marker_color='rgb(26, 118, 255)'

                ))

fig.add_trace(go.Bar(x=years,

                y=quar_data_T['Intangibles'],

                name='Intangibles',

#                 marker_color='rgb(26, 118, 255)'

                ))



fig.update_layout(

    title='Asset Statement',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title="AED '000",

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.15, # gap between bars of adjacent location coordinates.

    bargroupgap=0.1 # gap between bars of the same location coordinate.

)

fig.show()
years = quar_data_T['Quarter']



fig = go.Figure(data=[

    go.Bar(name='Cash and balances with central banks', x=years, y=quar_data_T['Cash and balances with central banks']),

    go.Bar(name='Reverse repurchase agreements', x=years, y=quar_data_T['Reverse repurchase agreements']),

    go.Bar(name='Loans and advances', x=years, y=quar_data_T['Loans and advances']),

    go.Bar(name='Due from banks and financial institutions', x=years, y=quar_data_T['Due from banks and financial institutions']),

    go.Bar(name='Investments (Trading + Non-trading)', x=years, y=quar_data_T['Investments (Trading + Non-trading)']),

    go.Bar(name='Investment properties', x=years, y=quar_data_T['Investment properties']),

    go.Bar(name='Intangibles', x=years, y=quar_data_T['Intangibles'])

])

# Change the bar mode

fig.update_layout(

    title='Asset Statement (Stacked Representation)',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title="AED '000",

        titlefont_size=16,

        tickfont_size=14,

    ),

    barmode='stack',

    bargap=0.15, # gap between bars of adjacent location coordinates.

    bargroupgap=0.1 # gap between bars of the same location coordinate.

)

fig.show()


labels = ['Cash and balances with central banks',

       'Due from banks and financial institutions',

       'Reverse repurchase agreements', 'Loans and advances',

       'Investments (Trading + Non-trading)', 'Investment properties',

       'Intangibles']

values = [176,35,24,382,133,9,19]



fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.update_layout(

    title='Asset Breakdown For the latest Quarter Q1/20')

fig.show()
import plotly.graph_objects as go



years = quar_data_T['Quarter']



fig = go.Figure()

fig.add_trace(go.Bar(x=years,

                y=quar_data_T['Due to banks and financial institutions'],

                name='Due to banks and financial institutions',

#                 marker_color='rgb(55, 83, 109)'

                ))

fig.add_trace(go.Bar(x=years,

                y=quar_data_T['Repurchase Agreements'],

                name='Repurchase Agreements',

#                 marker_color='rgb(26, 118, 255)'

                ))

fig.add_trace(go.Bar(x=years,

                y=quar_data_T['Commercial paper'],

                name='Commercial paper',

#                 marker_color='rgb(26, 118, 255)'

                ))

fig.add_trace(go.Bar(x=years,

                y=quar_data_T['Customers’ deposits'],

                name='Customers’ deposits',

                marker_color='rgb(55, 83, 109)'

                ))

fig.add_trace(go.Bar(x=years,

                y=quar_data_T['Term borrowings'],

                name='Term borrowings',

#                 marker_color='rgb(26, 118, 255)'

                ))

fig.add_trace(go.Bar(x=years,

                y=quar_data_T['Share capital (net of treasury shares)'],

                name='Share capital (net of treasury shares)',

#                 marker_color='rgb(26, 118, 255)'

                ))

fig.add_trace(go.Bar(x=years,

                y=quar_data_T['Share Premium'],

                name='Share Premium',

#                 marker_color='rgb(26, 118, 255)'

                ))

fig.add_trace(go.Bar(x=years,

                y=quar_data_T['Tier 1 capital notes'],

                name='Tier 1 capital notes',

#                 marker_color='rgb(26, 118, 255)'

                ))



fig.update_layout(

    title='Liabilities Statement',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title="AED '000",

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.15, # gap between bars of adjacent location coordinates.

    bargroupgap=0.1 # gap between bars of the same location coordinate.

)

fig.show()


labels = ['Due to banks and financial institutions', 'Repurchase Agreements',

       'Commercial paper', 'Customers’ deposits', 'Term borrowings',

        'Share capital (net of treasury shares)',

       'Share Premium', 'Tier 1 capital notes']

values = [60,48,17,497,57,11,53,11]



fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.update_layout(

    title='Liabilities Breakdown For the latest Quarter Q1/20')

fig.show()
years = quar_data_T['Quarter']

# Create traces

fig = go.Figure()

fig.add_trace(go.Scatter(x=years, y=quar_data_T['Cost to income ratio (ex-integration cost)'],

                    mode='lines+markers',

                    name='Cost to income ratio (ex-integration cost)'))



fig.update_layout(

    title='Efficiency',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title="Cost to income ratio (ex-integration cost)",

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    )

)



fig.show()
import plotly.graph_objects as go



years = quar_data_T['Quarter']



fig = go.Figure()

fig.add_trace(go.Bar(x=years,

                y=quar_data_T['Capital Adequacy Ratio'],

                name='Capital Adequacy Ratio',

                marker_color='rgb(55, 83, 109)'

                ))

fig.add_trace(go.Scatter(x=years, y=quar_data_T['Capital Adequacy Ratio'],

                    mode='lines+markers',

                    name='Capital Adequacy Ratio'))



fig.update_layout(

    title='Capital Adequacy Ratio',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title="Capital Adequacy Ratio",

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.15, # gap between bars of adjacent location coordinates.

    bargroupgap=0.1 # gap between bars of the same location coordinate.

)

fig.show()
years = quar_data_T['Quarter']

# Create traces

fig = go.Figure()

fig.add_trace(go.Scatter(x=years, y=quar_data_T['Loans to Assets'],

                    mode='lines+markers',

                    name='Loans to Assets'))

fig.add_trace(go.Scatter(x=years, y=quar_data_T['Loans to Deposits'],

                    mode='lines+markers',

                    marker_color='rgb(55, 83, 109)',

                    name='Loans to Deposits'))



fig.update_layout(

    title='Liquidity',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title="%",

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    )

)



fig.show()
years = quar_data_T['Quarter']

# Create traces

fig = go.Figure()

fig.add_trace(go.Scatter(x=years, y=quar_data_T['Leverage ratio (Assets/Equity)'],

                    mode='lines+markers',

                    name='Leverage ratio (Assets/Equity)'))



fig.update_layout(

    title='Leverage ratio (Assets/Equity)',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title="%",

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    )

)



fig.show()


years = quar_data_T['Quarter']

# Create traces

fig = go.Figure()

fig.add_trace(go.Scatter(x=years, y=quar_data_T['Net Profit Growth Rate'],

                    mode='lines+markers',

#                          marker_color='rgb(55, 83, 109)',

                    name='Net Profit Growth Rate'))

fig.add_trace(go.Scatter(x=years, y=quar_data_T['Operating Income Growth Rate'],

                    mode='lines+markers',

#                          marker_color='rgb(55, 83, 109)',

                    name='Operating Income Growth Rate'))





fig.update_layout(

    title='Profit & Income Growth Rate',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title="%",

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    )

)



fig.show()


years = quar_data_T['Quarter']

# Create traces

fig = go.Figure()



fig.add_trace(go.Scatter(x=years, y=quar_data_T['Customers’ deposits Growth Rate'],

                    mode='lines+markers',

#                          marker_color='rgb(55, 83, 109)',

                    name='Customers’ deposits Growth Rate'))

fig.add_trace(go.Scatter(x=years, y=quar_data_T['Loan Growth Rate'],

                    mode='lines+markers',

#                          marker_color='rgb(55, 83, 109)',

                    name='Loans & Advances Growth Rate'))





fig.update_layout(

    title='Deposit & Loan Advances Growth Rate',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title="%",

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    )

)



fig.show()