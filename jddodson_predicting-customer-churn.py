import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



customers_dataset = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
# Now we can look at the customer data and determine what data would be useful to look at. 

customers_dataset.head()
customers_dataset.dtypes
customers_dataset['TotalCharges'] = pd.to_numeric(customers_dataset['TotalCharges'], errors='coerce')
customers_dataset.dtypes
customers_dataset.isnull().sum()
# first I'll store the number rows in the dataset in a variable

total_rows = len(customers_dataset.index)



# next I'll get the percentage of the 11 missing rows from the entire set 

percentage = (11 / total_rows) * 100

print('{}%'.format(round(percentage, 2))) # prints the percentage rounded to 2 decimal points
customers_dataset.dropna(inplace=True)

customers_dataset.isnull().sum()
customers_dataset.iloc[0]
# create a new dataframe by selecting all the rows and all columns starting form column 1 - ignoring the customerID column

cust_df = customers_dataset.iloc[:, 1:]
cust_df.iloc[0]
cust_df['Churn'].replace(to_replace='Yes', value=1, inplace=True)

cust_df['Churn'].replace(to_replace='No', value=0, inplace=True)
cust_df.head()['Churn']
customer_dummies = pd.get_dummies(cust_df)
customer_dummies.head()
import plotly.graph_objects as go



churn_yes = (customer_dummies['Churn'] == 1).sum()

churn_no = (customer_dummies['Churn'] == 0).sum()



labels = ['Churn Rate', 'Retention Rate']

values = [churn_yes, churn_no]



fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5)])

fig.update_layout(

    title='Customer Churn Rate',

    

)

fig.show()
churn_correlation = customer_dummies.corr()['Churn'].sort_values(ascending=False)

fig = go.Figure(go.Bar(x=churn_correlation.keys(), 

                       y=churn_correlation, 

                       name='Churn Correlation',

                       marker={'color': churn_correlation, 'colorscale': 'Viridis'}))



fig.update_layout(yaxis={'categoryorder':'category ascending'})

fig.update_layout(

    title='Correlation to Churn',

    

    xaxis_tickfont_size=10,

    xaxis_tickangle=-45,

    yaxis=dict(

        title='Correlation',

        titlefont_size=16,

        tickfont_size=14,

    ),

    bargap=0.15, # gap between bars of adjacent location coordinates.

)

fig.show()
# simply calling customers_dataset.Contract.unique() will return a list of each unique value

# But I print each value with a for-loop to show you how you may can use this method to manipulate data

# within a for-loop

for contract in customers_dataset.Contract.unique():

    print(contract)
contract_amounts = []

contracts = customers_dataset.Contract.unique()



contract_amounts.append((customers_dataset['Contract'] == 'Month-to-month').sum())

contract_amounts.append((customers_dataset['Contract'] == 'One year').sum())

contract_amounts.append((customers_dataset['Contract'] == 'Two year').sum())



fig = go.Figure([go.Bar(x=contracts, y=contract_amounts)])

fig.update_layout(

    title_text = 'Customer Contracts',

    

)

fig.show()
contract_types = (customers_dataset['Contract'].value_counts(normalize=True) * 100).keys().tolist()

contract_totals = (customers_dataset['Contract'].value_counts(normalize=True) * 100).values.tolist()



# round the values up 

for i in range(len(contract_totals)):

    contract_totals[i] = round(contract_totals[i])



# create a list of values with the % sign added for the bar labels

bar_labels = ['{} %'.format(x) for x in contract_totals] 



fig = go.Figure([

    go.Bar(x=contract_types, 

           y=contract_totals,

           text = bar_labels,

           textposition = 'auto')])

fig.update_layout(

    title_text = 'Customer Contracts %',

    

)

fig.show()
# first we need to get the percentage of churn for each contract type

month_to_month = customers_dataset.loc[customers_dataset['Contract'] == 'Month-to-month']

m2m_churn = int(round((month_to_month['Churn'].value_counts(normalize=True) * 100)['Yes']))



one_year = customers_dataset.loc[customers_dataset['Contract'] == 'One year']

one_churn = int(round((one_year['Churn'].value_counts(normalize=True) * 100)['Yes']))



two_year = customers_dataset.loc[customers_dataset['Contract'] == 'Two year']

two_churn = int(round((two_year['Churn'].value_counts(normalize=True) * 100)['Yes']))





# Next we need to create some lists to keep our data

contract_types = (customers_dataset['Contract'].value_counts(normalize=True) * 100).keys().tolist()

contract_totals = (customers_dataset['Contract'].value_counts(normalize=True) * 100).values.tolist()

churn_rates = [m2m_churn, one_churn, two_churn]

retention_rates = [100 - m2m_churn, 100 - one_churn, 100 - two_churn]



# round the values up 

for i in range(len(contract_totals)):

    contract_totals[i] = round(contract_totals[i])

    

# create labels for the different bar types

total_labels = ['{} %'.format(x) for x in contract_totals]

churn_labels = ['{} %'.format(x) for x in churn_rates]

retention_labels = ['{} %'.format(x) for x in retention_rates]







fig = go.Figure(data=[

#     go.Bar(name='Contract Percentage', x=contract_types, y=contract_totals, text=total_labels,

#             textposition='auto',),

    go.Bar(name='Churn Rate', x=['Month-to-Month', 'One Year', 'Two Year'], y=churn_rates, text=churn_labels,

            textposition='auto'),

    go.Bar(name='Retention Rate', x=['Month-to-Month', 'One Year', 'Two Year'], y=retention_rates, text=retention_labels,

            textposition='auto')

])

# Change the bar mode

fig.update_layout(barmode='stack', title_text = 'Churn Rates By Contract Type')

fig.show()
# First get all of the churn data and create a series of the contract type percentages

churn = customers_dataset.loc[customers_dataset['Churn'] == 'Yes']

churn_series = (churn['Contract']).value_counts(normalize=True) * 100



# build graph lists of contracts and values

contract_types = churn_series.keys().tolist()

percentages = churn_series.values.tolist()



# round the percentages

for i in range(len(percentages)):

    percentages[i] = round(percentages[i], 1)



fig = go.Figure(data=[go.Pie(

    labels=contract_types, 

    values=percentages, 

    hole=.3)])



fig.update_layout(

    title_text = 'Contract Types of Churned Customers'

)

fig.show()



# Build a chart that shows the the average monthly cost per contract type



import plotly.express as px

customers = customers_dataset

fig = px.histogram(customers, x="Churn", y="MonthlyCharges", histfunc="avg", color='Churn',facet_col="Contract")

fig.update_layout(title_text='Average Monthly Cost by Contract Type')

fig.show()

# Create dataframes of each contract type to perform operations on

m2m = customers_dataset.loc[customers_dataset['Contract'] == 'Month-to-month']

oneYear = customers_dataset.loc[customers_dataset['Contract'] == 'One year']

twoYear = customers_dataset.loc[customers_dataset['Contract'] == 'Two year']





fig = go.Figure()



fig.add_trace(

    go.Indicator(

        mode = "number+gauge+delta", 

        value = twoYear.loc[twoYear['Churn'] == 'Yes']['MonthlyCharges'].mean(),

        domain = {'x': [0.25, 1], 'y': [0.7, 0.9]}, 

        title = {'text' :"<b>Two Year</b>"},

        delta = {'reference': twoYear['MonthlyCharges'].mean()}, 

        gauge = {

            'shape': "bullet", 

            'axis': {'range': [ twoYear['MonthlyCharges'].min(),  twoYear['MonthlyCharges'].max()]}, 

            'bar': {'color': "#7BE0AD"},

            'threshold': {

                'line': {'color': "black", 'width': 3}, 

                'thickness': 0.75, 

                'value': twoYear['MonthlyCharges'].mean()},

            'steps': [

                {'range': [ twoYear['MonthlyCharges'].min(), twoYear.loc[twoYear['Churn'] == 'No']['MonthlyCharges'].mean()], 'color': "#064789"}, 

                {'range': [ twoYear.loc[twoYear['Churn'] == 'No']['MonthlyCharges'].mean(),  twoYear['MonthlyCharges'].max()], 'color': "#EBF2FA"}]}))



fig.add_trace(

    go.Indicator(

        mode = "number+gauge+delta", 

        value = oneYear.loc[oneYear['Churn'] == 'Yes']['MonthlyCharges'].mean(),

        domain = {'x': [0.25, 1], 'y': [0.4, 0.6]}, 

        title = {'text' :"<b>One Year</b>"},

        delta = {'reference': oneYear['MonthlyCharges'].mean()}, 

        gauge = {

            'shape': "bullet", 

            'axis': {'range': [ oneYear['MonthlyCharges'].min(),  oneYear['MonthlyCharges'].max()]}, 

            'bar': {'color': "#7BE0AD"},

            'threshold': {

                'line': {'color': "black", 'width': 3}, 

                'thickness': 0.75, 

                'value': oneYear['MonthlyCharges'].mean()},

            'steps': [

                {'range': [ oneYear['MonthlyCharges'].min(), oneYear.loc[oneYear['Churn'] == 'No']['MonthlyCharges'].mean()], 'color': "#064789"}, 

                {'range': [ oneYear.loc[oneYear['Churn'] == 'No']['MonthlyCharges'].mean(),  oneYear['MonthlyCharges'].max()], 'color': "#EBF2FA"}]}))







fig.add_trace(go.Indicator(

    mode = "number+gauge+delta", 

    value = m2m.loc[m2m['Churn'] == 'Yes']['MonthlyCharges'].mean(),

    domain = {'x': [0.25, 1], 'y': [0.08, 0.25]}, 

    title = {'text' :"<b>Monthly</b>"},

    delta = {'reference': m2m['MonthlyCharges'].mean()}, 

    gauge = {

        'shape': "bullet", 

        'axis': {'range': [ m2m['MonthlyCharges'].min(),  m2m['MonthlyCharges'].max()]}, 

        'bar': {'color': "#7BE0AD"},

        'threshold': {

            'line': {'color': "black", 'width': 3}, 

            'thickness': 0.75, 

            'value': m2m['MonthlyCharges'].mean()},

        'steps': [

            {'range': [ m2m['MonthlyCharges'].min(), m2m.loc[m2m['Churn'] == 'No']['MonthlyCharges'].mean()], 'color': "#064789"}, 

            {'range': [ m2m.loc[m2m['Churn'] == 'No']['MonthlyCharges'].mean(),  m2m['MonthlyCharges'].max()], 'color': "#EBF2FA"}]}

))



fig.update_layout(title_text = 'Average Monthly Charges by Contract Agreement', height = 500, margin = {'t':50, 'b':10, 'l':10}, template='plotly_dark')

fig.show()