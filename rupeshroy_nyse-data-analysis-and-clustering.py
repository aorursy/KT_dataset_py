import sys

sys.executable
# 2.1 Display multiple outputs from a jupyter cell

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
#2.2 Import various packages required

import pandas as pd

import numpy as np

import plotly.express as px

from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

from sklearn.preprocessing import StandardScaler

from yellowbrick.cluster import SilhouetteVisualizer

import os
#2.3 Load Compan'y fundamental financial data

fundamentals = pd.read_csv("../input/nyse/fundamentals.csv",parse_dates = ['Period Ending'],index_col='Unnamed: 0')

securities =   pd.read_csv("../input/nyse/securities.csv",parse_dates = ['Date first added'])
#2.4 Rename columns, removing spaces and other symbols

renamed_columns = {'Ticker Symbol':'TickerSymbol',

'Period Ending':'PeriodEnding',

'Accounts Payable':'AccountsPayable',

'Accounts Receivable':'AccountsReceivable',

'Add\'l income/expense items':'Addlincome_expenseitems',

'After Tax ROE':'AfterTaxROE',

'Capital Expenditures':'CapitalExpenditures',

'Capital Surplus':'CapitalSurplus',

'Cash Ratio':'CashRatio',

'Cash and Cash Equivalents':'CashandCashEquivalents',

'Changes in Inventories':'ChangesinInventories',

'Common Stocks':'CommonStocks',

'Cost of Revenue':'CostofRevenue',

'Current Ratio':'CurrentRatio',

'Deferred Asset Charges':'DeferredAssetCharges',

'Deferred Liability Charges':'DeferredLiabilityCharges',

'Depreciation':'Depreciation',

'Earnings Before Interest and Tax':'EarningsBeforeInterestandTax',

'Earnings Before Tax':'EarningsBeforeTax',

'Effect of Exchange Rate':'EffectofExchangeRate',

'Equity Earnings/Loss Unconsolidated Subsidiary':'EquityEarnings_LossUnconsolidatedSubsidiary',

'Fixed Assets':'FixedAssets',

'Goodwill':'Goodwill',

'Gross Margin':'GrossMargin',

'Gross Profit':'GrossProfit',

'Income Tax':'IncomeTax',

'Intangible Assets':'IntangibleAssets',

'Interest Expense':'InterestExpense',

'Inventory':'Inventory',

'Investments':'Investments',

'Liabilities':'Liabilities',

'Long-Term Debt':'Long-TermDebt',

'Long-Term Investments':'Long-TermInvestments',

'Minority Interest':'MinorityInterest',

'Misc. Stocks':'MiscStocks',

'Net Borrowings':'NetBorrowings',

'Net Cash Flow':'NetCashFlow',

'Net Cash Flow-Operating':'NetCashFlow-Operating',

'Net Cash Flows-Financing':'NetCashFlows-Financing',

'Net Cash Flows-Investing':'NetCashFlows-Investing',

'Net Income':'NetIncome',

'Net Income Adjustments':'NetIncomeAdjustments',

'Net Income Applicable to Common Shareholders':'NetIncomeApplicabletoCommonShareholders',

'Net Income-Cont. Operations':'NetIncome-ContOperations',

'Net Receivables':'NetReceivables',

'Non-Recurring Items':'Non-RecurringItems',

'Operating Income':'OperatingIncome',

'Operating Margin':'OperatingMargin',

'Other Assets':'OtherAssets',

'Other Current Assets':'OtherCurrentAssets',

'Other Current Liabilities':'OtherCurrentLiabilities',

'Other Equity':'OtherEquity',

'Other Financing Activities':'OtherFinancingActivities',

'Other Investing Activities':'OtherInvestingActivities',

'Other Liabilities':'OtherLiabilities',

'Other Operating Activities':'OtherOperatingActivities',

'Other Operating Items':'OtherOperatingItems',

'Pre-Tax Margin':'Pre-TaxMargin',

'Pre-Tax ROE':'Pre-TaxROE',

'Profit Margin':'ProfitMargin',

'Quick Ratio':'QuickRatio',

'Research and Development':'ResearchandDevelopment',

'Retained Earnings':'RetainedEarnings',

'Sale and Purchase of Stock':'SaleandPurchaseofStock',

'Sales, General and Admin.':'SalesGeneralandAdmin.',

'Short-Term Debt / Current Portion of Long-Term Debt':'Short-TermDebt_CurrentPortionofLong-TermDebt',

'Short-Term Investments':'Short-TermInvestments',

'Total Assets':'TotalAssets',

'Total Current Assets':'TotalCurrentAssets',

'Total Current Liabilities':'TotalCurrentLiabilities',

'Total Equity':'TotalEquity',

'Total Liabilities':'TotalLiabilities',

'Total Liabilities & Equity':'TotalLiabilities&Equity',

'Total Revenue':'TotalRevenue',

'Treasury Stock':'TreasuryStock',

'For Year':'ForYear',

'Earnings Per Share':'EarningsPerShare',

'Estimated Shares Outstanding':'EstimatedSharesOutstanding'

 }

fundamentals.rename(renamed_columns,inplace = True , axis = 1)
#2.5 Check NA in columns

fundamentals.isna().any()
#2.5 Update columns containing NA

fundamentals.loc[fundamentals['ForYear'].isnull() == True ,'ForYear'] = fundamentals['PeriodEnding'].dt.year

fundamentals.loc[fundamentals['CurrentRatio'].isnull() == True,'CurrentRatio'] = 0

fundamentals.loc[fundamentals['CashRatio'].isnull() == True,'CashRatio'] = 0

fundamentals.loc[fundamentals['QuickRatio'].isnull() == True,'QuickRatio'] = 0

fundamentals.loc[fundamentals['EarningsPerShare'].isnull() == True,'EarningsPerShare'] = 0

fundamentals.loc[fundamentals['EstimatedSharesOutstanding'].isnull() == True,'EstimatedSharesOutstanding'] = 0

fundamentals.loc[fundamentals['ForYear'] == 1215,'ForYear'] = 2015
# 2.6 Removing outliners

fundamentals['ForYear'].value_counts()

fundamentals.drop(fundamentals[(fundamentals['ForYear'] == 2003.0) | (fundamentals['ForYear'] == 2004.0) | (fundamentals['ForYear'] == 2007.0) | (fundamentals['ForYear'] == 2006.0) | (fundamentals['ForYear'] == 2017.0)].index,inplace=True)

fundamentals['ForYear'] = fundamentals['ForYear'].astype('int')
fundamentals.columns

securities.columns
# 4.1.1 Calculating Mean Companywise for all the years considered

companywise_mean=fundamentals.groupby('TickerSymbol').mean().reset_index()
# 4.1.2 Considering Cash Ratio

px.histogram(data_frame = companywise_mean.sort_values('CashRatio',ascending = False).head(10),

             x = 'TickerSymbol',

             y= 'CashRatio',

             histfunc='avg',

             width = 800,

             height = 500

             )

#fig.update_xaxes(tickangle=90, tickfont=dict(family='Rockwell', color='crimson', size=14))
# 4.1.3 Considering Quick Ratio

px.histogram(data_frame = companywise_mean.sort_values('QuickRatio',ascending = False).head(10),

             x = 'TickerSymbol',

             y= 'QuickRatio',

             histfunc='avg',

             width = 800,

             height = 500

             )
# 4.1.4 Considering Current Ratio

px.histogram(data_frame = companywise_mean.sort_values('CurrentRatio',ascending = False).head(10),

             x = 'TickerSymbol',

             y= 'CurrentRatio',

             histfunc='avg',

             width = 800,

             height = 500

             )
# 4.1.5 Considering Profit Margin

px.histogram(data_frame = companywise_mean.sort_values('ProfitMargin',ascending = False).head(10),

             x = 'TickerSymbol',

             y= 'ProfitMargin',

             histfunc='avg',

             width = 800,

             height = 500

             )
# 4.1.6 Considering Research and Development

px.histogram(data_frame = companywise_mean.sort_values('ResearchandDevelopment',ascending = False).head(10),

             x = 'TickerSymbol',

             y= 'ResearchandDevelopment',

             histfunc='avg',

             width = 800,

             height = 500

             )
# 4.1.7 Considering Goodwill

px.histogram(data_frame = companywise_mean.sort_values('Goodwill',ascending = False).head(10),

             x = 'TickerSymbol',

             y= 'Goodwill',

             histfunc='avg',

             width = 800,

             height = 500

             )
# 4.1.8 Considering NetIncome

px.histogram(data_frame = companywise_mean.sort_values('NetIncome',ascending = False).head(10),

             x = 'TickerSymbol',

             y= 'NetIncome',

             histfunc='avg',

             width = 800,

             height = 500

             )
# 4.1.9 Distribution of Earning per share and Profit margin

fig = px.density_contour(

                         data_frame =companywise_mean,

                         x = 'EarningsPerShare',

                         y = 'ProfitMargin',

                         range_x=[0,10]

                        )

fig.update_traces(

                  contours_coloring="fill",

                  contours_showlabels = True

                 )
#4.2.1 Join fundamental dataset with securities 

fundamentals_security = pd.merge(fundamentals,securities, left_on = "TickerSymbol", right_on="Ticker symbol",how="left")
# 4.2.2 Histogram to show profit margin of various sectors across the years

px.histogram(data_frame =fundamentals_security,

                      x = 'GICS Sector',

                      y = 'ProfitMargin',

                      marginal = 'violin',

                      color = 'ForYear',

                      histfunc = 'avg'

                #opacity = 0.2

             )
# 4.2.3 Histogram to show profit margin of various sectors across the years

px.histogram(data_frame =fundamentals_security,

                      x = 'GICS Sector',

                      y = 'AfterTaxROE',

                      marginal = 'box',

                      color = 'ForYear', 

                      histfunc = 'avg'

                #opacity = 0.2

             )
# 4.2.4 Histogram to show Earning Per Share of various sectors across the years

px.histogram(data_frame =fundamentals_security,

                      x = 'GICS Sector',

                      y = 'EarningsPerShare',

                      marginal = 'box',

                      color = 'ForYear', 

                      histfunc = 'avg'

                #opacity = 0.2

             )
# 4.2.5 Histogram to show Earning Per Share of various sectors across the years

px.histogram(data_frame =fundamentals_security,

                      x = 'GICS Sector',

                      y = 'NetIncome',

                      marginal = 'box',

                      color = 'ForYear', 

                      histfunc = 'avg'

                #opacity = 0.2

             )
# 4.2.6 Histogram to show average Net Income of various sectors over the years

fig = px.histogram(data_frame = fundamentals_security,

             x = 'GICS Sector',

             y= 'NetIncome',

             histfunc='avg',

             width = 800,

             height = 1000

             )

fig.update_xaxes(tickangle=90, tickfont=dict(family='Rockwell', color='crimson', size=14))
# 5.1 Dataset containing mean of all parameters 

companywise_mean=fundamentals.groupby('TickerSymbol').mean().reset_index()
# 5.2 A seperate column Net Income category has been added. 

companywise_mean['NetIncome_Category'] = companywise_mean['NetIncome'].map(lambda x : 0 if x<=0  else 1 if x/10000000<500 else 2)
# 5.3 create a copy of original dataset df which contains only float and int columns to draw Andrew Curve 

df = companywise_mean.select_dtypes(include = ['float64','int64']).copy()

df.drop(columns = ['ForYear'], inplace = True)
# 5.4 Draw Andrew Curve to find if there is any pattern

pd.plotting.andrews_curves(df,'NetIncome_Category')
# 6.1 creating a seperate array to store values of NetIncome Category and droping from  dataframe df

y = df['NetIncome_Category'].values

df.drop(columns = ['NetIncome_Category'], inplace = True)
# 6.2 Use StandardScaler class for scaling of dataframe df

ss = StandardScaler()

ss.fit(df)

XX = ss.transform(df)
# 6.3 Split dataset into training and test,use train for clustering and test to check test to cluster based on parameter Net Income category

XX_train,XX_test,_,y_test = train_test_split(XX,y,test_size=0.25)
# 6.4 Draw Elbow curve to show relation between no. of clusters and SSE(Sum of squared Error)

sse =[]

for i in list(range(10)):

    n_cluster = i+1

    clf = KMeans(n_clusters = n_cluster)

    clf.fit(XX_train)

    sse.append(clf.inertia_ )  #append SSE value for this no. of clusters

    

    

sns.lineplot(range(1, 11), sse)    
# 6.5 create clusters and find cluster informations

cls = KMeans(n_clusters = 3)      # instantiate KMean object

cls.fit(XX_train)                  # Get info about X_train

cls.cluster_centers_.shape        # shape of cluster centres

cls.labels_                       # Cluster labels for every observation

cls.labels_.size                

cls.inertia_                      # display value of SSE
# 6.6 Predict clustering for test data

y_pred = cls.predict(XX_test)

y_pred
#6.7 check accuracy of prediction 

np.sum(y_pred==y_test)/y_test.size
#6.8 Calculate Silhouette score for the clusters

silhouette_score(XX_train, cls.labels_)
#6.8 Use Yellow brick for plotting Silhouette score for each  cluster

visualizer = SilhouetteVisualizer(cls, colors='yellowbrick')

visualizer.fit(XX_train)

visualizer.show()
#6.9 using scatter plot to check cluster differentiality

dx = pd.Series(XX_test[:, 1])

dy = pd.Series(XX_test[:,2])

sns.scatterplot(dx,dy, hue = y_pred)