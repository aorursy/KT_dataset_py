# 1.0 Call libraries
# 1.1 Data manipulation library
import pandas as pd
import numpy as np
# 1.2 OS related package
import os
# 1.3 Modeling librray
# 1.3.1 Scale data
from sklearn.preprocessing import StandardScaler
# 1.3.2 Split dataset
from sklearn.model_selection import train_test_split
# 1.3.3 Class to develop kmeans model
from sklearn.cluster import KMeans
# 1.4 Plotting library
import seaborn as sns
import plotly.express as px
# 1.5 How good is clustering?
from sklearn.metrics import silhouette_score
# 1.6 Set numpy options to display wide array
np.set_printoptions(precision = 3,          # Display upto 3 decimal places
                    threshold=np.inf        # Display full array
                    )
# 2.1 Read csv file -
df_fundamentals = pd.read_csv("../input/nyse/fundamentals.csv")
# 2.2 Explore dataset
df_fundamentals.shape # 1781 Rows , 79 Columns
df_fundamentals.head() # First Five Rows
#2.3 Drop Unnamed Column 
df_fundamentals.drop(df_fundamentals.columns[df_fundamentals.columns.str.contains('Unnamed',case = False)],axis = 1, inplace = True)
#2.4 Get Columns
df_fundamentals.columns.values
#2.5Rename columns by replacing spaces in column_names with underscore and also remove other symbols
#from column names such as: , (comma), . (full-stop), / (backslash) etc.
#That is clean the column names and assign these new names to your dataset.
#2.6 Columns stored in a List.
list1 = df_fundamentals.columns
#2.7 Prepare a List contains column name space replaced with Underscore(_)
list2= [
'Ticker_Symbol', 'Period_Ending', 'Accounts_Payable',
       'Accounts_Receivable', 'Addl_income_expense_items', 'After_Tax_ROE',
       'Capital_Expenditures', 'Capital_Surplus', 'Cash_Ratio',
       'Cash_and_Cash_Equivalents', 'Changes_in_Inventories', 'Common_Stocks',
       'Cost_of_Revenue', 'Current_Ratio', 'Deferred_Asset_Charges',
       'Deferred_Liability_Charges', 'Depreciation',
       'Earnings_Before_Interest_and_Tax', 'Earnings_Before_Tax',
       'Effect_of_Exchange_Rate',
       'Equity_Earnings_Loss_Unconsolidated_Subsidiary', 'Fixed_Assets',
       'Goodwill', 'Gross_Margin', 'Gross_Profit', 'Income_Tax',
       'Intangible_Assets', 'Interest_Expense', 'Inventory', 'Investments',
       'Liabilities', 'Long_Term_Debt', 'Long_Term_Investments',
       'Minority_Interest', 'Misc_Stocks', 'Net_Borrowings', 'Net_Cash_Flow',
       'Net_Cash_Flow_Operating', 'Net_Cash_Flows_Financing',
       'Net_Cash_Flows_Investing', 'Net_Income', 'Net_Income_Adjustments',
       'Net_Income_Applicable_to_Common_Shareholders',
       'Net_Income_Cont_Operations', 'Net_Receivables', 'Non_Recurring_Items',
       'Operating_Income', 'Operating_Margin', 'Other_Assets',
       'Other_Current_Assets', 'Other_Current_Liabilities', 'Other_Equity',
       'Other_Financing_Activities', 'Other_Investing_Activities',
       'Other_Liabilities', 'Other_Operating_Activities',
       'Other_Operating_Items', 'Pre_Tax_Margin', 'Pre_Tax_ROE',
       'Profit_Margin', 'Quick_Ratio', 'Research_and_Development',
       'Retained_Earnings', 'Sale_and_Purchase_of_Stock',
       'Sales, General_and_Admin',
       'Short_Term_Debt_Current_Portion_of_Long_Term_Debt',
       'Short_Term_Investments', 'Total_Assets', 'Total_Current_Assets',
       'Total_Current_Liabilities', 'Total_Equity', 'Total_Liabilities',
       'Total_Liabilities_Equity', 'Total_Revenue', 'Treasury_Stock',
       'For_Year', 'Earnings_Per_Share', 'Estimated_Shares_Outstanding']
#2.8 Create a dictionary which contains old & new columns list
dict1= dict(zip(list1,list2)) 
#2.9 Function 'ChngColName' will replace the special characets from column name
def chngColName(d,x):
   x.rename(d,axis=1,inplace=True)
   return x  
chngColName(dict1,df_fundamentals)
#3.0  Drop Rows having NaN/NULL values
df_fundamentals.dropna()
#3.1  Drop Columns having NaN/NULL Value
df_fundamentals.dropna(axis=1,inplace=True)
#3.2 After Dropping Null Rows & Columns Explore the Dataset.
df_fundamentals.shape #1781 Rows, 72 Columns
#3.3 Create 2 new columns 'Long_Term_Liabilities' & 'Fixed_Assets'
df_fundamentals['Long_Term_Liabilties']=df_fundamentals['Total_Liabilities']-df_fundamentals['Total_Current_Liabilities']
df_fundamentals['Fixed_Assets']=df_fundamentals['Total_Assets']-df_fundamentals['Total_Current_Assets']
#3.4 Describe Dataframe
df_fundamentals1= df_fundamentals.describe()
df_fundamentals1
#3.5 Draw a Line graph for 'Total_Current_Assets','Fixed_Assets','Total_Current_Liabilities','Long_Term_Liabilties' columns
df_fundamentals1[['Total_Current_Assets','Fixed_Assets','Total_Current_Liabilities','Long_Term_Liabilties']].describe().plot(kind='line')
# It shows Fixed Assets & Long Term Liabilities are more than short term liabilities and current assets. 
# Current Assets include cash, receivables(in short time) whereas fixed assets include plants,machinary, building, land.
# Long term liabilities include Term Loan, Caital whereas current liabilities means short term loans.
# 3.5 Bar Chart to relate 'Total Liabilities' & 'Total Assets'
df_fundamentals1[['Total_Liabilities','Total_Assets']].describe().plot(kind='bar')
#3.6 Group data by Ticker Symbols and take a mean of all numeric variables.
df_fundamentals2=df_fundamentals.groupby(['Ticker_Symbol']).mean()
df_fundamentals2
#3.7 A line chart to relate 'Cost_of_Revenue','Total_Revenue'
df_fundamentals2[['Cost_of_Revenue','Total_Revenue']].plot(kind='bar')
# 3.8 Read csv file
df_prices = pd.read_csv("../input/nyse/prices.csv")
#3.9 Explore Data
df_prices.shape    #851264 Rows, 7 Columns
df_prices.columns.values
#4.0 Describe 
df_prices.describe()
# 4.1 Copy 'close' column to another variable 'y'.
# Copy 'open','low', 'high', 'volume' to another variable 'x'
#     We will not use 'close' column in clustering
# Similarly 'date' column will also not be used

x=df_prices[['open','low', 'high', 'volume']].values
y=df_prices['close'].values
# 4.2 Scale data using StandardScaler
ss = StandardScaler()     # Create an instance of class
ss.fit(x)                # Train object on the data
z = ss.transform(x)      # Transform data
z[:5, :]                  # See first 5 rows
# 4.3 Split dataset into train/test
z_train, z_test, _, y_test = train_test_split( z,               # np array without target
                                               y,               # Target
                                               test_size = 0.25 # test_size proportion
                                               )
# 4.4 Examine the results
z_train.shape    #(638448, 4)          
z_test.shape     #(212816, 4)    
# 4.5 Develop model
# 4.6 Create an instance of modeling class
#     We will have two clusters
clf = KMeans(n_clusters = 2)
# 4.7 Train the object over data
clf.fit(z_train)
# 4.8 So what are our clusters?
clf.cluster_centers_
clf.cluster_centers_.shape     # (2,4)
clf.labels_                    # Cluster labels for every observation
clf.labels_.size               # 638448
clf.inertia_                   # Sum of squared distance to respective centriods, SSE

# 4.9 Make prediction over our test data and check accuracy
y_pred = clf.predict(z_test)
y_pred
# 5.0 How good is prediction
np.sum(y_pred == y_test)/y_test.size

# 5.1 Are clusters distiguisable?
#     We plot 1st and 2nd columns of X
#     Each point is coloured as per the
#     cluster to which it is assigned (y_pred)
dx = pd.Series(z_test[:, 0])
dy = pd.Series(z_test[:,1])
sns.scatterplot(dx,dy, hue = y_pred)
# 5.2 Scree plot:
sse = []
for i,j in enumerate(range(10)):
    # 5.2.1 How many clusters?
    n_clusters = i+1
    # 5.2.2 Create an instance of class
    clf = KMeans(n_clusters = n_clusters)
    # 5.2.3 Train the kmeans object over data
    clf.fit(z_train)
    # 5.2.4 Store the value of inertia in sse
    sse.append(clf.inertia_ )

# 5.3 Plot the line now
sns.lineplot(range(1, 11), sse)

