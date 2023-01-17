import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import numpy as np
import seaborn as sns
from os import path
from PIL import Image

import matplotlib.pyplot as plt
% matplotlib inline

pd.set_option('display.max_rows', 100)
online_retail = pd.read_csv(
    "/kaggle/input/onlineretail/OnlineRetail.csv",
    sep=",",
    dtype={"CustomerID": "object"},
    encoding="unicode_escape",
)
online_retail.columns
online_retail.head()
"""
!pip uninstall pandas-profiling
!pip install pandas-profiling[notebook,html]
"""

"""
from pandas_profiling import ProfileReport
online_retail_profile = ProfileReport(online_retail, title='Pandas Profiling Report', html={'style':{'full_width':True}})
online_retail_profile
"""
online_retail["InvoiceDate"] = pd.to_datetime(online_retail["InvoiceDate"])

# Create an additional column for date as year and month
online_retail["date"] = online_retail["InvoiceDate"].dt.strftime("%Y-%m")

# Create a new column for the total expenditure of that product in the purchase.
online_retail["total_sales_amount"] = (
    online_retail["UnitPrice"] * online_retail["Quantity"]
)
min(online_retail['InvoiceDate'])
max(online_retail['InvoiceDate'])
# Remove days from December 2011
online_retail = online_retail[online_retail.InvoiceDate < '2011-12-01']
# Remove rows were Unit price is 0.0
online_retail = online_retail[online_retail['UnitPrice'] !=0.0]
# Add column for cancelations 
online_retail['cancelation'] = online_retail['InvoiceNo'].apply(lambda x: x.startswith("C"))
online_retail['cancelation'] = online_retail['total_sales_amount'] <0
# Only positive purchases (No cancelations)
online_retail_purchases = online_retail[online_retail['cancelation'] == False] 
# Group by Purchase (Invoice)
invoices = (
    online_retail_purchases.groupby(["InvoiceNo"])[["Quantity", "total_sales_amount"]]
    .agg("sum")
    .reset_index()
)
# Correlation Quantity & total_sales_amount
sns.set_style({'axes.grid' : False})
sns.set_style("darkgrid")
ax = sns.regplot(x=invoices["Quantity"], y=invoices["total_sales_amount"], marker="+", fit_reg=False)
ax.set_title('Purchases')
ax.set(xlabel='Quantity (units)', ylabel='Total Amount')
plt.show()
# Removing outlier 
invoices = invoices[invoices["Quantity"] < 20000]
# Correlation graph quantity & total_sales_amount
sns.set_style({'axes.grid' : False})
sns.set_style("darkgrid")
ax = sns.regplot(x=invoices["Quantity"], y=invoices["total_sales_amount"], marker="+", fit_reg=False)
ax.set_title('Purchases')
ax.set(xlabel='Quantity (units)', ylabel='Total Amount')
plt.show()
corrMatrix = invoices.corr()
sns.heatmap(corrMatrix, annot=True)
invoices[invoices['total_sales_amount'] == max(invoices['total_sales_amount'])]
invoices[invoices['total_sales_amount'] == min(invoices['total_sales_amount'])]
np.sqrt(invoices['total_sales_amount'].var())
monthly_purchases = (
    online_retail_purchases.groupby(["date"])[["total_sales_amount"]]
    .agg("sum")
    .reset_index()
)
monthly_purchases = monthly_purchases.sort_values(by=["date"])


def millions(x, pos):
    "The two args are the value and tick position"
    return "$%1.1fM" % (x * 1e-6)


formatter = FuncFormatter(millions)

fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(formatter)
monthly_purchases.plot(
    kind="line", x="date", y="total_sales_amount", ax=ax, title="Monthly Purchases"
)
plt.show()
monthly_customers = online_retail.groupby(["date", "CustomerID"]).count().reset_index()
monthly_customers = monthly_customers[["date", "CustomerID"]]
monthly_customers[["date", "unique_customers"]] = (
    monthly_customers.groupby("date")["CustomerID"].count().reset_index()
)

fig, ax = plt.subplots()
monthly_customers.plot(
    kind="line", x="date", y="unique_customers", ax=ax, title="Monthly Customers"
)
plt.xticks(rotation=45)
plt.show()

customers_purchases = (
    online_retail_purchases.groupby(["CustomerID"])[["Quantity", "total_sales_amount"]]
    .agg("sum")
    .reset_index()
)
# Calculate Quantiles
customers_purchases['total_sales_amount'].quantile([.25, .5, 0.75,  0.80, 0.90, 0.95 , 0.99])
# Threshold
threshold = 3500
minority_customers = customers_purchases[
    customers_purchases["total_sales_amount"] <= threshold
]

mayority_customers = customers_purchases[
    customers_purchases["total_sales_amount"] > threshold
]
sns.boxplot(y=minority_customers['total_sales_amount'])
minority_customers['total_sales_amount'].quantile([.25, .5, 0.75, 0.99])
sns.distplot(a=minority_customers['total_sales_amount'], hist=True, kde=False, rug=False )
# By total amount spent 
customers_purchases.sort_values(by=['total_sales_amount'], ascending=False)
cancelations = online_retail[online_retail["cancelation"] == True]

# Graph with the number of Invoices that are cancelations vs purchases
online_retail.cancelation.value_counts().sort_values().plot(
    kind="barh", title="Purchase Cancelled"
)
top_ten_countries = online_retail['Country'].value_counts()
top_ten_countries = top_ten_countries.iloc[0:11]
top_ten_countries.plot(kind='bar', title = 'Top 10 - Purchases by Country')
products = (
    online_retail_purchases.groupby(["StockCode", "Description"])[["Quantity"]]
    .agg("sum")
    .reset_index()
)
products.sort_values(by=['Quantity'], ascending=False)[0:10]
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
descriptions = online_retail_purchases.Description.str.cat(sep=' ')
# Start with one review:
text = descriptions

# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
!pip install lifetimes
from dateutil import parser
import datetime
from dateutil import relativedelta
from lifetimes import BetaGeoFitter
from lifetimes.utils import summary_data_from_transaction_data
from sklearn.metrics import mean_squared_error
transactional_purchases = (
    online_retail_purchases.groupby(["InvoiceNo", "CustomerID", "InvoiceDate"])[
        ["Quantity", "total_sales_amount"]
    ]
    .agg("sum")
    .reset_index()
)
transactional_purchases.head()
# Configurable experimental variables 
SPLIT_DATE = "2011-06-01" # Date to be used to end train date
PERIOD_LENGTH = 2 # Months
date = parser.parse(SPLIT_DATE)
end_date = date + relativedelta.relativedelta(months=2)
end_test_date = end_date.strftime("%Y-%m-%d")

train = transactional_purchases[transactional_purchases["InvoiceDate"] < SPLIT_DATE]
test = transactional_purchases[
    (transactional_purchases["InvoiceDate"] >= SPLIT_DATE)
    & (transactional_purchases["InvoiceDate"] < end_test_date)
]

print("Start Train dataset date {}".format(train["InvoiceDate"].min()))
print("End Train dataset date {}".format(train["InvoiceDate"].max()))
print("---------------------------------------------")
print("Start Test dataset date {}".format(test["InvoiceDate"].min()))
print("End Test dataset date {}".format(test["InvoiceDate"].max()))
# Create Features (Frequency, Recency and T) for customers
features_train = summary_data_from_transaction_data(
    train,
    customer_id_col="CustomerID",
    datetime_col="InvoiceDate",
    monetary_value_col="total_sales_amount",
    freq="D",
)
features_train.reset_index(level=0, inplace=True)
features_train
# Fit to the BG/NBD model 
bgf = BetaGeoFitter(penalizer_coef=0.1)
bgf.fit(features_train['frequency'], features_train['recency'], features_train['T']) 
print(bgf)
customers = features_train[['CustomerID']]
# Predict future total amount spent for individual customers (next period)
t = PERIOD_LENGTH*30  # Days (2 Months aprox)
customers["pred_n_purchases"] = bgf.predict(
    t, features_train["frequency"], features_train["recency"], features_train["T"]
)
y_predictions = customers[["CustomerID", "pred_n_purchases"]]

test_n_purchases = (
    test["CustomerID"]
    .value_counts()
    .rename_axis("CustomerID")
    .to_frame("true_n_purchases")
)

dataset = pd.merge(y_predictions, test_n_purchases, on='CustomerID', how='left')
dataset['true_n_purchases'].fillna(0, inplace= True) # No sales 
# Example prediction individual customer 14646
dataset[dataset.CustomerID=='14646']
y_true_n_purchases = dataset['true_n_purchases']
y_pred_n_purchases = dataset['pred_n_purchases']
mean_squared_error(y_true_n_purchases, y_pred_n_purchases, squared=False)
# Amount spent by customer current period
test_amount_spent = (
    test.groupby(["CustomerID"])[["total_sales_amount"]]
    .agg("sum")
    .reset_index()
    .rename(columns={"total_sales_amount": "true_amount_spent_next_period"})
)
# In order to fit model remove the onems that have 0 en monetary value
f_r_t_summary2 = features_train[features_train['monetary_value']>0]
# Fit GammaGamma model 
from lifetimes import GammaGammaFitter

ggf = GammaGammaFitter(penalizer_coef = 0)
ggf.fit(f_r_t_summary2['frequency'],
        f_r_t_summary2['monetary_value'])
print(ggf)
f_r_t_summary2["pred_amount_spent_next_period"] = ggf.customer_lifetime_value(
    bgf, #the model to use to predict the number of future transactions
    f_r_t_summary2['frequency'],
    f_r_t_summary2['recency'],
    f_r_t_summary2['T'],
    f_r_t_summary2['monetary_value'],
    time=PERIOD_LENGTH, # months
    discount_rate=0.01 # monthly discount rate
)
y_predictions = f_r_t_summary2[["CustomerID", "pred_amount_spent_next_period"]]
dataset = pd.merge(customers['CustomerID'], y_predictions, on='CustomerID', how='left')
dataset = pd.merge(dataset, test_amount_spent, on='CustomerID', how='left')
dataset['pred_amount_spent_next_period'].fillna(0, inplace= True) # Fill No sales with 0
dataset['true_amount_spent_next_period'].fillna(0, inplace= True) # Fill No sales with 0
dataset
y_true_amount_spent = dataset['true_amount_spent_next_period']
y_pred_amount_spent = dataset['pred_amount_spent_next_period']
mean_squared_error(y_true_amount_spent, y_pred_amount_spent, squared=False)
def create_features_split(transactions, split_date, period_length, datetime_col, total_sales_col):
    """
    Taket historic transactional level data and returns train and test dataset in 
    costumer level useful to be used by machine learning models. 
    
    Arguments:
        transactions - Dataframe at transaction level with war list of purchases.
        split_date - Date to be used to end train date
        period_length - The length of period in Months.
        datetime_col - Column of date time
        
    Returns:
        train - Dataframe at customer level to be used for training
        test - Dataframe at customer level to be used for testing 
        
    """

    train = transactions[transactions[datetime_col] < split_date]

    date = parser.parse(split_date)
    end_test_date = date + relativedelta.relativedelta(months=period_length)
    end_test_date = end_test_date.strftime("%Y-%m-%d")

    train_transactions = transactions[transactions[datetime_col] < split_date]
    test_transactions = transactions[transactions[datetime_col] < end_test_date]

    print("Creating Train ...")
    train = _transactions_to_dataset(
        train_transactions,
        split_date,
        period_length,
        "InvoiceDate",
        "CustomerID",
        total_sales_col,
    )

    print("Creating Test ...")
    test = _transactions_to_dataset(
        test_transactions,
        end_test_date,
        period_length,
        "InvoiceDate",
        "CustomerID",
        total_sales_col,
    )

    return train, test


def _transactions_to_dataset(
    transactions,
    end_date,
    period_length,
    datetime_col,
    customer_id_col,
    total_sales_col,
):
    """
    Take historic transactions and create a dataset with basics staticts features,
    number of purchases from past, current and next period and amount spent from 
    past, current and next period.
    
    Begining dataset: t0
    Past period: t1 - t2
    Current period: t2 - t3
    Target period: t3 - t4
    
    Arguments: 
        transactions - Dataframe at transaction level with war list of purchases.
        end_date - Last date to use to create dataset
        period_length - The length of period in Months.
        customer_id_col - Name of column with the ids of costumers
        total_sales_col - Name of column of the total amount spent in purchase
    
    Returns: 
        dataset - Data for customer level with number of transactions and total 
                    amount spent in the last, current and next period
    
    """

    t4 = end_date
    t3 = (
        parser.parse(t4) - relativedelta.relativedelta(months=period_length)
    ).strftime("%Y-%m-%d")
    t2 = (
        parser.parse(t3) - relativedelta.relativedelta(months=period_length)
    ).strftime("%Y-%m-%d")
    t1 = (
        parser.parse(t2) - relativedelta.relativedelta(months=period_length)
    ).strftime("%Y-%m-%d")
    t0 = transactions[datetime_col].min().strftime("%Y-%m-%d")

    # Define time periods
    transactions_dev = transactions[transactions[datetime_col] < t3]

    current_period = transactions_dev[
        (transactions_dev[datetime_col] >= t2) & (transactions_dev[datetime_col] < t3)
    ]

    past_period = transactions_dev[
        (transactions_dev[datetime_col] >= t1) & (transactions_dev[datetime_col] < t2)
    ]

    target_period = transactions[
        (transactions[datetime_col] >= t3) & (transactions[datetime_col] < t4)
    ]

    # Basic Features (Frequency, Recency and T) since t0
    features_train = summary_data_from_transaction_data(
        transactions_dev,
        customer_id_col=customer_id_col,
        datetime_col=datetime_col,
        monetary_value_col=total_sales_col,
        freq="D",
    )
    features_train.reset_index(level=0, inplace=True)

    # Purchases by customers current period
    purchases_current_period = (
        current_period[customer_id_col]
        .value_counts()
        .rename_axis(customer_id_col)
        .to_frame("purchases_current_period")
    )
    purchases_current_period.reset_index(level=0, inplace=True)

    # Purchases by customer past period
    purchases_past_period = (
        past_period[customer_id_col]
        .value_counts()
        .rename_axis(customer_id_col)
        .to_frame("purchases_past_period")
    )

    purchases_past_period.reset_index(level=0, inplace=True)

    # Amount spent by customer current period
    amount_spent_current_period = (
        current_period.groupby([customer_id_col])[[total_sales_col]]
        .agg("sum")
        .reset_index()
        .rename(columns={"total_sales_amount": "amount_spent_current_period"})
    )

    # Amount spent by customer last period
    amount_spent_past_period = (
        past_period.groupby([customer_id_col])[[total_sales_col]]
        .agg("sum")
        .reset_index()
        .rename(columns={"total_sales_amount": "amount_spent_past_period"})
    )

    # Create Targets
    purchases_target = (
        target_period[customer_id_col]
        .value_counts()
        .rename_axis(customer_id_col)
        .to_frame("purchases_next_period")
    )

    amount_spent_target = (
        target_period.groupby([customer_id_col])[[total_sales_col]]
        .agg("sum")
        .reset_index()
        .rename(columns={total_sales_col: "amount_spent_next_period"})
    )

    # Join the Datasets
    dataset = pd.merge(
        features_train, purchases_past_period, on=customer_id_col, how="left"
    )
    dataset = pd.merge(
        dataset, purchases_current_period, on=customer_id_col, how="left"
    )
    dataset = pd.merge(dataset, purchases_target, on=customer_id_col, how="left")
    dataset = pd.merge(
        dataset, amount_spent_past_period, on=customer_id_col, how="left"
    )
    dataset = pd.merge(
        dataset, amount_spent_current_period, on=customer_id_col, how="left"
    )
    dataset = pd.merge(dataset, amount_spent_target, on=customer_id_col, how="left")

    # Fill NA (No sales) with 0
    dataset["purchases_past_period"].fillna(0, inplace=True)
    dataset["purchases_current_period"].fillna(0, inplace=True)
    dataset["purchases_next_period"].fillna(0, inplace=True)
    dataset["amount_spent_past_period"].fillna(0, inplace=True)
    dataset["amount_spent_current_period"].fillna(0, inplace=True)
    dataset["amount_spent_next_period"].fillna(0, inplace=True)

    print("Data statistics starts from {}".format(t0))
    print("Past period from [{} to {})".format(t1, t2))
    print("Current period from [{} to {})".format(t2, t3))
    print("Next period from [{} to {})".format(t3, t4))

    return dataset
train, test = create_features_split(
    transactional_purchases,
    split_date=SPLIT_DATE,
    period_length=PERIOD_LENGTH,
    datetime_col="InvoiceDate",
    total_sales_col="total_sales_amount"
)
train.columns
feature_cols = ['frequency', 'recency', 'T', 'monetary_value',
       			'purchases_past_period', 'purchases_current_period',
       			'amount_spent_past_period','amount_spent_current_period']

y_col = ['purchases_next_period']
X_train = train[feature_cols]
y_train = train[y_col]

X_test = test[feature_cols]
y_test = test[y_col]
from sklearn.ensemble import RandomForestRegressor

clf = RandomForestRegressor(random_state=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
mean_squared_error(y_pred, y_test, squared=False)
y_col = ['amount_spent_next_period']
X_train = train[feature_cols]
y_train = train[y_col]

X_test = test[feature_cols]
y_test = test[y_col]
clf = RandomForestRegressor(random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
mean_squared_error(y_pred, y_test, squared=False)