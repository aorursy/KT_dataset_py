# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Reading the sellers dataset
df_sellers = pd.read_csv("../input/brazilian-ecommerce/olist_sellers_dataset.csv")

# Reading the costumers dataset
df_costumers = pd.read_csv("../input/brazilian-ecommerce/olist_customers_dataset.csv")

# Reading the orders datasets
df_orders = pd.read_csv("../input/brazilian-ecommerce/olist_orders_dataset.csv")
df_order_items = pd.read_csv("../input/brazilian-ecommerce/olist_order_items_dataset.csv")
df_order_payments = pd.read_csv("../input/brazilian-ecommerce/olist_order_payments_dataset.csv")
df_order_reviews = pd.read_csv("../input/brazilian-ecommerce/olist_order_reviews_dataset.csv")

# Reading the products dataset
df_products = pd.read_csv("../input/brazilian-ecommerce/olist_products_dataset.csv")

# Reading the localization dataset
df_localizations = pd.read_csv("../input/brazilian-ecommerce/olist_geolocation_dataset.csv")

# Reading the category name translator
df_translator = pd.read_csv("../input/brazilian-ecommerce/product_category_name_translation.csv")
df_products
# Reading the leads dataset 
df_leads = pd.read_csv('../input/marketing-funnel-olist/olist_marketing_qualified_leads_dataset.csv')
df_leads.head(10)
# Reading the leads dataset 
df_closed_deals = pd.read_csv('../input/marketing-funnel-olist/olist_closed_deals_dataset.csv')
df_closed_deals.head(10)
df_text = df_order_reviews[["review_score", "review_comment_title"]].dropna()
df_text 
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()

X_counts = count_vect.fit_transform(df_text["review_comment_title"].to_list())
X_counts.shape
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)
X_tfidf.shape
# marketing funnel dataset (NaNs are leads that did not close a deal)
df_funnel = df_leads.merge(df_closed_deals, on='mql_id', how='left')
df_funnel.head(10)
df_funnel.dropna(subset=["seller_id"])
df_funnel_items = df_funnel.dropna(subset=["seller_id"]).merge(df_order_items, on='seller_id', how='left')
df_funnel_items = df_funnel_items.dropna(subset=["product_id"])
df_funnel_items
df_funnel_products = df_funnel_items.merge(df_products, on="product_id", how="left")
df_funnel_products
from datetime import datetime, date, timedelta

df = df_funnel.dropna(subset=["seller_id"])
df["won_date"] = df["won_date"].apply(lambda d: datetime.strptime(d,"%Y-%m-%d %H:%M:%S"))
df.head()

representant_struc = {}
for representant_id in df["sr_id"].unique():
    # Get the seller convertions...
    df_seller = df.where(df["sr_id"] == representant_id).dropna(subset=["won_date", "business_segment"])
    
    representant_struc[representant_id] = {
        "num_convertions": len(df_seller),
        "first_convertion": df_seller["won_date"].min(),
        "last_convertion": df_seller["won_date"].max(),
        "num_categories": len(df_seller["business_segment"].unique())
    }
    
pd_struc = {
    "representant": [],
    "categories": [],
    "convertions": [],
    "performance": [],
    "time_as_representant": []
}

for representant_id in representant_struc:
    # Time as seller in OList
    time_as_representant = (representant_struc[representant_id]["last_convertion"] - representant_struc[representant_id]["first_convertion"]).days
    
    # Convertions per time -> Performance metric
    convertions_per_time = representant_struc[representant_id]["num_convertions"] / time_as_representant if time_as_representant != 0 else representant_struc[representant_id]["num_convertions"]
    
    pd_struc["representant"].append(representant_id)
    pd_struc["performance"].append(convertions_per_time)
    pd_struc["categories"].append(representant_struc[representant_id]["num_categories"])
    pd_struc["convertions"].append(representant_struc[representant_id]["num_convertions"])
    pd_struc["time_as_representant"].append(time_as_representant)

plot_df = pd.DataFrame(pd_struc)
plot_df
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=plot_df["categories"],
    y=plot_df["convertions"],
    mode="markers"
))

fig.update_layout(template="xgridoff",
                  title="Correlations of convertions and categories",
                  xaxis_title="Number of categories",
                  yaxis_title="Number of convertions")

iplot(fig)
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=plot_df["categories"],
    y=plot_df["performance"],
    mode="markers"
))

fig.update_layout(template="xgridoff",
                  title="Correlations of categories and performance",
                  xaxis_title="Number of categories",
                  yaxis_title="Performance")

iplot(fig)
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=plot_df["time_as_representant"],
    y=plot_df["performance"],
    mode="markers"
))

fig.update_layout(template="xgridoff",
                  title="Correlations of time as represetant and performance",
                  xaxis_title="Time as representant",
                  yaxis_title="Performance")

iplot(fig)
df = df_funnel_items.merge(df_orders, on="order_id", how="left")

df["order_approved_at"] = df["order_approved_at"].apply(lambda d: datetime.strptime(d, "%Y-%m-%d %H:%M:%S"))

df.head()
df_performance = df.groupby(by="seller_id").sum().reset_index()
df_performance
seller_time = {
    "seller_id": [],
    "time_as_seller": []
}

for seller_id in df["seller_id"].unique():
    
    df_seller = df.where(df["seller_id"] == seller_id).dropna(subset=["seller_id"])
    
    first_order_date = df_seller["order_approved_at"].min()
    last_order_date = df_seller["order_approved_at"].max()
    
    time_as_seller = (last_order_date - first_order_date).days / 30
    
    seller_time["seller_id"].append(seller_id)
    seller_time["time_as_seller"].append(time_as_seller if time_as_seller != 0 else 1)

df_performance = df_performance.merge(pd.DataFrame(seller_time), on="seller_id", how="left")

df_performance["performance"] = df_performance["price"] / df_performance["time_as_seller"]

df_performance_seller = df_performance[["price", "freight_value", "time_as_seller", "performance"]]
df = df_order_items.merge(df_orders, on='order_id', how='left').dropna()
df["order_approved_at"] = df["order_approved_at"].apply(lambda d: datetime.strptime(d, "%Y-%m-%d %H:%M:%S"))

df.head()
df_best_products = df_order_items.groupby(by="product_id").count().sort_values(by="order_id", ascending=False).reset_index()
df_best_products = df_best_products[["product_id", "order_id"]]
df_best_products.columns = ["product_id", "order_count"]
df_best_products.head()
def build_product_history(product_id, df):
    """
    """
    # Build the dataframe of the particular product
    prod_df = df.where(df["product_id"] == product_id).dropna(subset=["product_id"])
    
    # Create the monthly group
    prod_df["month_group"] = prod_df["order_approved_at"].apply(lambda d: datetime(d.year, d.month, 1))
    
    # Group the product values
    res_df = prod_df.groupby(by="month_group").count().reset_index()
    res_df = res_df[["month_group", "order_id"]]
    res_df.columns = ["month_group", "order_count"]
    
    # Product mean price computing 
    res_df["mean price"] = prod_df.groupby(by="month_group").mean().reset_index()["price"]
    
    return res_df.sort_values(by="month_group")


def build_product_history_week(product_id, df):
    """
    """
    # Build the dataframe of the particular product
    prod_df = df.where(df["product_id"] == product_id).dropna(subset=["product_id"])
    
    # Create the monthly group
    prod_df["week"] = prod_df["order_approved_at"].apply(lambda d: d.week)
    prod_df["year"] = prod_df["order_approved_at"].apply(lambda d: d.year)
    
    # Group the product values
    res_df = prod_df.groupby(by=["week", "year"]).count().reset_index()
    res_df = res_df[["week", "year", "order_id"]]
    res_df.columns = ["week", "year", "order_count"]
    
    # Product mean price computing 
    res_df["mean price"] = prod_df.groupby(by=["week", "year"]).mean().reset_index()["price"]
    
    return res_df
my_prod_df = build_product_history("aca2eb7d00ea1a7b8ebd4e68314663af", df)
my_prod_df
df.where(df['product_id'] == "aca2eb7d00ea1a7b8ebd4e68314663af").dropna().max()

