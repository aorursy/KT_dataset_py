# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in a

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import dateparser
from ipywidgets import widgets
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
complaints = pd.read_csv("../input/Consumer_Complaints.csv").set_index("Complaint ID")
complaints.head()
complaints.columns
complaints.isna().sum()
xnan_cols = ['Consumer Complaint', 'Company Public Response',
            'Tags', 'Consumer consent provided?', 'Unnamed: 18']
complaints = complaints.drop(xnan_cols, axis =1)

complaints['year'] = complaints['Date received'].map(lambda x: x.split("-")[-1] if "-" in x else x.split("/")[-1])
complaints['month'] = complaints['Date received'].map(lambda x: x.split("-")[0] if "-" in x else x.split("/")[0])
complaints.year.head()
complaints.year.unique()
sns.countplot(x='year', data = complaints[complaints["year"]!='2018'])
sns.countplot(x="month", data=complaints)
mapping_old2new = {
    "Auto": "Auto debt",
    "Credit card": "Credit card debt",
    "Federal student loan": "Federal student loan debt",
    "Medical": "Medical debt",
    "Mortgage": "Mortgage debt",
    "Non-federal student loan": "Private student loan debt",
    "Other (i.e. phone, health club, etc.)": "Other debt",
    "Payday loan": "Payday loan debt",
    "Non-federal student loan": "Private student loan",
    "Federal student loan servicing": "Federal student loan",
    "Credit repair": "Credit repair services",
    "Credit reporting": "Credit reporting",
    "Conventional adjustable mortgage (ARM)": "Conventional home mortgage",
    "Conventional fixed mortgage": "Conventional home mortgage",
    "Home equity loan or line of credit": "Home equity loan or line of credit (HELOC)",
    "Other": "Other type of mortgage",
    "Other mortgage": "Other type of mortgage",
    "Second mortgage":"Other type of mortgage",
    "Credit card": "General-purpose credit card or charge card",
    "General purpose card": "General-purpose prepaid card",
    "Gift or merchant card": "Gift card",
    "Electronic Benefit Transfer / EBT card": "Government benefit card",
    "Government benefit payment card": "Government benefit card",
    "ID prepaid card": "Student prepaid card",
    "Other special purpose card":  "Other prepaid card",
    "Store credit card": "Other prepaid card",
    "Transit card": "Other prepaid card",
    "(CD) Certificate of deposit": "CD (Certificate of Deposit)",
    "Other bank product/service": "Other banking product or service",
    "Cashing a check without an account": "Other banking product or service",
    "Vehicle lease": "Lease",
    "Vehicle loan": "Loan",
    "Check cashing": "Check cashing service",
    "Mobile wallet": "Mobile or digital wallet",
    "Traveler’s/Cashier’s checks": "Traveler's check or cashier's check"
}
prod2sub = {
    "Auto debt": "Debt collection",
    "Credit card debt": "Debt collection",
    "Federal student loan debt": "Debt collection",
    "I do not know": "Debt collection",
    "Medical debt": "Debt collection",
    "Mortgage debt": "Debt collection",
    "Private student loan debt": "Debt collection",
    "Other debt": "Debt collection",
    "Payday loan debt": "Debt collection",
    "Credit repair services": "Credit reporting, credit repair services, or other personal consumer reports",
    "Credit reporting": "Credit reporting, credit repair services, or other personal consumer reports",
    "Other personal consumer report": "Credit reporting, credit repair services, or other personal consumer reports",
    "Conventional home mortgage": "Mortgage",
    "FHA mortgage": "Mortgage",
    "Home equity loan or line of credit (HELOC)": "Mortgage",
    "Other type of mortgage": "Mortgage",
    "Reverse mortgage": "Mortgage",
    "VA mortgage": "Mortgage",
    "General-purpose credit card or charge card": "Credit card or prepaid card",
    "General-purpose prepaid card": "Credit card or prepaid card",
    "Gift card": "Credit card or prepaid card",
    "Government benefit card": "Credit card or prepaid card",
    "Student prepaid card": "Credit card or prepaid card",
    "Payroll card": "Credit card or prepaid card",
    "Other prepaid card": "Credit card or prepaid card",
    "CD (Certificate of Deposit)": "Checking or savings account",
    "Checking account": "Checking or savings account",
    "Other banking product or service": "Checking or savings account",
    "Savings account": "Checking or savings account",
    "Lease": "Vehicle loan or lease",
    "Loan": "Vehicle loan or lease",
    "Federal student loan": "Student loan",
    "Private student loan": "Student loan",
    "Installment loan": "Payday loan, title loan, or personal loan",
    "Pawn loan": "Payday loan, title loan, or personal loan",
    "Payday loan": "Payday loan, title loan, or personal loan",
    "Personal line of credit": "Payday loan, title loan, or personal loan",
    "Title loan": "Payday loan, title loan, or personal loan",
    "Check cashing service": "Money transfer, virtual currency, or money service",
    "Debt settlement": "Money transfer, virtual currency, or money service",
    "Domestic (US) money transfer": "Money transfer, virtual currency, or money service",
    "Foreign currency exchange": "Money transfer, virtual currency, or money service",
    "International money transfer": "Money transfer, virtual currency, or money service",
    "Mobile or digital wallet": "Money transfer, virtual currency, or money service",
    "Money order": "Money transfer, virtual currency, or money service",
    "Refund anticipation check": "Money transfer, virtual currency, or money service",
    "Traveler's check or cashier's check": "Money transfer, virtual currency, or money service",
    "Virtual currency": "Money transfer, virtual currency, or money service"
}
def get_subprods(x):
    
    if x['Sub-product'] in mapping_old2new:
        if x['Sub-product']=="Other":
            if x['Product']=='Mortage':
                return("Other type of mortgage")
            else:
                return("Other debt")
        else:
            return(mapping_old2new[x['Sub-product']])
    else:
        return(x['Sub-product'])

complaints['Sub-product'] = complaints[['Product','Sub-product']].apply(lambda x: 
                                                                       get_subprods(x), axis =1)
prodmap = {
    "Payday loan": "Payday loan, title loan, or personal loan",
    "Credit reporting": "Credit reporting, credit repair services, or other personal consumer reports",
    "Credit card": "Credit card or prepaid card"
}
def get_product(x):
    
    if not isinstance(x['Sub-product'], str):
        if x['Product'] in prodmap:
            return(prodmap[x['Product']])
        else:
            return(x['Product'])
    else:
        return(prod2sub[x['Sub-product']])
    
complaints['Product'] = complaints[['Product','Sub-product']].apply(lambda x: get_product(x), axis =1)
products = complaints.Product.unique()
print(len(products), products)
plt.figure(figsize = (25,8))
plt.xticks(rotation=50, fontsize=15, ha="right")
plt.yticks(fontsize=15)
g = sns.countplot(x='Product', data = complaints,  order = complaints["Product"].value_counts().index)
g.set_xlabel("Product", fontsize=25)
g.set_ylabel("Count", fontsize=25)
from ipywidgets import widgets, interact, interactive
w = widgets.ToggleButtons(
    options=complaints.Product.value_counts().index,
    description='Product:',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
#     icons=['check'] * 3
)

@interact(product = w)
def plot_subproduct(product=w):
    plt.figure(figsize = (10,5))
    plt.xticks(rotation=50, fontsize=15, ha="right")
    plt.yticks(fontsize=15)
    g = sns.countplot(x='Sub-product', data = complaints[complaints.Product==product], 
                                order=complaints[complaints.Product==product]["Sub-product"].value_counts().index)
    g.set_xlabel("Sub-products({product})".format(product=product), fontsize=25)
    g.set_ylabel("Count", fontsize=25)
from ipywidgets import widgets, interact, interactive

w = widgets.ToggleButtons(
    options=['Mortgage', 'Credit reporting, credit repair services, or other personal consumer reports',
                             'Debt collection'],
    description='Product:',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
#     icons=['check'] * 3
)
z = widgets.ToggleButtons(
    options= complaints[complaints.Product=="Mortgage"]['Sub-product'].unique(),
    description='Sub-Product:',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
#     icons=['check'] * 3
)

def op_update_product(*args):
    z.options = [x for x in complaints[complaints.Product == w.value ]['Sub-product'].unique() 
                         if isinstance(x, str)]

w.observe(op_update_product, 'value')

@interact(product = w, subproduct=z)
def plot_issue(product="Mortgage", subproduct="VA mortgage"):
    #op_update_product()
    num_vals = len(complaints[(complaints.Product == product) & 
                                                   (complaints['Sub-product'] == subproduct)]['Issue'].value_counts().index)
    xdim=10
    if num_vals<6:
        xdim=8
    elif num_vals<12:
        xdim = 15
    else:
        xdim=20
        
    plt.figure(figsize = (xdim,6))
    plt.xticks(rotation=50, fontsize=15, ha="right")
    plt.yticks(fontsize=15)
    g = sns.countplot(x='Issue', data = complaints[(complaints.Product == product) & 
                                                   (complaints['Sub-product'] == subproduct)], 
                      order = complaints[(complaints.Product == product) & 
                                                   (complaints['Sub-product'] == subproduct)]['Issue'].value_counts().index)
    g.set_xlabel("Issues related to {Product}".format(Product=product), fontsize=25)
    g.set_ylabel("Count", fontsize=25)


companies = complaints.groupby('Company').Company.count().sort_values(ascending=False)
print(len(companies))
companies.head()
# top 15 companies with largest number of complaints.
plt.figure(figsize = (25,8))
plt.xticks(rotation=50, fontsize=15, ha="right")
plt.yticks(fontsize=15)
g = sns.barplot(companies.index[0:15], companies.values[0:15])
g.set_xlabel("Worst 15 Companies with largest number of complaints", fontsize=25)
g.set_ylabel("Count", fontsize=25)

c = widgets.ToggleButtons(
    options= companies.index[0:6],
    description='Product:',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
#     icons=['check'] * 3
)

@interact(company = c)
def plot_companies(company="EQUIFAX, INC."):
    issue_company = (complaints[complaints['Company']==company].groupby("Issue").
                                Issue.count().sort_values(ascending=False)[0:10])
    plt.figure(figsize = (15,8))
    plt.xticks(rotation=50, fontsize=15, ha="right")
    plt.yticks(fontsize=15)
    g = sns.barplot( issue_company.index, issue_company.values)
    g.set_xlabel("Issues with {company}".format(company=company), fontsize=25)
    g.set_ylabel("Count", fontsize=25)

# what was the company response
plt.figure(figsize = (15,6))
plt.xticks(rotation="50", ha="right", size = 10)
g = sns.countplot(x ="Company Response to Consumer", data = complaints, 
                  order = complaints['Company Response to Consumer'].value_counts().index)
g.set_xlabel("Response to Consumer complaint", fontsize=25)
g.set_ylabel("Count", fontsize=25)
c = widgets.ToggleButtons(
    options= companies.index[0:6],
    description='Product:',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
#     icons=['check'] * 3
)
@interact(company = c)
def plot_response(company = "EQUIFAX, INC."):
    plt.figure(figsize = (15,6))
    plt.xticks(rotation="50", ha="right", size = 10)
    g = sns.countplot(x ="Company Response to Consumer", 
                      data = complaints[complaints.Company==company],
                      order = complaints[complaints.Company==company]["Company Response to Consumer"].value_counts().index)
    g.set_xlabel("{company} response to Consumer complaint".format(company = company), fontsize=25)
    g.set_ylabel("Count", fontsize=25)
!wget http://www2.census.gov/geo/tiger/GENZ2017/shp/cb_2017_us_state_500k.zip -O state.zip
!unzip state.zip
import geopandas as gpd
states_2_rm = ["AK", "PR", "GU", "MP", "VI", "AS","HI"]
map_df = gpd.read_file('cb_2017_us_state_500k.shp')

#select only continuous us states
map_df = map_df[~map_df['STUSPS'].isin(states_2_rm)]

#plot of the us state boundary map
fig, ax = plt.subplots(1, figsize=(15, 8))
map_df.plot(ax=ax)
v=map_df.apply(lambda x: ax.annotate(s=x.NAME, xy=x.geometry.centroid.coords[0], ha='center'),axis=1)
# top 20 states with most number of complaints
gp_state = complaints.groupby('State')
plt.figure(figsize = (15,8))
plt.xticks(rotation="50", ha="right", size = 10)
state_count = gp_state.State.count().sort_values(ascending=False)
sns.barplot(state_count.index[0:20], state_count.values[0:20])
merged_df = map_df.merge(pd.Series.to_frame(gp_state.State.count()), 
                         left_on="STUSPS", right_index=True, how="left")
fig, ax = plt.subplots(1, figsize=(15, 8))
sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(vmin=0.0, vmax=143662))
sm._A = []
merged_df.plot(column = "State", ax = ax, cmap="Blues", linewidth=1.0, edgecolor='black')
fig.colorbar(sm)
# let's check distribution of  top 4 products across states
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, ax  = plt.subplots(4, figsize=(10, 25), sharex=True, sharey=True)
def prod2state( ax, product="Mortgage"):
    gp_state = complaints[complaints.Product==product].groupby('State').State.count()
    vmin = gp_state.min()
    vmax = gp_state.max()
    merged_df = map_df.merge(pd.Series.to_frame(gp_state), 
                         left_on="STUSPS", right_index=True, how="left")
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(vmin = vmin, vmax = vmax))
    sm._A = []
    merged_df.plot(column = "State", ax = ax, cmap="Blues", linewidth=1.0, edgecolor='black')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(sm, cax=cax)
    ax.set_title(product)

#fig.tight_layout(w_pad=10)
prod2state(ax[0], 'Mortgage')
prod2state(ax[1], "Debt collection")
prod2state(ax[2], 'Credit reporting, credit repair services, or other personal consumer reports')
prod2state(ax[3], 'Credit card or prepaid card')
companies.index[0:6]
fig, ax  = plt.subplots(6, figsize=(10, 35), sharex=True, sharey=True)
def prod2state( ax, company):
    gp_state = complaints[complaints.Company==company].groupby('State').State.count()
    vmin = gp_state.min()
    vmax = gp_state.max()
    merged_df = map_df.merge(pd.Series.to_frame(gp_state), 
                         left_on="STUSPS", right_index=True, how="left")
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(vmin = vmin, vmax = vmax))
    sm._A = []
    merged_df.plot(column = "State", ax = ax, cmap="Blues", linewidth=1.0, edgecolor='black')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(sm, cax=cax)
    ax.set_title(company)
prod2state(ax[0], 'EQUIFAX, INC.')
prod2state(ax[1], "BANK OF AMERICA, NATIONAL ASSOCIATION")
prod2state(ax[2], 'Experian Information Solutions Inc.')
prod2state(ax[3], 'TRANSUNION INTERMEDIATE HOLDINGS, INC.')
prod2state(ax[4], 'WELLS FARGO & COMPANY')
prod2state(ax[5], 'JPMORGAN CHASE & CO.')