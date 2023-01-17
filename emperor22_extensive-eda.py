import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set(style='ticks')

import numpy as np

import string
df = pd.read_csv('../input/consumer-complaints-financial-products/Consumer_Complaints.csv')

df.head(20)
df.info()
def prettify_column_name(col):

    punctuation_stripped = col.translate(str.maketrans('', '', string.punctuation))

    col__ = punctuation_stripped.replace(' ', '_')

    return col__.lower()



df.columns = [prettify_column_name(i) for i in df.columns]
def missing_value(df=df):

    total = df.isnull().sum()

    pct = df.isnull().sum() * 100 / df.shape[0]

    missing = pd.concat([total, round(pct, 1)], axis=1, keys=['Total', 'Percentage'])

    return missing.sort_values(by='Percentage', ascending=False)



missing_value()
def total_unique_values(df=df):

    cols = pd.Series(df.columns.values)

    len_unique = pd.Series([len(df[i].unique()) for i in df.columns.values])

    unique = pd.concat([cols, len_unique], axis=1, keys=['Column', 'Total Unique Values'])

    return unique.sort_values(by='Total Unique Values', ascending=False)



total_unique_values()
# seeing unique values of variables with least total unique values starting from product

unique = total_unique_values()



for product in unique['Column'].iloc[10:]:

    print(f"""Unique values for {product.upper()}: 

    {df[product].unique()}

    """)
# let's see the subproducts for each product



for product in df['product'].unique():

    a = df.loc[df['product'] == product]

    print(f"""Subproducts of {product.upper()}:

    {a['subproduct'].unique()}

    """)
# Total complaints per company (percentage)





fig_dims = (5, 10)

fig, ax = plt.subplots(figsize=fig_dims)

df['company'].value_counts(normalize=True).sort_values(ascending=True)[-50:].plot(ax=ax, kind='barh')

ax.set_title('Percentage of complaints per company (Top 50)')
def plot_frequency(col, dimension=(10, 5)):

    fig_dims = dimension

    fig, ax = plt.subplots(figsize=fig_dims)

    df[col].value_counts(normalize=True).sort_values(ascending=True).apply(lambda x: x*100).plot(ax=ax, kind='bar', grid=True, yticks=range(0, 110, 10))

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    ax.set_title(f"Percentage of {col.replace('_', ' ')}")



def plot_frequency_h(col, dimension=(5,5)):

    fig_dims = dimension

    fig, ax = plt.subplots(figsize=fig_dims)

    df[col].value_counts(normalize=True).sort_values(ascending=True).apply(lambda x: x*100).plot(ax=ax, kind='barh', grid=True, xticks=range(0, 110, 10))

    ax.set_title(f"Percentage of {col.replace('_', ' ')}")

    #ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
# Total complaints per product (percentage)



plot_frequency_h('product', (5, 4))
plot_frequency('consumer_consent_provided')
plot_frequency('timely_response')
plot_frequency('consumer_disputed')
plot_frequency_h('company_public_response', (6, 5))
plot_frequency_h('company_response_to_consumer')
# See which companies have the highest dispute rates

df['consumer_disputed_'] = df['consumer_disputed'].apply(lambda x: True if x=='Yes' else False)

grp = df.groupby(['company'])

cd_avg = grp['consumer_disputed_'].mean()  #cd for consumer dispute
# See which companies have dispute rates above 80%



def dispute_rate_above_n_percent(pct):

    filt = cd_avg.loc[cd_avg.values >= pct/100]

    print(f"Percentage of company with {pct}% or above consumer dispute rate: {len(filt) * 100/len(df['company'].unique())}")

    print(f"Number of companies with {pct}% or above consumer dispute rate: {len(filt)}")

    companies = [i for i in filt.index]

    return companies, filt



companies, filt = dispute_rate_above_n_percent(80)

filt
# See which companies have dispute rates below 5%



def dispute_rate_below_n_percent(pct):

    filt = cd_avg.loc[cd_avg.values <= pct/100]

    print(f"Percentage of company with {pct}% or below consumer dispute rate: {len(filt) * 100/len(df['company'].unique())}")

    print(f"Number of companies with {pct}% or below consumer dispute rate: {len(filt)}")

    companies = [i for i in filt.index]

    return companies, filt



companies, filt = dispute_rate_below_n_percent(5)

filt
# See which companies have the highest 20% dispute rate



def highest_dispute_rate(pct):

    nth = np.percentile(cd_avg.values, 100-pct)

    print(f'{100-pct}th percentile of dispute rate: {nth}')

    return cd_avg.loc[cd_avg.values >= nth].sort_values(ascending=True)



highest_dispute_rate(20)
# See which companies have the lowest 20% dispute rate



def lowest_dispute_rate(pct):

    nth = np.percentile(cd_avg.values, pct)

    print(f'{pct}th percentile of dispute rate: {nth}')

    return cd_avg.loc[cd_avg.values <= nth].sort_values()



lowest_dispute_rate(50)
# See which companies have the 100% timely response rate



df['timely_response_'] = df['timely_response'].apply(lambda x: True if x=='Yes' else False)

grp2 = df.groupby(['company'])

tr_avg = grp2['timely_response_'].mean() #tr for timely response

filt = tr_avg.loc[tr_avg.values == 1]

print(f"Percentage of company with 100% timely response rate: {len(filt) * 100/len(df['company'].unique())}")

filt
# See which companies have 0% timely response rate



filt = tr_avg.loc[tr_avg.values == 0]

print(f"Percentage of company with 0% timely response rate: {len(filt) * 100/len(df['company'].unique())}")

filt
# See which companies have timely response rate above 90%



def timely_response_above_n_percent(pct):

    filt = tr_avg.loc[tr_avg.values >= pct/100].sort_values(ascending=True)

    print(f"Percentage of company with {pct}% or above timely response rate: {len(filt) * 100/len(df['company'].unique())}")

    print(f"Number of companies with {pct}% or above timely response rate: {len(filt)}")

    companies = [i for i in filt.index]

    return companies, filt



companies, filt = timely_response_above_n_percent(90)

filt
# See which companies have timely response rate below 20%



def timely_response_below_n_percent(pct):

    filt = tr_avg.loc[tr_avg.values <= pct/100].sort_values(ascending=True)

    print(f"Percentage of company with {pct}% or below timely response rate: {len(filt) * 100/len(df['company'].unique())}")

    print(f"Number of companies with {pct}% or below timely response rate: {len(filt)}")

    companies = [i for i in filt.index]

    return companies, filt



companies, filt = timely_response_below_n_percent(20)

filt
# See which companies have the highest 20% timely response rate

def highest_timely_response(pct):

    nth = np.percentile(tr_avg.values, 100-pct)

    print(f'{100-pct}th percentile of timely response rate: {nth}')

    return tr_avg.loc[tr_avg.values >= nth].sort_values(ascending=True)

   

highest_timely_response(20)
# See which companies have the lowest 20% timely response rate

def lowest_timely_response(pct):

    nth = np.percentile(tr_avg.values, pct)

    print(f'{pct}th percentile of timely response rate: {nth}')

    return tr_avg.loc[tr_avg.values <= nth].sort_values()



lowest_timely_response(20)
# See if there are common companies in both the list of lowest timely response rate

# and highest dispute rate



cd1, cd2 = dispute_rate_above_n_percent(90) #these are filled with 100% dispute rate companies

tr1, tr2 = timely_response_below_n_percent(20)



cd_set = set(cd1)

intersection = list(cd_set.intersection(tr1))

intersection
def plot_issues_per_product(product, df=df, top_n_issues=15):

    filt = df.loc[df['product'] == product]

    g = sns.countplot(y='issue', data=filt, order=filt['issue'].value_counts().iloc[0:top_n_issues].index)

    g.set_title(f'Issues with product {product}')

    return g



products = df['product'].unique()

for i in products:

    plot_issues_per_product(i)

    plt.show()
# Issues per subproduct



filt = df.loc[df['subproduct'].notnull(), ['product', 'issue', 'subproduct']]



for product in filt['product'].unique():

    filt2 = filt.loc[df['product'] == product]

    g = sns.catplot(y='issue', col='subproduct', data=filt2, kind='count')

    g.fig.subplots_adjust(top=0.9)

    g.fig.suptitle(f'Product: {product.upper()}', fontsize=15)

    plt.show()
# A utility function to gather information on general metrics for a company



def plot_company(company, add_height=0):

    cols = ['consumer_disputed', 'timely_response', 'product', 'company_public_response', 'company_response_to_consumer']

    fig, axes = plt.subplots(nrows=5, figsize=(10, 20+add_height))

    for index, value in enumerate(cols):

        df.loc[df['company'] == company][value].value_counts(normalize=True).apply(lambda x: x*100).plot(title=f"{company}'s percentage of {value.replace('_', ' ')}",kind='barh', ax=axes[index])



plot_company('Bank of America', 2)
# Utility functions to plot issues of a company's product



def get_column_unique_values(company, col):

    list_ = [i for i in df.loc[df['company'] == company][col].unique()]

    return list_



def plot_issues_per_product(company, product, add_height=0):

    title = f'Percentage of issues regarding {product} of {company}'

    fig, ax = plt.subplots(figsize=(5, 5+add_height))

    df.loc[(df['company'] == company) & (df['product'] == product)]['issue'].value_counts(normalize=True).apply(lambda x:  x*100).plot(kind='barh', ax=ax,title=title)
# You can use the first function to get all products of a company and then plot issues regarding that product

company='Bank of America'



get_column_unique_values(company, 'product')



plot_issues_per_product('Bank of America', 'Credit card', 5)