## Import required libraries

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from matplotlib.colors import ListedColormap

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

import plotly.offline as pyo

import plotly.graph_objects as go

import plotly.express as px

pyo.init_notebook_mode()



%matplotlib inline

sns.set()

titleid=0
# #@title Enter dataset file path in drive

# #@markdown Enter the full data path (e.g: /content/drive/My Drive/DS 200/)



# mount_point = './'  #@param {type: "string"}

# data_dir_path = './'  #@param {type: "string"}

# is_zip_file = True  #@param {type: "boolean"}

# file_name = 'data-science-for-good-kiva-crowdfunding.zip'  #@param {type: "string"}

# #@markdown ---

# # load data provided by Kiva

# dir_path = 'data-science-for-good-kiva-crowdfunding/' # local path

# if use_colab:

#     from google.colab import drive

#     drive.mount(mount_point)
# if is_zip_file:

#   import zipfile

#   zip_ref = zipfile.ZipFile(f'{data_dir_path}/{file_name}', 'r')

#   zip_ref.extractall(dir_path)

#   zip_ref.close()
dir_path = '../input/data-science-for-good-kiva-crowdfunding/'
loans = pd.read_csv(dir_path + 'kiva_loans.csv')

mpi_region_location = pd.read_csv(dir_path + 'kiva_mpi_region_locations.csv')

theme_ids = pd.read_csv(dir_path + 'loan_theme_ids.csv')

themes_by_region = pd.read_csv(dir_path + 'loan_themes_by_region.csv')
loans.head(3)
loans.shape
loans.describe()
loans.columns
plt.figure(figsize=(15,10))

topK = 20

loan_amount_country = loans.groupby('country').sum()['funded_amount'].sort_values(ascending=False)

sns.barplot(y = loan_amount_country.head(topK).index, x=loan_amount_country.head(topK));

plt.ylabel('Country')

plt.xlabel('Total Funded Amount (USD 10M)')

titleid += 1

plt.title(f'Figure {titleid}: Top {topK} countries sorted by Total Funded Amount');
plt.figure(figsize=(15,10))

loan_amount_country = loans.groupby('country').mean()['funded_amount'].sort_values(ascending=False)

sns.barplot(y = loan_amount_country.head(topK).index, x=loan_amount_country.head(topK));



plt.ylabel('Country')

plt.xlabel('Average Funded Amount (USD 10M)')

titleid += 1

plt.title(f'Figure {titleid}: Top {topK} countries sorted by Total Funded Amount');
loans.query(r'country == "Cote D\'Ivoire"')
plt.figure(figsize=(15,10))

useful_countries = list(loans.groupby('country').count().query('id >= 5').index)

loan_amount_country = loans.groupby('country').mean()['funded_amount'][useful_countries].sort_values(ascending=False)

sns.barplot(y = loan_amount_country.head(topK).index, x=loan_amount_country.head(topK));



plt.ylabel('Country')

plt.xlabel('Average Funded Amount (USD 10M)')

titleid += 1

plt.title(f'Figure {titleid}: Top {topK} countries sorted by Total Funded Amount (Without Outliers)');
topK = 40

mpis_country = mpi_region_location.groupby('country').mean()['MPI'].dropna()

countries = loan_amount_country.head(topK)

common_countries = mpis_country.index.intersection(countries.index)

df = pd.DataFrame({'mpis': mpis_country[common_countries], 'amount': countries[common_countries]} )

sns.lmplot(x='mpis', y='amount', data=df)

titleid += 1

plt.title(f'Figure {titleid}: Relationship between funded_amount and MPI');
## Let's take the top 20 countries with total amount of funding

top_countries_list = list(loans.groupby('country').sum()['funded_amount'].sort_values(ascending=False).head(10).index)

top_country_loans = loans.query(f'country in {top_countries_list}').copy()

top_country_loans.head()

# We seen some NaN values in the region and tags columns

for col in top_country_loans.columns:

  print(f'There are {len(top_country_loans[top_country_loans[col].isnull()])} records with NaN values in {col}.')
import datetime

top_country_loans.loc[:, 'posted_time'] = pd.to_datetime(top_country_loans.posted_time)

## Lets add the year column separately to track growth over time.

## Parse datetime and set the day to the 1st of every month and club amounts together for easier plotting

top_country_loans.loc[:, 'posted_month_year'] = top_country_loans.posted_time.apply(lambda x: datetime.datetime(x.year, x.month, 1)) 



loan_data = top_country_loans.groupby(['posted_month_year', 'country']).sum()['funded_amount'].reset_index().sort_values(by='posted_month_year')

loan_data
fig = go.Figure()



for index, country in enumerate(loan_data.country.unique()):

  query = f"country == '{country}'"

  fig.add_trace(

      go.Scatter(

          y=loan_data.query(query)['funded_amount'], 

          x=loan_data.query(query)['posted_month_year'], 

          text=country,

          mode='lines',

          name=country,

          visible="legendonly" if index <= 4 else True

      )

  )



fig.update_layout(

      width=1200,

      height=1000,

      autosize=True,

      template="plotly_white",

  )



titleid += 1

fig.update_layout(

    title=f"Figure {titleid}: Funded Amount in top 10 countries over time",

    xaxis_rangeslider_visible=True,

    xaxis_title="Year",

    yaxis_title="Loan Amount in USD"

)



fig.show()
top_country_loans[top_country_loans.posted_time.dt.year == 2017][top_country_loans.posted_time.dt.month == 7]
sector_loans = loans.groupby('sector').sum()[['funded_amount', 'loan_amount']].reset_index()

sector_loans
from plotly.subplots import make_subplots



fig = go.Figure(data=[go.Pie(labels=sector_loans.sector, values=sector_loans.funded_amount)])

    

titleid += 1

fig.update_layout(dict(

    title=f'Figure {titleid}: Funded Loan Amount per Sector',

    width=800,

    height=800

))



fig.show()
top_country_sector_loans = top_country_loans.groupby(['sector', 'country']).sum()['funded_amount'].reset_index()

top_country_sector_loans
fig = go.Figure(

    data=[

          go.Pie(

              labels=top_country_sector_loans.query("country == 'Paraguay'").sector, 

              values=top_country_sector_loans.query("country == 'Paraguay'").funded_amount

              )

          ]

        )



titleid += 1  

fig.update_layout(dict(

    title=f'Figure {titleid}: Funded Loan Amount per Sector',

    width=800,

    height=800

))



fig.update_layout(

    updatemenus=[

        go.layout.Updatemenu(

            buttons=list([

                dict(

                    args=[

                        {

                            "labels": [top_country_sector_loans.query(f"country == '{country}'").sector], 

                            "values": [top_country_sector_loans.query(f"country == '{country}'").funded_amount],

                            "title": [

                                f'Funded Loan Amount per Sector for {country}'

                            ]

                        }],

                    label=country,

                    method="restyle"

                ) for country in list(top_country_sector_loans.country.unique())

            ]),

            direction="down",

            pad={"r": 10, "t": 10},

            showactive=True,

            x=0.1,

            xanchor="left",

            y=1.8,

            yanchor="top"

        ),

    ]

)



fig.show()
loans.query("repayment_interval == 'irregular'").head(2)
repayment_data = loans.groupby(['repayment_interval', 'country', 'sector', 'activity']).sum()['funded_amount'].reset_index()

repayment_data
fig = go.Figure(

    data=go.Splom(

        dimensions=[

                    dict(

                        label=label,

                         values=list(repayment_data[label])

                        ) for label in list(repayment_data.columns)

                    ]

))



titleid += 1

fig.update_layout(

    title=f'Figure {titleid}: Loan Repayment Interval',

    dragmode='select',

    width=1200,

    height=1200,

    hovermode='closest',

)



fig.show()
# for col in list(repayment_data.iloc[:, 1:].columns:

payment_interval_dist = repayment_data.groupby(['country', 'repayment_interval']).count()['sector'].reset_index()

payment_interval_dist
fig = go.Figure(

    data=[

          go.Pie(

              labels=payment_interval_dist.repayment_interval, 

              values=payment_interval_dist.sector

              )

          ]

        )

    

titleid += 1

fig.update_layout(dict(

    title=f'Figure {titleid}: Distribution of Repayment Intervals for Each Country',

    width=800,

    height=800

))



fig.update_layout(

    updatemenus=[

        go.layout.Updatemenu(

            buttons=list([

                dict(

                    args=[

                        {

                            "labels": [payment_interval_dist.query(f'country == "{country}"').repayment_interval], 

                            "values": [payment_interval_dist.query(f'country == "{country}"').sector],

                            "title": [

                                f'Funded Loan Amount per Repayment Interval for {country}'

                            ]

                        }],

                    label=country,

                    method="restyle"

                ) for country in list(payment_interval_dist.country.unique())

            ]),

            direction="down",

            pad={"r": 10, "t": 10},

            showactive=True,

            x=0.1,

            xanchor="left",

            y=1.8,

            yanchor="top"

        ),

    ]

)



fig.show()
# for col in list(repayment_data.iloc[:, 1:].columns:

payment_interval_dist_sector = repayment_data.groupby(['sector', 'repayment_interval']).count()['country'].reset_index()

payment_interval_dist_sector.head(10)
fig = go.Figure(

    data=[

          go.Pie(

              labels=payment_interval_dist_sector.repayment_interval, 

              values=payment_interval_dist_sector.country

              )

          ]

        )

    

titleid += 1

fig.update_layout(dict(

    title=f'Figure {titleid}: Distribution of Repayment Intervals for Each Sector',

    width=800,

    height=800

))



fig.update_layout(

    updatemenus=[

        go.layout.Updatemenu(

            buttons=list([

                dict(

                    args=[

                        {

                            "labels": [payment_interval_dist_sector.query(f'sector == "{sector}"').repayment_interval], 

                            "values": [payment_interval_dist_sector.query(f'sector == "{sector}"').country],

                            "title": [

                                f'Funded Loan Amount per Repayment Interval Type for {sector}'

                            ]

                        }],

                    label=sector,

                    method="restyle"

                ) for sector in list(payment_interval_dist_sector.sector.unique())

            ]),

            direction="down",

            pad={"r": 10, "t": 10},

            showactive=True,

            x=0.1,

            xanchor="left",

            y=1.8,

            yanchor="top"

        ),

    ]

)



fig.show()
loan_amount_per_country = loans.groupby(['country', 'country_code']).agg(['sum', 'count'])['funded_amount'].reset_index().set_index('country')

loan_amount_per_country['sum'] =loan_amount_per_country['sum'].astype(int)

loan_amount_per_country
# # The country codes are not in ISO Alpha format, let's set these.



iso_list = mpi_region_location.groupby(['ISO', 'country']).mean()['MPI'].reset_index().set_index('country')



loan_amount_per_country['country_code'] = iso_list['ISO']

## Let's drop the countries without any country codes

## For the countries with NaN values, set the code to the first three letters.

## Future: maybe scrape wikipedia/un sites to get info. Currently the data is not easily scrapped.



loan_amount_per_country.loc[loan_amount_per_country.country_code.isnull(), 'country_code'] = loan_amount_per_country[loan_amount_per_country.country_code.isnull()].apply(lambda row: row.name[:3].upper(), axis=1)

loan_amount_per_country
titleid += 1

fig = px.scatter_geo(loan_amount_per_country.reset_index(), locations="country_code", color="country",

                     hover_name="country", size="sum",

                     projection="natural earth",

                     title=f"Figure {titleid}: Geo-Visualization of Funded Amount")

fig.show()
"""

Let's now analyze the number of loans, the funded amount and the MPI of countries.

"""

loan_amount_per_country['MPI'] = iso_list['MPI']

loan_amount_per_country = loan_amount_per_country.dropna() ## Drop NaN MPI values to not skew the plot



loan_amount_per_country.head(10)
# # # ## Analyze class size and cost of attendance

fig = px.scatter(

    loan_amount_per_country.reset_index(), 

    x="MPI",

    y="sum", 

    color="country", 

    hover_name="country",

    size="count"

)



titleid += 1

fig.update_layout(

    height=900,

    width=900,

    xaxis_title="MPI",

    yaxis_title="Total Funded Amount",

    title=f'Figure {titleid}: MPI vs Funded Amount vs Number of Funded Loans'

)

fig.show()
country_peer_network = loans.groupby(['country', 'country_code']).agg(['sum', 'count'])[['funded_amount', 'lender_count']].reset_index().iloc[:, :5]

country_peer_network.head(3)
## Let's squash the column levels

country_peer_network['funded_amount_sum'] = country_peer_network['funded_amount']['sum']

country_peer_network['funded_amount_count'] = country_peer_network['funded_amount']['count']

country_peer_network['lender_count_sum'] = country_peer_network['lender_count']['sum']



country_peer_network.head(10)
# # # ## Analyze class size and cost of attendance

fig = px.scatter(

    country_peer_network, 

    x="lender_count_sum",

    y="funded_amount_sum", 

    color="country", 

    hover_name="country",

    size="funded_amount_count"

)



titleid += 1

fig.update_layout(

    height=900,

    width=900,

    xaxis_title="Total Lender Count",

    yaxis_title="Total Funded Amount",

    title=f'Figure {titleid}: Lender Count vs Funded Amount vs Number of Funded Loans'

)

fig.show()
country_loans = pd.merge(

    loans, 

    mpi_region_location, 

    how='inner', 

    on='country'

    )[['country','MPI', 'lat', 'lon', 'lender_count', 'funded_amount']].groupby('country').agg(

    {

      'lat': 'mean', 

      'lon': 'mean', 

      'lender_count': 'sum', 

      'funded_amount': 'sum', 

      'MPI': 'mean'

    }).reset_index()

country_loans = country_loans.dropna()

country_loans.head(2)
titleid += 1

fig = px.scatter(

    country_loans, 

    x='lat', 

    y='lon',

    title=f'Figure {titleid}: Region vs total lender count',

    size="lender_count",

    hover_name="country"

)

fig.show()



titleid += 1



fig = px.scatter(

    country_loans, 

    x='lat', 

    y='lon',

    title=f'Figure {titleid}: Region vs total average MPI',

    size="MPI",

    hover_name="country"

)



fig.show()
from sklearn.cluster import KMeans



kmeans = KMeans(n_clusters=10, random_state=0).fit(country_loans[['lat', 'lon']])



country_loans['cluster'] = kmeans.labels_
from sklearn.metrics import silhouette_score



silhouette_score(country_loans[['lat', 'lon']], kmeans.predict(country_loans[['lat', 'lon']]))
## Let's get the optimal number of clusters

geo_labels = ['lat', 'lon']



for n_clusters in range(5, 15):

    labels = KMeans(n_clusters=n_clusters, random_state=10).fit_predict(country_loans[geo_labels])

    silhouette_avg = silhouette_score(country_loans[geo_labels], labels)



    print(f"For n_clusters = {n_clusters}, The average silhouette_score is : {silhouette_avg}")
## We see that the optimal n_clusters is 12

kmeans = KMeans(n_clusters=12).fit(country_loans[['lat', 'lon']])



country_loans['cluster'] = kmeans.labels_
titleid += 1

fig = px.scatter(

    country_loans, 

    x='lat', 

    y='lon',

    title=f'Figure {titleid}: Clustered countries vs total lender count',

    size="lender_count",

    hover_name="country",

    color='cluster'

)

fig.show()
## We now analyze the trend between clusters, MPI and funded_amounts

clustered_loans = country_loans.groupby(['cluster']).mean().reset_index()

clustered_loans
titleid += 1

px.scatter(

    clustered_loans, 

    x='MPI', 

    y='funded_amount',

    color='cluster', 

    trendline='ols', 

    title=f'Figure {titleid}: Average Funded Amount vs Average MPI',

    labels={

        'MPI': 'Average Multidimensional Poverty Index (MPI)',

        'funded_amount':'Average Funded Amount'

    }

    )
country_loans.query('cluster == 2').country
country_loans.query('cluster == 4')
display(loans.isnull().sum())

display(mpi_region_location.isnull().sum())
loans = loans.dropna()

mpi_region_location = mpi_region_location.dropna()
# data design

mpi_data = mpi_region_location.groupby(['country', 'region']).mean()['MPI']

loans_data = loans.groupby(['country', 'region']).mean().drop(columns=['id', 'partner_id'])

data_index = mpi_data.index.intersection(loans_data.index)
# loss function

def l2(pred, true):

    return ((pred - true) ** 2).mean()
loans_data
model = LinearRegression()



X_0 = loans_data.loc[data_index]

y_0 = mpi_data.loc[data_index]
# train

def train_linear_regression(X, y):

    model = LinearRegression()



    kf = KFold(n_splits=5, shuffle=True)

    train_loss = []

    val_loss = []

    for train_id, val_id in kf.split(X):

        X_train = X.iloc[train_id]

        y_train = y.iloc[train_id]

        X_val = X.iloc[val_id]

        y_val = y.iloc[val_id]

        model.fit(X_train, y_train)

        train_loss.append(l2(model.predict(X_train), y_train))

        val_loss.append(l2(model.predict(X_val), y_val))

    print('training loss = {}\nvalidation loss = {}'.format(np.mean(train_loss),np.mean(val_loss)))      
# train

train_linear_regression(X_0, y_0)

      

# use all data to train

model.fit(X_0, y_0)

plt.figure(figsize=(14, 14))

for i in range(X_0.shape[1]):

    plt.subplot(2, 2, i + 1)

    plt.xlabel(X_0.columns[i])

    plt.ylabel('residual')

    plt.scatter(X_0.iloc[:, i], y_0 - model.predict(X_0))

titleid += 1

plt.suptitle(f'Figure {titleid}: Residual Plots');
# data design

# y: loan_amount

# X: mpi, sector(dummy encoding)

X_1 = loans[['country', 'region', 'loan_amount']]

X_1 = pd.concat([X_1, pd.get_dummies(loans['sector'])], axis=1) # one hot encode sector values

X_1 = X_1.merge(mpi_region_location[['country', 'region', 'MPI']])

y_1 = X_1['loan_amount']

X_1 = X_1.drop(columns=['country', 'region', 'loan_amount'])
data = loans[['country', 'region', 'loan_amount', 'sector']]

data = data.merge(mpi_region_location[['country', 'region', 'MPI']])

plt.figure(figsize=(10, 8))

titleid += 1

plt.title(f'Figure {titleid}: Loan Amount Distribution by Sector and MPI')

sns.scatterplot(x='MPI', y='loan_amount', hue='sector', data=data);
plt.figure(figsize=(10, 8))

sectors = data['sector'].value_counts().head(5).index.to_series(name='sector')

titleid += 1

plt.title(f'Figure {titleid}: Loan Amount(<=10000) Distribution by Major Sectors and MPI')

sns.scatterplot(x='MPI', y='loan_amount', hue='sector', data=data.merge(sectors).query('loan_amount <= 10000'));
train_linear_regression(X_1, y_1)
def train_random_forest_regressor(X, y, **params):

    model = RandomForestRegressor(**params)



    kf = KFold(n_splits=5, shuffle=True)

    train_loss = []

    val_loss = []

    for train_id, val_id in kf.split(X):

        X_train = X.iloc[train_id]

        y_train = y.iloc[train_id]

        X_val = X.iloc[val_id]

        y_val = y.iloc[val_id]

        model.fit(X_train, y_train)

        train_loss.append(l2(model.predict(X_train), y_train))

        val_loss.append(l2(model.predict(X_val), y_val))

    return train_loss, val_loss    
train_loss, val_loss = train_random_forest_regressor(X_1, y_1, n_estimators=10)

print('training loss = {}\nvalidation loss = {}'.format(np.mean(train_loss), np.mean(val_loss)))
def heatmap_random_forest_regressor(ns, depths, X, y):

    loss = np.zeros((len(ns), len(depths)))

    for i in range(len(ns)):

        for j in range(len(depths)):

            _, val_loss = train_random_forest_regressor(X, y, n_estimators=ns[i], criterion='mse', max_depth=depths[j])

            loss[i, j] = np.mean(val_loss)

    sns.heatmap(loss, cmap='YlGnBu', xticklabels=depths, yticklabels=ns)

    plt.xlabel('max_depth')

    plt.ylabel('n_estimators')

    global titleid

    titleid += 1

    plt.title(f'Figure {titleid}: Random Forest Regressor MSE')

    return loss
ns_reg = np.arange(1, 21, 5)

depths_reg = np.arange(1, 61, 5)

loss_forest_0 = heatmap_random_forest_regressor(ns_reg, depths_reg, X_1, y_1)
fine_ns_reg = np.arange(5, 41, 5)

fine_depths_reg = np.arange(7, 23, 3)

loss_forest_1 = heatmap_random_forest_regressor(fine_ns_reg, fine_depths_reg, X_1, y_1)
loss_forest_1.min()
# data design

X_logi = data[['MPI', 'loan_amount']]

y_logi = data['sector']
def train_logistic_regression(X, y):

    model = LogisticRegression(solver='liblinear', multi_class='auto')

    kf = KFold(n_splits=5, shuffle=True)

    train_acc = []

    val_acc = []

    for train_id, val_id in kf.split(X):

        X_train = X.iloc[train_id]

        y_train = y.iloc[train_id]

        X_val = X.iloc[val_id]

        y_val = y.iloc[val_id]

        model.fit(X_train, y_train)

        train_acc.append((model.predict(X_train) == y_train).mean())

        val_acc.append((model.predict(X_val) == y_val).mean())

    print('training accuracy = {}\nvalidation accuracy = {}'.format(np.mean(train_acc),np.mean(val_acc)))
train_logistic_regression(X_logi, y_logi)
def show_logistic_confusion_matrix(X, y):

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression(solver='liblinear', multi_class='auto')

    model.fit(X_train, y_train)

    global titleid

    titleid += 1

    plt.title(f'Figure {titleid}: Confusion Matrix for Logistic Regression')

    sns.heatmap(confusion_matrix(y_val, model.predict(X_val)), cmap='YlGnBu')



show_logistic_confusion_matrix(X_logi, y_logi)
def train_random_forest_classifier(X, y, heatmap=False, **params):

    model = RandomForestClassifier(**params)

    kf = KFold(n_splits=5, shuffle=True)

    train_acc = []

    val_acc = []

    for train_id, val_id in kf.split(X):

        X_train = X.iloc[train_id]

        y_train = y.iloc[train_id]

        X_val = X.iloc[val_id]

        y_val = y.iloc[val_id]

        model.fit(X_train, y_train)

        train_acc.append((model.predict(X_train) == y_train).mean())

        val_acc.append((model.predict(X_val) == y_val).mean())

        if heatmap:

            print(model.classes_)

            sns.heatmap(confusion_matrix(y_val, model.predict(X_val)), cmap='YlGnBu')

        plt.show()

    return train_acc, val_acc
train_acc, val_acc = train_random_forest_classifier(X_logi, y_logi, n_estimators=10, criterion='gini')

print('training accuracy = {}\nvalidation accuracy = {}'.format(np.mean(train_acc),np.mean(val_acc)))
def draw_decisicon_boundary(X, y, model):

    sns_cmap = ListedColormap(np.array(sns.color_palette('Paired'))[:, :])



    xx, yy = np.meshgrid(np.linspace(X.iloc[:, 0].min(), X.iloc[:, 0].max(), 30), np.linspace(X.iloc[:, 1].min(), X.iloc[:, 1].max(), 30))

    Z_string = model.predict(np.c_[xx.ravel(), yy.ravel()])

    categories, Z_int = np.unique(Z_string, return_inverse = True)

    Z_int = Z_int.reshape(xx.shape)

    cs = plt.contourf(xx, yy, Z_int, cmap = sns_cmap)

    proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) for pc in cs.collections]

    plt.legend(proxy, categories)

    plt.xlabel('MPI')

    plt.ylabel('funded amount')

    global titleid

    titleid += 1

    plt.title(f'Figure {titleid}: Decision Boundaries of Random Forest Classifier');
def show_forest_confusion_matrix(X, y):

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(n_estimators=10, criterion='gini')

    model.fit(X_train, y_train)

    plt.figure(figsize=(10, 15))

    plt.subplot(2,1,1)

    global titleid

    titleid += 1

    plt.title(f'Figure {titleid}: Confusion Matrix for Random Forest Classifier')

    sns.heatmap(confusion_matrix(y_val, model.predict(X_val)), cmap='YlGnBu')

    plt.subplot(2,1,2)

    draw_decisicon_boundary(X, y, model)



show_forest_confusion_matrix(X_logi, y_logi)
def heatmap_random_forest_classifier(ns, depths):

    acc = np.zeros((len(ns), len(depths)))

    for i in range(len(ns)):

        for j in range(len(depths)):

            _, val_acc = train_random_forest_classifier(X_logi, y_logi, n_estimators=ns[i], criterion='gini', max_depth=depths[j])

            acc[i, j] = np.mean(val_acc)

    sns.heatmap(acc, cmap='YlGnBu', xticklabels=depths, yticklabels=ns)

    plt.xlabel('max_depth')

    plt.ylabel('n_estimators')

    global titleid

    titleid += 1

    plt.title(f'Figure {titleid}: Random Forest Classifier Accuracy')

    return acc
ns = np.arange(1, 42, 10)

depths = np.arange(8, 25, 5)

acc_forest_0 = heatmap_random_forest_classifier(ns, depths)
acc_forest_0.max()
model = RandomForestClassifier(n_estimators=25, max_depth=15, criterion='gini')

X_train, _ , y_train, _ = train_test_split(X_logi, y_logi, test_size=0.2)

model.fit(X_train, y_train)

plt.figure(figsize=(10, 10))

draw_decisicon_boundary(X_logi, y_logi, model)
loans.columns
## Clean gender column to keep a single gender value, which is the first gender value

loans.borrower_genders = loans.borrower_genders.str.split(", ").apply(lambda x: list(set(x))[0])
## We know there was a linear relationship between lender_count and funded_amount, let's analyze the slopes of these

## and split based on certain columns

def split_show_linear(df, separator, use_plotly=False):

    temp_df = df.groupby([separator, 'lender_count']).sum()['funded_amount'].reset_index()

    if use_plotly:

        global titleid

        titleid += 1

        fig = px.scatter(temp_df, x="lender_count", y="funded_amount", facet_col=separator, color=separator, trendline="ols",

                         title=f'Figure {titleid}')

        fig.show()

    else:

        sns.lmplot(x="lender_count", y="funded_amount", hue=separator, data=temp_df)

    

separator='borrower_genders'

split_show_linear(loans, separator, True)
#The data seems overplottled for low lender counts, let's group this into two segments

separator='borrower_genders'

split_show_linear(loans.query('lender_count < 200'), separator, True)
separator='borrower_genders'

split_show_linear(loans.query('lender_count >= 200'), separator, True)
titleid += 1

px.bar(

    loans.groupby(['borrower_genders', 'sector']).sum()['funded_amount'].reset_index(),

    x='borrower_genders',

    y='funded_amount',

    color='sector',

    barmode='group',

    title=f'Figure {titleid}: Gender specific funded amounts per sector'

)
## Let's analyze the spread of repayment interval with respect to funded_amounts and sectors

plt.figure(figsize=(20,10));

temp_df = loans.groupby(['repayment_interval', 'sector']).sum()['funded_amount'].reset_index()

titleid += 1

px.bar(

    temp_df, 

    color='sector', 

    y='funded_amount', 

    x='repayment_interval', 

    barmode='group',

    title=f'Figure {titleid}: Loan Repayment Intervals vs Funded Amounts per sector',

)
## We join two dataframes to get geo location values as well.

mpi_region_location.country = mpi_region_location.country.astype('str')

loans.country = loans.country.astype('str')

full_data = pd.merge(loans, mpi_region_location, on='country', how='left')
numerical_features = ['MPI', 'lat', 'lon', 'funded_amount', 'loan_amount', 'term_in_months', 'lender_count']

categorical_features = ['sector', 'activity', 'country', 'region_x', 'currency']



features = numerical_features + categorical_features

features
## Let's split the data into train and test

outcome_variable = 'repayment_interval'

full_data = full_data[features + [outcome_variable]] ## Select only certain starting features

full_data = full_data.dropna()

X_train, X_test, Y_train, Y_test = train_test_split(full_data[features], full_data[outcome_variable])
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

## Clean and transform data

scaler = StandardScaler()

centered_data = scaler.fit_transform(X_train[numerical_features])



pca = PCA()

pca.fit(centered_data)
fig, ax = plt.subplots(1, figsize=(6,5))



# plot explained variance as a fraction of the total explained variance

ax.plot(pca.explained_variance_ratio_)



# mark the 1th principal component

ax.axvline(5, c='k', linestyle='--')



ax.set_xlabel('PC index')

ax.set_ylabel('% explained variance')

titleid += 1

ax.set_title(f'Figure {titleid}: Scree plot')

fig.tight_layout()
sns.barplot(x=numerical_features, y=pca.components_[0])

titleid += 1

plt.title(f'Figure {titleid}: 1st PC')

plt.xticks(rotation=90);
sns.barplot(x=numerical_features, y=pca.components_[1])

titleid += 1

plt.title(f'Figure {titleid}: 2nd PC')

plt.xticks(rotation=90);
sns.barplot(x=numerical_features, y=pca.components_[-1])

titleid += 1

plt.title(f'Figure {titleid}: last PC')

plt.xticks(rotation=90);
numerical_features.remove('loan_amount')



features = numerical_features + categorical_features

features
def one_hot_encode(dataframe, categorical_features):

  train = pd.DataFrame()

  for feature in categorical_features:

    train = pd.concat([train, pd.get_dummies(dataframe[categorical_features])], axis=1)



  return train
### Note: Collab RAM crashes because one hot encoding region and currency takes a lot of RAM

## let's reduce the variables and remove region, activity and currency

categorical_features = ['country', 'sector']

one_hot_encoded = one_hot_encode(X_train, categorical_features)
def merge_category_numerical(dataframe, numerical_features, one_hot_encoded_matrix):

  return pd.concat([dataframe[numerical_features], one_hot_encoded_matrix], axis=1)
X_concat = merge_category_numerical(X_train, numerical_features, one_hot_encoded)
## We convert the multi class problem into a binary class problem since

## the characteristics for one time and irregular payments would largely be the 

## same.

def merge_classes(Y):

  return Y.apply(lambda x: 'regular' if x == 'monthly' else 'irregular')



Y_train = merge_classes(Y_train)

Y_test = merge_classes(Y_test)
#@title Logistic Classification - Set Hyperparameters

#@markdown Set the hyperparameters for the Logistic Classification model



penalty = 'l2' #@param ["l2", "l1"] {allow-input: true}

penalty_weight =   10#@param {type: "number"}

class_weight = 'balanced' #@param ["balanced", "None"] {allow-input: true}

#@markdown ---

hyperparameters = {

    'penalty': penalty,

    'C': penalty_weight,

    'class_weight': None if class_weight == 'None' else class_weight

}
def get_model(hyperparameters):

  print(f'Setting hyperparameters: {hyperparameters}')

  return LogisticRegression(solver='liblinear', **hyperparameters)
def create_sparse_dataframe(dataframe):

  return dataframe.to_sparse()
X_concat_sparse = create_sparse_dataframe(X_concat)
def train_logistic_classifier_payment_interval(X_train, Y_train, params):

  kf = KFold(n_splits=5, shuffle=False)

  train_acc = []

  val_acc = []

  class_model = get_model(params)

  for train, test in kf.split(X_train):

    class_model.fit(X_train.iloc[train], Y_train.iloc[train])

    train_acc.append(class_model.score(X_train.iloc[train], Y_train.iloc[train]))

    val_acc.append(class_model.score(X_train.iloc[test], Y_train.iloc[test]))



  print(f'validation accuracy: {np.mean(val_acc)}, {np.mean(train_acc)}')

  return class_model
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

class_model = train_logistic_classifier_payment_interval(X_concat_sparse, Y_train, hyperparameters)
# Run on test set

X_test_concat = merge_category_numerical(X_test, numerical_features, one_hot_encode(X_test, categorical_features))
metric = confusion_matrix(Y_test, class_model.predict(X_test_concat))
fig = go.Figure(

      data=go.Heatmap(

            z=metric ,

            x=['predicted irregular', 'predicted regular'],

            y=['actual irregular', 'actual regular']

          )

    )

titleid += 1

fig.update_layout(

    title=f"Figure {titleid}: Confusion Matrix for predicted classes"

)

fig.show()