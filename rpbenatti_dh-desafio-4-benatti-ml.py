import numpy as np

import pandas as pd
base_url = '/kaggle/input/web-traffic-time-series-forecasting/'



key_1 = pd.read_csv(base_url+'key_1.csv')

train_1 = pd.read_csv(base_url+'train_1.csv')

sample_submission_1 = pd.read_csv(base_url+'sample_submission_1.csv')
print(train_1.shape, key_1.shape, sample_submission_1.shape)
train_1.info()
train_1.head()
# Creating a list of wikipedia main sites 

sites = ["wikipedia.org", "commons.wikimedia.org", "www.mediawiki.org"]



# Function to create a new column having the site part of the article page

def filter_by_site(page):

    for site in sites:

        if site in page:

            return site



# Creating a new column having the site part of the article page

train_1['Site'] = train_1.Page.apply(filter_by_site)
train_1['Site'].value_counts(dropna=False)
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()



plt.figure(figsize=(12, 6))

plt.title("Number of Wikipedia Articles by Sites", fontsize="18")

train_1['Site'].value_counts().plot.bar(rot=0);
# Checking which country codes exist in the article pages

train_1.Page.str.split(pat=".wikipedia.org", expand=True).iloc[:,0].str[-3:].value_counts().index.to_list()
# Creating a list of country codes

train_1.Page.str.split(pat=".wikipedia.org", expand=True).iloc[:,0].str[-2:].value_counts().index.to_list()[0:7]
# Checking which agents + access exist in the article pages and creating a list with them

train_1.Page.str.split(pat=".wikipedia.org", expand=True).iloc[:,1].str[1:].value_counts().index.to_list()
# Creating the list of country codes and agents

countries = train_1.Page.str.split(pat=".wikipedia.org", expand=True).iloc[:,0].str[-2:].value_counts().index.to_list()[0:7]

agents = train_1.Page.str.split(pat=".wikipedia.org", expand=True).iloc[:,1].str[1:].value_counts().index.to_list()



# Function to create a new column having the country code part of the article page

def filter_by_country(page):

    for country in countries:

        if "_"+country+"." in page:

            return country



# Creating a new column having the country code part of the article page

train_1['Country'] = train_1.Page.apply(filter_by_country)



# Function to create a new column having the agent + access part of the article page

def filter_by_agent(page):

    for agent in agents:

        if agent in page:

            return agent



# Creating a new column having the agent part of the article page

train_1['Agent'] = train_1.Page.apply(filter_by_agent)
# Understanding what are the NaN values for the Country column

# It seems that the URL page does not contain the country code for those cases



pd.DataFrame(train_1.Page[train_1['Country'].isna() == True]).sample(10)
plt.figure(figsize=(12, 6))

plt.title("Number of Wikipedia Articles by Country", fontsize="18")

train_1['Country'].value_counts(dropna=False).plot.bar(rot=0);
train_1['Agent'].value_counts(dropna=False)
plt.figure(figsize=(12, 6))

plt.title("Number of Wikipedia Articles by Agents/Access", fontsize="18")

train_1['Agent'].value_counts().plot.bar(rot=0);
# Creating a sample dataset from the Train dataset for analysis

train_1_sample = train_1.drop(['Site','Country','Agent'], axis=1).sample(6, random_state=42)

train_1_sample
# Transposing the sample dataset to have Date Time at the index

train_1_sampleT = train_1_sample.drop('Page', axis=1).T

train_1_sampleT.columns = train_1_sample.Page.values

train_1_sampleT.shape
train_1_sampleT.head()
# Plotting the Series from the sample dataset 

plt.figure(figsize=(16,8))



for k, v in enumerate(train_1_sampleT.columns):

    plt.subplot(2, 3, k + 1)

    plt.title( str(v.split(".org")[0])+".org"+"\n"+str(v.split(".org")[1]) )

    train_1_sampleT[v].plot()



plt.tight_layout();
# Plotting the Series from the sample dataset at the same graph

plt.figure(figsize=(15,8))



for v in train_1_sampleT.columns:

    plt.plot(train_1_sampleT[v])

    plt.legend(loc='upper center');
# Plotting the histograms for the Series from the sample dataset

plt.figure(figsize=(16,8))



for k, v in enumerate(train_1_sampleT.columns):

    plt.subplot(2, 3, k + 1)

    plt.title( str(v.split(".org")[0])+".org"+"\n"+str(v.split(".org")[1]) )

    sns.distplot(train_1_sampleT[v])



plt.tight_layout();
# Checking that the number of visits to the Wikipedia Articles have Gaussian Distribution (p-value=0)

from scipy.stats import kstest, ks_2samp



pages = list(train_1_sampleT.columns)



print("Kolgomorov-Smirnov - Normality Test")

print()



for p in pages:

    print(p,':', kstest(train_1_sampleT[p], 'norm', alternative = 'less'))    
# List of the main Wikipedia Article sites

sites
# Creating sample datasets from the train dataset and filtering them by sites

train_1_sample_site0 = train_1[train_1['Site'] == sites[0]].drop(['Site','Country','Agent'], axis=1).sample(6, random_state=42)

train_1_sample_site1 = train_1[train_1['Site'] == sites[1]].drop(['Site','Country','Agent'], axis=1).sample(6, random_state=42)

train_1_sample_site2 = train_1[train_1['Site'] == sites[2]].drop(['Site','Country','Agent'], axis=1).sample(6, random_state=42)



# Transposing them to have the Date Time as index

train_1_sampleT_site0 = train_1_sample_site0.drop('Page', axis=1).T

train_1_sampleT_site0.columns = train_1_sample_site0.Page.values

train_1_sampleT_site1 = train_1_sample_site1.drop('Page', axis=1).T

train_1_sampleT_site1.columns = train_1_sample_site1.Page.values

train_1_sampleT_site2 = train_1_sample_site2.drop('Page', axis=1).T

train_1_sampleT_site2.columns = train_1_sample_site2.Page.values
# Plotting the Series from the sample datasets

plt.figure(figsize=(16,8))



for k, v in enumerate(train_1_sampleT_site0.columns):

    plt.subplot(2, 3, k + 1)

    plt.title( str(v.split(".org")[0])+".org"+"\n"+str(v.split(".org")[1]) )

    train_1_sampleT_site0[v].plot()



plt.tight_layout();
# Plotting the Series from the sample datasets at the same graph

plt.figure(figsize=(15,8))



for v in train_1_sampleT_site0.columns:

    plt.plot(train_1_sampleT_site0[v])

    plt.legend(loc='upper center');
# Plotting the Series from the sample datasets

plt.figure(figsize=(16,8))



for k, v in enumerate(train_1_sampleT_site1.columns):

    plt.subplot(2, 3, k + 1)

    plt.title( str(v.split(".org")[0])+".org"+"\n"+str(v.split(".org")[1]) )

    train_1_sampleT_site1[v].plot()



plt.tight_layout();
# Plotting the Series from the sample datasets at the same graph

plt.figure(figsize=(15,8))



for v in train_1_sampleT_site1.columns:

    plt.plot(train_1_sampleT_site1[v])

    plt.legend(loc='upper center');
# Plotting the Series from the sample datasets

plt.figure(figsize=(16,8))



for k, v in enumerate(train_1_sampleT_site2.columns):

    plt.subplot(2, 3, k + 1)

    plt.title( str(v.split(".org")[0])+".org"+"\n"+str(v.split(".org")[1]) )

    train_1_sampleT_site2[v].plot()



plt.tight_layout();
# Plotting the Series from the sample datasets at the same graph

plt.figure(figsize=(15,8))



for v in train_1_sampleT_site2.columns:

    plt.plot(train_1_sampleT_site2[v])

    plt.legend(loc='upper center');
train_1_sampleT_site2.columns[4]
# List of the Wikipedia Article country codes

countries
# Creating a sample dataset from the train dataset for countries having "de" code

train_1_sample_de = train_1[train_1['Country'] == countries[2]].drop(['Site','Country','Agent'], axis=1).sample(6, random_state=42)



# Transposing the sample dataset to have Date Time at the index

train_1_sampleT_de = train_1_sample_de.drop('Page', axis=1).T

train_1_sampleT_de.columns = train_1_sample_de.Page.values
# Plotting the Series from the sample dataset

plt.figure(figsize=(16,8))



for k, v in enumerate(train_1_sampleT_de.columns):

    plt.subplot(2, 3, k + 1)

    plt.title( str(v.split(".org")[0])+".org"+"\n"+str(v.split(".org")[1]) )

    train_1_sampleT_de[v].plot()



plt.tight_layout();
# Plotting the Series from the sample datasets at the same graph

plt.figure(figsize=(15,8))



for v in train_1_sampleT_de.columns:

    plt.plot(train_1_sampleT_de[v])

    plt.legend(loc='upper center');
# Picked up one Time Series for the prophet modeling

train_1_sampleT.columns[1]
# Creating a dataframe for the Time Series from the train_1 samples dataset

data = pd.DataFrame(train_1_sampleT.iloc[:,1].copy())

data.columns = ['y']

data.head()
plt.figure(figsize=(15, 7))

plt.plot(data.y.values, label="actual", linewidth=2.0);
# Adding the lag of the target variable from 1 step back up to 7

for i in range(1, 8):

    data["lag_{}".format(i)] = data.y.shift(i)
data.tail()
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import TimeSeriesSplit 
# for time-series cross-validation set 3 folds

# ~180 days by fold from total of 550 days

tscv = TimeSeriesSplit(n_splits=3)
def timeseries_train_test_split(X, y, test_size):

    """

        Perform train-test split with respect to time series structure

    """

    

    # get the index after which test set starts

    test_index = int(len(X)*(1-test_size))

    

    X_train = X.iloc[:test_index]

    y_train = y.iloc[:test_index]

    X_test = X.iloc[test_index:]

    y_test = y.iloc[test_index:]

    

    return X_train, X_test, y_train, y_test
y = data.dropna().y

X = data.dropna().drop(['y'], axis=1)



# reserve 33% of data for testing

# so test size would be ~180 days

X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.33)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# Function for the MAPE error

def mean_absolute_percentage_error(y_true, y_pred): 

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



# Function for the SMAPE error

def smape(y_true, y_pred):

    denominator = (np.abs(y_true) + np.abs(y_pred))

    diff = np.abs(y_true - y_pred) / denominator

    diff[denominator == 0] = 0.0

    return 200 * np.mean(diff)
def plotModelResults(model, X_train=X_train, X_test=X_test, plot_intervals=False, plot_anomalies=False):

    """

        Plots modelled vs fact values, prediction intervals and anomalies

    

    """

    

    prediction = model.predict(X_test)

    

    plt.figure(figsize=(15, 7))

    plt.plot(prediction, "g", label="prediction", linewidth=2.0)

    plt.plot(y_test.values, label="actual", linewidth=2.0)

    

    if plot_intervals:

        cv = cross_val_score(model, X_train, y_train, 

                                    cv=tscv, 

                                    scoring="neg_mean_absolute_error")

        mae = cv.mean() * (-1)

        deviation = cv.std()

        

        scale = 1.96

        lower = prediction - (mae + scale * deviation)

        upper = prediction + (mae + scale * deviation)

        

        plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)

        plt.plot(upper, "r--", alpha=0.5)

        

        if plot_anomalies:

            anomalies = np.array([np.NaN]*len(y_test))

            anomalies[y_test<lower] = y_test[y_test<lower]

            anomalies[y_test>upper] = y_test[y_test>upper]

            plt.plot(anomalies, "o", markersize=10, label = "Anomalies")

    

    mape_error = mean_absolute_percentage_error(prediction, y_test)

    smape_error = smape(prediction, y_test)

    plt.title("MAPE: "+str(mape_error)+"\n"+"SMAPE: "+str(smape_error))

    plt.legend(loc="best")

    plt.tight_layout()

    plt.grid(True);

    

def plotCoefficients(model):

    """

        Plots sorted coefficient values of the model

    """

    

    coefs = pd.DataFrame(model.coef_, X_train.columns)

    coefs.columns = ["coef"]

    coefs["abs"] = coefs.coef.apply(np.abs)

    coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)

    

    plt.figure(figsize=(15, 7))

    coefs.coef.plot(kind='bar')

    plt.grid(True, axis='y')

    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed');
from sklearn.linear_model import LinearRegression



# Linear Regression

lr = LinearRegression()

lr.fit(X_train, y_train)
plotModelResults(lr, plot_intervals=True)

plotCoefficients(lr)
data.index = pd.to_datetime(data.index)

data["weekday"] = data.index.weekday

data['is_weekend'] = data.weekday.isin([5,6])*1

data.tail(7)
plt.figure(figsize=(16, 5))

plt.title("Encoded features")

#data.weekday.plot()

data.is_weekend.plot()

plt.grid(True);
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
y = data.dropna().y

X = data.dropna().drop(['y'], axis=1)



X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.33)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
print(X_train_scaled.shape, y_train.shape, X_test_scaled.shape, y_test.shape)
# Linear Regression using Scaled Data

lr = LinearRegression()

lr.fit(X_train_scaled, y_train)
plotModelResults(lr, X_train=X_train_scaled, X_test=X_test_scaled, plot_intervals=True)

plotCoefficients(lr)
plt.figure(figsize=(10, 8))

sns.heatmap(X_train.corr());
from sklearn.linear_model import LassoCV, RidgeCV



ridge = RidgeCV(cv=tscv)

ridge.fit(X_train_scaled, y_train)



plotModelResults(ridge, 

                 X_train=X_train_scaled, 

                 X_test=X_test_scaled, 

                 plot_intervals=True, plot_anomalies=True)

plotCoefficients(ridge)
lasso = LassoCV(cv=tscv)

lasso.fit(X_train_scaled, y_train)



plotModelResults(lasso, 

                 X_train=X_train_scaled, 

                 X_test=X_test_scaled, 

                 plot_intervals=True, plot_anomalies=True)

plotCoefficients(lasso)
from xgboost import XGBRegressor 



xgb = XGBRegressor()

xgb.fit(X_train_scaled, y_train);
plotModelResults(xgb, X_train=X_train_scaled, X_test=X_test_scaled, plot_intervals=True)
# Creating a dataframe for the Time Series from the train_1 samples dataset

data = pd.DataFrame(train_1_sampleT.iloc[:,1].copy())

data.columns = ['y']

data.index = pd.to_datetime(data.index)

future = pd.DataFrame(index=pd.date_range(start='2017-01-01', end='2017-03-01'), columns=data.columns).fillna(0)

data_future = data.append(future)
for i in range(1, 61):

    data_future["lag_{}".format(i)] = data_future.y.shift(i)



data_future["weekday"] = data_future.index.weekday

data_future['is_weekend'] = data_future.weekday.isin([5,6])*1

data_future.tail(7)
data_future.shape
X_train = data_future.iloc[:550,:].dropna().drop(['y'], axis=1)

y_train = data_future.iloc[:550,:].dropna().y



X_test = data_future.iloc[550:,:].dropna().drop(['y'], axis=1)

y_test = data_future.iloc[550:,:].dropna().y



print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
lr = LinearRegression()

lr.fit(X_train_scaled, y_train)
prediction = lr.predict(X_test)
plotModelResults(lr, X_train=X_train_scaled, X_test=X_test_scaled, plot_intervals=True)

plotCoefficients(lr)
# train_1_sampleT.columns[1]+"_"+"2017-01-01"

# train_1_sampleT.columns[1]+"_"+"2017-01-01" in list(key_1.Page.values)