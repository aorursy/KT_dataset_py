!pip install pmdarima
import pandas as pd

import numpy as np

%matplotlib inline



from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.seasonal import seasonal_decompose

from pmdarima import auto_arima



import seaborn as sns

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

from scipy import stats

from scipy.stats import skew



import warnings

warnings.filterwarnings('ignore')





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/train.csv', index_col='date', parse_dates=True)

test = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/test.csv', index_col='date', parse_dates=True)
train.head()
train.tail()
test.head()
test.tail()
train.dtypes, test.dtypes
train.isnull().sum(), test.isnull().sum()
train['sales'] = train['sales'].astype('float64')
Q1 = train.quantile(0.25)

Q3 = train.quantile(0.75)

IQR = Q3 - Q1

print(IQR)
train_without_outliers = train[~((train < (Q1 - 1.5*Q3)) | (train > (Q3 + 1.5*Q1))).any(axis=1)]
data = train_without_outliers.append(test)
data.head()
data.tail()
data.dtypes
data.shape
data.isnull().sum()
corr = data.corr()

corr['sales'][:-2]
for i in (data.select_dtypes(include ='object').columns):

    if(i != 'sales'):

        data_crosstab = pd.crosstab(data[i], data['sales'], margins = False)

        stat, p, dof, expected = stats.chi2_contingency(data_crosstab)

        prob=0.95

        alpha = 1.0 - prob

        if p <= alpha:

            print(i, ' : Dependent (reject H0)')

        else:

            print(i, ' : Independent (fail to reject H0)')
corr_matrix = data.corr().abs()



# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



# Find index of feature columns with correlation greater than 0.95

to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
data = data.drop(data[to_drop], axis=1)
data.drop('store', axis=1, inplace=True)

data.drop('item', axis=1, inplace=True)
def fixing_skewness(df):

    numeric_feats = df.dtypes[df.dtypes != object].index

    

    skew_feats = df[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)

    

    high_skew = skew_feats[abs(skew_feats) > 0.5].index

    

    for i in high_skew:

        df[i] = boxcox1p(df[i], boxcox_normmax(df[i] + 1))

#         print(i)

        

fixing_skewness(data)
def overfit_reducer(df):

    overfit = []

    for i in df.columns:

        count = df[i].value_counts()

        zero_index_value = count.iloc[0]

        

        if (((zero_index_value / len(df)) * 100) > 99.94):

            overfit.append(i)

            

    overfit = list(overfit)

    return overfit
#Finding the list of overfitted features using above user-defined function

overfitted_features = overfit_reducer(data)

#Dropping the overfitted columns from the final dataframes

data.drop(overfitted_features, axis=1, inplace=True)
data1 = data.dropna()
data1['sales'][:60].plot(figsize=(12,8))
result = seasonal_decompose(data1['sales'], model='additive', period=365)

result.plot();
from statsmodels.tsa.stattools import adfuller



def adf_test(series,title=''):

    """

    Pass in a time series and an optional title, returns an ADF report

    """

    print(f'Augmented Dickey-Fuller Test: {title}')

    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data

    

    labels = ['ADF test statistic','p-value','# lags used','# observations']

    out = pd.Series(result[0:4],index=labels)



    for key,val in result[4].items():

        out[f'critical value ({key})']=val

        

    print(out.to_string())          # .to_string() removes the line "dtype: float64"

    

    if result[1] <= 0.05:

        print("Strong evidence against the null hypothesis")

        print("Reject the null hypothesis")

        print("Data has no unit root and is stationary")

    else:

        print("Weak evidence against the null hypothesis")

        print("Fail to reject the null hypothesis")

        print("Data has a unit root and is non-stationary")
# adf_test(data1['sales'])
# auto_arima(data1['sales'], trace=True, n_jos=-1).summary()
from statsmodels.tsa.arima_model import ARIMA,ARIMAResults

model = ARIMA(data1['sales'],order=(2,0,0))

results = model.fit()

results.summary()
start=len(train_without_outliers)

end=len(train_without_outliers)+len(test)-1

forecast = results.predict(start=start, end=end,typ='levels').rename('ARIMA(2,0,0) Forecast')
# Plot predictions against known values

title = 'Real sales versus Forecasted sales'



ax = data1['sales'].plot(legend=True,figsize=(12,6),title=title)

forecast.plot(legend=True)

ax.autoscale(axis='x',tight=True)

# ax.set(xlabel=xlabel, ylabel=ylabel)

# ax.yaxis.set_major_formatter(formatter);
forecast = forecast.reset_index(drop=True)
test = test.reset_index(drop=True)
submission_df = pd.DataFrame()

submission_df['id'] = test['id']

submission_df['sales'] = forecast

submission_df['sales'] = submission_df['sales'].astype('int64')
submission_df.to_csv('/kaggle/working/submission.csv', index=False)