import warnings

warnings.filterwarnings("ignore")

from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter('ignore', ConvergenceWarning)

from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.utils.testing import ignore_warnings

from sklearn.exceptions import ConvergenceWarning

from pandas import read_csv

from pandas import datetime

from pandas import DataFrame

from matplotlib import pyplot

from pandas.plotting import autocorrelation_plot

from sklearn.metrics import mean_squared_error
def parser(x):

    return datetime.strptime('190'+x, '%Y-%m')

series = read_csv('../input/shampoo/shampoo1.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

print(series.head())

series.plot()

pyplot.show()



autocorrelation_plot(series)

pyplot.show()
import numpy as np

np.random.seed(0)



series1=series.copy()+np.random.randint(-50,50,len(series))+200

series2=series.copy()+np.random.randint(10,110,len(series))+150

series3=series.copy()+np.random.randint(-70,30,len(series))+100

combs = series1+series2+series3+series
series.plot()

pyplot.title('series')

pyplot.show()



series1.plot()

pyplot.title('series1')

pyplot.show()



series2.plot()

pyplot.title('series2')

pyplot.show()



series3.plot()

pyplot.title('series3')

pyplot.show()



combs.plot()

pyplot.title('Aggregated Time Series')

pyplot.show()
@ignore_warnings(category=ConvergenceWarning)

def repeated(series):

    # pridiction 

    X = series.values

    size = int(len(X) * 0.7)

    train, test = X[0:size], X[size:len(X)]

    history = [x for x in train]

    predictions = list()

    for t in range(len(test)):

        model = SARIMAX(history, order=(1, 1, 1), seasonal_order=(1, 1, 1, 2))

        model_fit = model.fit(disp=False)

        output = model_fit.forecast()

        yhat = output[0]

        predictions.append(yhat)

        obs = test[t]

        history.append(obs)

    error = sum([abs(i-j) for (i,j) in zip(test,predictions)])

    return error
@ignore_warnings(category=ConvergenceWarning)

def Evaluation():

    individual_series = [series, series1, series2, series3]

    summed_indi_errors = 0

    for i in individual_series:

        error = repeated(i)

        summed_indi_errors+=error

    comb_error = repeated(combs)

    diff = summed_indi_errors - comb_error

    print("Difference between individual and combined error",diff)

    percentage_difference = (diff/summed_indi_errors)*100

    print("percentage_difference", percentage_difference)

Evaluation()