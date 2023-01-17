import pandas as pd
import numpy as np
from scipy import stats # if you want to import everything
from scipy.stats import kstest # specific import
x = np.linspace(-15, 15, 9)
stats.kstest(x, 'norm')
np.random.seed(987654321) # set random seed to get the same result
stats.kstest('norm', False, N=100)
np.random.seed(987654321)
stats.kstest(stats.norm.rvs(size=100), 'norm')
np.random.seed(987654321)
x = stats.norm.rvs(loc=0.2, size=100)
stats.kstest(x,'norm', alternative = 'less')
stats.kstest(x,'norm', alternative = 'greater')
stats.kstest(x,'norm', mode='asymp')
np.random.seed(987654321)
stats.kstest(stats.t.rvs(100,size=100),'norm')
np.random.seed(987654321)
stats.kstest(stats.t.rvs(3,size=100),'norm')
from scipy import stats # if you want to import everything
from scipy.stats import ks_2samp # specific import
np.random.seed(12345678)  #fix random seed to get the same result
n1 = 200  # size of first sample
n2 = 300  # size of second sample
rvs1 = stats.norm.rvs(size=n1, loc=0., scale=1)
rvs2 = stats.norm.rvs(size=n2, loc=0.5, scale=1.5)
stats.ks_2samp(rvs1, rvs2)
rvs3 = stats.norm.rvs(size=n2, loc=0.01, scale=1.0)
rvs4 = stats.norm.rvs(size=n2, loc=0.0, scale=1.0)
stats.ks_2samp(rvs1, rvs4)
from scipy.stats import kstest

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.plotly as py

init_notebook_mode(connected=True)
# Importing GitHub file

url ="https://raw.githubusercontent.com/plotly/datasets/master/wind_speed_laurel_nebraska.csv"
data = pd.read_csv(url, sep=',')
data.head()
df = data[0:10]

table = ff.create_table(df)
iplot(table)
x = data['10 Min Sampled Avg']

ks_results = kstest(x, cdf='norm')

matrix_ks = [['', 'DF', 'Test Statistic', 'p-value'],
             ['Sample Data', len(x) - 1, ks_results[0], ks_results[1]]]

ks_table = ff.create_table(matrix_ks, index=True)
iplot(ks_table, filename='ks-table')
np.random.seed(12345678)
x = np.random.normal(0, 1, 1000)
y = np.random.normal(0, 1, 1000)
z = np.random.normal(1.1, 0.9, 1000)
ks_2samp(x, y)
ks_2samp(x, z)
pd.show_versions()