import numpy as np
import pandas as pd 
from scipy import stats

from matplotlib import pyplot as plt
# generate a univariate data sample
np.random.seed(142)
datas = sorted(stats.lognorm.rvs(s=0.5, loc=1, scale=1000, size=1000))

data = pd.DataFrame(datas, columns=['values'])


print('Size of data:', data.shape)
def gen_graph(value):
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    plt.hist(value, color='g', alpha=0.5)

    plt.subplot(1, 2, 2)
    stats.probplot(value, dist="norm", plot=plt)

    plt.show()
gen_graph(data['values'])
data_log = np.log(data['values'])

gen_graph(data_log)
data_rec = np.reciprocal(data['values'])

# or

data_rec_2 = 1/ data['values']

gen_graph(data_rec)
data_square = np.sqrt(data['values'])

# or 
data_square_2 = (data)**(1/2)

gen_graph(data_square)
data_cube = np.cbrt(data['values'])

gen_graph(data_cube)
data_expo =  data['values'] ** (1/5)

gen_graph(data_expo)
data_boxcox, a = stats.boxcox(data['values'])

gen_graph(data_boxcox)
data_yeo, a = stats.yeojohnson(data['values'])

gen_graph(data_yeo)
