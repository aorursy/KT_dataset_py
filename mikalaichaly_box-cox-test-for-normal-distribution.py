# setup defaults

import numpy as np
import pandas as pd
pd.options.display.max_columns = 12
# Disable warnings in Anaconda
import warnings
warnings.simplefilter('ignore')
# We will display plots right inside Jupyter Notebook
%matplotlib inline
import matplotlib.pyplot as plt
# We will use the Seaborn library
import seaborn as sns
sns.set()
# Graphics in SVG format are more sharp and legible
%config InlineBackend.figure_format = 'png'
# Increase the default plot size
from pylab import rcParams
rcParams['figure.figsize'] = 5, 4
def RunNormalization(original_data, name):
    
    norm_data = np.array(stats.boxcox(original_data)[0])
    
    original_data.sort()
    norm_data.sort()
    #print(type(norm_data))
    df = pd.DataFrame()
    df['norm'] = norm_data
    df['orig'] = original_data

    # plot both together to compare
    fig, ax=plt.subplots(1, 2, sharey=True)

    sns.distplot(original_data, ax=ax[0])
    ax[0].set_title("Original Data (%s)" % (name))
    sns.distplot(norm_data, ax=ax[1])
    ax[1].set_title("Normalized data")


    f = plt.figure()
    plt.scatter(original_data, norm_data)
    plt.xlabel('original (%s)' % (name))
    plt.ylabel('normalized')

    sns.jointplot('orig', 'norm',
                  data=df, kind="kde", color="g");
    
    #test for normal distribution
    k2, p = stats.normaltest(norm_data)
    alpha = 0.05
    print("p = {:g}".format(p))
    if p < alpha:  # null hypothesis: x comes from a normal distribution
         print("NEGATIVE: The null hypothesis can be rejected")
    else:
         print("POSITIVE: The null hypothesis cannot be rejected")
from scipy import stats
np.random.seed(42)
sample_size = 500

original_data = np.random.exponential(size = sample_size)
RunNormalization(original_data, 'EXP')
original_data = np.random.uniform(size = sample_size, low=10, high=42)
RunNormalization(original_data, 'Uniform')





