# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import other libraries

import pywt

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

sns.set(style='whitegrid', palette="deep", font_scale=0.7, rc={"figure.figsize": [16, 8]})
# declare global variables

ROOT="/kaggle/input/m5-forecasting-accuracy/"
sell_prices_df = pd.read_csv(ROOT+"sell_prices.csv")

sample_submission_df = pd.read_csv(ROOT+"sample_submission.csv")

calendar_df = pd.read_csv(ROOT+"calendar.csv")

sales_train_val_df = pd.read_csv(ROOT+"sales_train_validation.csv")
sell_prices_df.head()
sell_prices_df.info()
sample_submission_df.head()
sample_submission_df.info()
calendar_df.head()
calendar_df.info()
sales_train_val_df.head()
sales_train_val_df.info()
ids = sorted(sales_train_val_df['id'].unique().tolist())
dcols = [c for c in sales_train_val_df.columns.tolist() if "d_" in c]
sales = []

for i in ids[0:6]:

    sales.append(sales_train_val_df[sales_train_val_df['id'] == i][dcols].values.flatten().tolist())
plt.figure()

for i, s in enumerate(sales):

    plt.subplot(2,3,i+1)

    plt.tight_layout()

    plt.scatter(np.arange(1, len(s)+1), s)

    plt.xlabel('Time')

    plt.ylabel('Sales')

    plt.title('Sales for ID: {}'.format(i))

plt.show()
plt.figure()

for i, s in enumerate(sales):

    plt.subplot(2,3,i+1)

    plt.tight_layout()

    plt.plot(np.arange(1, len(s[0:100])+1), s[0:100])

    plt.xlabel('Time')

    plt.ylabel('Sales')

    plt.title('Sales (over 100 days) for ID: {}'.format(i))

plt.legend(loc='best')

plt.show()
def wavelet_denoising(x, wavelet="db4", level=1):

    """ Wavelet Denoising of Forecasting data"""

    axis = None

    coeff = pywt.wavedec(x, wavelet, mode="per")

    mad = np.mean(np.absolute(coeff[-level] - np.mean(coeff[-level], axis)), axis)

    sigma = (1/0.6745) * mad



    uthresh = sigma * np.sqrt(2*np.log(len(x)))

    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])



    return pywt.waverec(coeff, wavelet, mode='per')



def avg_smoothing(x, kernel_size=3, stride=1):

    """Average Smoothing of Forecasting data"""

    sample = []

    start = 0

    end = kernel_size

    while end <= len(x):

        start = start + stride

        end = end + stride

        sample.extend(np.ones(end - start) * np.mean(x[start:end]))

    return list(np.array(sample))
plt.figure()

for i, s in enumerate(sales):

    plt.subplot(2,3,i+1)

    plt.tight_layout()

    plt.plot(np.arange(1, len(s[0:100])+1), s[0:100])

    plt.plot(np.arange(1, len(s[0:100])+1), wavelet_denoising(s[0:100]))

    plt.xlabel('Time')

    plt.ylabel('Sales')

    plt.title('Wavelet Denoising of Sales (over 100 days) for ID: {}'.format(i))

plt.legend(loc='best')

plt.show()
plt.figure()

for i, s in enumerate(sales):

    smooth_s = avg_smoothing(s[0:100])

    plt.subplot(2,3,i+1)

    plt.tight_layout()

    plt.plot(np.arange(1, len(s[0:100])+1), s[0:100])

    plt.plot(np.arange(1, len(smooth_s)+1), smooth_s)

    plt.xlabel('Time')

    plt.ylabel('Sales')

    plt.title('Wavelet Denoising of Sales (over 100 days) for ID: {}'.format(i))

    plt.xlim([0,100])

plt.legend(loc='best')

plt.show()
def rolling_avg_plot(col="", label=""):

    """Rolling average plot per label"""

    means = []

    past_sales_df = pd.DataFrame()

    

    colList = sales_train_val_df[col].unique().tolist()

    past_sales = sales_train_val_df.set_index('id')[dcols].T.merge(calendar_df.set_index('d')['date'], 

                                                                   left_index=True, right_index=True, validate='1:1').set_index('date')

    past_sales_df['date'] = past_sales.index.tolist()

    

    plt.figure()

    plt.suptitle('Rolling Average of Sales per {}'.format(label))

    plt.subplot(3,3,1)

    for s in colList:

        store_items = [c for c in past_sales.columns if s in c] # fetch items for the current colList item

        data = past_sales[store_items].sum(axis=1).rolling(90).mean().tolist() # calculate rolling average (90 day window)

        means.append(np.mean(past_sales[store_items].sum(axis=1)))

        past_sales_df[s] = data

        plt.plot(np.arange(0, len(data)), data, label=s)

    plt.xticks([])

    plt.xlabel('Time')

    plt.ylabel('Average Sales')

    plt.legend(loc='best')

    

    plt.subplot(3,3,2)    

    past_sales_df.boxplot()

    plt.xlabel(label)

    plt.ylabel('Average Sales')

    

    plt.subplot(3,3,3) 

    means_df = pd.DataFrame(np.transpose([means, colList]))

    means_df.columns = ["Average Sales", label]

    

    sns.barplot(x=label, y="Average Sales", data=means_df)

    plt.show()

    

    return means
store_means = rolling_avg_plot("store_id", "stores")
state_means = rolling_avg_plot("state_id", "states")
cat_means = rolling_avg_plot("cat_id", "categories")
dept_means = rolling_avg_plot("dept_id", "departments")
ext_factors = [c for c in calendar_df.columns.tolist() if "event" in c or "snap" in c]

ext_factors
calendar_df.info()
calendar_df[ext_factors[0]].unique()
plt.figure()

for i, ext in enumerate(ext_factors[1:]):

    plt.subplot(3,2,i+1)

    plt.tight_layout()

    chart = sns.countplot(x=ext, data=calendar_df)

plt.show()