# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import probplot
import pylab

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.



# csv dosyasını oku
shoes = pd.read_csv("../input/womens-shoes-prices/7210_1.csv")
shoes.head()
shoes.columns
shoes.describe()
# Değişken normal dağılıp dağılmadığnı kontrol et
# Çoğu noktanın kırmızı çizgi boyunc aolması gerekiyor
probplot(shoes["prices.amountMin"],dist="norm",plot=pylab)
shoes
# count each brand occurence
shoes.groupby(by=["brand"])['brand'].count()
# count occurence descending order
# select two brands for t-test (Ralph Lauren and Skechers in this case)
shoes["brand"].value_counts()
# compare two brands : Ralph Lauren and Skechers
# make a t-test
ralphLauren = shoes["prices.amountMin"][shoes["brand"] == "Ralph Lauren"]
skechers = shoes["prices.amountMin"][shoes["brand"] == "Skechers"]
ttest_ind(ralphLauren, skechers, equal_var=False)
ralphLauren.mean()
skechers.mean()
plt.hist(ralphLauren, alpha=0.5, label="ralph")
plt.hist(skechers, alpha=0.5, label="skech")
plt.legend(loc="upper right")
plt.title("Prices of Ralph and Skecher")