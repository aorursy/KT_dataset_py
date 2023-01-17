# Do some feature extraction.

!pip install fastai==0.7
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input/Data"))

# Any results you write to the current directory are saved as output.
import pandas as pd
STOCKS = '../input/Data/Stocks/'
FILE = 'fb.us.txt'
stocks = pd.read_csv(STOCKS+FILE)

# TODO detrend data by removing average stock movement (how tho?)
import numpy as np
close = np.array(stocks['Close'])
return_raw_prev1 = (close[1:]- close[:-1]) / close[:-1]
# These are like, if you bought stock yesterday and sold it today, what's your ROI
stocks['ReturnsRawPrev1'] = None
stocks['ReturnsRawPrev1'][1:] = return_raw_prev1
return_raw_prev5 = (close[5:] - close[:-5]) / close[:-5]
stocks['ReturnsRawPrev5'] = None
stocks['ReturnsRawPrev5'][5:] = return_raw_prev5
return_raw_prev10 = (close[10:] - close[:-10]) / close[:-10]
stocks['ReturnsRawPrev10'] = None
stocks['ReturnsRawPrev10'][10:] = return_raw_prev10
stocks.head(15)

# This is if you buy stock today and sell it tomorrow, what will your return be? Target variable you want to predict
stocks['ReturnsRawFut1'] = None
stocks['ReturnsRawFut1'][:-1] = return_raw_prev1
stocks['ReturnsRawFut5'] = None
stocks['ReturnsRawFut5'][:-5] = return_raw_prev5
stocks['ReturnsRawFut10'] = None
stocks['ReturnsRawFut10'][:-10] = return_raw_prev10

from fastai.structured import add_datepart

add_datepart(stocks, 'Date')
stocks.head(15)
stocks.head(15)
stocks.to_csv('fb.us.txt')
!ls ..