import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

from datetime import datetime



bitUSD = pd.read_csv('../input/bitstampUSD_1-min_data_2012-01-01_to_2017-10-20.csv')

bitUSD['Timestamp'] = bitUSD['Timestamp'].apply(lambda x: datetime.fromtimestamp(x))



bitUSD.tail()
%matplotlib inline

%pylab inline



print(bitUSD.shape)

bitUSD = bitUSD[bitUSD['Timestamp'] > '2017-07-01']

print(bitUSD.shape)



bitUSD['Hour'] = bitUSD['Timestamp'].apply(lambda x: x.hour)

bitUSD['DoW'] = bitUSD['Timestamp'].apply(lambda x: x.weekday())



bitUSD.plot.scatter(x='Hour', y='Weighted_Price', alpha=0.01)
bitUSD.plot.scatter(x='Hour', y='Volume_(BTC)', alpha=0.1)
bitUSD.plot.scatter(x='DoW', y='Weighted_Price', alpha=0.1)
bitUSD.plot.scatter(x='DoW', y='Volume_(BTC)', alpha=0.1)