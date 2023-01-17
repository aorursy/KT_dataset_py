import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
eth = pd.read_csv('../input/all/eth.csv', index_col=False)
eth.index = pd.to_datetime(eth.date);
btc = pd.read_csv('../input/all/btc.csv', index_col=False)
btc.index = pd.to_datetime(btc.date);
eth.tail(5)
btc.tail(5)
ax=eth.plot(y='blockSize',logy=True,figsize=[20,10])
ax=eth.plot(y='generatedCoins',secondary_y=True,ax=ax)
ax
sns.jointplot(x='price(USD)',y='activeAddresses',data=eth,height=10)
p=pd.DataFrame({ 'ETH txVolume(USD)': eth['txVolume(USD)'], 'BTC txVolume(USD)': btc['txVolume(USD)'] }, index=eth.index)
ax=p.plot(logy=True,figsize=[20,10])
ax