import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

data = pd.read_csv('../input/assignment-solution/11. Assignment Dataset/chennai_house_price_prediction.csv')
data.shape
data.head()
data.describe()
data.isnull().sum()
data.dtypes
data['SALES_PRICE'].plot.hist(bins = 50)
plt.xlabel('Sales', fontsize=12)
(data['SALES_PRICE'].loc[data['SALES_PRICE']<18000000]).plot.hist(bins=50)
data['AREA'].value_counts().plot(kind = 'bar')
data.drop_duplicates()
data.plot.scatter('INT_SQFT','SALES_PRICE')
fig, ax = plt.subplots()
colors = {'Commercial':'red', 'House':'blue', 'Others':'green'}
ax.scatter(data['INT_SQFT'], data['SALES_PRICE'], c=data['BUILDTYPE'].apply(lambda x: colors[x]))
plt.show()
fig, axs = plt.subplots(2, 2)

fig.set_figheight(10)
fig.set_figwidth(10)

axs[0, 0].scatter(data['QS_BEDROOM'], data['SALES_PRICE'])    # QS_BEDROOM and sale price
axs[0, 0].set_title('QS_BEDROOM')

axs[0, 1].scatter(data['QS_BATHROOM'], data['SALES_PRICE'])   # QS_BATHROOM and price
axs[0, 1].set_title('QS_BATHROOM')

axs[1, 0].scatter(data['QS_ROOMS'], data['SALES_PRICE'])    # QS_ROOMS and sale price
axs[1, 0].set_title('QS_ROOMS')

axs[1, 1].scatter(data['QS_OVERALL'], data['SALES_PRICE'])    # QS_OVERALL and sale price
axs[1, 1].set_title('QS_OVERALL')


temp = data.groupby(['BUILDTYPE', 'PARK_FACIL']).SALES_PRICE.median()
temp.plot(kind = 'bar', stacked = True)
import numpy as np
import pandas as pd
df = pd.DataFrame(
np.random.rand(100, 5),
columns=['a', 'b', 'c', 'd', 'e'])
df.to_csv('/kaggle/working/df.csv',index=False)