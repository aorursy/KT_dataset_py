#!pip install --upgrade dtale #downloading dtale
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import dtale
#import dtale.app as dtale_app

#dtale_app.USE_COLAB = True
#dtale.show(data, ignore_duplicate=True)
data = pd.read_csv(r'../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
print(data.head(10))
print(data.columns)
print(data.shape)
data.dtypes
data.isnull().sum()
data.fillna({'reviews_per_month':0}, inplace=True) #filling null values of 4 columns with respective values
data.fillna({'name':"NoName"}, inplace=True)
data.fillna({'host_name':"NoName"}, inplace=True)
data.fillna({'last_review':"NotReviewed"}, inplace=True)
data.isnull().sum()
data['price'].head(10)
data['price'].describe()
hist_price = data["price"].hist()
hist_price = data["price"][data['price'] <= 1000].hist()
hist_price = data["price"][data['price'] > 1000].hist()
price_greater_than_2000 = data[data['price'] > 1000].value_counts()
print(price_greater_than_2000)
data["price"][data["price"]<250].hist()
data["price"][data["price"]<250].describe()
data['neighbourhood'].value_counts()
neigh =data.groupby("neighbourhood").filter(lambda x: x['neighbourhood'].count() > 200)
print(len(neigh['neighbourhood']))
neigh1 = data.groupby("neighbourhood").filter(lambda x: x['neighbourhood'].count() == 1)
print(len(neigh1["neighbourhood"]))
data['neighbourhood_group'].value_counts()
ng_price = data.groupby("neighbourhood_group")["price"].mean()
print(ng_price)
data1 = data

#sns.catplot(x = "neighbourhood_group", y = "price", hue = "room_type", kind = "swarm", data = data1)
#https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python/#21.-Histogram-for-Categorical-Variable


x_var = 'neighbourhood_group'
groupby_var = 'room_type'

df_agg = data.loc[:, [x_var, groupby_var]].groupby(groupby_var)
#vals = [df[x_var].values.tolist() for i, df in df_agg]
vals = [data['price'].values.tolist() for i in df_agg]

plt.figure(figsize=(16,9), dpi= 80)
colors = [plt.cm.Spectral(i / float(len(vals) - 1 )) for i in range(len(vals))]
n, bins, patches = plt.hist(vals, data[x_var].unique().__len__(), stacked = True, density = False, color = colors[:len(vals)])

plt.legend({group:col for group, col in zip(np.unique(data[groupby_var]).tolist(), colors[:len(vals)])})
#plt.title(f"Stacked Histogram of ${x_var}$ colored by ${groupby_var}$", fontsize=22)
plt.xlabel(x_var)
plt.ylabel("Price")
plt.ylim(0, 40)
plt.xticks(ticks = bins, labels = np.unique(data[x_var]).tolist(), rotation = 90, horizontalalignment = 'left')
plt.show()
data = data.drop(columns = ["id","host_name"])
data["name_length"] = data['name'].map(str).apply(len)
# Max and Min name length
print(data["name_length"].max())
print(data["name_length"].min())
print(data["name_length"].idxmax())
print(data["name_length"].idxmin())
data.at[25832, 'name']
data.at[4033, 'name']
data.plot.scatter(x = "name_length", y = "number_of_reviews" )
data["number_of_reviews"].corr(data["name_length"])
data[data["name_length"] < 50].plot.scatter(x = "price", y = "name_length")
data["name_length"].corr(data["price"])
data.name_length.hist()
data['room_type'].value_counts()
rt_price = data.groupby("room_type")["price"].mean()
print(rt_price)
data["minimum_nights"].describe()
data["minimum_nights"].hist()
data["minimum_nights"][data["minimum_nights"] < 10].hist()
data['price'].corr(data['minimum_nights'])
data["minimum_nights"][data["minimum_nights"] > 30]
data.loc[(data.minimum_nights > 30), "minimum_nights"] = 30
data['price'].corr(data['minimum_nights'])
data["availability_365"].describe()
data["availability_365"].hist()
data.drop(["name", 'last_review', "latitude", 'longitude', 'host_id'], axis = 1, inplace = True)
corr = data.corr(method = 'pearson')
plt.figure(figsize = (15,8))
sns.heatmap(corr, annot = True)
data.columns
