import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import pytz as tz
#from datetime import datetime
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
from matplotlib import cm
from collections import OrderedDict
df=pd.read_csv(".../snapptrip.csv")
print(df.info())
print(df.shape)
print(df.isnull().sum())
df.head()
def parse_datetime(s):
    tzone = tz.timezone("America/New_York") #parse_datetime
    utc = datetime.strptime(s, '%Y-%m-%dT%H:%M:%SZ')
    return tz.utc.localize(utc).astimezone(tzone)
df['ts'] = df['departure_date'].apply(lambda x: parse_datetime(x))
df['departure_date'] = df['departure_date'].drop('departure_date',axis=1,errors='ignore')

#local date and time
df['departure_date']  = df['departure_date'].astype(object).apply(lambda x : x.date())
df['departure_date']  = df['departure_date'].astype(object).apply(lambda x : x.time())
#Observing correlation
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')
#The most flights in 2 years 2019-2018
plt.figure(figsize=(8,5))
sns.countplot(df['departure-Year'] , color='#c9090F')
plt.show()
plt.figure(figsize=(8,5))
sns.countplot(df['paid-Year'])
#Which airlines has the most flights
plt.figure(figsize=(7,8))
df['airline'].value_counts()[:5].plot(kind='pie',autopct='%1.1f%%',shadow=True,legend = True)
plt.show()
group = data.groupby(['paid-Year','airline']).sum()
total_sold = group["Tickets sold"].groupby(level=0, group_keys=False)
print(total_sold.nlargest(5))
group = data.groupby(['paid-Year','paid-Month']).sum()
total_sold = group["Tickets sold"].groupby(level=0, group_keys=False)
print(total_sold.nlargest(12))
# Plots of Total price by Hour, Year and month
sns.pairplot(data.dropna(),
             hue='Tickets sold',
             x_vars=['paid-Hour','paid-Month','paid-Year'],
             y_vars='Total Price (after discount)',
             height=5,
             plot_kws={'alpha':0.1, 'linewidth':0}
            )
plt.suptitle('Total price by Hour, Year and month of Year')
plt.show()
total_sold.plot(figsize=(16,4),legend=True)
plt.title('،Ticket Sales Yy Years - Months')
plt.show()
group2 = data.groupby(['paid-Year','paid-Month']).sum()
total_sold_per_hour = group2['Total Price (after discount)'].groupby(level=0, group_keys=False)
total_sold_per_hour.plot(figsize=(15,4),legend=True)
plt.title('Price Volatility By Year - Month')
plt.show()
# Feature Selection
X_df = data.drop(columns='Original Price')
y = data['Original Price'].values
#One Hot Encoding
encoder = OneHotEncoder()
X = encoder.fit_transform(X_df.values)
print('type of X is :',X.dtype)
for category in encoder.categories_:
    print(category[:5])
print(df.groupby(['airline']).mean())
print(df.groupby(['airline']).std())
#df.groupby(['airline']).count()
data=df.drop(['paid_date','departure_date','DepDate','PDate','paidTime','departureTime','paidYear','departureYear'] , axis=1)
#Total number of tickets sold by airline-year and month-year
group = data.groupby(['paid-Year','airline']).sum()
total_sold = group["Tickets sold"].groupby(level=0, group_keys=False)
print(total_sold.nlargest(5))
group = data.groupby(['paid-Year','paid-Month']).sum()
total_sold = group["Tickets sold"].groupby(level=0, group_keys=False)
print(total_sold.nlargest(12))
# Plots of Total price by Hour, Year and month
sns.pairplot(data.dropna(),
             hue='Tickets sold',
             x_vars=['paid-Hour','paid-Month','paid-Year'],
             y_vars='Total Price (after discount)',
             height=5,
             plot_kws={'alpha':0.1, 'linewidth':0}
            )
plt.suptitle('Total price by Hour, Year and month of Year')
plt.show()
total_sold.plot(figsize=(16,4),legend=True)
plt.title('،Ticket Sales Yy Years - Months')
plt.show()
group2 = data.groupby(['paid-Year','paid-Month']).sum()
total_sold_per_hour = group2['Total Price (after discount)'].groupby(level=0, group_keys=False)
total_sold_per_hour.plot(figsize=(15,4),legend=True)
plt.title('Price Volatility By Year - Month')
plt.show()
pd.plotting.scatter_matrix (data.loc[1:10,:],figsize = (10,10))
plt.show()
pd.plotting.scatter_matrix(data, alpha=0.2, figsize=(10, 10), diagonal='kde')
plt.show()
# Choosing Influential Variables by ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesClassifier

X_df = data.drop(columns='Original Price')
array=X_df.values
X=array[:,1:15]
y = data['Original Price'].array
Y=y.astype('int')

# feature extraction
forest = ExtraTreesClassifier()
forest.fit(X, Y)
importances = forest.feature_importances_
print(importances)
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color='gold', yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from matplotlib import cm
from colorspacious import cspace_converter
from collections import OrderedDict
new_df['airline'] = new_df['airline'].astype(str)
label_encoder = LabelEncoder()
new_df['airline'] = label_encoder.fit_transform(new_df['airline'])
colormap = plt.cm.spring
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1, size=15)
sns.heatmap(new_df.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
#we have multicollinearity in our variable, there are 2 ways for handling : apply pca or using ridge regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

array = new_df.values
Xdf = new_df.drop(columns='log_price')
array=Xdf.values
X=array[:,0:4]

y = new_df['log_price'].array
# feature extraction
X=scale(Xdf)
pca = PCA(n_components=2)
fit = pca.fit(X)

# summarize components
print("Explained Variance: %s",fit.explained_variance_)
print("Explained Variance Ration: %s",fit.explained_variance_ratio_)
print("Explained Variance cumulative: %s",fit.explained_variance_ratio_.cumsum())
print("Singular values",fit.singular_values_) 
print(fit.components_)




