import numpy as np
import pandas as pd
data = pd.read_csv("../input/del data.csv")

# Any results you write to the current directory are saved as output.

data['first_order'] = pd.to_datetime(data['first_order'])
data['recent_order'] = pd.to_datetime(data['recent_order'])
data.info()
data.describe()
#to check null for all columns in df
data.isnull().sum()
# as of now i am replacing it by 0 but i'll correct my data later and will populate these values
data = data.fillna(0)

#Avg resturant distance is negative in 44 cases, for simplicity I am calling them as 0.I am taking them
#as absolute value of that number
data['avg_distancefromresturant'] = abs(data['avg_distancefromresturant'])
(data['orders_last7days'] > 0.0).count()
#orders last 7 days
#orders last 28 days 
data['orders_last7days'] = pd.to_numeric(data['orders_last7days'], downcast = 'integer')
#Order Count, AOV, Avg distance from Restaurant
#deriving new columns
data['avg_all'] = data['amount']/data['total_orders']
data['avg_last7d'] = data['amount_last7days']/data['orders_last7days']
data['avg_last4w'] = data['amount_last4weeks']/data['orders_last4weeks']

data1 = data[['customer_id','total_orders','orders_last7days', 'orders_last4weeks',
              'avg_distancefromresturant','avg_deliverytime',
              'avg_all','avg_last7d','avg_last4w']]
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# Changing the datatype of Customer Id as per Business understanding

data['customer_id'] = data['customer_id'].astype(str)
#Compute last order date and first orderdate to get the recency of customers
data['recent'] = data['recent_order']-data['first_order'] 
data
#remove time from number of days
# Extract number of days only

data['recent'] = data['recent'].dt.days
# Outlier Analysis of Amount Frequency and Recency
import matplotlib.pyplot as plt
import seaborn as sns
attributes = ['recent','avg_all','avg_last7d','avg_last4w']
plt.rcParams['figure.figsize'] = [10,8]
sns.boxplot(data = data[attributes], orient="v", palette="Set2" ,whis=1.5,saturation=1, width=0.7)
plt.title("Outliers Variable Distribution", fontsize = 14, fontweight = 'bold')
plt.ylabel("Range", fontweight = 'bold')
plt.xlabel("Attributes", fontweight = 'bold')
data.describe()
#removing outliers 
q1 = data.recent.quantile(0.05)
q3 = data.recent.quantile(0.95)
IQR = q3-q1
data = data[(data.recent>= q1- 1.5*IQR) & (data.recent<= q3+1.5*IQR)]

q1 = data.avg_all.quantile(0.05)
q3 = data.avg_all.quantile(0.95)
IQR = q3-q1
data = data[(data.avg_all>= q1- 1.5*IQR) & (data.avg_all<= q3+1.5*IQR)]

q1 = data.avg_last7d.quantile(0.05)
q3 = data.avg_last7d.quantile(0.95)
IQR = q3-q1
data = data[(data.avg_last7d>= q1- 1.5*IQR) & (data.avg_last7d<= q3+1.5*IQR)]

q1 = data.avg_last4w.quantile(0.05)
q3 = data.avg_last4w.quantile(0.95)
IQR = q3-q1
data = data[(data.avg_last4w>= q1- 1.5*IQR) & (data.avg_last4w<= q3+1.5*IQR)]
data.describe()
data.describe()
#rescaling/normalizing the attributes
data_normalized = data[['recent', 'avg_all', 'avg_last7d','avg_last4w']]
scaler = StandardScaler()
data_normalized = pd.DataFrame(scaler.fit_transform(data_normalized))
data_normalized.columns = ['recent', 'avg_all', 'avg_last7d','avg_last4w']
# k-means with some arbitrary k

kmeans = KMeans(n_clusters=4, max_iter=50)
kmeans.fit(data_normalized)
kmeans.labels_
# Elbow-curve/SSD
#sum of squared errors
ssd = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8,9,10]
for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(data_normalized)
    
    ssd.append(kmeans.inertia_)
    
# plot the SSDs for each n_clusters
plt.plot(ssd)