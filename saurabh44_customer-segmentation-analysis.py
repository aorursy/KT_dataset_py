# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import plotly.offline as px
px.init_notebook_mode(connected=True)
px.offline.init_notebook_mode(connected=True)
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from kmodes.kmodes import KModes
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Reading all data files....
rates = pd.read_csv('../input/rates.csv')
reservations = pd.read_csv('../input/reservations.csv')
# Join rates and reservation databases on basis of RateId
reservation_with_rates= pd.merge(left=reservations, right=rates, left_on='RateId', right_on='RateId')
# To find Day of Week using given StartUTC.
reservation_with_rates['StartUtc'] = pd.to_datetime(reservation_with_rates['StartUtc'])
reservation_with_rates['Day of Week'] = reservation_with_rates['StartUtc'].dt.day_name()
# Plot popular choices of booking Rates Names
fig = px.histogram(reservation_with_rates, x="RateName")
fig.show()
#  Plot booking rate choice with different Customer Segment (eg. AgeGroup, Gender)
print("AgeGroup v/s RateName")
test5 = reservation_with_rates.groupby(['RateName','AgeGroup'])['AgeGroup'].count().unstack('RateName').plot(kind='bar', stacked=True)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.gcf().set_size_inches(15, 10)
plt.show()
print("Gender v/s RateName")
test5 = reservation_with_rates.groupby(['RateName','Gender'])['Gender'].count().unstack('RateName').plot(kind='bar', stacked=True)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.gcf().set_size_inches(15, 10)
plt.show()
# selecting guest data who pursue online check-in using column IsOnlineCheckin
online_reservation_with_rates = reservation_with_rates[reservation_with_rates.IsOnlineCheckin == 1]
# Plot typical guests who pursue online check-in using BusinessSegment column.
fig = px.histogram(online_reservation_with_rates, x="BusinessSegment")
fig.show()
# Plot variation in typical guests who pursue online check-in using BusinessSegment and Day of Week columns.
print("Online Check-in v/s Business Segment v/s Weekday")
test = online_reservation_with_rates.groupby(['BusinessSegment','Day of Week'])['Day of Week'].count().unstack('BusinessSegment').plot(kind='bar', stacked=True)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.gcf().set_size_inches(15, 10)
plt.show()
# Selecting guest data having null value in CancellationReason and 1 in IsOnlineCheckin column
non_Cancelled_reservation_with_rates = online_reservation_with_rates[online_reservation_with_rates.CancellationReason.notnull()]
non_Cancelled_reservation_with_rates.head()
# Select dataframe which best describes guest segment from given data.
guest_Segment_df=reservation_with_rates[["AgeGroup","Gender","NationalityCode","BusinessSegment","CancellationReason"]]

# Column NationalityCode has 1096 rows with Null values. Replacing these null values with value 'Other'.
guest_Segment_df['NationalityCode'].fillna("Other",inplace=True)
# Column CancellationReason code has 1806 rows with Null values. Replacing these null values with value '100'.
guest_Segment_df['CancellationReason'].fillna(100,inplace=True)

# Categorical encoding of NationalityCode column. Converting categorical values to binary values using one hot encoding method
guest_Segment_df['NationalityCode'] = pd.Categorical(guest_Segment_df['NationalityCode'])
dfDummies_Nationality = pd.get_dummies(guest_Segment_df['NationalityCode'], prefix = 'nationality')
Nationality_guest_Segment_df = pd.concat([guest_Segment_df, dfDummies_Nationality], axis=1)

# Categorical encoding of BusinessSegment column. Converting categorical values to binary values using one hot encoding method
Nationality_guest_Segment_df['BusinessSegment'] = pd.Categorical(Nationality_guest_Segment_df['BusinessSegment'])
dfDummies_Business = pd.get_dummies(Nationality_guest_Segment_df['BusinessSegment'], prefix = 'business')
business_Segment_guest_Segment_df = pd.concat([Nationality_guest_Segment_df, dfDummies_Business], axis=1)

# Encoding of CancellationReason column. Converting ordinal values (0<1<2<3..) values to binary values (1s and 0s) using one hot encoding method
business_Segment_guest_Segment_df['CancellationReason'] = pd.Categorical(business_Segment_guest_Segment_df['CancellationReason'])
dfDummies_cancellation = pd.get_dummies(business_Segment_guest_Segment_df['CancellationReason'], prefix = 'cancellation')
cancellation_guest_Segment_df = pd.concat([business_Segment_guest_Segment_df, dfDummies_cancellation], axis=1)

# Encoding of Gender column. Converting ordinal values (0<1<2) values to binary values (1s and 0s) using one hot encoding method
cancellation_guest_Segment_df['Gender'] = pd.Categorical(cancellation_guest_Segment_df['Gender'])
dfDummies_gender = pd.get_dummies(cancellation_guest_Segment_df['Gender'], prefix = 'gender')
encoded_guest_Segment_df = pd.concat([cancellation_guest_Segment_df, dfDummies_gender], axis=1)

labelencoder = LabelEncoder()
encoded_guest_Segment_df['AgeGroup'] = labelencoder.fit_transform(encoded_guest_Segment_df['AgeGroup'])
encoded_guest_Segment_df = encoded_guest_Segment_df.drop(["BusinessSegment", "NationalityCode", "CancellationReason","Gender"], axis=1)
distortions = []
K = range(1,10)
for k in K:
    kmodeModel = KModes(n_clusters=k).fit(encoded_guest_Segment_df)
    kmodeModel.fit(encoded_guest_Segment_df)
    distortions.append(sum(np.min(cdist(encoded_guest_Segment_df, kmodeModel.cluster_centroids_, 'hamming'), axis=1)) / encoded_guest_Segment_df.shape[0])

# Plot the elbow graph
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
# Scaling all the features in the dataframe to normalize the data in a particular range. 
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(encoded_guest_Segment_df)
# Number of clusters are 4 (derived from elbow method)
kmodes = KModes(n_clusters=4, random_state=0) 
y = kmodes.fit_predict(X_scaled)
reservation_with_rates['Cluster_Segment'] = y

# Grouping NightCost_Sum and OccupiedSpace_Sum on the basis of Cluster_Segment
nightcost_Sum_reservation_with_rates = reservation_with_rates.groupby("Cluster_Segment",as_index=False)["NightCost_Sum","OccupiedSpace_Sum"].sum()
# Calculating Night Cost per occupied space from above dataframe
nightcost_Sum_reservation_with_rates['Night Cost per occupied space'] = nightcost_Sum_reservation_with_rates['NightCost_Sum']/nightcost_Sum_reservation_with_rates['OccupiedSpace_Sum']
profit_reservation_with_rates = nightcost_Sum_reservation_with_rates.sort_values('Night Cost per occupied space')

# Plotting Night Cost per occupied space v/s Cluster_Segment (Most and Least profitable Guest Segment)
fig = px.bar(profit_reservation_with_rates, x="Cluster_Segment", y="Night Cost per occupied space", orientation='v')
fig.show()