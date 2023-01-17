import pandas as pd 
import numpy as np
from scipy import stats 
from mlxtend.preprocessing import  minmax_scaling
import seaborn as sns 
import matplotlib.pyplot as plt
kickstarters_data=pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")
kickstarters_data
np.random.seed(0)
orignal_data=np.random.exponential(size=1000)
orignal_data

scaled_data = minmax_scaling(orignal_data , columns = [0])
scaled_data
fig, ax=plt.subplots(1,2)
sns.distplot(orignal_data,ax=ax[0])
ax[0].set_title("original Data")
ax[1].set_title("scaled Data")
sns.distplot(scaled_data,ax=ax[1])
normalized_data=stats.boxcox(orignal_data)
normalized_data

fig, ax=plt.subplots(1,2)
sns.distplot(orignal_data, ax=ax[0])
ax[0].set_title('original data')
ax[1].set_title('normalized data')
sns.distplot(normalized_data[0] ,ax=ax[1])
kickstarters_data.head()

usd_goal_real_column=kickstarters_data['usd_goal_real']
usd_goal_real_column

scaled_usd_real_column=minmax_scaling(usd_goal_real_column, columns =[0])
scaled_usd_real_column
fig, ax=plt.subplots(1,2)
ax[0].set_title("original column data");
ax[1].set_title("scaled column data");
sns.distplot(usd_goal_real_column, ax=ax[0])
sns.distplot(scaled_usd_real_column ,ax=ax[1])
kickstarters_data.head()
goal_column=kickstarters_data["goal"]
goal_column
scaled_goal_column=minmax_scaling(goal_column, columns =[0])
scaled_goal_column
fig , ax=plt.subplots(1,2)
ax[0].set_title(" original goal column data")
ax[1].set_title(" scaled goal column data")
sns.distplot(goal_column , ax=ax[0])
sns.distplot(scaled_goal_column ,ax=ax[1])
kickstarters_data.head()
postive_pledege_usd_index=kickstarters_data.usd_pledged_real > 0
postive_pledege_usd_index
positive_pledge_usa =kickstarters_data["usd_pledged_real"].loc[postive_pledege_usd_index]
positive_pledge_usa
normalized_usd_pledged_real =stats.boxcox(positive_pledge_usa)
normalized_usd_pledged_real
fig, ax=plt.subplots(1,2)
ax[0].set_title("original data")
ax[1].set_title("normalized data")
sns.distplot(positive_pledge_usa,ax=ax[0])
sns.distplot(normalized_usd_pledged_real[0] ,ax=ax[1])
kickstarters_data.head()
index_pledge=kickstarters_data['pledged'] > 0

index_pledge
positive_pledge = kickstarters_data['pledged'].loc[index_pledge]
positive_pledge
normailzed_pledge_data=stats.boxcox(positive_pledge)[0]
normailzed_pledge_data
fig , ax=plt.subplots(1,2)
ax[0].set_title("original data")
ax[1].set_title("normalized data")
sns.distplot(positive_pledge,ax=ax[0])
sns.distplot(normailzed_pledge_data,ax=ax[1])
