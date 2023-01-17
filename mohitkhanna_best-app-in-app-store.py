# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("ggplot")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
appstore_df= pd.read_csv("../input/AppleStore.csv", sep=",", index_col="id", header=0)
app_description_df=pd.read_csv("../input/appleStore_description.csv", sep=",", index_col="id", header=0)
print("The size of the appstore data is {0}".format(appstore_df.shape))
print("The dimension of the appstore data is {0}".format(appstore_df.ndim))
print("The size of the app description data is {0}".format(app_description_df.shape))
print("The dimension of the appstore data is {0}".format(app_description_df.ndim))
combined_app_data= pd.merge(appstore_df,app_description_df,left_index=True, right_index=True)
combined_app_data.shape
combined_app_data.head()
assert len(combined_app_data["Unnamed: 0"].value_counts())==combined_app_data.shape[0]
combined_app_data=combined_app_data.drop("Unnamed: 0", axis=1)
combined_app_data=combined_app_data.drop("track_name_y", axis=1)
combined_app_data=combined_app_data.drop("size_bytes_y", axis=1)
combined_app_data.shape
columns_list=combined_app_data.columns.tolist()
print("The columns in our data frame are {0}".format(columns_list))
combined_app_data.info()
combined_app_data.describe()
combined_app_data.isna().sum().to_frame().T
data_type_df=combined_app_data.dtypes.value_counts().reset_index()
data_type_df.columns=["Data type", "count"]
data_type_df
categorical_column_list=combined_app_data.loc[:,combined_app_data.dtypes=="object"].columns.tolist()
print("The categorical columns is out data are {0} \n".format(categorical_column_list))
numerical_column_list= [col for col in columns_list if col not in categorical_column_list]
print("The numerical columns is out data are {0} \n".format(numerical_column_list))
combined_app_data.loc[combined_app_data.rating_count_tot.idxmax()].to_frame().T
combined_app_data.sort_values(by="rating_count_tot", ascending=False).track_name_x.head(5).reset_index().T
combined_app_data.sort_values(by="user_rating", ascending=False).track_name_x.head(5).reset_index().T
combined_app_data.sort_values(by=["rating_count_tot","user_rating"], ascending=False).track_name_x.head(5).reset_index().T
combined_app_data["isFree"]= np.where(combined_app_data.price==0.00, "Free app", "Paid app")
combined_app_data.isFree.value_counts().plot.bar()
fig, ax = plt.subplots()
fig.set_size_inches(3.7, 6.27)
sns.barplot(x="isFree", y="rating_count_tot", data=combined_app_data,ax=ax)
combined_app_data.groupby("user_rating")['track_name_x'].count().plot.bar()
combined_app_data["lang.num"].value_counts().sort_index().to_frame().T
combined_app_data["lang.num_descrete"]=pd.cut(combined_app_data["lang.num"], bins=[0,5,10,20, 50,80],labels=["<5","5-10","10-20","20-50", "50-80"])
combined_app_data["lang.num_descrete"].value_counts().plot.bar()
fig, ax = plt.subplots()
fig.set_size_inches(19.7, 8.27)
xx=sns.pointplot(x="lang.num", y="rating_count_tot", data=combined_app_data, hue="lang.num_descrete",ax=ax)
ax.set(xlabel='No of language support', ylabel='Total rating count')
plt.show()
fig, ax = plt.subplots()
fig.set_size_inches(19.7, 8.27)
combined_app_data.prime_genre.value_counts().plot.bar()
fig, ax = plt.subplots()
fig.set_size_inches(29.7, 8.27)
sns.barplot(x="prime_genre", y="rating_count_tot", data=combined_app_data,ax=ax)
fig, ax = plt.subplots()
fig.set_size_inches(19.7, 8.27)
combined_app_data["sup_devices.num"].value_counts().plot.bar()
fig, ax = plt.subplots()
fig.set_size_inches(29.7, 8.27)
sns.barplot(x="sup_devices.num", y="rating_count_tot", data=combined_app_data,ax=ax)
fig, ax = plt.subplots()
fig.set_size_inches(29.7, 8.27)
sns.barplot(x="ipadSc_urls.num", y="rating_count_tot", data=combined_app_data,ax=ax)
plt.scatter(x=combined_app_data.size_bytes_x, y=combined_app_data.rating_count_tot)















