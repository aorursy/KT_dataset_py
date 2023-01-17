# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing necessery libraries for future analysis of the dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
#%matplotlib notebook
import seaborn as sns
import pandas_profiling 

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from wordcloud import WordCloud
data_dir = '/kaggle/input/new-york-city-airbnb-open-data/'
fina_name = 'AB_NYC_2019.csv'
airbnb_df=pd.read_csv(os.path.join(data_dir,fina_name))
airbnb_df.head(10)
airbnb_df.shape
airbnb_df.info()
airbnb_df.head(4).T
airbnb_df.isnull().sum()
airbnb_df.describe().T
airbnb_df.select_dtypes(include=['object']).nunique()
airbnb_df.room_type.value_counts()                 
distinct_room_type = airbnb_df.room_type.value_counts()
distinct_room_type.plot(kind='bar')
distinct_neighbourhood_group = airbnb_df.neighbourhood_group.value_counts()
distinct_neighbourhood_group.plot(kind='bar')
distinct_neighbourhood_group
plt.figure(figsize=(10,10))
a = sns.scatterplot(data=airbnb_df, x='longitude', y='latitude', hue='neighbourhood_group')
plt.title('Map of airbnb neighbourhood distribution', fontsize=15)
plt.xlabel('Latitude')
plt.ylabel("Longitude")
plt.legend(frameon=False, fontsize=13)
plt.figure(figsize=(10,10))
a = sns.scatterplot(data=airbnb_df, x='longitude', y='latitude', hue='availability_365')
plt.title('Map of airbnb Based on Availability', fontsize=15)
plt.xlabel('Latitude')
plt.ylabel("Longitude")
plt.legend(frameon=False, fontsize=13)
airbnb_df.availability_365.value_counts()
plt.subplots(figsize=(25,15))
wordcloud = WordCloud(
                          background_color='white',
                          width=1920,
                          height=1080
                         ).generate(" ".join(airbnb_df.name.apply(lambda x: str(x))))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('airbnb_df_prop_name.png')
plt.show()
plt.subplots(figsize=(25,15))
wordcloud = WordCloud(
                          background_color='white',
                          width=1920,
                          height=1080
                         ).generate(" ".join(airbnb_df.neighbourhood))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('neighbourhood.png')
plt.show()
airbnb_df.neighbourhood.value_counts().head(10)
top_host=airbnb_df.host_id.value_counts().head(10)
top_host
top_host.plot(kind='bar')
top_host.index
index = top_host.index
index[0]
airbnb_df_top_host = airbnb_df[airbnb_df.host_id.isin(index)]
airbnb_df_top_host['neighbourhood_group'].value_counts()
#airbnb_df_top_host.sort_values(by='id',ascending=False)
airbnb_df_top_host_table = pd.pivot_table(airbnb_df_top_host, index=['host_id','neighbourhood_group'])
airbnb_df_top_host_table.sort_values(by='calculated_host_listings_count',ascending=False)
airbnb_df.groupby(airbnb_df['neighbourhood_group']).count()
neighbourhood_groups=[]

for neighbourhood in airbnb_df['neighbourhood_group'].unique():
    sub_neighbourhood =airbnb_df.loc[airbnb_df['neighbourhood_group'] == neighbourhood]
    price_sub_neighbourhood=sub_neighbourhood[['price']]
    #neighbourhood_groups.append(price_sub_neighbourhood.describe())
    i=price_sub_neighbourhood.describe(percentiles=[.25, .50, .75])
    #i=i.iloc[3:]
    i.reset_index(inplace=True)
    i.rename(columns={'index':'Stats'}, inplace=True)
    i.rename(columns={'price':neighbourhood}, inplace=True)
    neighbourhood_groups.append(i)
    
neighbourhood_groups_stat_df=[df.set_index('Stats') for df in neighbourhood_groups]
neighbourhood_groups_stat_df=neighbourhood_groups_stat_df[0].join(neighbourhood_groups_stat_df[1:])
neighbourhood_groups_stat_df.T
## Above code can be simplified by 
airbnb_df.groupby('neighbourhood_group').price.describe()
airbnb_df.groupby('neighbourhood_group').room_type.describe()
airbnb_df.drop(['name', 'host_name', 'last_review', 'id'], inplace=True, axis=1)
airbnb_df['reviews_per_month'].fillna(value=0, inplace=True)
airbnb_df.sample(2).T
plt.figure(figsize=(15,8))
sns.heatmap(airbnb_df.corr(), annot=True, linewidths=0.1)
labelEncoder_nbhgp = preprocessing.LabelEncoder()
labelEncoder_nbh = preprocessing.LabelEncoder()
labelEncoder_rm_type = preprocessing.LabelEncoder()

airbnb_df['neighbourhood_group']=labelEncoder_nbhgp.fit_transform(airbnb_df['neighbourhood_group'])
airbnb_df['neighbourhood']=labelEncoder_nbh.fit_transform(airbnb_df['neighbourhood'])
airbnb_df['room_type']=labelEncoder_rm_type.fit_transform(airbnb_df['room_type'])

labelEncoder_rm_type.classes_
airbnb_df.sample(2)
X = airbnb_df.drop(['price'], inplace=False, axis=1)
y = airbnb_df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)
# The coefficients for each of the independent attributes    
    
for idx, column in enumerate(X.columns) :
    print("The coefficient for {} is {}".format(column,regression_model.coef_[idx]))
accuracy = regression_model.score(X_test, y_test)
print("The accuracy for our model is {}".format(accuracy))
plt.figure(figsize=(15,8))
sns.heatmap(airbnb_df.corr(), annot=True, linewidths=0.1)
airbnb_df.corr()['price']
import lightgbm as lgb
def create_lgb_model(x_train, x_val, y_train, y_val):
    params = {
            "objective" : "regression",
            "metric" : "rmse",
            "num_leaves" : 30,
            "learning_rate" : 0.1,
            "bagging_fraction" : 0.7,
            "feature_fraction" : 0.7,
            "bagging_frequency" : 5,
            "bagging_seed" : 2018,
            "verbosity" : -1
        }
    lgtrain = lgb.Dataset(x_train, label=y_train)
    lgval = lgb.Dataset(x_val, label=y_val)
    evals_result = {}
    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=20, evals_result=evals_result)
    return model,evals_result
    


model,evals_result = create_lgb_model(X_train, X_test, y_train, y_test)
pred_test = model.predict(X_test, num_iteration=model.best_iteration)
fig, ax = plt.subplots(figsize=(12,18))
lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=15)
plt.show()
