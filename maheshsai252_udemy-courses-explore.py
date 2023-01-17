import pandas as pd

data = pd.read_csv("../input/udemy-courses/clean_dataset.csv")
data.head()
data.describe()
len(data['course_title'].value_counts())
data.shape
data_paid= data[data['is_paid']==True]
data_paid.shape
data_paid.head()
data_free=data[data['is_paid']==False]
data_free.shape
data_free.head()
data_free.sort_values(by='num_subscribers',ascending=False)
data_paid.sort_values(by='num_subscribers',ascending=False)
import seaborn as sns

import matplotlib.pyplot as plt
sns.set_style('ticks')

fig, ax = plt.subplots()

fig.set_size_inches(11.7, 8.27)

fig.set

sns.scatterplot(x="price", y="num_subscribers",hue="num_subscribers",ax=ax ,data=data_paid).set(title = 'price vs subscribers(paid)')



ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)
data_paid[data_paid['num_subscribers']==max(data_paid['num_subscribers'])]
sns.set_style('ticks')

fig, ax = plt.subplots()

fig.set_size_inches(11.7, 8.27)

fig.set

sns.scatterplot(x="price", y="num_lectures",hue="num_lectures",ax=ax ,data=data_paid).set(title = 'price vs number of lectures(paid)',xlabel= "price")





ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)
data_paid['subject'].value_counts()
data_paid[data_paid['price']=='200']['subject'].value_counts()
data_free['subject'].value_counts()
sns.countplot(x='subject', data=data_paid)
sns.countplot(x='subject', data=data_free)
import re



data[data['course_title'].str.contains(r'Data')== True]
sns.set_style('ticks')

fig, ax = plt.subplots()

fig.set_size_inches(11.7, 8.27)

fig.set

sns.scatterplot(x="price", y="engagement",hue="num_lectures",ax=ax ,data=data_paid).set(title = 'price vs engagement(paid)')





ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)

data_paid[data_paid['engagement']==1.0]
sns.set_style('ticks')

fig, ax = plt.subplots()

fig.set_size_inches(11.7, 8.27)

sns.set_palette("Blues_d")

sns.scatterplot(x="num_lectures", y="engagement",hue="num_lectures",ax=ax ,data=data_paid).set(title = 'engagement vs number of lectures(paid)')



ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)

data_paid[data_paid['num_lectures']==max(data_paid['num_lectures'])]
sns.set_style('ticks')

fig, ax = plt.subplots()

fig.set_size_inches(11.7, 8.27)

fig.set

sns.scatterplot(x="num_subscribers", y="num_reviews",hue="num_reviews",ax=ax ,data=data_paid).set(title = 'price vs number of lectures(paid)')





ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)
data_paid_10=data_paid.sort_values(by='num_subscribers',ascending=False)[0:10].sort_values("num_subscribers", ascending=False).reset_index(drop=True).reset_index()[['course_id','course_title','num_subscribers','num_reviews','price']]
data_paid_10


sns.set_style('ticks')

fig, ax = plt.subplots()

fig.set_size_inches(11.7, 8.27)

fig.set

sns.barplot(x="course_title", y="num_subscribers",ax=ax ,data=data_paid_10).set(title = 'price vs number of lectures(paid)')

plt.xticks(rotation=90)



ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)
data_free_10=data_free.sort_values(by='num_subscribers',ascending=False)[0:10].sort_values("num_subscribers", ascending=False).reset_index(drop=True).reset_index()[['course_id','course_title','num_subscribers','num_reviews','price']]
data_free_10


sns.set_style('ticks')

fig, ax = plt.subplots()

fig.set_size_inches(11.7, 8.27)

fig.set

sns.barplot(x="course_title", y="num_subscribers",ax=ax ,data=data_free_10).set(title = 'price vs number of lectures(paid)')

plt.xticks(rotation=90)





ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)
data_paid['subject'].value_counts()
data_paid_business = data_paid[data_paid['subject']=='Business Finance']
data_paid_business['price']=data_paid_business['price'].apply(lambda x:int(x))

type(data_paid_business['price'][0])
sns.distplot(data_paid_business['price'])



ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)
data_paid_development = data_paid[data_paid['subject']=='Web Development']
data_paid_development['price']=data_paid_development['price'].apply(lambda x:int(x))
sns.distplot(data_paid_development['price'])
data_paid_musical = data_paid[data_paid['subject']=='Musical Instruments']
data_paid_musical['price']=data_paid_musical['price'].apply(lambda x:int(x))
sns.distplot(data_paid_musical['price'])
data_paid_musical[data_paid_musical['price']==200]
data1=data

datat=data1.drop(['course_id','course_title','url','num_reviews','published_timestamp','engagement','content_multiplier'],axis=1)
data.isnull().sum()
datat.head()
data['level'].value_counts()
sns.countplot(x='level',data=data)
datat.corr()
y = datat['num_subscribers']

x = datat.drop(['num_subscribers'],axis=1)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x['is_paid'] = x['is_paid'].apply(lambda x:str(x))

x['price'] = x['price'].apply(lambda x: 0 if x=='Free' else x)

x['price'] = x['price'].apply(lambda x:int(x))
#x['price'] = scaler.fit_transform(x.price.values.reshape(-1, 1))
#x['num_lectures'] = scaler.fit_transform(x.num_lectures.values.reshape(-1, 1))
#x['content_duration'] = scaler.fit_transform(x.content_duration.values.reshape(-1, 1))
#x['content_time_value'] = scaler.fit_transform(x.content_time_value.values.reshape(-1, 1))
x = pd.get_dummies(x)
x.columns

from sklearn.ensemble import RandomForestRegressor 

from sklearn.model_selection import train_test_split
regressor =  RandomForestRegressor(n_estimators = 100, random_state = 0) 

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
feat_importances = pd.Series(regressor.feature_importances_, index=x.columns)

feat_importances.plot(kind='barh')
from sklearn.metrics import mean_absolute_error as mse

mse_sub = mse(y_pred,y_test)
mse_sub


from xgboost import XGBRegressor
regressor = XGBRegressor()

regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
mse_sub = mse(y_pred,y_test)
mse_sub
feature_important = regressor.get_booster().get_score(importance_type='weight')

keys = list(feature_important.keys())

values = list(feature_important.values())



data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)

data.plot(kind='barh')