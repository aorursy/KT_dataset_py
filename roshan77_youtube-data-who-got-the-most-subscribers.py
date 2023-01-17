import numpy as np 
import pandas as pd 

import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm
df = pd.read_csv('../input/data.csv')
df.head()
df.tail()
df.info()
#df['Subscribers'] = df['Subscribers'].convert_objects(convert_numeric=True)
#df['Video Uploads'] = df['Video Uploads'].convert_objects(convert_numeric=True)

df['Subscribers'] = pd.to_numeric(df['Subscribers'], errors='coerce')
df['Video Uploads'] = pd.to_numeric(df['Video Uploads'], errors='coerce')
df.head(20).plot.bar(x = 'Channel name', y = 'Subscribers')
plt.title('Number of subscribers of top 20 channels')
df.head(20).plot.bar(x = 'Channel name', y = 'Video views')
plt.title('Number of video views of top 20 channels')
df.head(20).plot.bar(x = 'Channel name', y = 'Video Uploads')
plt.title('Number of video uploads of top 20 channels')
df.sort_values(by = ['Subscribers'], ascending = False).head(20).plot.bar(x = 'Channel name', y = 'Subscribers')
plt.title('Top 20 channels with maximum number of subscribers')
df.sort_values(by = ['Video views'], ascending = False).head(20).plot.bar(x = 'Channel name', y = 'Video views')
plt.title('Top 20 channels with maximum number of video views')
df.sort_values(by = ['Video Uploads'], ascending = False).head(20).plot.bar(x = 'Channel name', y = 'Video Uploads')
plt.title('Top 20 channels with maximum number of video uploads')
df.sort_values(by = ['Subscribers'], ascending = False).plot(x = 'Channel name', y = 'Subscribers')
plt.xlabel('Ranking by subscribers')
plt.ylabel('Number of subscribers')
df.sort_values(by = ['Video views'], ascending = False).plot(x = 'Channel name', y = 'Video views')
plt.xlabel('Ranking by video views')
plt.ylabel('Number of video views')
df.sort_values(by = ['Video Uploads'], ascending = False).plot(x = 'Channel name', y = 'Video Uploads')
plt.xlabel('Ranking by video uploads')
plt.ylabel('Number of video uploads')
grade_name = list(set(df['Grade']))
grade_name
df_by_grade = df.set_index(df['Grade'])

count_grade = list()
for grade in grade_name:
    count_grade.append(len(df_by_grade.loc[[grade]]))
df_by_grade.head()
print(count_grade)
print(grade_name)
grade_name[2] = 'missing'
labels = grade_name
sizes = count_grade

explode1 = (0.2, 0.2, 0.5, 0.2, 0.2, 0.2)
color_list = ['green',  'red', 'gold', 'blue', 'lightskyblue', 'brown']

patches, texts = plt.pie(sizes, colors = color_list, explode = explode1, 
                         shadow = False, startangle = 90, radius = 3)
plt.legend(patches, labels, loc = "best")
plt.axis('equal')
plt.title('Classification of channels by grades')
plt.show()
df.describe()
props = dict(boxes="gold", whiskers="Black", medians="Black", caps="Black")
df.plot.box(color=props, patch_artist=True)
plt.yscale('log')
plt.ylabel('Log count')
plt.subplots(figsize=(8, 5))
sns.heatmap(df.corr(), cmap = 'RdGy')
plt.title('Correlation Matrix Plot')
df_clean = df.dropna()
sns.pairplot(df_clean)
X = df_clean[['Video Uploads', 'Video views']]
Y = df_clean[['Subscribers']]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
lm = LinearRegression()
lm.fit(X_train.dropna(),y_train.dropna())
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions, color = 'red')
plt.xlabel('Y in test set')
plt.ylabel('Predicted Y')
sns.residplot(y_test, predictions,  color="g")
plt.ylabel('d')
plt.xlabel('instances')
plt.title('standardized residual plot')
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
coefficients = pd.DataFrame(X.columns)
coefficients['coefficients']=lm.coef_[0]
coefficients
model = sm.OLS(Y, X).fit() 
predictions = model.predict(X_test)
model.summary()
df['Subscribers'].hist(bins = 200)
plt.xlabel('Number of subscribers')
plt.ylabel('Number of channels')
df['Video views'].hist(bins = 200)
plt.xlabel('Number of video views')
plt.ylabel('Number of channels')
df['Video Uploads'].hist(bins = 200)
plt.xlabel('Number of video uploads')
plt.ylabel('Number of channels')
np.log(df['Subscribers']).hist(bins = 20)
plt.xlabel('Log of number of subscribers')
plt.ylabel('Number of channels')
np.log(df['Video views']).hist(bins = 20)
plt.xlabel('Log of number of video views')
plt.ylabel('Number of channels')
np.log(df['Video Uploads']).hist(bins= 20)
plt.xlabel('Log of number of video uploads')
plt.ylabel('Number of channels')
df_log = pd.DataFrame()
df_log['Video_uploads_log'] = np.log(df_clean['Video Uploads'])
df_log['Video_views_log'] = np.log(df_clean['Video views'])
df_log['Subscribers_log'] = np.log(df_clean['Subscribers'])
df_log.head()
df_log.tail()
plt.subplots(figsize=(8, 5))
sns.heatmap(df_log.corr(), cmap = 'RdGy')
sns.pairplot(df_log)
X2 = df_log[['Video_uploads_log', 'Video_views_log']]
Y2 = df_log[['Subscribers_log']]
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, Y2, test_size = 0.2)
lm2 = LinearRegression()
lm2.fit(X2_train.dropna(),y2_train.dropna())
predictions2 = lm2.predict(X2_test)
plt.scatter(y2_test,predictions2, color = 'red')
plt.xlabel('Y in test set')
plt.ylabel('Predicted Y')
sns.residplot(y2_test, predictions2,  color="g")
plt.ylabel('d')
plt.xlabel('instances')
plt.title('standardized residual plot')
print('MAE:', metrics.mean_absolute_error(y2_test, predictions2))
print('MSE:', metrics.mean_squared_error(y2_test, predictions2))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y2_test, predictions2)))
coefficients2 = pd.DataFrame(X2.columns)
coefficients2['coefficients']=lm2.coef_[0]
coefficients2
model2 = sm.OLS(Y2, X2).fit() 
predictions2 = model2.predict(X2_test)
model2.summary()
p = coefficients2['coefficients'][0]
q = coefficients2['coefficients'][1]
def pred_from_log(x, y):
    return x ** p + y ** q
X_test.head()
vid_upl_test = np.array(X_test['Video Uploads'])
vid_viw_test = np.array(X_test['Video views'])
prediction_log = pred_from_log(vid_upl_test, vid_viw_test)
plt.scatter(predictions, prediction_log, color = 'r', alpha = 0.5)
plt.xlabel('prediction without log transformation')
plt.ylabel('prediction with log transformation')
plt.scatter(range(len(X_test)), predictions - prediction_log, color = 'red', alpha = 0.5)
plt.xlabel('count of test data')
plt.ylabel('difference of prediction with and without log')

