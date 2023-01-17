# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt #data visualization
import seaborn as sns #statistical data visualization
from sklearn.model_selection import train_test_split #evaluating estimator performance
from sklearn.linear_model import LinearRegression ##evaluating estimator performance
from sklearn import metrics #machine learning.
import statsmodels.api as sm #data frames
df = pd.read_csv('../input/YOUTUBE.csv')
df.head()
df.tail()
df.info()
df['Subscribers'] = df['Subscribers'].convert_objects(convert_numeric=True)
df['Video Uploads'] = df['Video Uploads'].convert_objects(convert_numeric=True)
#30 premiere valeur 
df.head(30).plot.bar(x = 'Channel name', y = 'Subscribers')
plt.title('Number of subscribers of top 30 channels')
#nombre de video vue
df.head(30).plot.bar(x = 'Channel name', y = 'Video views')
plt.title('Number of video views of top 30 channels')
df.head(30).plot.bar(x = 'Channel name', y = 'Video Uploads')
plt.title('Number of video uploads of top 30 channels')
df.sort_values(by = ['Subscribers'], ascending = False).head(30).plot.bar(x = 'Channel name', y = 'Subscribers')
plt.title('Top 30 channels with maximum number of subscribers')
df.sort_values(by = ['Video views'], ascending = False).head(30).plot.bar(x = 'Channel name', y = 'Video views')
plt.title('Top 30 channels with maximum number of video views')
df.sort_values(by = ['Video Uploads'], ascending = False).head(30).plot.bar(x = 'Channel name', y = 'Video Uploads')
plt.title('Top 30 channels with maximum number of video uploads')
df.describe()
#En regardant l'intrigue ci-dessous, on voit que le nombre d'abonnés est positivement corrélé au nombre de téléspectateurs. Cela est prévu. 
#Mais le nombre d'abonnés est négativement corrélé au nombre de vidéos téléchargées par cette chaîne. Cela pourrait être surprenant. 
#Les canaux vidéo attirant le plus grand nombre de viwers et d’abonnés téléchargent un plus petit nombre de vidéos.
plt.subplots(figsize=(7, 7))
sns.heatmap(df.corr(), cmap = 'PRGn')
plt.title('Correlation Matrix Plot')
#Les données contiennent des valeurs non numériques. Donc, si les données nettoyées 
#sont présentées sur la matrice du diagramme de dispersion des corrélations, 
#la conclusion susmentionnée concernant la corrélation de trois variables est plus évidente.
#datacleaning
df_clean = df.dropna()
sns.pairplot(df_clean)
X = df_clean[['Video Uploads', 'Video views']]
Y = df_clean[['Subscribers']]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
lm = LinearRegression()
lm.fit(X_train.dropna(),y_train.dropna())
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions, color = 'yellow')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
sns.residplot(y_test, predictions,  color="R")
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
df['Subscribers'].hist(bins = 100)
plt.xlabel('Number of subscribers')
plt.ylabel('Number of channels')
df['Video views'].hist(bins = 100)
plt.xlabel('Number of video views')
plt.ylabel('Number of channels')
df['Video Uploads'].hist(bins = 100)
plt.xlabel('Number of video uploads')
plt.ylabel('Number of channels')

np.log(df['Subscribers']).hist(bins = 200)
plt.xlabel('Log of number of subscribers')
plt.ylabel('Number of channels')
np.log(df['Video views']).hist(bins = 200)
plt.xlabel('Log of number of video views')
plt.ylabel('Number of channels')
np.log(df['Video Uploads']).hist(bins= 200)
plt.xlabel('Log of number of video uploads')
plt.ylabel('Number of channels')
df_log = pd.DataFrame()
df_log['Video_uploads_log'] = np.log(df_clean['Video Uploads'])
df_log['Video_views_log'] = np.log(df_clean['Video views'])
df_log['Subscribers_log'] = np.log(df_clean['Subscribers'])
df_log.head()
df_log.tail()

plt.subplots(figsize=(7, 7))
sns.heatmap(df_log.corr(), cmap = 'PRGn')
#A partir du tracé de corrélation ci-dessus, le coefficient de corrélation des variables 
#n'a pas été modifié après la transformation du journal. Au moins, la corrélation positive reste positive et inversement.
#Mais si nous regardons le nuage de points ci-dessous, la corrélation négative entre 
#les téléchargements vidéo et les abonnés semble avoir disparu. C'est l'effet de 
#la transformation de log qui ne doit pas être confondu en pensant qu'ils ont des corrélations positives.
sns.pairplot(df_log)
X2 = df_log[['Video_uploads_log', 'Video_views_log']]
Y2 = df_log[['Subscribers_log']]
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, Y2, test_size = 0.2)
lm2 = LinearRegression()
lm2.fit(X2_train.dropna(),y2_train.dropna())
predictions2 = lm2.predict(X2_test)
plt.scatter(y2_test,predictions2, color = 'blue')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
sns.residplot(y2_test, predictions2,  color="blue")
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
#Dans la suite, la prédiction faite par la transformation de journal est comparée à celle effectuée directement.
#La relation est mentionnée dans l'enveloppe ci-dessus.
p = coefficients2['coefficients'][0]
q = coefficients2['coefficients'][1]
def pred_from_log(x, y):
    return x ** p + y ** q
X_test.head()
vid_upl_test = np.array(X_test['Video Uploads'])
vid_viw_test = np.array(X_test['Video views'])
prediction_log = pred_from_log(vid_upl_test, vid_viw_test)
#Il est bon que les deux prédictions soient fortement corrélées.

plt.scatter(predictions, prediction_log, color = 'blue', alpha = 0.5)
plt.xlabel('prediction without log transformation')
plt.ylabel('prediction with log transformation')
#Le tracé direct de la différence montre que la transformation de log a tendance à 
#prédire une valeur plus élevée que celle sans log, le cas échéant.
#Il n'y a aucun moyen de prédire une baisse cependant.
plt.scatter(range(len(X_test)), predictions - prediction_log, color = 'blue', alpha = 0.5)
plt.xlabel('count of test data')
plt.ylabel('difference of prediction with and without log')