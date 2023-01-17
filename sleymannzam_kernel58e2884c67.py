# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn    import metrics, svm
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import  linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
covid_data = pd.read_csv("/kaggle/input/corona-virus-report/covid_19_clean_complete.csv")
covid_data.head()

covid_data.describe()
pd.DataFrame(covid_data[covid_data['Country/Region']=='Italy']['Date'].groupby(covid_data['Province/State']))
pd.DataFrame(covid_data[covid_data['Country/Region']=='Italy']['Confirmed'].groupby(covid_data['Date']).sum()).reset_index()
pd.DataFrame(covid_data[covid_data['Country/Region']=='Italy']['Deaths'].groupby(covid_data['Date']).sum()).reset_index()
data=pd.DataFrame(covid_data[covid_data['Country/Region']=='Italy'].groupby(covid_data['Date'])['Confirmed','Deaths','Recovered'].sum().sort_values(by='Confirmed')).reset_index()
data.head(20)
ax = plt.axes()
ax.scatter(data.Deaths, data.Recovered)

ax.set(xlabel='Died bodies',
       ylabel='Saved bodies',
       title='Died and Recovered Bodies');
df2=pd.Series(data['Date'],name="Date")
df3=pd.Series(data['Confirmed'],name="Confirmed")
df4=pd.Series(data['Deaths'],name="Deaths")
df5=pd.Series(data['Recovered'],name="Recovered")
df_italy=pd.concat([df2,df3, df4,df5], axis=1)
df_italy.head(10)
italy =df_world.copy()
italy_values = (italy['Country'] == 'Italy').astype(int)
fields = list(italy.columns[:-1])  
correlations = italy[fields].corrwith(italy_values)
correlations.sort_values(inplace=True)
correlations
ax = correlations.plot(kind='bar')
ax.set(ylim=[-1, 1], ylabel='italy correlation');
plt.figure()
df_italy.boxplot(column=['Confirmed','Deaths','Recovered'])

fig,axs=plt.subplots(2,2) 
axs[0, 0].boxplot(df_world['Confirmed'])
axs[0, 0].set_title('Count of Case')

axs[0, 1].boxplot(df_world['Recovered'])
axs[0, 1].set_title('Count of Recovered Case')

axs[1, 0].boxplot(df_world['Deaths'])
axs[1, 0].set_title('Count of Death Case')
from sklearn.model_selection import train_test_split


X = df_italy.iloc[:,2:5]
y = df_italy['Recovered']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, random_state=0)
print("Dimension of Dataframe: ",df_italy.shape)
print("Dimension of Training data: ",X_train.shape, y_train.shape)
print("Dimension of Test data: ",X_test.shape,y_test.shape)
from sklearn import preprocessing
from sklearn import utils

lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(y)
print(utils.multiclass.type_of_target(y))
print(utils.multiclass.type_of_target(y_train.astype('int')))
print(utils.multiclass.type_of_target(encoded))

lab_enc = preprocessing.LabelEncoder()
Y_train = lab_enc.fit_transform(y_train)
print("\nLineer Regresyon")
lm = linear_model.LinearRegression().fit(X_train, Y_train)
y_true1 , y_pred1 =y_test,lm.predict(X_test)
y_pred = lm.predict(X_test)
print("\nTahmin değerleri: ",y_pred1)
plt.scatter(y_true1, y_pred1,c='orange')
plt.scatter(y_true1, y_test,c='green')
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.figure(figsize=(8,8))
plt.plot(df_italy['Confirmed']);
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(df_italy.corr());
sns.pairplot(df_italy,kind="reg");
from sklearn.neighbors import KNeighborsClassifier

Knn = KNeighborsClassifier(n_neighbors=5,weights= 'distance').fit(X_train, y_train)

y_predict = Knn.predict(X_train)
y_predict
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score 
GNB = GaussianNB()
Naive_Bayes = {'Gausian algoritması': GaussianNB(),
      'Bernuilli algoritması': BernoulliNB(),
        'Multinomial algoritması': MultinomialNB()}
score = {}
X = df_italy[['Confirmed','Recovered']]
y = df_italy['Deaths'].values
for key, model in Naive_Bayes.items():
    sc = cross_val_score(model, X, y, cv=7, n_jobs=10, scoring='accuracy')
    score[key] = np.mean(sc)
print(score)
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
X_train,X_test,y_train,y_test = train_test_split(X,y, random_state=48)
rf_model = RandomForestClassifier(n_estimators=75, 
                               bootstrap = True,
                               max_features = 'sqrt')
rf_model.fit(X_train, y_train)
y_predict=rf_model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_predict))
import plotly.graph_objects as go
fig1_1=go.Figure()
fig1_1.add_trace(go.Scatter(x=df_italy['Date'],y=df_italy['Deaths'],mode='lines+markers',marker=dict(size=10,color=1)))
