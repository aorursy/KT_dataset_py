# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
CoronaV = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')

print(CoronaV.head(10))

print('\n')

print(CoronaV.info())
#normalize the dataset

CoronaV = CoronaV.drop('Sno', axis = 1)

CoronaV.columns = ['State', 'Country', 'Date', 'Confirmed', 'Deaths', 'Recovered']

CoronaV['Date'] = CoronaV['Date'].apply(pd.to_datetime).dt.normalize() 
CoronaV.info()
CoronaV[['State','Country','Date','Confirmed']].drop_duplicates().shape[0] == CoronaV.shape[0]



CoronaV.describe(include = 'all')
CoronaV[['Country','State']][CoronaV['State'].isnull()].drop_duplicates()
CoronaV[CoronaV['Country'].isin(list(CoronaV[['Country','State']][CoronaV['State'].isnull()]['Country'].unique()))]['State'].unique()
CoronaV.State.unique()
CoronaV.Country.unique()
print(CoronaV[CoronaV['Country'].isin(['China', 'Mainland China'])].groupby('Country')['State'].unique())

print(CoronaV[CoronaV['Country'].isin(['China', 'Mainland China'])].groupby('Country')['Date'].unique())
CoronaV['Country'] = CoronaV['Country'].replace(['Mainland China'], 'China') #set 'Mainland China' to 'China'

sorted(CoronaV.Country.unique())
print(CoronaV.head())
china = CoronaV[CoronaV['Country']=='China']

china.head()
import matplotlib.pyplot as plt

%matplotlib inline



plt.rcParams["figure.figsize"] = (12,9)

ax1 = china[['Date','Confirmed']].groupby(['Date']).sum().plot()

ax1.set_ylabel("Total Number of Confirmed Cases")

ax1.set_xlabel("Date")



ax2 = china[['Date','Deaths', 'Recovered']].groupby(['Date']).sum().plot()

ax2.set_ylabel("Total N")

ax2.set_xlabel("Date")

import plotly.graph_objects as go

from plotly.subplots import make_subplots



fig = make_subplots(rows=1, cols=3, specs=[[{"type" : "pie"}, {"type" : "pie"},{"type" : "pie"}]],

                    subplot_titles=("number of provience in countries", "Deaths", "Recovers"))



fig.add_trace(

    go.Pie(labels=CoronaV.groupby('Country')['State'].nunique().sort_values(ascending=False)[:10].index,

           values=CoronaV.groupby('Country')['State'].nunique().sort_values(ascending=False)[:10].values),

    row=1, col=1

)



fig.add_trace(

    go.Pie(labels=CoronaV[CoronaV.Deaths > 0].groupby('Country')["Deaths"].sum().index,

           values=CoronaV[CoronaV.Deaths > 0].groupby('Country')["Deaths"].sum().values),

    row=1, col=2

)

fig.add_trace(

    go.Pie(labels=CoronaV.groupby('Country')["Recovered"].sum().sort_values(ascending=False).index[:4],

           values=CoronaV.groupby('Country')["Recovered"].sum().sort_values(ascending=False).values[:4]),

    row=1, col=3

)



fig.update_layout(height=400, showlegend=True)

fig.show()



CoronaV['Date'] = pd.to_datetime(CoronaV['Date'])

CoronaV['Day'] = CoronaV['Date'].apply(lambda x : x.day)

CoronaV['Hour'] = CoronaV['Date'].apply(lambda x : x.hour)



CoronaV = CoronaV[CoronaV['Confirmed'] != 0]

CoronaV

global_case = CoronaV.groupby('Country')['Confirmed','Deaths','Recovered'].sum().reset_index()

global_case.head()

global_case
CoronaV.groupby(['Date','Country']).agg({

    'Confirmed': pd.Series.nunique,

}).reset_index().pivot(index='Date',columns='Country',values='Confirmed').plot.barh(stacked=True,figsize=(26,10),colormap='gist_rainbow')
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import seaborn as sns

plt.rcParams["figure.figsize"] = (17,10)

nums = china.groupby(["State"])['Confirmed'].aggregate(sum).reset_index().sort_values('Confirmed', ascending= False)

ax = sns.barplot(x="Confirmed", y="State", order = nums['State'], data=china, ci=None) 

ax.set_xlabel("Total Confirmed Cases")



def get_ci(N,p):

    lci = (p - 1.96*(((p*(1-p))/N) ** 0.5))*100

    uci = (p + 1.96*(((p*(1-p))/N) ** 0.5))*100

    return str(np.round(lci,3)) + "% - " + str(np.round(uci,3)) + '%'



final = CoronaV[CoronaV.Date==np.max(CoronaV.Date)]

final = final.copy()



final['CFR'] = np.round((final.Deaths.values/final.Confirmed.values)*100,3)

final['CFR 95% CI'] = final.apply(lambda row: get_ci(row['Confirmed'],row['CFR']/100),axis=1)

global_cfr = np.round(np.sum(final.Deaths.values)/np.sum(final.Confirmed.values)*100, 3)

final.sort_values('CFR', ascending= False).head(10)



tops = final.sort_values('CFR', ascending= False)

tops = tops[tops.CFR >0]

df = final[final['CFR'] != 0]

plt.rcParams["figure.figsize"] = (10,5)

ax = sns.barplot(y="CFR", x="State", order = tops['State'], data=df, ci=None) 

ax.axhline(global_cfr, alpha=.5, color='r', linestyle='dashed')

ax.set_title('Case Fatality Rates (CFR) as of 30 Jan 2020')

ax.set_ylabel('CFR %')

print('Average CFR % = ' + str(global_cfr))



from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import LocalOutlierFactor

scaler = StandardScaler()

scd = scaler.fit_transform(final[['Confirmed','Deaths','Recovered']])

clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1) #LOF is very sensitive to the choice of n_neighbors. Generally, n_neighbors = 20 works better

clf.fit(scd)

lofs = clf.negative_outlier_factor_*-1

final['LOF Score'] = lofs

tops = final.sort_values('LOF Score', ascending= False)

plt.rcParams["figure.figsize"] = (20,12)

ax = sns.barplot(x="LOF Score", y="State", order = tops['State'], data=final, ci=None) 

ax.axvline(1, alpha=.5, color='g', linestyle='dashed')

ax.axvline(np.median(lofs), alpha=.5, color='b', linestyle='dashed')

ax.axvline(np.mean(lofs) + 3*np.std(lofs), alpha=.5, color='r', linestyle='dashed')

final.sort_values('LOF Score', ascending=False)
from sklearn.cluster import KMeans

plt.rcParams["figure.figsize"] = (5,5)

wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=1897)

    kmeans.fit(scd)

    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)

plt.title('Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('Within Cluster Sum of Squares')

plt.show()

kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=1897)

clusters = np.where(kmeans.fit_predict(scd) == 0, 'Cluster 1', 'Cluster 2')

clusters
from sklearn import decomposition

pca = decomposition.PCA(n_components=3)

pca.fit(scd)

X = pca.transform(scd)

print(pca.explained_variance_ratio_.cumsum())



plt.rcParams["figure.figsize"] = (7,7)

ax = sns.scatterplot(X[:,0], X[:,1], marker = 'X', s = 80, hue=clusters)

ax.set_title('K-Means Clusters of States')

ax.set_xlabel('Principal Component 1')

ax.set_ylabel('Principal Component 2')
pd.DataFrame(final.State.values, clusters)
X = CoronaV['Deaths'].values.reshape(-1,1)

y = CoronaV['Recovered'].values.reshape(-1,1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()  

regressor.fit(X_train, y_train) #training the algorithm
#To retrieve the intercept:

print(regressor.intercept_)#For retrieving the slope:

print(regressor.coef_)
y_pred = regressor.predict(X_test)

y_pred
import matplotlib.pyplot as plt

import seaborn as sns

import seaborn as seabornInstance

plt.figure(figsize=(15,10))

plt.tight_layout()

seabornInstance.distplot(CoronaV['Recovered'])
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

df
df1 = df.head(25)

df1.plot(kind='bar',figsize=(16,10))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
# transform data

from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

X_train_scale = scaler.fit_transform(X_train)

X_test_scale = scaler.transform(X_test)



# split training feature and target sets into training and validation subsets

from sklearn.model_selection import train_test_split



X_train_sub, X_validation_sub, y_train_sub, y_validation_sub = train_test_split(X_train_scale, y_train, random_state=0)



# import machine learning algorithms

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

X_scale = min_max_scaler.fit_transform(X)

X_scale
from sklearn.model_selection import train_test_split

X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)
from keras.models import Sequential

from keras.layers import Dense

model = Sequential([Dense(32, activation='relu', input_shape=(1,)), Dense(32, activation='relu'), Dense(1, activation='sigmoid'),])
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
hist = model.fit(X_train, Y_train, batch_size=32, epochs=500, validation_data=(X_val, Y_val))
model.evaluate(X_test, Y_test)[1]
import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='upper right')

plt.show()
plt.plot(hist.history['accuracy'])

plt.plot(hist.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='lower right')

plt.show()
model_2 = Sequential([Dense(1000, activation='relu', input_shape=(1,)), Dense(1000, activation='relu'), Dense(1000, activation='relu'), Dense(1000, activation='relu'),    Dense(1, activation='sigmoid'),])
model_2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
hist_2 = model_2.fit(X_train, Y_train, batch_size=32, epochs=100, validation_data=(X_val, Y_val))
plt.plot(hist_2.history['accuracy'])

plt.plot(hist_2.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='lower right')

plt.show()
from keras.layers import Dropout

from keras import regularizers



model_3 = Sequential([Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(1,)),    Dropout(0.3),    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),    Dropout(0.3),    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)), Dropout(0.3), Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)), Dropout(0.3), Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)),])

model_3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



hist_3 = model_3.fit(X_train, Y_train, batch_size=32, epochs=100, validation_data=(X_val, Y_val))
plt.plot(hist_3.history['loss'])

plt.plot(hist_3.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='upper right')

plt.ylim(top=1.2, bottom=0)

plt.show()
plt.plot(hist_3.history['accuracy'])

plt.plot(hist_3.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='lower right')

plt.show()
model_3.evaluate(X_test, Y_test)[1]
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

clf = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)

y_pred = cross_val_predict(clf, X, y, cv=10)

conf_mat = confusion_matrix(y, y_pred)



conf_mat
from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(y, y_pred)

print(conf_mat)
import seaborn as sn

sn.set(font_scale=1.4) # for label size

sn.heatmap(conf_mat, annot=True, annot_kws={"size": 10}) # font size

plt.ylabel('Actual')

plt.xlabel('Predicted')
from sklearn.metrics import accuracy_score 

# True Positives

TP = confusion[1, 1]# True Negatives

TN = confusion[0, 0]# False Positives

FP = confusion[0, 1]# False Negatives

FN = confusion[1, 0]



print((TP + TN) / float(TP + TN + FP + FN))

print(accuracy_score(y, y_pred))
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import BernoulliNB

from sklearn.metrics import confusion_matrix

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

print("Precision Score : ",precision_score(y, y_pred, 

                                           pos_label='positive',

                                           average='micro'))

print("Recall Score : ",recall_score(y, y_pred, 

                                           pos_label='positive',

                                           average='micro'))