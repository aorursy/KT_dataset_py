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
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn import utils
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # fancy statistics plots
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import geopandas as gpd
import scipy
from scipy.optimize import curve_fit
import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from IPython.display import HTML
import seaborn as sns
from plotly.subplots import make_subplots
%matplotlib inline


import plotly.tools as tls
import cufflinks as cf
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset=pd.read_csv('../input/us-counties-covid-19-dataset/us-counties.csv')
dataset
dataset.head()
dataset.tail()
print(dataset.shape)
print(dataset.head(10))
dataset['cases'].mean()
dataset['cases'].median()
dataset['cases'].std()
def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.2f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center") 

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)
gb_cntry = dataset.groupby(['county', 'date'], as_index = False).sum()
gb_cntry['Mortality_rate'] = gb_cntry['deaths'] / gb_cntry['cases'] 
gb_cntry['Mortality_rate'].fillna(0, inplace = True)

## Plot 5 countries with the most cases as of last day in training set

plt.figure(figsize = (10, 8))
ax = sns.barplot(x = 'county', y = 'cases',
            data = gb_cntry[gb_cntry['date'] == max(gb_cntry['date'])].sort_values(['cases'], ascending = False)[:5])
plt.title('Number of COVID19 cases in the top 5 most infected districts')

show_values_on_bars(ax)
dataset.plot(x='cases', y='deaths', style='*')
import seaborn as sns
corr = dataset.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
covid_19 = dataset.groupby(['date']).sum().reset_index()
fig, ax = plt.subplots(figsize=(16,9))
ax.plot(covid_19["date"],
        covid_19["cases"],
        color="r");
ax.set_title("us counties confirmed cases per day");
ax.spines["top"].set_visible(False);
ax.spines["right"].set_visible(False);
fig, ax = plt.subplots(figsize=(16,9))
ax.plot(covid_19["date"],
        covid_19["deaths"],
        color="r");
ax.set_title("us counties confirmed deaths per day");
ax.spines["top"].set_visible(False);
ax.spines["right"].set_visible(False);
dataset_reg=dataset.groupby(['county']).agg({'cases':'sum','deaths':'sum'}).sort_values(["cases"],ascending=False).reset_index()
dataset_reg.head(10)
fig = go.Figure(data=[go.Table(
    columnwidth = [50],
    header=dict(values=('county', 'cases', 'deaths'),
                fill_color='#104E8B',
                align='center',
                font_size=14,
                font_color='white',
                height=40),
    cells=dict(values=[dataset_reg['county'].head(10), dataset_reg['cases'].head(10), dataset_reg['deaths'].head(10)],
               fill=dict(color=['#509EEA', '#A4CEF8',]),
               align='right',
               font_size=12,
               height=30))
])

fig.show()
fig = px.pie(dataset_reg.head(10),
             values="cases",
             names="county",
             title="cases",
             template="seaborn")
fig.update_traces(rotation=90, pull=0.05, textinfo='value+label')
fig.show()
dataset.isnull().sum().sum()
dataset.isnull().sum()
def eksik_deger_tablosu(dataset):
    eksik_deger=dataset.isnull().sum()
    eksik_deger_yuzde=100* dataset.isnull().sum()/len(dataset)
    eksik_deger_tablo= pd.concat([eksik_deger,eksik_deger_yuzde], axis=1)
    eksik_deger_tablo_son=eksik_deger_tablo.rename(
    columns = {0 : 'Eksik Değerler',1: '% Değeri'} )
    return eksik_deger_tablo_son

eksik_deger_tablosu(dataset)
dataset.dropna()
label_encoder = preprocessing.LabelEncoder() 
dataset['state_Encoded']= label_encoder.fit_transform(dataset['state'])
dataset['county_Encoded']= label_encoder.fit_transform(dataset['county'])

dataset
dataset =dataset.drop(columns ='date')
dataset =dataset.drop(columns ='county')
dataset =dataset.drop(columns ='state')
dataset =dataset.drop(columns ='fips')
dataset =dataset.drop(columns ='cases')
dataset
scatter_matrix(dataset)
pyplot.show()
dataset.hist()
pyplot.show()
dataset.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
pyplot.show()
array = dataset.values
X = array[:,0:5]
y = array[:,0:1]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

print("Dataframe boyutu: ",dataset.shape)
print("Eğitim verisi boyutu: ",X_train.shape, Y_train.shape)
print("Test verisi boyutu: ",X_validation.shape, Y_validation.shape)
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
#Decision Trees
cellTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
print(cellTree) # it shows the default parameters
  #I fit the data with the training
cellTree.fit(X_train,Y_train)
  #now predictions
yhat_dt = cellTree.predict(X_validation)

  #Accuracy evaluation
acc = metrics.accuracy_score(Y_validation, yhat_dt)
print('karar agaci icin accuracy: ',acc)

#karar agaci icin confusion matrix ve metrik degerler
cellTree_dt = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
#train model with cv of 10 burda modeli 10 cross validasyon ile scorelari verdik
cv_scores_dt = cross_val_score(cellTree_dt, X,y, cv=10)
#print each cv score (accuracy) and average them
print(cv_scores_dt)
print('cv_scores mean:{}'.format(np.mean(cv_scores_dt)))
from sklearn.metrics import classification_report
prec_dt = classification_report(yhat_dt,Y_validation)
print(prec_dt)
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors = 3)
# fit the models
neigh = knn_model.fit(X_train,Y_train)
#predict the mode;
yhatknn=neigh.predict(X_validation)

  #Accuracy evaluation
accknn = metrics.accuracy_score(Y_validation, yhatknn)
print('en yakin komsular icin accuracy',accknn)

#knn=3 icin confusion matrix ve metrik degerler
knn_knn = KNeighborsClassifier(n_neighbors = 3)
#train model with cv of 10 burda modeli 10 cross validasyon ile scorelari verdik
cv_scores_knn = cross_val_score(knn_knn, X,y, cv=10)
#print each cv score (accuracy) and average them
print(cv_scores_knn)
print('cv_scores mean:{}'.format(np.mean(cv_scores_knn)))

#knn scores
from sklearn.metrics import classification_report
prec_knn = classification_report(yhatknn,Y_validation)
print(prec_knn)
from sklearn.naive_bayes import GaussianNB
#call the models
gnb = GaussianNB()
  #fit the model
gnb.fit(X_train, Y_train) 
  #predict
yhatgnb = gnb.predict(X_validation)
accgnb = metrics.accuracy_score(Y_validation, yhatgnb)
print('gaussian naive bayes icin accuracy',accgnb)


#gaussian naive bayes icin confusion matrix ve metrik degerler
clf_gnb = GaussianNB()
#train model with cv of 10 burda modeli 10 cross validasyon ile scorelari verdik
cv_scores_gnb = cross_val_score(clf_gnb, X,y, cv=10)
#print each cv score (accuracy) and average them
print(cv_scores_gnb)
print('cv_scores mean:{}'.format(np.mean(cv_scores_gnb)))

#klasifikasyon tablosu
from sklearn.metrics import classification_report
prec_gnb = classification_report(yhatgnb,Y_validation)
print(prec_gnb)
