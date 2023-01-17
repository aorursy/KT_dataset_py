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
!pip install sweetviz
!pip install dexplot
import pandas as pd
import numpy as np 

# Data Visualization Libraries
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import plotly
import plotly.express as px
import plotly.offline as py
import plotly.graph_objs as go
from IPython.display import display, Markdown
import sweetviz as sv
import dexplot as dxp

#hide warnings
import warnings
warnings.filterwarnings('ignore')

#display max columns of pandas dataframe
pd.set_option('display.max_columns', None)

from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
df_battle = pd.read_csv("../input/game-of-thrones/battles.csv")
df_death = pd.read_csv("../input/game-of-thrones/character-deaths.csv")
df_pred_1 = pd.read_csv("../input/game-of-thrones/character-predictions.csv")
df_pred = df_pred_1.copy()
df_battle.head(2)
df_death.head(2)
df_pred.head(2)
# Helper Function - Missing data check
def missing_data(data):
    missing = data.isnull().sum()
    available = data.count()
    total = (missing + available)
    percent = (data.isnull().sum()/data.isnull().count()*100).round(4)
    return pd.concat([missing, available, total, percent], axis=1, keys=['Missing', 'Available', 'Total', 'Percent']).sort_values(['Missing'], ascending=False)
missing_data(df_battle)
# Year-wise battles
sns.countplot(df_battle['year'])
plt.title('Yearwise battle summary')
plt.show()
# Win - Lose distribution
sns.countplot(x='attacker_outcome',data = df_battle)
plt.title('Win-Loss Distribution')
plt.show()
dxp.count(val='attacker_king', data=df_battle, split = 'attacker_outcome', stacked=True,title='King w/ most attacks')
dxp.count(val='defender_king', data=df_battle, split = 'attacker_outcome', stacked=True,title='King who defended the most')
# new column to count the alliance during an attack
df_battle['alliance_count'] = (4 - df_battle[["attacker_1", "attacker_2", "attacker_3", "attacker_4"]].isnull().sum(axis = 1))
dxp.count(val='alliance_count', data=df_battle, split = 'attacker_outcome', stacked=True,title='Alliance Count vs Battle Outcome')
dxp.count(val='attacker_king', data=df_battle, split = 'alliance_count', stacked=True,title='Alliance Count vs Attacker King')
# new column to count the alliance while defending
df_battle['alliance_count_defend'] = (4 - df_battle[["defender_1", "defender_2", "defender_3", "defender_4"]].isnull().sum(axis = 1))
dxp.count(val='alliance_count_defend', data=df_battle, split = 'attacker_outcome', stacked=True,title='Alliance Count vs Battle Outcome')
dxp.count(val='defender_king', data=df_battle, split = 'alliance_count', stacked=True,title='Alliance Count vs Defender King')
dxp.count(val='battle_type', data=df_battle, split = 'attacker_outcome', stacked=True,title='Battle Type vs Battle Outcome')
dxp.count(val='battle_type', data=df_battle, split = 'attacker_king',title='Battle Type vs Attacked King')
dxp.count(val='battle_type', data=df_battle, split = 'summer',title='Battle Type vs Season', stacked=True)
dxp.count(val='summer', data=df_battle, split = 'attacker_outcome',title='Battle Outcome vs Season', stacked=True)
fig = px.bar(df_battle, 
             x=df_battle['attacker_king'].fillna("Other"),
             y=df_battle['attacker_size'].fillna(0),
             orientation='v',
             height=800,
             title='Battle outcome vs Army size',
            color=df_battle['attacker_outcome'].fillna("Unknown"))
fig.show()
#No of deaths vs Year
plt.figure(figsize=(15,10))
plt.subplot(2,1,1)
sns.countplot(df_battle['year'])
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Year', fontsize=18)
plt.ylabel('Count', fontsize=18)
plt.title('No of deaths vs Year',fontsize=20)
plt.show()

#No of deaths vs Gender
plt.figure(figsize=(15,10))
plt.subplot(2,1,2)
sns.countplot(df_death['Gender'])
plt.title('Gender vs Death', fontsize=20)
plt.xlabel('Gender', fontsize=18)
plt.ylabel('Count', fontsize=18)
plt.xticks(np.arange(2),('Female','Male'),fontsize=16)
plt.yticks(fontsize=16)
plt.show()
#No of deaths vs Year
plt.figure(figsize=(15,10))
plt.subplot(2,1,1)
sns.countplot(df_death['Death Year'])
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Year', fontsize=18)
plt.ylabel('Count', fontsize=18)
plt.title('No of deaths vs Year',fontsize=20)
plt.show()

#No of deaths vs Gender
plt.figure(figsize=(15,10))
plt.subplot(2,1,2)
sns.countplot(df_death['Gender'])
plt.title('Gender vs Death', fontsize=20)
plt.xlabel('Gender', fontsize=18)
plt.ylabel('Count', fontsize=18)
plt.xticks(np.arange(2),('Female','Male'),fontsize=16)
plt.yticks(fontsize=16)
plt.show()
# Data cleaning for character predictions dataset
cult = {
    'Summer Islands': ['summer islands', 'summer islander', 'summer isles'],
    'Ghiscari': ['ghiscari', 'ghiscaricari',  'ghis'],
    'Asshai': ["asshai'i", 'asshai'],
    'Lysene': ['lysene', 'lyseni'],
    'Andal': ['andal', 'andals'],
    'Braavosi': ['braavosi', 'braavos'],
    'Dornish': ['dornishmen', 'dorne', 'dornish'],
    'Myrish': ['myr', 'myrish', 'myrmen'],
    'Westermen': ['westermen', 'westerman', 'westerlands'],
    'Westerosi': ['westeros', 'westerosi'],
    'Stormlander': ['stormlands', 'stormlander'],
    'Norvoshi': ['norvos', 'norvoshi'],
    'Northmen': ['the north', 'northmen'],
    'Free Folk': ['wildling', 'first men', 'free folk'],
    'Qartheen': ['qartheen', 'qarth'],
    'Reach': ['the reach', 'reach', 'reachmen'],
}

def get_cult(value):
    value = value.lower()
    v = [k for (k, v) in cult.items() if value in v]
    return v[0] if len(v) > 0 else value.title()
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
df_pred.loc[:, "culture"] = [get_cult(x) for x in df_pred.culture.fillna("")]
data = df_pred.groupby(["culture", "isAlive"]).count()["S.No"].unstack().copy(deep = True)
data.loc[:, "total"]= data.sum(axis = 1)
p = data[data.index != ""].sort_values("total")[[0, 1]]
p.reset_index(level=0, inplace=True)
p.rename(columns={0: 'dead', 1: 'alive'}, inplace=True)
p = p.fillna(0)

fig = go.Figure()
fig.add_trace(go.Bar(x=p['culture'],
                     y=p['dead'],
                     name="Dead"))
fig.add_trace(go.Bar(x=p['culture'],
                     y=p['alive'],
                     name="Alive"))

fig.update_layout(
    xaxis = dict(
        tickmode = 'array',
        tickvals = p['culture'])
)
df_pred.columns
#Data prep

df_pred.loc[:, "culture"] = [get_cult(x) for x in df_pred.culture.fillna("")]
df_pred.loc[:, "title"] = pd.factorize(df_pred.title)[0]
df_pred.loc[:, "culture"] = pd.factorize(df_pred.culture)[0]
df_pred.loc[:, "mother"] = pd.factorize(df_pred.mother)[0]
df_pred.loc[:, "father"] = pd.factorize(df_pred.father)[0]
df_pred.loc[:, "heir"] = pd.factorize(df_pred.heir)[0]
df_pred.loc[:, "house"] = pd.factorize(df_pred.house)[0]
df_pred.loc[:, "spouse"] = pd.factorize(df_pred.spouse)[0]
df_pred['houseSize'] = df_pred['house'].map(df_pred['house'].value_counts())
df_pred['houseAlive'] = df_pred['house'].map(df_pred['house'].value_counts())
df_pred['houseDead'] = df_pred['house'].map(df_pred['house'].value_counts())
df_pred['houseDeathRate'] = df_pred['houseDead']/df_pred['houseSize']
df_pred.drop(["name", "alive", "pred", "plod", "isAlive", "dateOfBirth", "DateoFdeath","S.No"], 1, inplace = True)
df_pred.columns = map(lambda x: x.replace(".", "").replace("_", ""), df_pred.columns)
df_pred.fillna(value = -1, inplace = True)
df_pred.head()
fig = go.Figure(
    data=[go.Bar(x=df_pred['actual'].unique(), y=df_pred['actual'].value_counts())],
    layout=go.Layout(
        title=go.layout.Title(text="Class Distribution")
    )
)

fig.show()
fig = go.Figure(data=[go.Pie(labels=['Alive','Dead'], values=df_pred['actual'].value_counts())])
fig.update_layout(
    title_text="Class Distribution")
fig.show()
# Helper function for ROC curve and dataset

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - ' + str_1)
    plt.legend()
    plt.show()

def plot_data(X,Y):
    plt.scatter(X[Y == 0,0], X[Y == 0, 1], label="Class 0-Non Fraud", alpha=0.4, c='b')
    plt.scatter(X[Y == 1,0], X[Y == 1, 1], label="Class 1-Fraud", alpha=0.4, c='g')
    plt.legend()
    
    return plt.show()
df_pred.describe()
y = df_pred.actual.values
X = df_pred.copy(deep=True)
X.drop(["actual"], 1, inplace = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
from sklearn.feature_selection import mutual_info_classif
mutual_infos = pd.Series(data=mutual_info_classif(X, y, discrete_features=False, random_state=1), index=X.columns)
mutual_infos.sort_values(ascending=False).plot(kind='bar',figsize = (12,6))
# logistic regression w/o handling class imbalance

from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix, classification_report 
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

str_1 = 'Logistic Regression w/o handling class imbalance'

lr = LogisticRegression() 

# train the model on train set 
lr.fit(X_train, y_train.ravel()) 
  
predictions = lr.predict(X_test) 
  
# print classification report 
print(classification_report(y_test, predictions))
# Predict probabilities for test data
probs_lr = lr.predict_proba(X_test)
# Probabilities of the positive class
probs_lr_pos = probs_lr[:, 1]
# AUC Score
auc = roc_auc_score(y_test, probs_lr_pos)
print('AUC: %.2f' % auc)
# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, probs_lr_pos)
# Plot ROC Curve 
plot_roc_curve(fpr, tpr)
from sklearn.metrics import f1_score, matthews_corrcoef
mathews_corr_lr = matthews_corrcoef(y_test, predictions) 
print("Matthews correlation coefficient for LR model is {:.3f}".format(mathews_corr_lr))
# Random forest model  w/o handling class imbalance
from sklearn.ensemble import RandomForestClassifier 
rfc = RandomForestClassifier()

#fit the model
rfc.fit(X_train, y_train.ravel()) 

predictions_rfc = rfc.predict(X_test)

# print classification report 
print(classification_report(y_test, predictions_rfc))
# Predict probabilities for test data
probs_rfc = rfc.predict_proba(X_test)
# Probabilities of the positive class
probs_rfc_pos = probs_rfc[:, 1]
# AUC Score
auc_rfc = roc_auc_score(y_test, probs_rfc_pos)
print('AUC: %.2f' % auc_rfc )
# ROC Curve
fpr_rfc, tpr_rfc, thresholds_rfc = roc_curve(y_test, probs_rfc_pos)
# Plot ROC Curve 
plot_roc_curve(fpr_rfc, tpr_rfc)
mathews_corr_rfc = matthews_corrcoef(y_test, predictions_rfc) 
print("Matthews correlation coefficient for RF model is {:.3f}".format(mathews_corr_rfc))
import keras
from keras.models import Sequential
from keras import backend as K
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers.core import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

n_inputs = X_train.shape[1]


# define model
model= Sequential()
model.add(Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(2, activation='softmax'))
model.summary()
model.compile(Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
history = model.fit(X_train, y_train, validation_split=0.2, batch_size=10, epochs=50, shuffle=True, verbose=2)
# summarize history for accuracy
plt.plot(history.history['sparse_categorical_accuracy'])
plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.title('model accuracy - BS= 10')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss - BS=10')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()