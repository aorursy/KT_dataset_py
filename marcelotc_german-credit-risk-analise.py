import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import pandas_profiling as pp

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv("../input/german-credit-risk/german_credit.csv", sep=",")
df.head()
df.isnull().values.any()
df.info()
import plotly.offline as py 

py.init_notebook_mode(connected=True) 

import plotly.graph_objs as go 

import plotly.tools as tls

import warnings 

from collections import Counter 



credit1 = go.Bar(x = df[df["Creditability"]== 1]["Creditability"].value_counts().index.values,

                y = df[df["Creditability"]== 1]["Creditability"].value_counts().values, name='Adimplentes')



credit0 = go.Bar(x = df[df["Creditability"]== 0]["Creditability"].value_counts().index.values,

                y = df[df["Creditability"]== 0]["Creditability"].value_counts().values, name='Inadimplentes')



data = [credit1, credit0]



layout = go.Layout()



layout = go.Layout(yaxis=dict(title='Quantidade'),xaxis=dict(title='Variável Classe'),title='Distribuição da Variável Classe', xaxis_type='category')



fig = go.Figure(data=data, layout=layout)



py.iplot(fig, filename='grouped-bar')
df_good = df.loc[df["Creditability"] == 1]['Age (years)'].values.tolist()

df_bad = df.loc[df["Creditability"] == 0]['Age (years)'].values.tolist()

df_age = df['Age (years)'].values.tolist()



#plot 1

credit1 = go.Histogram(x=df_good, histnorm='percent', name="Adimplentes")



#plot 2

credit0 = go.Histogram(x=df_bad, histnorm='percent', name="Inadimplentes")



#plot 3

creditT = go.Histogram(x=df_age, histnorm='percent', name="Geral")



#Grid

fig = tls.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],

                          subplot_titles=('Adimplentes','Inadimplentes', 'Geral'))



#Figs

fig.append_trace(credit1, 1, 1)

fig.append_trace(credit0, 1, 2)

fig.append_trace(creditT, 2, 1)



fig['layout'].update(showlegend=True, title='Distribuição Idade %', bargap=0.05)

py.iplot(fig)
print ('Adimplente % ',round(df['Creditability'].value_counts()[1]/len(df)*100,2))

print ()

print (df['Credit Amount'][df.Creditability == 1].describe().round(2))

print ()

print ()

print ('Inadimplente % ',round(df['Creditability'].value_counts()[0]/len(df)*100,2))

print ()

print (df['Credit Amount'][df.Creditability == 0].describe().round(2))
ax = sns.boxplot(x="Creditability", y="Credit Amount", data=df, order=[1, 0])

plt.xlabel('1: Adimplente       0: Inadimplente')

plt.gcf().set_size_inches(12, 8)
plt.figure(figsize=(14,12))

sns.heatmap(df.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True,  linecolor='white', annot=True)

plt.show()
credit1 = go.Bar(x = df[df["Creditability"]== 1]["Type of apartment"].value_counts().index.values,

                 y = df[df["Creditability"]== 1]["Type of apartment"].value_counts().values, name='Adimplente')



credit0 = go.Bar(x = df[df["Creditability"]== 0]["Type of apartment"].value_counts().index.values,

                 y = df[df["Creditability"]== 0]["Type of apartment"].value_counts().values, name="Inadimplente")



data = [credit1, credit0]



layout = go.Layout(title='Moradia')



fig = go.Figure(data=data, layout=layout)



fig.update_xaxes (ticktext = [ "Alugada" ,  "Própria" ,  "Outras"],

                  tickvals = ["1", "2", "3"])



py.iplot(fig)
prf = pp.ProfileReport(df)

prf
feature_names = df.iloc[:, 1:21].columns

target = df.iloc[:1, 0:1].columns



data_features = df[feature_names]

data_target = df[target]
feature_names
target
import numpy as np



from sklearn.model_selection import train_test_split

np.random.seed(123)

X_train, X_test, y_train, y_test = train_test_split(data_features, data_target, 

                                                    train_size = 0.70, test_size = 0.30, random_state = 1)
from sklearn.ensemble import RandomForestClassifier 



rf = RandomForestClassifier()
rf.fit(X_train, y_train) 
def PrintStats(cmat, y_test, pred):

    tpos = cmat[0][0]

    fneg = cmat[1][1]

    fpos = cmat[0][1]

    tneg = cmat[1][0]
def RunModel(model, X_train, y_train, X_test, y_test):

    model.fit(X_train, y_train.values.ravel())

    pred = model.predict(X_test)

    matrix = confusion_matrix(y_test, pred)

    return matrix, pred
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

import scikitplot as skplt
cmat, pred = RunModel(rf, X_train, y_train, X_test, y_test)
import scikitplot as skplt

skplt.metrics.plot_confusion_matrix(y_test, pred)
accuracy_score(y_test, pred)
print (classification_report(y_test, pred))
bad_records = len(df[df.Creditability == 0]) 

bad_indices = df[df.Creditability == 0].index

good_indices = df[df.Creditability == 1].index



under_sample_indices = np.random.choice(good_indices, bad_records, False)

df_undersampled = df.iloc[np.concatenate([bad_indices, under_sample_indices]),:]

X_undersampled = df_undersampled.iloc[:,1:21]

Y_undersampled = df_undersampled.Creditability

X_undersampled_train, X_undersampled_test, Y_undersampled_train, Y_undersampled_test = train_test_split(X_undersampled, Y_undersampled, test_size = 0.30)
rf_undersampled = RandomForestClassifier() 

cmat, pred = RunModel(rf_undersampled, X_undersampled_train, Y_undersampled_train, X_undersampled_test, Y_undersampled_test)

PrintStats(cmat, Y_undersampled_test, pred)
skplt.metrics.plot_confusion_matrix(Y_undersampled_test, pred)
accuracy_score(Y_undersampled_test, pred)
print (classification_report(Y_undersampled_test, pred))
rf = RandomForestClassifier()

cmat, pred = RunModel(rf, X_undersampled_train, Y_undersampled_train, X_test, y_test)

PrintStats(cmat, y_test, pred)
skplt.metrics.plot_confusion_matrix(y_test, pred)
accuracy_score(y_test, pred)
from sklearn.metrics import classification_report

print (classification_report(y_test, pred))
from sklearn.model_selection import GridSearchCV
param_grid = {"criterion": ['entropy', 'gini'],

              "n_estimators": [25, 50, 75],

              "n_jobs": [1, 2, 3, 4],

              "max_features": ['auto', 0.1, 0.2, 0.3]}



grid_search_rf = GridSearchCV(rf, param_grid, scoring="precision")

grid_search_rf.fit(y_test, pred)



rf = grid_search_rf.best_estimator_ 

grid_search_rf.best_params_, grid_search_rf.best_score_
rf_undersampled = RandomForestClassifier(criterion = 'entropy', max_features = 'auto', n_estimators = 25, n_jobs = 1)

cmat, pred = RunModel(rf_undersampled, X_undersampled_train, Y_undersampled_train, X_undersampled_test, Y_undersampled_test)

PrintStats(cmat, Y_undersampled_test, pred)
skplt.metrics.plot_confusion_matrix(Y_undersampled_test, pred)
accuracy_score(Y_undersampled_test, pred)
print (classification_report(Y_undersampled_test, pred))
cmat, pred = RunModel(rf, X_undersampled_train, Y_undersampled_train, X_test, y_test)

PrintStats(cmat, y_test, pred)
skplt.metrics.plot_confusion_matrix(y_test, pred)
accuracy_score(y_test, pred)
print (classification_report(y_test, pred))
from sklearn import metrics   



clf = RandomForestClassifier(criterion='entropy', n_estimators = 25, n_jobs = 1, max_features='auto')

clf.fit(X_train, y_train)



y_pred_probability = clf.predict_proba(X_test)[::,1]

fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_probability)

auc = metrics.roc_auc_score(y_test, pred)

plt.plot(fpr,tpr,label="RandomForest, auc="+str(auc))

plt.legend(loc=4)

plt.show()