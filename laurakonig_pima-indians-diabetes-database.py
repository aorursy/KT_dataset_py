#Importing packages



#Data manipulation and linear algebra

import pandas as pd

import numpy as np



#Plots

%matplotlib inline

from matplotlib import pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

import plotly.offline as py

import plotly.figure_factory as ff



#Data processing and modeling

from sklearn.model_selection import train_test_split

from sklearn import ensemble

from sklearn.linear_model import LogisticRegression

from sklearn import model_selection

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc, accuracy_score
#Importing dataset

dbt = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")



#EDA with Pandas Profiling

from pandas_profiling import ProfileReport

file = ProfileReport(dbt)

file.to_file(output_file='output.html')
dbt.info()
dbt.head()
desc = dbt.describe()

print(desc)
D = dbt[(dbt['Outcome'] == 1)]

N = dbt[(dbt['Outcome'] == 0)]



def target_count():

    trace = go.Bar( x = dbt['Outcome'].value_counts().values.tolist(), 

                    y = ['healthy','diabetic' ], 

                    orientation = 'h', 

                    text=dbt['Outcome'].value_counts().values.tolist(), 

                    textfont=dict(size=15),

                    textposition = 'auto',

                    opacity = 0.8,marker=dict(

                    color=['green', 'red'],

                    line=dict(color='#000000',width=1.8)))



    layout = dict(title =  'Nombre de cas de diabète')



    fig = dict(data = [trace], layout=layout)

    py.iplot(fig)



#------------PERCENTAGE-------------------

def target_percent():

    trace = go.Pie(labels = ['Non diabétique','Diabétique'], values = dbt['Outcome'].value_counts(), 

#                   textfont=dict(size=15), opacity = 0.8,

                   marker=dict(colors=['green', 'red'], 

                               line=dict(color='#000000', width=1.5)))





    layout = dict(title =  'Distribution des cas de diabète')



    fig = dict(data = [trace], layout=layout)

    py.iplot(fig)

target_count()

target_percent()
# Define missing plot to detect all missing values in dataset

def missing_plot(dataset, key) :

    

    null_feat = pd.DataFrame(len(dataset[key]) - dataset.isnull().sum(), columns = ['Count'])



    trace = go.Bar(x = null_feat.index, y = null_feat['Count'] ,opacity = 0.5,  textposition = 'auto',marker=dict(color = 'blue',

            line=dict(color='#000000',width=1.5)))



    layout = dict(title =  "Valeurs manquantes")



    fig = dict(data = [trace], layout=layout)

    py.iplot(fig)
missing_plot(dbt, 'Outcome')
dbt1 = dbt

dbt1[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = dbt1[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
missing_plot(dbt1, 'Outcome')
dbt2 = dbt1.fillna(value = {'Insulin':dbt1.Insulin.median()})

plt.hist(dbt1.Insulin, bins=80)

plt.hist(dbt2.Insulin, bins=80)
def median_outcome(var):   

#    temp = dbt1[dbt1[var].notnull()]

    median = dbt[[var, 'Outcome']].groupby(['Outcome'])[[var]].median()

    return median
median_outcome('BMI')
temp = np.array(median_outcome('BMI'))

dbt.loc[(dbt['Outcome'] == 0 ) & (dbt['BMI'].isnull()), 'BMI'] = 30.1

dbt.loc[(dbt['Outcome'] == 1 ) & (dbt['BMI'].isnull()), 'BMI'] = 34.3
median_outcome('Glucose')
temp = np.array(median_outcome('Glucose'))

dbt.loc[(dbt['Outcome'] == 0 ) & (dbt['Glucose'].isnull()), 'Glucose'] = 107

dbt.loc[(dbt['Outcome'] == 1 ) & (dbt['Glucose'].isnull()), 'Glucose'] = 140
median_outcome('Insulin')
temp = np.array(median_outcome('Insulin'))

dbt.loc[(dbt['Outcome'] == 0 ) & (dbt['Insulin'].isnull()), 'Insulin'] = 102.5

dbt.loc[(dbt['Outcome'] == 1 ) & (dbt['Insulin'].isnull()), 'Insulin'] = 169.5
median_outcome('BloodPressure')
temp = np.array(median_outcome('BloodPressure'))

dbt.loc[(dbt['Outcome'] == 0 ) & (dbt['BloodPressure'].isnull()), 'BloodPressure'] = 70

dbt.loc[(dbt['Outcome'] == 1 ) & (dbt['BloodPressure'].isnull()), 'BloodPressure'] = 74.5
median_outcome('SkinThickness')
dbt.loc[(dbt['Outcome'] == 0 ) & (dbt['SkinThickness'].isnull()), 'SkinThickness'] = 27

dbt.loc[(dbt['Outcome'] == 1 ) & (dbt['SkinThickness'].isnull()), 'SkinThickness'] = 32
missing_plot(dbt, 'Outcome')
plt.style.use('ggplot') 

f, ax = plt.subplots(figsize=(12, 16))

ax.set_facecolor('#ffffff')

ax.set(xlim=(-5, 250))

plt.ylabel('Paramètres')

ax = sns.boxplot(data = dbt1, 

  orient = 'h', 

  palette = 'Set2')
def plot_distribution(col, size_bin) :  

    # 2 datasets

    tmp1 = D[col]

    tmp2 = N[col]

    hist_data = [tmp1, tmp2]

    

    group_labels = ['diabetic', 'healthy']

    colors = ['#FFD700', '#7EC0EE']



    fig = ff.create_distplot(hist_data, group_labels, colors = colors, show_hist = True, bin_size = size_bin, curve_type='kde')

    fig['layout'].update(title = col)



    py.iplot(fig, filename = 'Density plot')
sns.violinplot(x="Outcome", y="Pregnancies", data=dbt, palette="Set2")
sns.boxplot(x="Outcome", y="Glucose", data=dbt, palette="Set2")
sns.violinplot(x="Outcome", y="BloodPressure", data=dbt, palette="Set2")
sns.violinplot(x="Outcome", y="SkinThickness", data=dbt, palette="Set2")
ax = sns.catplot(x="Outcome", y="Insulin", data=dbt, palette="Set2", kind="box", height=8)
sns.boxplot(x="Outcome", y="BMI", data=dbt, palette="Set2")
sns.violinplot(x="Outcome", y="DiabetesPedigreeFunction", data=dbt, palette="Set2")
sns.swarmplot(x="Outcome", y="Age", data=dbt, palette="Set2")
X = dbt.drop(['Outcome'], axis=1)

Y = dbt.Outcome
#Splitting the data into training and test

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)
print(x_train.shape)

print(x_test.shape)
lr = LogisticRegression()

lr.fit(x_train,y_train)

y_lr = lr.predict(x_test)
print(confusion_matrix(y_test,y_lr))
print(accuracy_score(y_test,y_lr))
print(classification_report(y_test, y_lr))
probas = lr.predict_proba(x_test)

dfprobas = pd.DataFrame(probas,columns=['proba_0','proba_1'])

dfprobas['y'] = np.array(y_test)

plt.figure(figsize=(10,10))

sns.distplot(1-dfprobas.proba_0[dfprobas.y==0], bins=50)

sns.distplot(dfprobas.proba_1[dfprobas.y==1], bins=50)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,probas[:, 1])

roc_auc = auc(false_positive_rate, true_positive_rate)

print (roc_auc)
plt.figure(figsize=(12,12))

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')        # plus mauvaise courbe

plt.plot([0,0,1],[0,1,1],'g:')     # meilleure courbe

plt.xlim([-0.1,1.2])

plt.ylim([-0.1,1.2])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
rf = ensemble.RandomForestClassifier()

rf.fit(x_train, y_train)

y_rf = rf.predict(x_test)

print(classification_report(y_test, y_rf))

cm = confusion_matrix(y_test, y_rf)

print(cm)
param_grid = {

              'n_estimators': [10, 100, 500],

              'min_samples_leaf': [1, 20, 50]

             }

estimator = ensemble.RandomForestClassifier()

rf_gs = model_selection.GridSearchCV(estimator, param_grid)

rf_gs.fit(x_train, y_train)

print(rf_gs.best_params_)
rf1 = ensemble.RandomForestClassifier(n_estimators=100, min_samples_leaf=20, max_features=4)

rf1.fit(x_train, y_train)

y_rf1 = rf.predict(x_test)

print(classification_report(y_test, y_rf1))
rf2 = rf_gs.best_estimator_

y_rf2 = rf2.predict(x_test)

print(classification_report(y_test, y_rf2))
importances = rf2.feature_importances_

indices = np.argsort(importances)

plt.figure(figsize=(8,5))

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), x_train.columns[indices])

plt.title('Importance des caracteristiques')
import xgboost as XGB

xgb  = XGB.XGBClassifier()

xgb.fit(x_train, y_train)

y_xgb = xgb.predict(x_test)

cm = confusion_matrix(y_test, y_xgb)

print(cm)

print(classification_report(y_test, y_xgb))