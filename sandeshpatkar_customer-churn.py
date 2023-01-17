import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pprint #better dictionary printing

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import plotly.figure_factory as ff

import plotly.express as px
churn = pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
churn.head()
churn.info()
print('Rows: ', churn.shape[0])

print('Columns: ', churn.shape[1])

print('\nFeatures:\n ', churn.columns.tolist())

print('\nUnique Value Count:\n ', churn.nunique())

print('\nMissing Value:\n ', churn.isnull().sum())
def values(cols):

    d = {}

    for col in cols:

        x = churn[col].unique()

        d[col] = x

    pprint.pprint(d)
print('Feature Values: \n')

values(churn.columns)
churn.TotalCharges.min()
churn['TotalCharges'] = churn['TotalCharges'].replace(' ',np.nan)
churn.head()
churn = churn[churn['TotalCharges'].notnull()]
churn['TotalCharges'] = churn['TotalCharges'].astype(float)
churn['SeniorCitizen'] = churn['SeniorCitizen'].replace({1: 'Yes', 0: 'No'})
churn['tenure'].min()
churn['tenure'].max()
def tenure_slabs(value):

    if value <= 12:

        return 'ten_0-12'

    elif (value > 12) & (value <= 24):

        return 'ten_12-24'

    elif (value > 24) & (value <= 36):

        return 'ten_24-36'

    elif (value > 36) & (value <= 48):

        return 'ten_36-48'

    elif (value > 48) & (value <= 60):

        return 'ten_48-60'

    elif (value > 60) & (value <= 72):

        return 'ten_60-72'
churn['tenure_duration'] = churn['tenure'].apply(tenure_slabs) #to categorical column
def make_df(data, col):

    df = pd.DataFrame(data[col].value_counts(normalize = True)*100)

    df = df.reset_index()

    return df
gen = make_df(churn, 'gender')

gen.head()
px.bar(gen, x = 'index', y = 'gender', title = 'Gender Distribution: Overall data')
sen = make_df(churn, 'SeniorCitizen')

sen.head()
px.bar(sen, x = 'index', y = 'SeniorCitizen', title = 'Senior Citizen Distribution: Overall data')
labels = churn['Churn'].value_counts().keys().tolist()

vals = churn['Churn'].value_counts().values.tolist()



fig = go.Figure(data = go.Pie(labels = labels, values = vals))

fig.update_traces(hoverinfo = 'label+value', marker = dict(colors = ['rgb(124,185,232)', 'gold']), hole = .5)

fig.update(layout_title_text = 'Customer Churn Data: Overall data', layout_showlegend = True)

fig.show()
churn.shape
y_churn = churn[churn['Churn'] == 'Yes'] #customers who have left the network

n_churn = churn[churn['Churn'] == 'No'] #customers who have stayed with the network
print('Number of people who left the telecom:', y_churn.shape[0])

print('Number of people who did not left the telecom:', n_churn.shape[0])
def plot_pie(labels, values):

    fig = go.Figure(data = go.Pie(labels = labels, values = values))

    fig.update_traces(hoverinfo='label+value', marker = dict(colors = ['royal blue', 'gold']), hole = .5)

    fig.show()

    

def label(col, churn):

    if churn == 1:

        x = y_churn[col].value_counts().keys().tolist()

        return x

    else:

        x = n_churn[col].value_counts().keys().tolist()

        return x



def values(col, churn):

    if churn == 1:

        x = y_churn[col].value_counts().values.tolist()

        return x

    else: 

        x = n_churn[col].value_counts().values.tolist()

        return x
from plotly.subplots import make_subplots
def make_pies(column, title):

    specs = [[{'type':'domain'}, {'type':'domain'}]]

    colors = ['rgb(124,185,232)','rgb(255,213,0)','rgb(25,77,0)','rgb(255,126,0)','rgb(153,255,102)']

    fig = make_subplots(rows = 1, cols = 2, specs = specs)

    fig.add_trace(go.Pie(labels = label(column, 1), values = values(column, 1), name = 'Churn', marker_colors = colors), 1,1)

    fig.add_trace(go.Pie(labels = label(column, 0), values = values(column, 0), name = 'Non-churn', marker_colors = colors), 1,2)

    fig.update_traces(hoverinfo = 'label+value', hole = 0.6)

    fig.update(layout_title_text = title +': Churn Vs. Non-churn customers')

    fig.update_layout(annotations = [dict(text = 'Churn',x=0.18, y=0.5, font_size=20, showarrow=False),

                                    dict(text = 'Non-churn',x=0.85, y=0.5, font_size=20, showarrow=False)])

    fig.show()
make_pies('gender', 'Gender')
make_pies('SeniorCitizen', 'Senior Citizen')
make_pies('DeviceProtection', 'Device Protection')
make_pies('OnlineBackup', 'Online Backup')
make_pies('OnlineSecurity', 'Online Security')
make_pies('StreamingMovies', 'Streaming Movies')
make_pies('StreamingTV','Streaming TV')
make_pies('TechSupport', 'Tech Support')
make_pies('Contract','Contract')
make_pies('SeniorCitizen', 'Senior Citizen')
make_pies('Partner','Partner')
make_pies('Dependents', 'Dependents')
make_pies('PhoneService', 'Phone Service')
make_pies('PaperlessBilling', 'Paperless Billing')
make_pies('PaymentMethod', 'Payment Method')
def make_hist(column, title):

    #fig = make_subplots(rows = 1, cols = 2)

    fig = go.Figure()

    fig.add_trace(go.Histogram(x = n_churn[column], name = 'Non-churn'))

    fig.add_trace(go.Histogram(x = y_churn[column], name = 'Churn'))

    #fig.append_trace(h1, 1,1)

    #fig.append_trace(h2, 1,2)

    fig.update_layout(title_text = title+': Churn Vs. Non-churn customers', 

                      xaxis_title_text = 'Value', yaxis_title_text = 'Count',

                     bargap = 0.2,

                     bargroupgap = 0.1

                     )

    fig.show()
make_hist('TotalCharges', 'Total Charge')
make_hist('tenure_duration', 'Tenure Duration')
avg_charges = churn.groupby('tenure_duration').mean().reset_index()
fig = px.bar(avg_charges, x = 'tenure_duration', y = 'MonthlyCharges')

fig.show()
churn.columns.tolist()
churn.nunique()
category_cols = []

for col in churn.columns.tolist():

    if (churn[col].nunique() <= 6):

        category_cols.append(col)

print(category_cols)
from sklearn import preprocessing



churn2 = churn.copy()

le = preprocessing.LabelEncoder()

churn2[category_cols] = churn2[category_cols].apply(le.fit_transform)

churn2.head()
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



scaled_values = scaler.fit_transform(churn2[numeric_cols])

scaled_values = pd.DataFrame(scaled_values, columns = numeric_cols)

scaled_values.head()
scaled_values.isnull().sum()
churn2.isnull().sum()
churn2 = churn2.drop(columns = numeric_cols, axis = 1)

churn2 = churn2.merge(scaled_values, how = 'left', left_index = True, right_index = True)

churn2.head()

churn2 = churn2.dropna()
correlation = churn2.corr()

correlation
corr_col = correlation.columns.tolist()

fig = go.Figure(data = go.Heatmap(z = correlation,

                                 x = corr_col,

                                 y = corr_col)

               )



fig.update_layout(title = 'Correlation Matrix', width = 800, height = 800)

fig.update_xaxes(tickangle = 90)

fig.show()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

from sklearn.metrics import roc_auc_score,roc_curve,scorer
t_cols = []

for i in churn2.columns:

    if (i != 'Churn') & (i != 'customerID'):

        t_cols.append(i)
train_data = churn2[t_cols]

target = churn2['Churn']
x_train, x_test, y_train, y_test = train_test_split(train_data, target, test_size = 0.3, random_state = 1)



lr = LogisticRegression(solver = 'liblinear')



lr.fit(x_train, y_train)
predictions_lr = lr.predict(x_test)
probs = lr.predict_proba(x_test)
acc_lr = lr.score(x_test, y_test)

print('The accuracy of this model is:', round(acc_lr, 3)*100, '%')

print('\n')



print('Classification Report:\n')

clf_report_lr = classification_report(y_test, predictions_lr)

print(clf_report_lr)

print('\n')



con_matrix = confusion_matrix(y_test, predictions_lr)

print('Confusion Matrix:\n')

print(con_matrix)

print('\n')



roc_auc = roc_auc_score(y_test, predictions_lr)

print('Area under the curve:',roc_auc)
#plotting confusion matrix

fig = go.Figure(data = go.Heatmap(z = con_matrix,

                                 x = ['Not Churn', 'Churn'],

                                 y = ['Not Churn', 'Churn'],

                                 colorscale = 'Cividis',

                                 showscale = False))

fig.update_layout(title = 'Confusion Matrix')

fig.show()
import scikitplot as skplt
#plotting ROC curve

skplt.metrics.plot_roc(y_test, probs, figsize = (8,8), title = 'ROC Curves: Logistic Regression Model')
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)
predict_tree = tree.predict(x_test)
def model_metrics(algo, x, y, preds):

    score = algo.score(x,y)

    print('The accuracy of this model is:', round(score, 3)*100, '%')

    print('\n')

    

    print('Classification Report:\n')

    clf_report = classification_report(y, preds)

    print(clf_report_lr)

    print('\n')

    

    con_matrix = confusion_matrix(y, preds)

    print('Confusion Matrix:\n')

    print(con_matrix)

    print('\n')

    

    roc_auc = roc_auc_score(y, preds)

    print('Area under the curve:',roc_auc)

    

    return con_matrix
c = model_metrics(tree, x_test, y_test, predict_tree )
def plot_confusion(con_mat, model_name):

    fig = go.Figure(data = go.Heatmap(z = con_mat,

                                      x = ['Not Churn','Churn'],

                                      y = ['Not Churn','Churn'],

                                      colorscale = 'Cividis',

                                      showscale = False

                                     ))

    fig.update_layout(title = 'Confusion Matrix: '+ model_name)

    fig.show()
plot_confusion(c, 'Decision Tree')
from sklearn.feature_selection import RFE



lr = LogisticRegression(solver = 'liblinear')

rfe = RFE(lr,10)

rfe = rfe.fit(x_train, y_train.values.ravel())



rfe.support_

rfe.ranking_

churn_rfe = pd.DataFrame({"rfe_support" :rfe.support_,

                       "columns" : [i for i in churn2.columns if i not in churn2[['customerID', 'Churn']]],

                       "ranking" : rfe.ranking_,

                      })

rfe_cols = churn_rfe[churn_rfe["rfe_support"] == True]["columns"].tolist()
rfe_cols
rfe_xtrain = x_train[rfe_cols]

rfe_ytrain = y_train

rfe_xtest = x_test[rfe_cols]

rfe_ytest = y_test
lr.fit(rfe_xtrain, rfe_ytrain)
predictions_rfe_lr = lr.predict(rfe_xtest)

probs_rfe = rfe.predict_proba(x_test)
print(rfe_xtest.shape)

print(rfe_ytest.shape)

#acc_rfe_lr = lr.score(rfe_xtest, rfe_ytest)

#print('Model Accuracy is:', acc_rfe_lr*100, '%')
predictions_rfe_lr
acc_lr_rfe = lr.score(rfe_xtest, rfe_ytest)

print('The accuracy of this model is:', round(acc_lr_rfe, 3)*100, '%')
c_rfe = model_metrics(rfe, x_test, y_test, predictions_rfe_lr)
plot_confusion(c_rfe, 'Recursive Feature Elimination')
skplt.metrics.plot_roc(y_test, probs_rfe, figsize = (8,8), title = 'ROC Curve: RFE Model')