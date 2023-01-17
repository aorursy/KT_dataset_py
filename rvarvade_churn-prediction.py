from __future__ import division



from datetime import datetime, timedelta,date

import pandas as pd

%matplotlib inline

from sklearn.metrics import classification_report,confusion_matrix

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from sklearn.cluster import KMeans
import plotly as py

import plotly.offline as pyoff

import plotly.graph_objs as go
import xgboost as xgb

from sklearn.model_selection import KFold, cross_val_score, train_test_split
pyoff.init_notebook_mode()
df_data = pd.read_csv('../input/churn-prediction/Churn.csv')

df_data.head()
df_data.info()
df_data.loc[df_data.Churn=='No','Churn'] = 0 

df_data.loc[df_data.Churn=='Yes','Churn'] = 1
df_data.groupby('gender').Churn.mean()
df_plot = df_data.groupby('gender').Churn.mean().reset_index()

plot_data = [

    go.Bar(

        x=df_plot['gender'],

        y=df_plot['Churn'],

        width = [0.5, 0.5],

        marker=dict(

        color=['green', 'blue'])

    )

]



plot_layout = go.Layout(

        xaxis={"type": "category"},

        yaxis={"title": "Churn Rate"},

        title='Gender',

        plot_bgcolor  = 'rgb(243,243,243)',

        paper_bgcolor  = 'rgb(243,243,243)',

    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)



df_plot = df_data.groupby('Partner').Churn.mean().reset_index()

plot_data = [

    go.Bar(

        x=df_plot['Partner'],

        y=df_plot['Churn'],

        width = [0.5, 0.5],

        marker=dict(

        color=['green', 'blue'])

    )

]



plot_layout = go.Layout(

        xaxis={"type": "category"},

        yaxis={"title": "Churn Rate"},

        title='Partner',

        plot_bgcolor  = 'rgb(243,243,243)',

        paper_bgcolor  = 'rgb(243,243,243)',

    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
df_plot = df_data.groupby('PhoneService').Churn.mean().reset_index()

plot_data = [

    go.Bar(

        x=df_plot['PhoneService'],

        y=df_plot['Churn'],

        width = [0.5, 0.5],

        marker=dict(

        color=['green', 'blue'])

    )

]



plot_layout = go.Layout(

        xaxis={"type": "category"},

        yaxis={"title": "Churn Rate"},

        title='Phone Service',

        plot_bgcolor  = 'rgb(243,243,243)',

        paper_bgcolor  = 'rgb(243,243,243)',

        

    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
df_plot = df_data.groupby('MultipleLines').Churn.mean().reset_index()

plot_data = [

    go.Bar(

        x=df_plot['MultipleLines'],

        y=df_plot['Churn'],

        width = [0.5, 0.5, 0.5],

        marker=dict(

        color=['green', 'blue', 'orange'])

    )

]



plot_layout = go.Layout(

        xaxis={"type": "category"},

        title='Multiple Lines',

        yaxis={"title": "Churn Rate"},

        plot_bgcolor  = 'rgb(243,243,243)',

        paper_bgcolor  = 'rgb(243,243,243)',

    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
df_plot = df_data.groupby('InternetService').Churn.mean().reset_index()

plot_data = [

    go.Bar(

        x=df_plot['InternetService'],

        y=df_plot['Churn'],

        width = [0.5, 0.5, 0.5],

        marker=dict(

        color=['green', 'blue', 'orange'])

    )

]



plot_layout = go.Layout(

        xaxis={"type": "category"},

        title='Internet Service',

        yaxis={"title": "Churn Rate"},

        plot_bgcolor  = 'rgb(243,243,243)',

        paper_bgcolor  = 'rgb(243,243,243)',

    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
df_plot = df_data.groupby('OnlineSecurity').Churn.mean().reset_index()

plot_data = [

    go.Bar(

        x=df_plot['OnlineSecurity'],

        y=df_plot['Churn'],

        width = [0.5, 0.5, 0.5],

        marker=dict(

        color=['green', 'blue', 'orange'])

    )

]



plot_layout = go.Layout(

        xaxis={"type": "category"},

        yaxis={"title": "Churn Rate"},

        title='Online Security',

        plot_bgcolor  = "rgb(243,243,243)",

        paper_bgcolor  = "rgb(243,243,243)",

    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
df_plot = df_data.groupby('OnlineBackup').Churn.mean().reset_index()

plot_data = [

    go.Bar(

        x=df_plot['OnlineBackup'],

        y=df_plot['Churn'],

        width = [0.5, 0.5, 0.5],

        marker=dict(

        color=['green', 'blue', 'orange'])

    )

]



plot_layout = go.Layout(

        xaxis={"type": "category"},

        title='Online Backup',

        plot_bgcolor  = "rgb(243,243,243)",

        paper_bgcolor  = "rgb(243,243,243)",

    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)

df_plot = df_data.groupby('DeviceProtection').Churn.mean().reset_index()

plot_data = [

    go.Bar(

        x=df_plot['DeviceProtection'],

        y=df_plot['Churn'],

        width = [0.5, 0.5, 0.5],

        marker=dict(

        color=['green', 'blue', 'orange'])

    )

]



plot_layout = go.Layout(

        xaxis={"type": "category"},

        title='Device Protection',

        plot_bgcolor  = "rgb(243,243,243)",

        paper_bgcolor  = "rgb(243,243,243)",

    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
df_plot = df_data.groupby('TechSupport').Churn.mean().reset_index()

plot_data = [

    go.Bar(

        x=df_plot['TechSupport'],

        y=df_plot['Churn'],

        marker=dict(

        color=['green', 'blue', 'orange'])

    )

]



plot_layout = go.Layout(

        xaxis={"type": "category"},

        title='Tech Support'    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
df_plot = df_data.groupby('StreamingTV').Churn.mean().reset_index()

plot_data = [

    go.Bar(

        x=df_plot['StreamingTV'],

        y=df_plot['Churn'],

        marker=dict(

        color=['green', 'blue', 'orange'])

    )

]



plot_layout = go.Layout(

        xaxis={"type": "category"},

        title='Streaming TV',

    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
df_plot = df_data.groupby('StreamingMovies').Churn.mean().reset_index()

plot_data = [

    go.Bar(

        x=df_plot['StreamingMovies'],

        y=df_plot['Churn'],

        marker=dict(

        color=['green', 'blue', 'orange'])

    )

]



plot_layout = go.Layout(

        xaxis={"type": "category"},

        title='Streaming Movies'

    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
df_plot = df_data.groupby('Contract').Churn.mean().reset_index()

plot_data = [

    go.Bar(

        x=df_plot['Contract'],

        y=df_plot['Churn'],

        marker=dict(

        color=['green', 'blue', 'orange'])

    )

]



plot_layout = go.Layout(

        xaxis={"type": "category"},

        title='Contract'

    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
df_plot = df_data.groupby('PaperlessBilling').Churn.mean().reset_index()

plot_data = [

    go.Bar(

        x=df_plot['PaperlessBilling'],

        y=df_plot['Churn'],

        marker=dict(

        color=['green', 'blue', 'orange'])

    )

]



plot_layout = go.Layout(

        xaxis={"type": "category"},

        title='Paperless Billing'    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
df_plot = df_data.groupby('PaymentMethod').Churn.mean().reset_index()

plot_data = [

    go.Bar(

        x=df_plot['PaymentMethod'],

        y=df_plot['Churn'],

        marker=dict(

        color=['green', 'blue', 'orange','red'])

    )

]



plot_layout = go.Layout(

        xaxis={"type": "category"},

        title='Payment Method'    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
df_data.head()
df_data.tenure.describe()
df_plot = df_data.groupby('tenure').Churn.mean().reset_index()



plot_data = [

    go.Scatter(

        x=df_plot['tenure'],

        y=df_plot['Churn'],

        mode='markers',

        name='Low',

        marker= dict(size= 7,

            line= dict(width=1),

            color= 'blue',

            opacity= 0.8

           ),

    )

]



plot_layout = go.Layout(

        yaxis= {'title': "Churn Rate"},

        xaxis= {'title': "Tenure"},

        title='Tenure based Churn rate',

        plot_bgcolor  = "rgb(243,243,243)",

        paper_bgcolor  = "rgb(243,243,243)",

    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
df_plot = df_data.groupby('MonthlyCharges').Churn.mean().reset_index()



plot_data = [

    go.Scatter(

        x=df_plot['MonthlyCharges'],

        y=df_plot['Churn'],

        mode='markers',

        name='Low',

        marker= dict(size= 7,

            line= dict(width=1),

            color= 'blue',

            opacity= 0.9

           ),

    )

]



plot_layout = go.Layout(

        yaxis= {'title': "Churn Rate"},

        xaxis= {'title': "Monthly Charges"},

        title='Monthly Charges based Churn rate'

    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)

df_plot = df_data.groupby('TotalCharges').Churn.mean().reset_index()



plot_data = [

    go.Scatter(

        x=df_plot['TotalCharges'],

        y=df_plot['Churn'],

        mode='markers',

        name='Low',

        marker= dict(size= 7,

            line= dict(width=1),

            color= 'blue',

            opacity= 0.8

           ),

    )

]



plot_layout = go.Layout(

        yaxis= {'title': "Churn Rate"},

        xaxis= {'title': "Total Charges"},

        title='Total Charges based Churn rate'

    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
sse = {}

df_cluster = df_data[['tenure']]

for k in range (1, 10):

    kmeans = KMeans(n_clusters=k, max_iter =1000).fit(df_cluster)

    df_cluster['clusters'] = kmeans.labels_

    sse[k] = kmeans.inertia_

plt.figure()

plt.plot(list(sse.keys()), list(sse.values()))

plt.xlabel("Number of cluster")

plt.show()

kmeans = KMeans(n_clusters=3)

df_data['TenureCluster'] = kmeans.fit_predict(df_data[['tenure']])
df_data.groupby('TenureCluster')['tenure'].describe()
def order_cluster(cluster_field_name, target_field_name, df, ascending):

    new_cluster_field_name = 'new_' + cluster_field_name

    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()

    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)

    df_new['index'] = df_new.index

    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)

    df_final = df_final.drop([cluster_field_name],axis=1)

    df_final = df_final.rename(columns={"index":cluster_field_name})

    return df_final
df_data = order_cluster('TenureCluster', 'tenure',df_data,True)
df_data.groupby('TenureCluster').tenure.describe()
df_data['TenureCluster'] = df_data["TenureCluster"].replace({0:'Low',1:'Mid',2:'High'})
df_plot = df_data.groupby('TenureCluster').Churn.mean().reset_index()

plot_data = [

    go.Bar(

        x=df_plot['TenureCluster'],

        y=df_plot['Churn'],

        marker=dict(

        color=['green', 'blue', 'orange','red'])

    )

]



plot_layout = go.Layout(

        xaxis={"type": "category","categoryarray":['Low','Mid','High']},

        title='Tenure Cluster vs Churn Rate'

    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
sse={}

df_cluster = df_data[['MonthlyCharges']]

for k in range(1, 10):

    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df_cluster)

    df_cluster["clusters"] = kmeans.labels_

    sse[k] = kmeans.inertia_ 

plt.figure()

plt.plot(list(sse.keys()), list(sse.values()))

plt.xlabel("Number of cluster")

plt.show()
kmeans = KMeans(n_clusters=3)

df_data['MonthlyChargeCluster'] = kmeans.fit_predict(df_data[['MonthlyCharges']])
df_data = order_cluster('MonthlyChargeCluster', 'MonthlyCharges',df_data,True)
df_data.groupby('MonthlyChargeCluster').MonthlyCharges.describe()
df_data['MonthlyChargeCluster'] = df_data["MonthlyChargeCluster"].replace({0:'Low',1:'Mid',2:'High'})
df_plot = df_data.groupby('MonthlyChargeCluster').Churn.mean().reset_index()

plot_data = [

    go.Bar(

        x=df_plot['MonthlyChargeCluster'],

        y=df_plot['Churn'],

        width = [0.5, 0.5, 0.5],

        marker=dict(

        color=['green', 'blue', 'orange'])

    )

]



plot_layout = go.Layout(

        xaxis={"type": "category","categoryarray":['Low','Mid','High']},

        title='Monthly Charge Cluster vs Churn Rate',

        plot_bgcolor  = "rgb(243,243,243)",

        paper_bgcolor  = "rgb(243,243,243)",

    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)

df_data.head()
df_data['TotalCharges'].describe()
df_data[pd.to_numeric(df_data['TotalCharges'], errors='coerce').isnull()]

len(df_data[pd.to_numeric(df_data['TotalCharges'], errors='coerce').isnull()])
df_data.loc[pd.to_numeric(df_data['TotalCharges'], errors='coerce').isnull(),'TotalCharges'] = np.nan
df_data = df_data.dropna()

df_data['TotalCharges'] = pd.to_numeric(df_data['TotalCharges'], errors='coerce')
sse={}

df_cluster = df_data[['TotalCharges']]

for k in range(1, 10):

    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df_cluster)

    df_cluster["clusters"] = kmeans.labels_

    sse[k] = kmeans.inertia_ 

plt.figure()

plt.plot(list(sse.keys()), list(sse.values()))

plt.xlabel("Number of cluster")

plt.show()
kmeans = KMeans(n_clusters=3)

df_data['TotalChargeCluster'] = kmeans.fit_predict(df_data[['TotalCharges']])
df_data = order_cluster('TotalChargeCluster', 'TotalCharges',df_data,True)

df_data.groupby('TotalChargeCluster').TotalCharges.describe()

df_data['TotalChargeCluster'] = df_data["TotalChargeCluster"].replace({0:'Low',1:'Mid',2:'High'})
df_plot = df_data.groupby('TotalChargeCluster').Churn.mean().reset_index()

plot_data = [

    go.Bar(

        x=df_plot['TotalChargeCluster'],

        y=df_plot['Churn'],

        width = [0.5, 0.5, 0.5],

        marker=dict(

        color=['green', 'blue', 'orange'])

    )

]



plot_layout = go.Layout(

        xaxis={"type": "category","categoryarray":['Low','Mid','High']},

        title='Total Charge Cluster vs Churn Rate',

        plot_bgcolor  = "rgb(243,243,243)",

        paper_bgcolor  = "rgb(243,243,243)",

    )

fig = go.Figure(data=plot_data, layout=plot_layout)

pyoff.iplot(fig)
df_data.info()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

dummy_columns = []



for column in df_data.columns:

    if df_data[column].dtype == object and column != 'customerID':

        if df_data[column].nunique() == 2:

            df_data[column] = le.fit_transform(df_data[column])

        else:

            dummy_columns.append(column)



#apply get dummies for selected columns

df_data = pd.get_dummies(data = df_data,columns = dummy_columns)                   
df_data.head()
df_data.columns
all_columns = []

for column in df_data.columns:

    column = column.replace(" ", "_").replace("(", "_").replace(")", "_").replace("-", "_")

    all_columns.append(column)



df_data.columns = all_columns
df_data.columns
X = df_data.drop(['Churn', 'customerID'], axis = 1)

y = df_data.Churn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 56)

xgb_model = xgb.XGBClassifier(max_depth = 5, learning_rate = 0.08, n_jobs = -1).fit(X_train, y_train)





print('Accuracy of XGB classifier on training set: {:.2f}'

       .format(xgb_model.score(X_train, y_train)))

print('Accuracy of XGB classifier on test set: {:.2f}'

       .format(xgb_model.score(X_test, y_test)))
y_pred = xgb_model.predict(X_test)
print(classification_report(y_test, y_pred))
from xgboost import plot_importance
fig, ax = plt.subplots(figsize=(10,8))

plot_importance(xgb_model, ax=ax)
from numpy import sort

from sklearn.feature_selection import SelectFromModel



thresholds = sort(xgb_model.feature_importances_)
feature_imp = pd.DataFrame(sorted(zip(xgb_model.feature_importances_, X_train.columns)), columns=['Value','Feature'])

features_df = feature_imp.sort_values(by="Value", ascending=False)

selected_features = list(features_df[features_df['Value']>=0.01]['Feature'])

print('The no. of features selected:',len(selected_features))
selected_features
features_df.head()
X_train_selected=X_train[selected_features]

X_test_selected=X_test[selected_features]
#building the model

xgb_model = xgb.XGBClassifier(max_depth=5, learning_rate=0.08, n_jobs=-1).fit(X_train_selected, y_train)



print('Accuracy of XGB classifier on training set: {:.2f}'

       .format(xgb_model.score(X_train_selected, y_train)))

print('Accuracy of XGB classifier on test set: {:.2f}'

       .format(xgb_model.score(X_test_selected, y_test)))