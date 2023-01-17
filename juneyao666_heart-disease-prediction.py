import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt 
import plotly.graph_objs as go # interactive plotting library
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve,confusion_matrix,auc
heart_data = pd.read_csv("../input/heart-disease-uci/heart.csv")
heart_data.head()
sexMap={1:"male",0:"female"}
heart_data['sex_label']=heart_data['sex'].map(sexMap)
cpMap={1:"typical angina",2:"atypical angina",3:"non-anginal pain",4:"asymptomatic"}
heart_data['cp_label']=heart_data['cp'].map(cpMap)
fbsMap={1:"true",0:"false"}
heart_data['fbs_label']=heart_data['fbs'].map(fbsMap)
restecgMap={0:"normal",1:"having ST-T wave abnormality",2:"showing probable or definite left ventricular hypertrophy by Estes' criteria"}
heart_data['restecg_label']=heart_data['restecg'].map(restecgMap)
exangMap={0:"no",1:"yes"}
heart_data['exang_label']=heart_data['exang'].map(exangMap)
slopeMap={1:"upsloping",2:"flat",3:"downsloping"}
heart_data['slope_label']=heart_data['slope'].map(slopeMap)
thalMap={1:"normal",2:"fixed defect",3:"reversable defect"}
heart_data['thal_label']=heart_data['thal'].map(thalMap)

targetMap={1:"heart disease",0:"health"}
heart_data['target_label']=heart_data['target'].map(targetMap)

heart_data['trestbps_label'] = heart_data.trestbps.apply(lambda x: "normal" if x<120  else("elevated" if x < 130 else ( "high stage1" if x < 140 else( "high stage2" if x < 180 else "hypertensive crisis" ))))
fig = px.violin(heart_data, y="age", x="sex_label", color="target_label", box=True, points="all",
                labels={
                    "sex_label": "Gender",
                     "target_label": "Health Condition"
                 },title="Age Distribution by Gender And Health Condition",
          hover_data=heart_data.columns)
fig.show()
fig = px.histogram(heart_data, x="trestbps", color="target_label",labels={
                     "trestbps": "Resting blood pressure",
                     "target_label": "Health Condition"
                 },title="Resting Blood Pressure Distribution by Health Condition",marginal="box")
fig.update_traces(opacity=0.85)
fig.show()
fig = px.histogram(heart_data, x="chol", color="target_label",labels={
                     "chol": "Serum cholestoral in mg/dl",
                     "target_label": "Health Condition"
                 },title="Serum Cholestoral Distribution by Health Condition",marginal="box")
fig.update_traces(opacity=0.85)
fig.show()
fig = px.histogram(heart_data, x="thalach", color="target_label",labels={
                     "thalach": "The person's maximum heart rate achieved",
                     "target_label": "Health Condition"
                 },title="Maximum Heart Rate Achieved Distribution by Health Condition",marginal="box")
fig.update_traces(opacity=0.85)
fig.show()
fig = px.histogram(heart_data, x="oldpeak", color="target_label",labels={
                     "oldpeak": "ST depression induced by exercise relative to rest",
                     "target_label": "Health Condition"
                 },title="Oldpeak Distribution by Health Condition",marginal="box")
fig.update_traces(opacity=0.85)
fig.show()
# Taking the count of each Sex value of the disease
heart_disease_cp= heart_data[heart_data.target.eq(1)]['cp_label'].value_counts()
heart_disease_cp = pd.DataFrame({'cp':heart_disease_cp.index, 'count':heart_disease_cp.values})

# Taking the count of each Sex value of the health
heart_health_cp = heart_data[heart_data.target.eq(0)]['cp_label'].value_counts()
heart_health_cp = pd.DataFrame({'cp':heart_health_cp.index, 'count':heart_health_cp.values})

fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
fig.add_trace(go.Pie(labels=heart_disease_cp['cp'], values=heart_disease_cp['count'], name="disease"),
              1, 1)
fig.add_trace(go.Pie(labels=heart_health_cp['cp'], values=heart_health_cp['count'], name="health"),
              1, 2)

# Use `hole` to create a donut-like pie chart
fig.update_traces(hole=.45, hoverinfo="label+percent+value+name")

fig.update_layout(
    title_text="The Chest Pain Distribution by Health Condition",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Heart Disease', x=0.14, y=0.5, font_size=16, showarrow=False),
                 dict(text='Health', x=0.82, y=0.5, font_size=16, showarrow=False)])
fig.show()
# Taking the count of each Sex value of the disease
heart_disease_restecg= heart_data[heart_data.target.eq(1)]['restecg_label'].value_counts()
heart_disease_restecg = pd.DataFrame({'restecg':heart_disease_restecg.index, 'count':heart_disease_restecg.values})

# Taking the count of each Sex value of the health
heart_health_restecg = heart_data[heart_data.target.eq(0)]['restecg_label'].value_counts()
heart_health_restecg = pd.DataFrame({'restecg':heart_health_restecg.index, 'count':heart_health_restecg.values})

fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
fig.add_trace(go.Pie(labels=heart_disease_restecg['restecg'], values=heart_disease_restecg['count'], name="disease"), 
              1, 1)
fig.add_trace(go.Pie(labels=heart_health_restecg['restecg'], values=heart_health_restecg['count'], name="health"),
              1, 2)

# Use `hole` to create a donut-like pie chart
fig.update_traces(hole=.45, hoverinfo="label+percent+value+name")

fig.update_layout(
    title_text="Resting Electrocardiographic Distribution by Health Condition",
    legend=dict( yanchor="top", y=-0.1, xanchor="left", x=0.6),
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Heart Disease', x=0.15, y=0.5, font_size=16, showarrow=False),
                 dict(text='Health', x=0.82, y=0.5, font_size=16, showarrow=False)])
fig.show()
# Taking the count of each Sex value of the disease
heart_disease_slope= heart_data[heart_data.target.eq(1)]['slope'].value_counts()
heart_disease_slope = pd.DataFrame({'slope':heart_disease_slope.index, 'count':heart_disease_slope.values})

# Taking the count of each Sex value of the health
heart_health_slope = heart_data[heart_data.target.eq(0)]['slope'].value_counts()
heart_health_slope = pd.DataFrame({'slope':heart_health_slope.index, 'count':heart_health_slope.values})

fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
fig.add_trace(go.Pie(labels=heart_disease_slope['slope'], values=heart_disease_slope['count'], name="disease"),
              1, 1)
fig.add_trace(go.Pie(labels=heart_health_slope['slope'], values=heart_health_slope['count'], name="health"),
              1, 2)

# Use `hole` to create a donut-like pie chart
fig.update_traces(hole=.45, hoverinfo="label+percent+value+name")

fig.update_layout(
    title_text="Slope Distribution by Health Condition",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Heart Disease', x=0.15, y=0.5, font_size=16, showarrow=False),
                 dict(text='Health', x=0.82, y=0.5, font_size=16, showarrow=False)])
fig.show()
# Taking the count of each Sex value of the disease
heart_disease_thal= heart_data[heart_data.target.eq(1)]['thal_label'].value_counts()
heart_disease_thal = pd.DataFrame({'thal':heart_disease_thal.index, 'count':heart_disease_thal.values})

# Taking the count of each Sex value of the health
heart_health_thal = heart_data[heart_data.target.eq(0)]['thal_label'].value_counts()
heart_health_thal = pd.DataFrame({'thal':heart_health_thal.index, 'count':heart_health_thal.values})

fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
fig.add_trace(go.Pie(labels=heart_disease_thal['thal'], values=heart_disease_thal['count'], name="disease"),
              1, 1)
fig.add_trace(go.Pie(labels=heart_health_thal['thal'], values=heart_health_thal['count'], name="health"),
              1, 2)

# Use `hole` to create a donut-like pie chart
fig.update_traces(hole=.45, hoverinfo="label+percent+value+name")

fig.update_layout(
    title_text="Thalassemia Distribution by Health Condition",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Heart Disease', x=0.15, y=0.5, font_size=16, showarrow=False),
                 dict(text='Health', x=0.82, y=0.5, font_size=16, showarrow=False)])
fig.show()
heart_data.corr()['target'].sort_values(ascending =False)[0:15]
fig = px.imshow(heart_data.corr(),color_continuous_scale='rdbu_r',zmin=-1,zmax=1)
fig.show()
data_X=heart_data.iloc[:, 0:13]
data_y=heart_data[['target']]
X_train,X_test,y_train,y_test = train_test_split(data_X, data_y, test_size = 0.25, random_state = 250)
std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.transform(X_test)
lg = LogisticRegression()  
lg.fit(X_train, y_train.values.ravel())
#y_predict = lg.predict(X_test)
y_scores_lg = lg.predict_proba(X_test)
fpr_lg, tpr_lg, threshold_lg = roc_curve(y_test, y_scores_lg[:, 1])
roc_auc_lg = auc(fpr_lg, tpr_lg)
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train, y_train.values.ravel())
y_scores_knn = knn.predict_proba(X_test)
fpr_knn, tpr_knn, threshold_knn = roc_curve(y_test, y_scores_knn[:, 1])
roc_auc_knn = auc(fpr_knn, tpr_knn)
NB = GaussianNB()
NB.fit(X_train, y_train.values.ravel())
y_scores_NB = NB.predict_proba(X_test)
fpr_NB, tpr_NB, threshold_NB = roc_curve(y_test, y_scores_NB[:, 1])
roc_auc_NB = auc(fpr_NB, tpr_NB)
fig = go.Figure()
# Create and style traces
line_lg=go.Scatter(x=fpr_lg, y=tpr_lg, name='LG AUC = %0.2f' % roc_auc_lg,
                         line=dict(color='firebrick', width=4))
line_knn=go.Scatter(x=fpr_knn, y=tpr_knn, name='KNN AUC = %0.2f' % roc_auc_knn,
                         line=dict(color='royalblue', width=4))
line_NB=go.Scatter(x=fpr_NB, y=tpr_NB, name='GaussianNB AUC = %0.2f' % roc_auc_NB,
                         line=dict(color='green', width=4))

fig.add_trace(line_lg)
fig.add_trace(line_knn)
fig.add_trace(line_NB)
fig.add_trace(go.Scatter(x=[0,1],y=[0,1],name='',line=dict(color='black',dash='dash')))
fig.update_layout(title='ROC Curve',
                   xaxis_title='False Positive Rate',
                   yaxis_title='True Positive Rate')


fig.show()