# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 
import pandas as pd

import seaborn as sns 
import matplotlib.pyplot as plt
# Plotly Libraris
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings("ignore")
train = pd.read_csv('../input/health-insurance-cross-sell-prediction/train.csv')
test = pd.read_csv('../input/health-insurance-cross-sell-prediction/test.csv')
sample_sub = pd.read_csv('../input/health-insurance-cross-sell-prediction/sample_submission.csv')
df=train.copy()
test.head(2)
df.head(2)
df.shape
df.info()
df.columns
df.isnull().values.any()
df.isnull().sum()
df.corr()
df[df.duplicated() == True]

df_gender=df['Gender'].value_counts().to_frame().reset_index().rename(columns={'index':'Gender','Gender':'count'})


fig = go.Figure([go.Pie(labels=df_gender['Gender'], values=df_gender['count'],hole=0.2)])

fig.update_traces(hoverinfo='label+percent', textinfo='value+percent', textfont_size=12,insidetextorientation='radial')

fig.update_layout(title="Gender Count",title_x=0.5)
fig.show()
# Violin Boxplot
df_agevi=df['Age']
fig = go.Figure(data=go.Violin(y=df_agevi, box_visible=True, line_color='black',
                               meanline_visible=True, fillcolor='lightseagreen', opacity=0.6,
                               x0='Age'))

fig.update_layout(yaxis_zeroline=False,title="Distribution Of Age",title_x=0.5)
fig.show()
fig = go.Figure(go.Box(y=df['Age'],name="Age ")) # to get Horizonal plot change axis   
fig.update_layout(title="Distribution of Age ",title_x=0.5)
fig.show()
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (15, 5)
sns.distplot(df['Age'], color = 'BlueViolet')
plt.title('Distribution of Age', fontsize = 20)
plt.show()
df['age_category']=np.where((df['Age']<20),"below 20",
                                 np.where((df['Age']>19)&(df['Age']<=30),"20-30",
                                    np.where((df['Age']>30)&(df['Age']<=50),"31-50",
                                                np.where(df['Age']>50,"Above 50","NULL"))))

age=df['age_category'].value_counts().to_frame().reset_index().rename(columns={'index':'age_category','age_category':'Count'})


fig = go.Figure(data=[go.Scatter(
    x=age['age_category'], y=age['Count'],
    mode='markers',
    marker=dict(
        color=age['Count'],
        size=age['Count']*0.0005,
        showscale=True
    ))])

fig.update_layout(title=' Age ',xaxis_title="Age Category",yaxis_title="Number Of People",title_x=0.5)
fig.show()
df_VD=df.groupby(by =['Gender','age_category','Vehicle_Damage'])['Age'].count().to_frame().reset_index().rename(columns={'Gender':'Gender','Vehicle_Damage':'Vehicle_Damage','age_category':'Age Category','Age':'Count'})
df_VD['Vehicle_Damage']=df_VD['Vehicle_Damage'].astype('category')
df_VD

fig = px.bar(df_VD, x="Vehicle_Damage", y="Count",color="Age Category",barmode="group",
             facet_row="Gender"
             )
fig.update_layout(title_text='Age Category With Vehicle Damage And Gender',title_x=0.5)
fig.show()
df_Vehicle_Damage=df['Vehicle_Damage'].value_counts().to_frame().reset_index().rename(columns={'index':'Vehicle_Damage','Vehicle_Damage':'count'})


fig = go.Figure([go.Pie(labels=df_Vehicle_Damage['Vehicle_Damage'], values=df_Vehicle_Damage['count'])])

fig.update_traces(hoverinfo='label+percent', textinfo='value+percent', textfont_size=12,insidetextorientation='radial')

fig.update_layout(title=" Vehicle Damage",title_x=0.5)
fig.show()
df_Vintage=df.groupby(by =['Gender','age_category'])['Vintage'].mean().to_frame().reset_index().rename(columns={'Gender':'Gender','age_category':'Age Category','Vintage':'Days'})
df_Vintage

fig = px.bar(df_Vintage, x="Age Category", y="Days",
             color="Gender",barmode="group")
               
fig.update_layout(title_text='Average Vintage Days With Gender,Age Class',title_x=0.5)
fig.show()
df_PSC=df.Policy_Sales_Channel.value_counts().to_frame().reset_index()[0:10]

df_PSC['index']='PSC_Cod '+df_PSC['index'].astype('str')


fig = go.Figure(go.Bar(
    x=df_PSC['index'],y=df_PSC['Policy_Sales_Channel'],
    marker={'color': df_PSC['Policy_Sales_Channel'], 
    'colorscale': 'sunsetdark'},  
    text=df_PSC['Policy_Sales_Channel'],
    textposition = "outside",
))
fig.update_layout(title_text=' Top 10 Policy Sales Channel Code',xaxis_title="Value",yaxis_title="Number Of People",title_x=0.5)
fig.show()
df_PSC=df.Policy_Sales_Channel.value_counts().to_frame().reset_index()[0:10]
indexs=df_PSC['index']
df_PSC_age=df.groupby('Policy_Sales_Channel')['Age'].mean().to_frame().reset_index()

avg_age = []
age_cod=[]
for number in indexs:
    
    df_PSC_age_top1=df_PSC_age[df_PSC_age['Policy_Sales_Channel']==number]['Age']
    avg_age.extend(df_PSC_age_top1)
        
         
df_PSC_age = pd.DataFrame(avg_age)
df_PSC_Cod= pd.DataFrame(indexs)


frames = [df_PSC_age, df_PSC_Cod]

result = pd.concat(frames,axis=1)

result.columns = ['Age', 'Cod']

result['Cod']='PSC_Cod '+result['Cod'].astype('str')

fig = go.Figure(go.Bar(
    x=result['Cod'],y=result['Age'],
    marker={'color': result['Age'], 
    'colorscale': 'sunsetdark'},  
    text=result['Age'],
    textposition = "outside",
))
fig.update_layout(title_text=' Top 10 Policy Sales Channel Code Average Age',xaxis_title="Policy Sales Channel",yaxis_title="Age",title_x=0.5)
fig.show()
df_DL=df.groupby(by =['Gender','Driving_License'])['Age'].count().to_frame().reset_index().rename(columns={'Gender':'Gender','age_category':'Age Category','Age':'count'})
df_DL['Driving_License']=df_DL['Driving_License'].astype('category')
df_DL

fig = px.bar(df_DL, x="Driving_License", y="count",
             color="Gender",barmode="group")
               
fig.update_layout(title_text='Gender With Driving License',title_x=0.5)
fig.show()
df_Driving_License=df['Driving_License'].value_counts().to_frame().reset_index().rename(columns={'index':'Driving_License','Driving_License':'count'})


fig = go.Figure([go.Pie(labels=df_Driving_License['Driving_License'], values=df_Driving_License['count'],hole=0.2)])

fig.update_traces(hoverinfo='label+percent', textinfo='value+percent', textfont_size=12,insidetextorientation='radial')

fig.update_layout(title=" Customer Driving License",title_x=0.5)
fig.show()
df_RGC=df.Region_Code.value_counts().to_frame().reset_index()[0:10]

df_RGC['index']='R_Cod '+df_RGC['index'].astype('str')


fig = go.Figure(go.Bar(
    x=df_RGC['index'],y=df_RGC['Region_Code'],
    marker={'color': df_RGC['Region_Code'], 
    'colorscale': 'portland'},  
    text=df_RGC['Region_Code'],
    textposition = "outside",
))
fig.update_layout(title_text='Top 10 Region Code',xaxis_title="Region Code",yaxis_title="Number Of People",title_x=0.5)
fig.show()


df_V_Age=df['Vehicle_Age'].value_counts().to_frame().reset_index().rename(columns={'index':'Vehicle_Age','Vehicle_Age':'Count'})


fig = go.Figure(data=[go.Scatter(
    x=df_V_Age['Vehicle_Age'], y=df_V_Age['Count'],
    mode='markers',
    marker=dict(
        color=df_V_Age['Count'],
        size=df_V_Age['Count']*0.0005,
        showscale=True
    ))])

fig.update_layout(title='Vehicle Age ',xaxis_title=" Vehicle Age ",yaxis_title="Number Of Vehicle",title_x=0.5)
fig.show()
df_VAge_AP_mean=df.groupby(by =['Vehicle_Age'])['Annual_Premium'].mean().to_frame().reset_index().rename(columns={'Vehicle_Age':'Vehicle_Age','Annual_Premium':'Annual_Premium'})
df_VAge_AP_mean


fig = go.Figure(go.Bar(
    x=df_VAge_AP_mean['Vehicle_Age'],y=df_VAge_AP_mean['Annual_Premium'],
    marker={'color': df_VAge_AP_mean['Annual_Premium'], 
    'colorscale': 'portland'},  
    text=df_VAge_AP_mean['Annual_Premium'],
    textposition = "outside",
))
fig.update_layout(title_text='Vehicle Age With Annual Premium',xaxis_title="Vehicle Age",yaxis_title="Premium Price",title_x=0.5)
fig.show()
df_Response=df['Response'].value_counts().to_frame().reset_index().rename(columns={'index':'Response','Response':'count'})


fig = go.Figure([go.Pie(labels=df_Response['Response'], values=df_Response['count'],hole=0.2)])

fig.update_traces(hoverinfo='label+percent', textinfo='value+percent', textfont_size=12,insidetextorientation='radial')

fig.update_layout(title=" Response",title_x=0.5)
fig.show()
df_gender_Previously_Insured=df.groupby(by =['Gender','Previously_Insured'])['Age'].count().to_frame().reset_index().rename(columns={'Gender':'Gender','Previously_Insured':'Previously_Insured','Age':'Count'})
df_gender_Previously_Insured['Previously_Insured']=df_gender_Previously_Insured['Previously_Insured'].astype('category')

fig = px.bar(df_gender_Previously_Insured, x="Gender", y="Count",color="Previously_Insured",barmode="group",
             
             )
fig.update_layout(title_text='Gender With Previously Insured',title_x=0.5)
fig.show()
df_gender_response=df.groupby(by =['Gender','Response'])['Age'].count().to_frame().reset_index().rename(columns={'Gender':'Gender','Response':'Response','Age':'Count'})
df_gender_response['Response']=df_gender_response['Response'].astype('category')

fig = px.bar(df_gender_response, x="Gender", y="Count",color="Response",barmode="group",
             
             )
fig.update_layout(title_text='Gender With Response',title_x=0.5)
fig.show()

df_gender_Vehicle_Age=df.groupby(by =['Response','Vehicle_Age'])['Age'].count().to_frame().reset_index().rename(columns={'Response':'Response','Vehicle_Age':'Vehicle_Age','Age':'Count'})
df_gender_Vehicle_Age['Response']=df_gender_Vehicle_Age['Response'].astype('category')

fig = px.bar(df_gender_Vehicle_Age, x="Response", y="Count",color="Vehicle_Age",barmode="group",
             
             )
fig.update_layout(title_text='Response With Vehicle Age',title_x=0.5)
fig.show()
print("Correlation Matrix")
plt.rcParams['figure.figsize']=(8,6)
sns.heatmap(df.corr(),cmap='coolwarm',linewidths=.5,fmt=".2f",annot = True);
test=pd.read_csv('../input/health-insurance-cross-sell-prediction/test.csv')
df_model=train.copy()
df_model=df_model.drop(['id'], axis=1)
df_model.head(5)
gender_map= {'Male':0,'Female':1}
vehicle_age_map= {'< 1 Year':0,'1-2 Year':1,'> 2 Years':2}
vehicle_damage_map= {'Yes':1,'No':0}

df_model['Gender']= df_model['Gender'].map(gender_map)
df_model['Vehicle_Age']= df_model['Vehicle_Age'].map(vehicle_age_map)
df_model['Vehicle_Damage']= df_model['Vehicle_Damage'].map(vehicle_damage_map)
df_model.head(5)
df_model['Region_Code']=df_model['Region_Code'].astype(int)
df_model['Policy_Sales_Channel']=df_model['Policy_Sales_Channel'].astype(int)

df_model.info()
from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors = 20, contamination = 0.1)
df_out=df_model.copy()
clf.fit_predict(df_out)
df_scores = clf.negative_outlier_factor_
df_scores[0:10]
np.sort(df_scores)[0:20]
threshold_value = np.sort(df_scores)[1]
threshold_value
Outlier_df= df_out[df_scores < threshold_value]
indexs=Outlier_df.index
Outlier_df
# Kick Outliers
for i in indexs:
    df_model.drop(i, axis = 0,inplace = True)
y=df_model['Response']
X=df_model.drop('Response',axis=1)

print('X shape :',X.shape)
print('y shape :',y.shape)
# Normalize
X = (X - np.min(X)) / (np.max(X) - np.min(X)).values
X.head(2)
# Data split
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.2,
                                               random_state=42)
print('X_train :',X_train.shape)
print('X_test :',X_test.shape)
print('y_train :',y_train.shape)
print('y_test :',y_test.shape)
from sklearn.linear_model import LogisticRegression
loj = LogisticRegression(solver = "liblinear")
loj_model = loj.fit(X_train,y_train)
loj_model
y_pred_loj = loj_model.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
# Cofusion Matrix
cm = confusion_matrix(y_test, y_pred_loj)
plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True, annot_kws = {'size':11}, cmap = 'PuBu',fmt=".1f");
print("Training Accuracy :", loj_model.score(X_train, y_train))

print("Testing Accuracy :", loj_model.score(X_test, y_test))
print(classification_report(y_test, y_pred_loj))
cross_val_score(loj_model, X_test, y_test, cv = 10).mean()
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb_model = nb.fit(X_train, y_train)
nb_model
y_pred_nb = nb_model.predict(X_test)
accuracy_score(y_test, y_pred_nb)
# Cofusion Matrix
cm = confusion_matrix(y_test, y_pred_nb)
plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu',fmt=".1f")
print(classification_report(y_test, y_pred_nb))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn_model = knn.fit(X_train, y_train)
knn_model
y_pred_knn = knn_model.predict(X_test)
accuracy_score(y_test, y_pred_knn)
# Cofusion Matrix
cm = confusion_matrix(y_test, y_pred_knn)
plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu',fmt=".1f")
print(classification_report(y_test, y_pred_knn))
from sklearn.neural_network import MLPClassifier
mlpc = MLPClassifier().fit(X_train, y_train)
y_pred_mlpc = mlpc.predict(X_test)
accuracy_score(y_test,y_pred_mlpc)
# Cofusion Matrix
cm = confusion_matrix(y_test, y_pred_mlpc)
plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu',fmt=".1f")
print(classification_report(y_test, y_pred_mlpc))
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier().fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_score(y_test, y_pred_rf)
# Cofusion Matrix
cm = confusion_matrix(y_test, y_pred_rf)
plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu',fmt=".1f")
print(classification_report(y_test, y_pred_rf))
Importance = pd.DataFrame({"Importance": rf_model.feature_importances_*100},
                         index = X_train.columns)
Importance.sort_values(by = "Importance", 
                       axis = 0, 
                       ascending = True).plot(kind ="barh", color = "r")

plt.xlabel("Variable Significance Levels")
from sklearn.ensemble import GradientBoostingClassifier
gbm_model = GradientBoostingClassifier().fit(X_train, y_train)
y_pred_gbm = gbm_model.predict(X_test)
accuracy_score(y_test, y_pred_gbm)
print(classification_report(y_test, y_pred_gbm))
# Cofusion Matrix
cm = confusion_matrix(y_test, y_pred_gbm)
plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu',fmt=".1f")
from xgboost import XGBClassifier
import xgboost as xgb
xgb_model = XGBClassifier().fit(X_train, y_train)
y_pred_xgb_model = xgb_model.predict(X_test)
accuracy_score(y_test, y_pred_xgb_model)
print(classification_report(y_test, y_pred_xgb_model))
# Cofusion Matrix
cm = confusion_matrix(y_test, y_pred_xgb_model)
plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu',fmt=".1f")
xgb.plot_importance(xgb_model)
from sklearn.metrics import roc_curve, auc

xgb_model=XGBClassifier().fit(X_train, y_train)

y_score = xgb_model.predict_proba(X_test)[:, 1]


fpr, tpr, thresholds = roc_curve(y_test, y_score)

fig = px.area(
    x=fpr, y=tpr,
    title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=700, height=500
)
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')
fig.show()
# Model Tuning
XGB_model = XGBClassifier(random_state = 42, max_depth = 8, n_estimators = 3000, 
                          reg_lambda = 1.2, reg_alpha = 1.2, 
                          min_child_weight = 1,objective = 'binary:logistic',
                         learning_rate = 0.15, gamma = 0.3, colsample_bytree = 0.5,
                          eval_metric = 'auc').fit(X_train, y_train)
y_pred_XGB_model = XGB_model.predict(X_test)
accuracy_score(y_test, y_pred_XGB_model)
print(classification_report(y_test, y_pred_XGB_model))
# Cofusion Matrix
cm = confusion_matrix(y_test, y_pred_XGB_model)
plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu',fmt=".1f")
from lightgbm import LGBMClassifier
LGB_model = LGBMClassifier(random_state=42, max_depth= 8,n_estimators=3000,
                    reg_lambda=1.2, reg_alpha=1.2, min_child_weight=1,verbose= 1,
                    learning_rate = 0.15, gamma = 0.3, colsample_bytree = 0.5,
                    eval_metric = 'auc', is_higher_better = 1, plot = True)
LGB_model.fit(X_train, y_train)
y_pred_lgbm_model = LGB_model.predict(X_test)
accuracy_score(y_test, y_pred_lgbm_model)
# Cofusion Matrix
cm = confusion_matrix(y_test, y_pred_lgbm_model)
plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu',fmt=".1f")
print(classification_report(y_test, y_pred_lgbm_model))
models = [
    knn_model,
    loj_model,
    nb_model,
    mlpc,
    gbm_model,
    LGB_model,
    xgb_model,
    XGB_model,
    rf_model,
      
]

for model in models:
    names = model.__class__.__name__
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("-"*28)
    print(names + ":" )
    print("Accuracy: {:.4%}".format(accuracy))
result = []

results = pd.DataFrame(columns= ["Models","Accuracy"])

for model in models:
    names = model.__class__.__name__
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)    
    result = pd.DataFrame([[names, accuracy*100]], columns= ["Models","Accuracy"])
    results = results.append(result)
    
    
sns.barplot(x= 'Accuracy', y = 'Models', data=results, color="r")
plt.xlabel('Accuracy %')
plt.title('Accuracy Ratios of Models'); 