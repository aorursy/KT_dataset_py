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


import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score,roc_curve
import statsmodels.formula.api as smf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from warnings import filterwarnings
filterwarnings('ignore')
heart=pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
df=heart.copy()
df.head()
df.info()
df.dtypes
df.shape
df.columns
df.describe().T
df.isnull().values.any()
df.isnull().sum()
df["age"].describe()
print("Mean Age: " + str(df["age"].mean()))
print("Max Age Value: " + str(df["age"].max()))
print("Min Age Value: " + str(df["age"].min()))
print("Median Age: " + str(df["age"].median()))

plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True,fmt='.1f')
plt.show()
# Bar Chart - Gradient & Text Position

df_age=df['age'].value_counts().reset_index().rename(columns={'index':'age','age':'Count'})

fig = go.Figure(go.Bar(
    x=df_age['age'],y=df_age['Count'],
    marker={'color': df_age['Count'], 
    'colorscale': 'Viridis'},  
    text=df_age['Count'],
    textposition = "outside",
))
fig.update_layout(title_text='Age Distribution',xaxis_title="Age",yaxis_title="Age Count ",title_x=0.5)
fig.show()

df_agevi=df['age']

fig = go.Figure(data=go.Violin(y=df_agevi, box_visible=True, line_color='black',
                               meanline_visible=True, fillcolor='lightseagreen', opacity=0.6,
                               x0='Age'))

fig.update_layout(yaxis_zeroline=False,title="Distribution of Age",title_x=0.5)
fig.show()
# Age Distribution 2

# Grouped Box Plot

age_29_40=df[(df.age>=29)&(df.age<40)]
age_41_50=df[(df.age>=40)&(df.age<50)]
age_50=df[(df.age>50)]

df_age_29_40=age_29_40['age']
df_age_41_50=age_41_50['age']
df_age_50=age_50['age']
df_age=df['age']

fig = go.Figure()
fig.add_trace(go.Box(y=df_age_29_40,
                     marker_color="cyan",
                     name="Age 29 - 40"))
fig.add_trace(go.Box(y=df_age_41_50,
                     marker_color="darkcyan",
                     name="Age 41- 50 "))
fig.add_trace(go.Box(y=df_age_50,
                     marker_color="royalblue",
                     name="Age 50+ " ))
fig.add_trace(go.Box(y=df_age,
                     marker_color="darkblue",
                     name="Age" ))

fig.update_layout(title="Distribution of Age With And Category ",title_x=0.5)
fig.show()
f, ax = plt.subplots(figsize=(10,6))
x = df['age']
ax = sns.distplot(x, bins=10)
plt.xlabel('Age')
plt.show()
# Pie with custom colors

df['age_category']=np.where((df['age']>28)&(df['age']<=40),'29-40',np.where(df['age']>50,'50+',
np.where((df['age']>40)&(df['age']<=50),'41-50',"Not Specified")))

df_age_category=df['age_category'].value_counts().to_frame().reset_index().rename(columns={'index':'age_category','age_category':'Count'})


colors=['lightcyan','cyan',"darkcyan"]

fig = go.Figure([go.Pie(labels=df_age_category['age_category'], values=df_age_category['Count'])])
fig.update_traces(hoverinfo='label+percent', textinfo='percent+value', textfont_size=15,
                 marker=dict(colors=colors, line=dict(color='#000000', width=2)))
fig.update_layout(title="Age Category",title_x=0.5)
fig.show()

# Bubble Plot with Color gradient

df['age_category']=np.where((df['age']>28)&(df['age']<=40),'29-40',np.where(df['age']>50,'50+',
np.where((df['age']>40)&(df['age']<=50),'41-50',"Not Specified")))

df_age_category=df['age_category'].value_counts().to_frame().reset_index().rename(columns={'index':'age_category','age_category':'Count'})

fig = go.Figure(data=[go.Scatter(
    x=df_age_category['age_category'], y=df_age_category['Count'],
    mode='markers',
    marker=dict(
        color=df_age_category['Count'],
        size=df_age_category['Count']*0.7,
        showscale=True
    ))])

fig.update_layout(title='Age Frequency ',xaxis_title="Age Category",yaxis_title="Age Count",title_x=0.5)
fig.show()
(sns
 .FacetGrid(df,
              hue = "target",
              height = 5,
              xlim = (0, 1000))
 .map(sns.kdeplot, "chol", shade= True)
 .add_legend()
);
# Bar Chart - Gradient & Text Position

mean_thalach=df.groupby('age_category')['thalach'].mean().to_frame().reset_index().rename(columns={'index':'age_category','thalach':'Mean'})

fig = go.Figure(go.Bar(
    x=mean_thalach['age_category'],y=mean_thalach['Mean'],
    marker={'color': mean_thalach['Mean'], 
    'colorscale': 'Viridis'},  
     text=mean_thalach['Mean'],
    textposition = "outside",
))
fig.update_layout(title_text='Average Max Heart Measurement With Age Category',xaxis_title="Age Category",yaxis_title=" Max Heart Measurement",title_x=0.5)
fig.show()
# Bar Chart - Gradient & Text Position

mean_chol=df.groupby('age_category')['chol'].mean().to_frame().reset_index().rename(columns={'index':'age_category','chol':'Mean'})

fig = go.Figure(go.Bar(
    x=mean_chol['age_category'],y=mean_chol['Mean'],
    marker={'color': mean_chol['Mean'], 
    'colorscale': 'Viridis'},  
    text=mean_chol['Mean'],
    textposition = "outside",
))
fig.update_layout(title_text='Average Cholesterol Measurement With Age Category',xaxis_title="Age Category",yaxis_title="Cholesterol Measurement",title_x=0.5)
fig.show()
# Bar Chart - Gradient & Text Position

mean_trestbps=df.groupby('age_category')['trestbps'].mean().to_frame().reset_index().rename(columns={'index':'age_category','trestbps':'Mean'})

fig = go.Figure(go.Bar(
    x=mean_trestbps['age_category'],y=mean_trestbps['Mean'],
    marker={'color': mean_trestbps['Mean'], 
    'colorscale': 'Viridis'},  
     text=mean_trestbps['Mean'],
    textposition = "outside",
))
fig.update_layout(title_text='Average Resting Blood Pressure Measurement With Age Category',xaxis_title="Age Category",yaxis_title="Resting Blood Pressure Measurement",title_x=0.5)
fig.show()
# Bar Chart

age_exang_values=df.groupby(by =['age_category','exang'])['age'].count().to_frame().reset_index().rename(columns={'index':'age_category','exang':'Exang','age':'Count'})
age_exang_values['Exang']=age_exang_values['Exang'].astype('category')
age_exang_values

fig = px.bar(age_exang_values, x="age_category", y="Count",
             color="Exang",barmode="group")
             
fig.update_layout(title_text='Exang With Age Category',xaxis_title="Age Category",title_x=0.5)
fig.show()
age_target_values=df.groupby(by =['age_category','target'])['age'].count().to_frame().reset_index().rename(columns={'index':'age_category','target':'Target','age':'Count'})
age_target_values['Target']=age_target_values['Target'].astype('category')


fig = px.bar(age_target_values, x="age_category", y="Count",
             color="Target",barmode="group")
               
fig.update_layout(title_text='Target With Age Category',title_x=0.5)
fig.show()
# Scatter plot - Category

fig = px.scatter(df, x='age', y='thalach',
                 color='exang') # Added color to basic scatter
fig.update_layout(title='Age Vs Maximum Heart Measurement With Exang ',xaxis_title="Age",yaxis_title="Maximum Heart Measurement",title_x=0.5)
fig.show()
# Scatter plot - Category

fig = px.scatter(df, x='age', y='thalach',
                 color='target',
                 color_continuous_scale='Viridis') 
fig.update_layout(title='Age Vs Maximum Heart Measurement With Target ',xaxis_title="Age",yaxis_title="Maximum Heart Measurement",title_x=0.5)
fig.show()
# Scatter plot - Category

fig = px.scatter(df, x='age', y='chol',
                 color='exang',
                 color_continuous_scale='fall'
                ) 
fig.update_layout(title='Age Vs Cholestoral Measurement  With Exang ',xaxis_title="Age",yaxis_title="Cholestoral",title_x=0.5)
fig.show()
# Scatter plot - Category

fig = px.scatter(df, x='age', y='chol',
                 color='target',
                 color_continuous_scale='earth'
                ) 
fig.update_layout(title='Age Vs Cholestoral Measurement With Target',xaxis_title="Age",yaxis_title="Cholestoral Measurement",title_x=0.5)
fig.show()
# Scatter plot - Category

fig = px.scatter(df, x='age', y='trestbps',
                 color='target',
                 color_continuous_scale='tropic'
                ) 
fig.update_layout(title='Age Vs Resting Blood Pressure Measurement With Target',
                  xaxis_title="Age",
                  yaxis_title="Resting Blood Pressure Measurement",
                  title_x=0.5)
fig.show()
# Scatter plot - Category

fig = px.scatter(df, x='age', y='trestbps',
                 color='exang',
                 color_continuous_scale='rdylbu'
                ) 
fig.update_layout(title='Age Vs Resting Blood Pressure Measurement With Target',
                  xaxis_title="Age",
                  yaxis_title="Resting Blood Pressure Measurement",
                  title_x=0.5)
fig.show()
# Bar Chart

df_cp=df.groupby(by =['age_category','cp'])['age'].count().to_frame().reset_index().rename(columns={'age_category':'Age Category','cp':'Cp Class','age':'Count'})
df_cp['Cp Class']=df_cp['Cp Class'].astype('category')
df_cp

fig = px.bar(df_cp, x="Age Category", y="Count",color="Cp Class",barmode="group",
             
             )
fig.update_layout(title_text='Chest Pain Type With Age Category',title_x=0.5)
fig.show()
# Bar Chart

df_age_sex=df.groupby(by =['age_category','sex'])['thalach'].count().to_frame().reset_index().rename(columns={'age_category':'Age Category','sex':'Sex','thalach':'Count'})
df_age_sex['Sex']=df_age_sex['Sex'].astype('category')

fig = px.bar(df_age_sex, x="Age Category", y="Count",
             color="Sex",barmode="group")
               
fig.update_layout(title_text='Sex With Age Class',title_x=0.5)
fig.show()
# Bar Chart

# Exang Counts

df_exang=df['exang'].value_counts().to_frame().reset_index().rename(columns={'index':'exang','exang':'Count'})

fig = go.Figure(go.Bar(
    x=df_exang['exang'],y=df_exang['Count'],
    marker={'color': df_exang['Count'], 
    'colorscale': 'Viridis'},  
    text=df_exang['Count'],
    textposition = "outside",
))
fig.update_layout(title_text='Exercise Induced Angina',xaxis_title="Exang Class",yaxis_title="Count",title_x=0.5)
fig.show()
# Bar Chart

df_sex_exang=df.groupby(by =['sex','exang'])['age'].count().to_frame().reset_index().rename(columns={'sex':'Sex','exang':'exang','age':'Count'})
df_sex_exang['exang']=df_sex_exang['exang'].astype('category')

fig = px.bar(df_sex_exang, x="Sex", y="Count",color="exang",barmode="group",
             
             )
fig.update_layout(title_text='Sex With Exang',title_x=0.5)
fig.show()
# Bar Chart

df_target_exang=df.groupby(by =['target','exang'])['age'].count().to_frame().reset_index().rename(columns={'target':'target','exang':'exang','age':'Count'})
df_target_exang['exang']=df_target_exang['exang'].astype('category')
df_target_exang['target']=df_target_exang['target'].astype('category')
fig = px.bar(df_target_exang, x="target", y="Count",color="exang",barmode="group",
             
             )
fig.update_layout(title_text='Target With Exang',title_x=0.5)
fig.show()
df_cp=df.groupby(by =['target','exang','sex'])['age'].count().to_frame().reset_index().rename(columns={'target':'Target','sex':'Sex','exang':'Exang','age':'Count'})
df_cp['Exang']=df_cp['Exang'].astype('category')
df_cp['Sex']=df_cp['Sex'].astype('category')

# Bar Chart

fig = px.bar(df_cp, x="Target", y="Count",color="Exang",barmode="group",
             facet_row="Sex"
             )
fig.update_layout(title_text='Exang With Target And Sex',title_x=0.5)
fig.show()
(sns
 .FacetGrid(df,
              hue = "target",
              height = 5,
              xlim = (0,700))
 .map(sns.kdeplot, "chol", shade= True)
 .add_legend()
);
(sns
 .FacetGrid(df,
              hue = "target",
              height = 5,
              xlim = (0, 400))
 .map(sns.kdeplot, "thalach", shade= True)
 .add_legend()
);
(sns
 .FacetGrid(df,
              hue = "target",
              height = 5,
              xlim = (0, 300))
 .map(sns.kdeplot, "trestbps", shade= True)
 .add_legend()
);
# Bar Chart
df_target=df['target'].value_counts().to_frame().reset_index().rename(columns={'index':'target','target':'Count'})

fig = go.Figure(go.Bar(
    x=df_target['target'],y=df_target['Count'],
    marker={'color': df_target['Count'], 
    'colorscale': 'Viridis'},  
    text=df_target['Count'],
    textposition = "outside",
))
fig.update_layout(title_text='Target',xaxis_title="Target Class",yaxis_title="Count",title_x=0.5)
fig.show()
# Bar Chart

df_sex_target=df.groupby(by =['sex','target'])['age'].count().to_frame().reset_index().rename(columns={'sex':'Sex','target':'Target','age':'Count'})
df_sex_target['Target']=df_sex_target['Target'].astype('category')

fig = px.bar(df_sex_target, x="Sex", y="Count",color="Target",barmode="group",
             
             )
fig.update_layout(title_text='Sex With Target',title_x=0.5)
fig.show()
# Bar Chart

df_cp=df['cp'].value_counts().reset_index().rename(columns={'index':'Cp Class','cp':'Count'})

fig = px.bar(df_cp, x="Cp Class", y="Count")
                          
fig.update_layout(title_text='Chest Pain',title_x=0.5)
fig.show()
# Bar Chart

df_cp=df.groupby(by =['sex','cp'])['age'].count().to_frame().reset_index().rename(columns={'sex':'Sex','cp':'Cp Class','age':'Count'})
df_cp['Cp Class']=df_cp['Cp Class'].astype('category')
df_cp

fig = px.bar(df_cp, x="Sex", y="Count",color="Cp Class",barmode="group",
             
             )
fig.update_layout(title_text='Chest Pain Type With Sex',title_x=0.5)
fig.show()
# Bar Chart

df_target=df.groupby(by =['target','cp'])['age'].count().to_frame().reset_index().rename(columns={'target':'Target','cp':'Cp Class','age':'Count'})
df_target['Cp Class']=df_target['Cp Class'].astype('category')


fig = px.bar(df_target, x="Target", y="Count",color="Cp Class",barmode="group",
             
             )
fig.update_layout(title_text='Chest Pain Type With Target',title_x=0.5)
fig.show()
# Bar Chart

df_cp=df.groupby(by =['exang','cp'])['age'].count().to_frame().reset_index().rename(columns={'exang':'Exang','cp':'Cp Class','age':'Count'})
df_cp['Cp Class']=df_cp['Cp Class'].astype('category')
df_cp['Exang']=df_cp['Exang'].astype('category')


fig = px.bar(df_cp, x="Exang", y="Count",color="Cp Class",barmode="group",
             
             )
fig.update_layout(title_text='Chest Pain Type With Exang',title_x=0.5)
fig.show()
df_cp=df.groupby(by =['target','cp','sex'])['age'].count().to_frame().reset_index().rename(columns={'target':'Target','sex':'Sex','cp':'Cp Class','age':'Count'})
df_cp['Cp Class']=df_cp['Cp Class'].astype('category')
df_cp['Sex']=df_cp['Sex'].astype('category')

# Bar Chart

fig = px.bar(df_cp, x="Target", y="Count",color="Cp Class",barmode="group",
             facet_row="Sex"
             )
fig.update_layout(title_text='Chest Pain Type With Target And Sex',title_x=0.5)
fig.show()
df_cp=df.groupby(by =['target','cp','sex','exang'])['age'].count().to_frame().reset_index().rename(columns={'target':'Target','sex':'Sex','exang':'Exang','cp':'Cp Class','age':'Count'})
df_cp['Cp Class']=df_cp['Cp Class'].astype('category')
df_cp['Sex']=df_cp['Sex'].astype('category')
df_cp['Exang']=df_cp['Exang'].astype('category')
# Bar Chart

fig = px.bar(df_cp, x="Target", y="Count",color="Cp Class",barmode="group",
             facet_row="Sex",facet_col="Exang"
             )
fig.update_layout(title_text='Chest Pain Type With Target,Sex And Exang',title_x=0.5)
fig.show()
heart=pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
df=heart.copy()
df.head()
df.info()
df.dtypes
df.columns
df.describe().T
#Conversion to categorical variables

df['cp']=df['cp'].astype('category')
df['slope']=df['slope'].astype('category')
df['restecg']=df['restecg'].astype('category')
df['thal']=df['thal'].astype('category')
df.dtypes
df.info()
df_num=df.select_dtypes(include = ['float64', 'int64']) 
df_num
from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors = 20, contamination = 0.1)
clf.fit_predict(df_num)
df_scores = clf.negative_outlier_factor_
df_scores[0:15]
np.sort(df_scores)[0:50]

threshold_value= np.sort(df_scores)[5]
threshold_value
Outlier_tf = df_scores > threshold_value
Outlier_tf
Outlier_df= df_num[df_scores < threshold_value]
indexs=Outlier_df.index
Outlier_df
# Kick Outliers
#for i in indexs:
#    df.drop(i, axis = 0,inplace = True)
df.head()
df.info()
df.shape
df.describe().T
df=pd.get_dummies(df,drop_first=True)
df.head()
df.info()
y=df['target']
X=df.drop('target',axis=1)
X.head()
X = (X - np.min(X)) / (np.max(X) - np.min(X)).values
X.head()
X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.2,
                                               random_state=42)
print('X_train',X_train.shape)
print('X_test',X_test.shape)
print('y_train',y_train.shape)
print('y_test',y_test.shape)
from sklearn.linear_model import LogisticRegression
loj = LogisticRegression(solver = "liblinear")
loj_model = loj.fit(X,y)
loj_model
loj_model.intercept_
loj_model.coef_
y_pred_loj = loj_model.predict(X_test)
confusion_matrix(y_test , y_pred_loj)
# Cofusion Matrix
cm = confusion_matrix(y_test, y_pred_loj)
plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu')
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
sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu')
cross_val_score(nb_model, X_test, y_test, cv = 10).mean()
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn_model = knn.fit(X_train, y_train)
knn_model
y_pred_knn = knn_model.predict(X_test)
accuracy_score(y_test, y_pred_knn)
print(classification_report(y_test, y_pred_knn))
knn_params = {"n_neighbors": np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, knn_params, cv=10)
knn_cv.fit(X_train, y_train)
print("Best Score_:" + str(knn_cv.best_score_))
print("Best Params: " + str(knn_cv.best_params_))
scoreList = []
for i in range(1,50):
    knn2 = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k
    knn2.fit(X_train, y_train)
    scoreList.append(knn2.score(X_test, y_test))
    
plt.plot(range(1,50), scoreList)
plt.xticks(np.arange(1,50,1))
plt.xlabel("K value")
plt.ylabel("Score")
plt.show()

acc = max(scoreList)*100
print("Maximum KNN Score is {:.2f}%".format(acc))
knn = KNeighborsClassifier(41)
knn_tuned = knn.fit(X_train, y_train)
y_pred_knn_tuned = knn_tuned.predict(X_test)
accuracy_score(y_test, y_pred_knn_tuned)
print(classification_report(y_test, y_pred_knn_tuned))
from sklearn.svm import SVC
svm_model = SVC(kernel = "linear").fit(X_train, y_train)
y_pred_svc = svm_model.predict(X_test)
accuracy_score(y_test, y_pred_svc)
svc_params = {"C": np.arange(1,10)}

svc = SVC(kernel = "linear")

svc_cv_model = GridSearchCV(svc,svc_params, 
                            cv = 10, 
                            n_jobs = -1, 
                            verbose = 2 )

svc_cv_model.fit(X_train, y_train)
print("Best Params: " + str(svc_cv_model.best_params_))
svc_tuned_linear = SVC(kernel = "linear", C = 7).fit(X_train, y_train)
y_pred_svc_tuned = svc_tuned_linear.predict(X_test)
accuracy_score(y_test, y_pred_svc_tuned)
# Cofusion Matrix
cm = confusion_matrix(y_test, y_pred_svc_tuned)
plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu')
svc_model = SVC(kernel = "rbf").fit(X_train, y_train)
y_pred_svc_model_rbf = svc_model.predict(X_test)
accuracy_score(y_test, y_pred_svc_model_rbf)
svc_params = {"C": [0.0001, 0.001, 0.1, 1, 5, 10 ,50 ,100],
             "gamma": [0.0001, 0.001, 0.1, 1, 5, 10 ,50 ,100]}
svc = SVC(kernel = "rbf")
svc_cv_model = GridSearchCV(svc, svc_params, 
                         cv = 10, 
                         n_jobs = -1,
                         verbose = 2)

svc_cv_model.fit(X_train, y_train)
print("Best Params: " + str(svc_cv_model.best_params_))
svc_tuned_rbf = SVC(kernel = "rbf",C = 50, gamma = 0.001).fit(X_train, y_train)
y_pred_svc_tuned_rbf = svc_tuned_rbf.predict(X_test)
accuracy_score(y_test, y_pred_svc_tuned_rbf)
# Cofusion Matrix
cm = confusion_matrix(y_test, y_pred_svc_tuned_rbf)
plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu')
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()
from sklearn.neural_network import MLPClassifier
mlpc = MLPClassifier().fit(X_train, y_train)
y_pred_mlpc = mlpc.predict(X_test)
accuracy_score(y_test,y_pred_mlpc)
mlpc_params = {"alpha": [0.1, 0.01, 0.02, 0.005, 0.0001],
              "hidden_layer_sizes": [(10,10,10),
                                     (100,100,100),
                                     (100,100)],
              "solver" : ["lbfgs","adam","sgd"],
              "activation": ["relu","logistic"]}
mlpc = MLPClassifier()
mlpc_cv_model = GridSearchCV(mlpc, mlpc_params, 
                         cv = 10, 
                         n_jobs = -1,
                         verbose = 2)

mlpc_cv_model.fit(X_train, y_train)
print("Best Params: " + str(mlpc_cv_model.best_params_))
mlpc_tuned = MLPClassifier(activation = "relu", 
                           alpha = 0.1, 
                           hidden_layer_sizes = (10, 10,10),
                          solver = "adam")
mlpc_tuned.fit(X_train, y_train)
y_pred_mlpc_tuned = mlpc_tuned.predict(X_test)
accuracy_score(y_test, y_pred_mlpc_tuned)
# Cofusion Matrix
cm = confusion_matrix(y_test, y_pred_mlpc_tuned)
plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu')
from sklearn.tree import DecisionTreeClassifier

cart = DecisionTreeClassifier()
cart_model = cart.fit(X_train, y_train)
y_pred_cart = cart_model.predict(X_test)
accuracy_score(y_test, y_pred_cart)
cart_grid = {"max_depth": range(1,20),
            "min_samples_split" : list(range(2,20)) }
cart = tree.DecisionTreeClassifier()
cart_cv = GridSearchCV(cart, cart_grid, cv = 10, n_jobs = -1, verbose = 2)
cart_cv_model = cart_cv.fit(X_train, y_train)
print("Best Params: " + str(cart_cv_model.best_params_))
cart = tree.DecisionTreeClassifier(max_depth =6, min_samples_split = 14)
cart_tuned = cart.fit(X_train, y_train)
y_pred_cart_tuned = cart_tuned.predict(X_test)
accuracy_score(y_test, y_pred_cart_tuned)
# Cofusion Matrix
cm = confusion_matrix(y_test, y_pred_cart_tuned)
plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu')
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier().fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_score(y_test, y_pred_rf)
rf_params = {"max_depth": [2,5,8,10],
            "max_features": [2,5,8],
            "n_estimators": [10,50,100,250],
            "min_samples_split": [2,5,10]}
rf_model = RandomForestClassifier()

rf_cv_model = GridSearchCV(rf_model, 
                           rf_params, 
                           cv = 10, 
                           n_jobs = -1, 
                           verbose = 2)
rf_cv_model.fit(X_train, y_train)
print("Best Params: " + str(rf_cv_model.best_params_))
rf_tuned = RandomForestClassifier(max_depth = 8, 
                                  max_features = 8, 
                                  min_samples_split = 10,
                                  n_estimators = 10)

rf_tuned.fit(X_train, y_train)
y_pred_rf_tuned = rf_tuned.predict(X_test)
accuracy_score(y_test, y_pred_rf_tuned)
# Cofusion Matrix
cm = confusion_matrix(y_test, y_pred_rf_tuned)
plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu')
Importance = pd.DataFrame({"Importance": rf_tuned.feature_importances_*100},
                         index = X_train.columns)
Importance.sort_values(by = "Importance", 
                       axis = 0, 
                       ascending = True).plot(kind ="barh", color = "r")

plt.xlabel("Variable Significance Levels")
from sklearn.ensemble import GradientBoostingClassifier
gbm_model = GradientBoostingClassifier().fit(X_train, y_train)
y_pred_gbm_model = gbm_model.predict(X_test)
accuracy_score(y_test, y_pred_gbm_model)
gbm_params = {"learning_rate" : [0.001, 0.01, 0.1, 0.05],
             "n_estimators": [50,250,100],
             "max_depth": [3,5,10],
             "min_samples_split": [2,5,10]}
gbm = GradientBoostingClassifier()

gbm_cv = GridSearchCV(gbm, gbm_params, cv = 10, n_jobs = -1, verbose = 2)
gbm_cv.fit(X_train, y_train)
print("Best Params: " + str(gbm_cv.best_params_))
gbm = GradientBoostingClassifier(learning_rate = 0.1, 
                                 max_depth = 3,
                                min_samples_split = 5,
                                n_estimators = 100)
gbm_tuned =  gbm.fit(X_train,y_train)
y_pred_gbm_tuned = gbm_tuned.predict(X_test)
accuracy_score(y_test, y_pred_gbm_tuned)
# Cofusion Matrix
cm = confusion_matrix(y_test, y_pred_gbm_tuned)
plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu')
#!pip install xgboost
from xgboost import XGBClassifier
xgb_model = XGBClassifier().fit(X_train, y_train)
y_pred_xgb_model = xgb_model.predict(X_test)
accuracy_score(y_test, y_pred_xgb_model)
xgb_params = {
        'n_estimators': [100, 50, 200],
        'subsample': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5,6,7,8],
        'learning_rate': [0.1,0.01,0.02,0.05]
        }
xgb = XGBClassifier()

xgb_cv_model = GridSearchCV(xgb, xgb_params, cv = 10, n_jobs = -1, verbose = 2)
xgb_cv_model.fit(X_train, y_train)
xgb_cv_model.best_params_
xgb = XGBClassifier(learning_rate = 0.02, 
                    max_depth =3,
                    n_estimators = 200,
                    subsample = 0.6)
xgb_tuned =  xgb.fit(X_train,y_train)
y_pred_xgb_tuned = xgb_tuned.predict(X_test)
accuracy_score(y_test, y_pred_xgb_tuned)
# Cofusion Matrix
cm = confusion_matrix(y_test, y_pred_xgb_tuned)
plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu')
#!conda install -c conda-forge lightgbm

from lightgbm import LGBMClassifier
lgbm_model = LGBMClassifier().fit(X_train, y_train)
y_pred_lgbm_model = lgbm_model.predict(X_test)
accuracy_score(y_test, y_pred_lgbm_model)
lgbm_params = {
        'n_estimators': [100, 50, 250,],
        'subsample': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5,6],
        'learning_rate': [0.1,0.01,0.02,0.05],
        "min_child_samples": [5,10,20]}
lgbm = LGBMClassifier()

lgbm_cv_model = GridSearchCV(lgbm, lgbm_params, 
                             cv = 10, 
                             n_jobs = -1, 
                             verbose = 2)

lgbm_cv_model.fit(X_train, y_train)
lgbm_cv_model.best_params_
lgbm = LGBMClassifier(learning_rate = 0.05, 
                       max_depth = 6,
                       n_estimators = 100,
                       min_child_samples = 10,
                       subsample = 0.6)
lgbm_tuned = lgbm.fit(X_train,y_train)
y_pred_lgbm_tuned = lgbm_tuned.predict(X_test)
accuracy_score(y_test, y_pred_lgbm_tuned)
# Cofusion Matrix
cm = confusion_matrix(y_test, y_pred_lgbm_tuned)
plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu')
models = [
    knn_tuned,
    loj_model,
    svc_tuned_linear,
    svc_tuned_rbf,
    nb_model,
    mlpc_tuned,
    cart_tuned,
    rf_tuned,
    gbm_tuned,
    lgbm_tuned,
    xgb_tuned
    
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
