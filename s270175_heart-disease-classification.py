import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import scipy 
from scipy import stats
import scipy.stats as ss

import warnings
warnings.filterwarnings("ignore")
from plotly.subplots import make_subplots
import plotly.graph_objects as go
np.random.seed(123)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import GridSearchCV
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.tree import plot_tree
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import shap

from IPython.display import HTML

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')
# !jupyter nbconvert --to html MathProject.ipynb
# !pip install shap
header = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 
              'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved','exercise_induced_angina', 
              'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']

df = pd.read_csv("../input/heart-disease-uci/heart.csv")
df.columns = header
df.target = df.target.replace({0:1, 1:0})
df.head(10)

missing_values =df.isna().sum().sum()
print(f'Missing values found: {missing_values}')
df['sex'][df['sex'] == 0] = 'female'
df['sex'][df['sex'] == 1] = 'male'

df['chest_pain_type'][df['chest_pain_type'] == 0] = 'typical angina'
df['chest_pain_type'][df['chest_pain_type'] == 1] = 'atypical angina'
df['chest_pain_type'][df['chest_pain_type'] == 2] = 'non-anginal pain'
df['chest_pain_type'][df['chest_pain_type'] == 3] = 'asymptomatic'

df['thalassemia'][df['thalassemia'] == 0] = 'non present'
df['thalassemia'][df['thalassemia'] == 1] = 'normal'
df['thalassemia'][df['thalassemia'] == 2] = 'fixed defect'
df['thalassemia'][df['thalassemia'] == 3] = 'reversable defect'

df['fasting_blood_sugar'][df['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'
df['fasting_blood_sugar'][df['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'

df['rest_ecg'][df['rest_ecg'] == 0] = 'normal'
df['rest_ecg'][df['rest_ecg'] == 1] = 'ST-T wave abnormality'
df['rest_ecg'][df['rest_ecg'] == 2] = 'left ventricular hypertrophy'

df['exercise_induced_angina'][df['exercise_induced_angina'] == 0] = 'no'
df['exercise_induced_angina'][df['exercise_induced_angina'] == 1] = 'yes'

df['st_slope'][df['st_slope'] == 0] = 'upsloping'
df['st_slope'][df['st_slope'] == 1] = 'flat'
df['st_slope'][df['st_slope'] == 2] = 'downsloping'

df['num_major_vessels'][df['num_major_vessels'] == 4] = 'no occlusion'
df['num_major_vessels'][df['num_major_vessels'] == 3] = 'slight occlusion'
df['num_major_vessels'][df['num_major_vessels'] == 2] = 'medium occlusion'
df['num_major_vessels'][df['num_major_vessels'] == 1] = 'high occlusion'
df['num_major_vessels'][df['num_major_vessels'] == 0] = 'severe occlusion'


categorical = ['sex','chest_pain_type','fasting_blood_sugar','rest_ecg','exercise_induced_angina','st_slope',
               'thalassemia','num_major_vessels']
numeric = ['age', 'resting_blood_pressure', 'cholesterol', 'max_heart_rate_achieved','target']

df[categorical] = df[categorical].astype('category')
df[numeric] = df[numeric].astype(int).astype('int64')
df.head(10)

df[['age', 'resting_blood_pressure', 'cholesterol', 'max_heart_rate_achieved','st_depression']].describe().T
palette = ['blue', 'red']
fig = px.histogram(df, x = 'target', nbins =5, color = 'target', 
                   title = 'Count healthy and unhealthy patients (healthy = 0)', height = 400, 
                   width = 500, color_discrete_sequence= palette)
fig.show()
countDisease, countNoDisease = df.groupby(['target']).size()
print(f'Percentage of people having heart disease: {(countDisease/(countDisease+countNoDisease))*100:.1f}%')
print(f'Percentage of people not having heart disease: {(countNoDisease/(countDisease+countNoDisease))*100:.1f}%')
df_cross = pd.crosstab(df.sex, df.target)
data = []
#use for loop on every name to create bar data
for i,x in enumerate(df_cross.columns):
    data.append(go.Bar(name=str(x), x=df_cross.index, y=df_cross[x], marker_color= palette[i]))

figure = go.Figure(data)
figure.update_layout(barmode = 'group', 
                     title="Sex vs Heart Disease (healthy = 0)", width = 500, height = 400)

# barmode = 'group','stack','relative'
#For you to take a look at the result use
figure.show()

countFemale, countMale = df.groupby(['sex']).size()
print(f'Percentage of female patients: {(countFemale/(countMale+countFemale))*100:.1f}%')
print(f'Percentage of male patients: {(countMale/(countMale+countFemale))*100:.1f}%')
df_cross = pd.crosstab(df.age, df.target)
data = []
#use for loop on every name to create bar data
for i,x in enumerate(df_cross.columns):
    data.append(go.Bar(name=str(x), x=df_cross.index, y=df_cross[x], marker_color= palette[i]))

figure = go.Figure(data)
figure.update_layout(barmode = 'group', 
                     title="Age vs Heart Disease (healthy = 0)")

# barmode = 'group','stack','relative'
#For you to take a look at the result use
figure.show()
for i,attr in enumerate(['age', 'resting_blood_pressure', 'cholesterol', 'max_heart_rate_achieved','st_depression']):
    
    f, (ax_box1,ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (0.2,0.8)}, 
                                                 figsize = (10,7))
    f.suptitle(f'{attr}')
    sns.boxplot(df[attr], ax=ax_box1, color = sns.color_palette('hsv')[i])
    sns.distplot(df[attr], ax=ax_hist, color = sns.color_palette('hsv')[i])
    ax_box1.set_xlabel('')


for attr in ['age', 'resting_blood_pressure', 'cholesterol', 'max_heart_rate_achieved','st_depression']:
    f, (ax_box1,ax_box2, ax_hist) = plt.subplots(3, sharex=True, gridspec_kw={"height_ratios": (.12, .12, .76)}, 
                                                 figsize = (12,8))
    f.suptitle(f'{attr} (0 = healthy, 1 = unhealthy)')
    # Add a graph in each part
    sns.boxplot(df[attr][df["target"]==0], ax=ax_box1, color = palette[0])
    sns.boxplot(df[attr][df["target"]==1], ax=ax_box2,color =palette[1])
    sns.distplot(df[attr][df["target"]==0], ax=ax_hist, color = palette[0])
    sns.distplot(df[attr][df["target"]==1], ax=ax_hist, color = palette[1])
    # Remove x axis name for the boxplot
    ax_box1.set_ylabel('0', rotation='horizontal')
    ax_box2.set_ylabel('1', rotation='horizontal')
    ax_box1.set_xlabel('')
    ax_box2.set_xlabel('')
    

for i,attr in enumerate(["age","resting_blood_pressure","cholesterol","max_heart_rate_achieved","st_depression"]):
    fig = px.box(df, x = "sex", y = attr, color = 'target', width = 800, height = 400, color_discrete_sequence= palette[::-1])
    fig.show()
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(df[['age', 'resting_blood_pressure', 
                                    'cholesterol', 'max_heart_rate_achieved','st_depression']]), columns = ['age', 'resting_blood_pressure', 
                                    'cholesterol', 'max_heart_rate_achieved','st_depression'])
X_scaled['target'] = df['target']
plt.figure(figsize=(16, 10))
sns.pairplot(X_scaled,
             hue='target', palette = palette,
           markers=['o','o'], plot_kws=dict(s=25, alpha=0.75, ci=None)
            )
plt.show()
plt.figure(figsize = (10,6))
sns.heatmap(df.corr(), annot=True)
f1, axs = plt.subplots(2,2,figsize=(15,10))

f1.tight_layout()

sns.countplot(df.sex,data=df,ax=axs[0][0], palette = "magma")
for p in axs[0][0].patches:
    height = p.get_height()
    axs[0][0].text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/float(len(df.sex))),
            ha="center")         
        
sns.countplot(df.exercise_induced_angina,data=df,ax=axs[0][1], palette ="magma")
for p in axs[0][1].patches:
    height = p.get_height()
    axs[0][1].text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/float(len(df.exercise_induced_angina))),
            ha="center") 
    
sns.countplot(df.st_slope,data=df,ax=axs[1][1],palette = "magma")
for p in axs[1][1].patches:
    height = p.get_height()
    axs[1][1].text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/float(len(df.st_slope))),
            ha="center") 
    
sns.countplot(df.chest_pain_type,data=df,ax=axs[1][0], palette ="magma")
for p in axs[1][0].patches:
    height = p.get_height()
    axs[1][0].text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/float(len(df.chest_pain_type))),
            ha="center") 

f2, axs2 = plt.subplots(2,2,figsize=(15,8))

f2.tight_layout()
sns.countplot(df.fasting_blood_sugar,data=df,ax=axs2[0][0], palette = "magma")
for p in axs2[0][0].patches:
    height = p.get_height()
    axs2[0][0].text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/float(len(df.fasting_blood_sugar))),
            ha="center") 
    
sns.countplot(df.rest_ecg,data=df,ax=axs2[1][0], palette = "magma")
for p in axs2[1][0].patches:
    height = p.get_height()
    axs2[1][0].text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/float(len(df.rest_ecg))),
            ha="center")
    
    
sns.countplot(df.num_major_vessels,data=df,ax=axs2[0][1],palette ="magma")
for p in axs2[0][1].patches:
    height = p.get_height()
    axs2[0][1].text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/float(len(df.num_major_vessels))),
            ha="center")
    
    
sns.countplot(df.thalassemia,data=df,ax=axs2[1][1], palette = "magma")
for p in axs2[1][1].patches:
    height = p.get_height()
    axs2[1][1].text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/float(len(df.thalassemia))),
            ha="center")
f1, axs = plt.subplots(2,2,figsize=(15,10))

f1.tight_layout()

sns.countplot(df.target,hue="sex",data=df,ax=axs[0][0], palette = "magma")
for p in axs[0][0].patches:
    height = p.get_height()
    axs[0][0].text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/float(len(df.sex))),
            ha="center") 
    
sns.countplot(df.target,hue="exercise_induced_angina",data=df,ax=axs[0][1], palette = "magma")
for p in axs[0][1].patches:
    height = p.get_height()
    axs[0][1].text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/float(len(df.exercise_induced_angina))),
            ha="center") 
    
sns.countplot(df.target,hue="st_slope",data=df,ax=axs[1][1],palette = "magma")
for p in axs[1][1].patches:
    height = p.get_height()
    axs[1][1].text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/float(len(df.st_slope))),
            ha="center") 


sns.countplot(df.target,hue="chest_pain_type",data=df,ax=axs[1][0], palette = "magma")
for p in axs[1][0].patches:
    height = p.get_height()
    axs[1][0].text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/float(len(df.chest_pain_type))),
            ha="center") 

f2, axs2 = plt.subplots(2,2,figsize=(15,8))

f2.tight_layout()
sns.countplot(df.target,hue="fasting_blood_sugar",data=df,ax=axs2[0][0], palette = "magma")
for p in axs2[0][0].patches:
    height = p.get_height()
    axs2[0][0].text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/float(len(df.fasting_blood_sugar))),
            ha="center") 

sns.countplot(df.target,hue="rest_ecg",data=df,ax=axs2[1][0], palette = "magma")
for p in axs2[1][0].patches:
    height = p.get_height()
    axs2[1][0].text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/float(len(df.rest_ecg))),
            ha="center")

sns.countplot(df.target,hue="num_major_vessels",data=df,ax=axs2[0][1],palette = "magma")
for p in axs2[0][1].patches:
    height = p.get_height()
    axs2[0][1].text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/float(len(df.num_major_vessels))),
            ha="center")

sns.countplot(df.target,hue="thalassemia",data=df,ax=axs2[1][1], palette = "magma")
for p in axs2[1][1].patches:
    height = p.get_height()
    axs2[1][1].text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/float(len(df.thalassemia))),
            ha="center")
X = pd.get_dummies(df, drop_first=True)
X.head(10)
rand = 42
target = X['target']
X = X.drop('target', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, target, train_size = 0.80, stratify = target, random_state = rand)
scaler = StandardScaler()
scaler.fit(X_train)

X_train_std = pd.DataFrame(scaler.fit_transform(X_train), index = X_train.index, columns = X_train.columns)
X_test_std = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)
plt.figure(figsize =(8,8))
for i in ["age","resting_blood_pressure","cholesterol","max_heart_rate_achieved","st_depression"]:
    sns.distplot(X_train_std[i], hist = False, label = i)
X_train_IQR = X_train_std
y_train_IQR = y_train

for attr in ['age', 'resting_blood_pressure', 'cholesterol','max_heart_rate_achieved','st_depression']:
    indexToDrop = []
    Q1 = X_train_IQR[attr].quantile(0.25)
    Q3 = X_train_IQR[attr].quantile(0.75)
    IQR = Q3 - Q1
    low = Q1 - IQR*1.5
    upp = Q3 + IQR*1.5
    ind = X_train_IQR[(X_train_IQR[attr] < low ) | (X_train_IQR[attr] > upp)].index
    if len(ind) != 0:
        X_train_IQR = X_train_IQR.drop(ind)
        y_train_IQR = y_train_IQR.drop(ind)
fig = make_subplots(rows=1, cols=5, subplot_titles=("Age", 
                                                    "resting blood pressure",
                                                    "cholesterol", 
                                                    "max_heart_rate_achieved",
                                                   "st_depression"))

fig.add_trace(
    go.Box(y=X_train_std.age, boxpoints = 'outliers', name = 'age'),
    row=1, col=1
)
fig.add_trace(
    go.Box(y=X_train_std.resting_blood_pressure, boxpoints = 'outliers', name = 'resting blood pressure'),
    row=1, col=2
)
fig.add_trace(
    go.Box(y=X_train_std.cholesterol, boxpoints = 'outliers', name = 'cholesterol'),
    row=1, col=3
)
fig.add_trace(
    go.Box(y=X_train_std.max_heart_rate_achieved, boxpoints = 'outliers', name = 'max heart rate'),
    row=1, col=4
)
fig.add_trace(
    go.Box(y=X_train_std.st_depression, boxpoints = 'outliers', name = 'ST depression'),
    row=1, col=5
)
fig.update_layout(height=450, width=1100,showlegend=True)
fig.update_traces(orientation='v')
fig.show()
fig = make_subplots(rows=1, cols=5, subplot_titles=("Age", 
                                                    "resting blood pressure",
                                                    "cholesterol", 
                                                    "max_heart_rate_achieved",
                                                   "st_depression"))

fig.add_trace(
    go.Box(y=X_train_IQR.age, boxpoints = 'outliers', name = 'age'),
    row=1, col=1
)
fig.add_trace(
    go.Box(y=X_train_IQR.resting_blood_pressure, boxpoints = 'outliers', name = 'resting blood pressure'),
    row=1, col=2
)
fig.add_trace(
    go.Box(y=X_train_IQR.cholesterol, boxpoints = 'outliers', name = 'cholesterol'),
    row=1, col=3
)
fig.add_trace(
    go.Box(y=X_train_IQR.max_heart_rate_achieved, boxpoints = 'outliers', name = 'max heart rate'),
    row=1, col=4
)
fig.add_trace(
    go.Box(y=X_train_IQR.st_depression, boxpoints = 'outliers', name = 'ST depression'),
    row=1, col=5
)
fig.update_layout(height=450, width=1100,showlegend=True)
fig.update_traces(orientation='v')
fig.show()
outlierDetector = IsolationForest(n_estimators=100, max_samples='auto', contamination=float(.06), \
                        max_features=1.0, bootstrap=False, n_jobs=-1, random_state=rand, verbose=0)
outlierDetector.fit(X_train_std)
pred = outlierDetector.predict(X_train_std)
indexToDrop = X_train_std[pred==-1].index

print(f'Number of outliers found: {len(indexToDrop)}')
pca =  PCA(n_components  = 3)
X_train_red  =  pd.DataFrame(pca.fit_transform(X_train_std), index = X_train_std.index)

X_train_red.columns = ['component1', 'component2','component3']
X_train_red['outliers'] = [1]*len(X_train_red)
X_train_red.loc[indexToDrop,'outliers'] = -1

fig = go.Figure(data=[go.Scatter3d(
    x=X_train_red['component1'], y=X_train_red['component2'], z=X_train_red['component3'],
    mode='markers',
    marker=dict(
        size=3,
        color = X_train_red['outliers'],
        colorscale=palette[::-1],
        opacity=0.8
    )
)])
fig.update_layout(title = '3D visualization of the outliers')
fig.show()
X_train_clean = X_train_std.drop(indexToDrop)
y_train_clean = y_train.drop(indexToDrop)
pca =  PCA(n_components = 3)
X_train_red  =  pd.DataFrame(pca.fit_transform(X_train_clean), index = X_train_clean.index)

X_train_red.columns = ['component1', 'component2','component3']

fig = go.Figure(data=[go.Scatter3d(
    x=X_train_red['component1'], y=X_train_red['component2'], z=X_train_red['component3'],
    mode='markers',
    marker=dict(
        size=7,
        color = y_train_clean,
        colorscale=palette,
        opacity=0.8
    )
)])
fig.update_layout(title = '3D visualization of the data')
fig.show()
X_train_clean = X_train_std.drop(indexToDrop)
y_train_clean = y_train.drop(indexToDrop)
pca = PCA().fit(X_train_clean)
nVar = len(X_train_clean.columns)

fig = make_subplots(rows=1, cols=1, subplot_titles=("Cumulative Variance Explained and Variance Explained"))

fig.add_trace(
    go.Scatter(x=np.arange(1,nVar+1), y=pca.explained_variance_ratio_.cumsum(),
        name='Cumulative Variance Explained',
        line=dict(color='blue'),
        connectgaps=True),
    row=1, col=1
)
fig.add_trace(
    go.Bar(x=np.arange(1,nVar+1), y=pca.explained_variance_ratio_,
        name='Variance Explained'),
    row=1, col=1
)
fig.add_shape(
            type="line",
            x0=10,
            y0=0,
            x1=10,
            y1=1,
            line=dict(
                color="green",
                width=2,
                dash="dot",
            )
)

fig.update_layout(width = 1000, height = 600)

fig.show()
pca = PCA(n_components = 0.70)
def build_pipeline(pipeline, params, X_train, y_train):
    grid = GridSearchCV(pipeline, params, scoring = 'f1_weighted',
                            return_train_score=True, n_jobs = -1)
    grid.fit(X_train, y_train)
    print(f"Best parameters set found on development set: {grid.best_params_}")
    return grid

def apply_model(grid, X_test, y_test):
    y = grid.predict(X_test)
    
    c = confusion_matrix(y_test, y)
    sensitivity = c[0,0]/(c[1,0]+c[0,0])
    acc = accuracy_score(y_test,y)
    f1 = f1_score(y_test,y)
    specificity = c[1,1]/(c[1,1]+c[0,1])    

    print(classification_report(y_test,y))
    print()
    plt.figure(figsize = (7,5))
    sns.heatmap(c, annot=True)
    plt.title('Confusion Matrix')
    print()
    
    return acc,f1,specificity,sensitivity
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

RF_PCA_ros_pip = Pipeline([('pca', pca), ('ros', RandomOverSampler(random_state=rand)),
                     ('classifier', RandomForestClassifier(random_state = rand))])

RF_PCA_pip = Pipeline([('pca', pca),('classifier', RandomForestClassifier(random_state = rand))])

params_tree = {'classifier__n_estimators': [25,50,100], 'classifier__max_depth' : [5, 8], 
               'classifier__max_features': ['auto', 1,2], 'classifier__criterion' :['gini']}
tree_pca_ros_metrics = []
tree_grid_pca_ros = build_pipeline(RF_PCA_ros_pip, params_tree, X_train_clean, y_train_clean)
tree_pca_ros_metrics = apply_model(tree_grid_pca_ros, X_test_std, y_test)

tree_pca_metrics = []
tree_grid_pca = build_pipeline(RF_PCA_pip, params_tree, X_train_clean, y_train_clean)
tree_pca_metrics = apply_model(tree_grid_pca, X_test_std, y_test)
svm_PCA_ros_pip = Pipeline([('pca', pca),('ros', RandomOverSampler(random_state=rand)),
                     ('classifier', SVC(random_state = rand))])
svm_PCA_pip = Pipeline([('pca', pca),('classifier', SVC(random_state = rand))])

params_svm = {'classifier__kernel':['linear', 'rbf'], 'classifier__C' : [0.01,0.1, 1, 10], 'classifier__gamma': ['auto','scale']}
svm_pca_ros_metrics = []
svm_grid_pca_ros = build_pipeline(svm_PCA_ros_pip, params_svm, X_train_clean, y_train_clean)
svm_pca_ros_metrics = apply_model(svm_grid_pca_ros, X_test_std, y_test)
svm_pca_metrics = []
svm_grid_pca = build_pipeline(svm_PCA_pip, params_svm, X_train_clean, y_train_clean)
svm_pca_metrics = apply_model(svm_grid_pca, X_test_std, y_test)
lr_PCA_ros_pip = Pipeline([('pca', pca),('ros', RandomOverSampler(random_state=rand)),
                     ('classifier', LogisticRegression(random_state = rand))])
lr_PCA_pip = Pipeline([('pca', pca),('classifier', LogisticRegression(random_state = rand))])

params_lr = {'classifier__penalty':['l1', 'l2'], 'classifier__C' : [0.01,0.1, 1, 10]}
lr_pca_ros_metrics = []
lr_grid_pca_ros = build_pipeline(lr_PCA_ros_pip, params_lr, X_train_clean, y_train_clean)
lr_pca_ros_metrics = apply_model(lr_grid_pca_ros, X_test_std, y_test)
lr_pca_metrics = []
lr_grid_pca = build_pipeline(lr_PCA_pip, params_lr, X_train_clean, y_train_clean)
lr_pca_metrics = apply_model(lr_grid_pca, X_test_std, y_test)
from sklearn.neighbors import KNeighborsClassifier

knn_PCA_ros_pip = Pipeline([('pca', pca),('ros', RandomOverSampler(random_state=rand)),
                     ('classifier', KNeighborsClassifier())])
knn_PCA_pip = Pipeline([('pca', pca), ('classifier', KNeighborsClassifier())])

params_knn = {'classifier__n_neighbors':[3,5,7,8], 'classifier__p':[1,2]}
knn_pca_ros_metrics = []
knn_grid_pca_ros = build_pipeline(knn_PCA_ros_pip, params_knn, X_train_clean, y_train_clean)
knn_pca_ros_metrics = apply_model(knn_grid_pca_ros, X_test_std, y_test)
knn_pca_metrics = []
knn_grid_pca = build_pipeline(knn_PCA_pip, params_knn, X_train_clean, y_train_clean)
knn_pca_metrics = apply_model(knn_grid_pca, X_test_std, y_test)
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
   
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(20, 5))
    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")
    
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes.legend(loc="best")
    return plt
fig, axes = plt.subplots(2, 2, figsize=(17, 10))

title = "Learning Curves (Random Forest) - PCA ROS"

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

plot_learning_curve(tree_grid_pca_ros.best_estimator_, title, X_train_clean, y_train_clean, axes = axes[0][0], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)

title = r"Learning Curves (SVM) - PCA ROS"
plot_learning_curve(svm_grid_pca_ros.best_estimator_, title,  X_train_clean, y_train_clean, axes = axes[0][1], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)

title = r"Learning Curves (Logistic Regression) - PCA ROS"

plot_learning_curve(lr_grid_pca_ros.best_estimator_, title,  X_train_clean, y_train_clean, axes = axes[1][0], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)

title = r"Learning Curves (KNN) - PCA ROS"
plot_learning_curve(knn_grid_pca_ros.best_estimator_, title,  X_train_clean, y_train_clean, axes = axes[1][1], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)
plt.show()
fig, axes = plt.subplots(2, 2, figsize=(17, 10))

title = "Learning Curves (Random Forest) - PCA "

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
plot_learning_curve(tree_grid_pca.best_estimator_, title, X_train_clean, y_train_clean, axes = axes[0][0], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)

title = r"Learning Curves (SVM) - PCA"
plot_learning_curve(svm_grid_pca.best_estimator_, title,  X_train_clean, y_train_clean, axes = axes[0][1], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)

title = r"Learning Curves (Logistic Regression) - PCA"
estimator = lr_grid_pca_ros.best_estimator_
plot_learning_curve(lr_grid_pca, title,  X_train_clean, y_train_clean, axes = axes[1][0], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)

title = r"Learning Curves (KNN) - PCA"

plot_learning_curve(knn_grid_pca.best_estimator_, title,  X_train_clean, y_train_clean, axes = axes[1][1], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)
plt.show()
tree_ros_pip = Pipeline([('ros', RandomOverSampler(random_state=rand)),
                     ('classifier', RandomForestClassifier(random_state = rand))])
tree_pip = Pipeline([('classifier', RandomForestClassifier(random_state = rand))])

params_tree = {'classifier__n_estimators': [25,50,100], 'classifier__max_depth' : [5], 
               'classifier__max_features': ['auto',1,2], 'classifier__criterion' :['gini']}
tree_ros_metrics = []
tree_grid_ros = build_pipeline(tree_ros_pip, params_tree, X_train_clean, y_train_clean)
tree_ros_metrics = apply_model(tree_grid_ros, X_test_std, y_test)
tree_metrics = []
tree_grid = build_pipeline(tree_pip, params_tree, X_train_clean, y_train_clean)
tree_metrics = apply_model(tree_grid, X_test_std, y_test)
importances = tree_grid_ros.best_estimator_[1].feature_importances_
std = np.std([importances for tree in tree_grid.best_estimator_[0].estimators_],
             axis=0)
indices = np.argsort(importances)

p = sns.color_palette('Spectral', len(indices))

plt.figure(figsize = (12,5))
plt.title("Feature importances - RandomForest")
plt.barh(range(X.shape[1]), importances[indices], color = 'blue', yerr=std[indices], align="center")
plt.yticks(range(X.shape[1]), X_train_clean.columns[indices], rotation = 'horizontal')
# plt.xlim([-1, X.shape[1]])
plt.show()

svm_ros_pip = Pipeline([('ros', RandomOverSampler(random_state=rand)),
                     ('classifier', SVC(random_state = rand))])
svm_pip = Pipeline([('classifier', SVC(random_state = rand))])

params_svm = {'classifier__kernel':['linear'], 'classifier__C' : [0.01,0.1, 1, 10]}
svm_ros_metrics = []
svm_grid_ros = build_pipeline(svm_ros_pip, params_svm, X_train_clean, y_train_clean)
svm_ros_metrics = apply_model(svm_grid_ros, X_test_std, y_test)
svm_metrics = []
svm_grid = build_pipeline(svm_pip, params_svm, X_train_clean, y_train_clean)
svm_metrics = apply_model(svm_grid, X_test_std, y_test)
imp,names = zip(*sorted(zip(svm_grid_ros.best_estimator_[1].coef_[0],X_train_clean.columns)))
colors = ['red' if c < 0 else 'blue' for c in imp]
p1 = sns.color_palette('Spectral', len(names))

plt.figure(figsize = (12,5))
plt.barh(range(len(names)), imp, align='center', color = colors)
plt.yticks(range(len(names)), names)
plt.title("Feature importances - SVM")
plt.legend()
plt.show()

lr_ros_pip = Pipeline([('ros', RandomOverSampler(random_state=rand)),
                     ('classifier', LogisticRegression(random_state = rand))])
lr_pip = Pipeline([('classifier', LogisticRegression(random_state = rand))])

params_lr = {'classifier__C' : [0.01,0.1, 1, 10]}
lr_ros_metrics = []
lr_grid_ros = build_pipeline(lr_ros_pip, params_lr, X_train_clean, y_train_clean)
lr_ros_metrics = apply_model(lr_grid_ros, X_test_std, y_test)
lr_metrics = []
lr_grid = build_pipeline(lr_pip, params_lr, X_train_clean, y_train_clean)
lr_metrics = apply_model(lr_grid, X_test_std, y_test)
imp,names = zip(*sorted(zip(lr_grid_ros.best_estimator_[1].coef_[0],X_train_clean.columns)))
colors = ['red' if c < 0 else 'blue' for c in imp]
p1 = sns.color_palette('Spectral', len(names))

plt.figure(figsize = (12,5))
plt.barh(range(len(names)), imp, align='center', color = colors)
plt.yticks(range(len(names)), names)
plt.title("Feature importances - Logistic Regression")
plt.legend()
plt.show()
knn_ros_pip = Pipeline([('ros', RandomOverSampler(random_state=rand)),
                     ('classifier', KNeighborsClassifier())])
knn_pip = Pipeline([('classifier', KNeighborsClassifier())])

params_knn = {'classifier__n_neighbors':[3,4,7,8], 'classifier__p':[1,2]}
knn_ros_metrics = []
knn_grid_ros = build_pipeline(knn_ros_pip, params_knn, X_train_clean, y_train_clean)
knn_ros_metrics = apply_model(knn_grid_ros, X_test_std, y_test)
knn_metrics = []
knn_grid = build_pipeline(knn_pip, params_knn, X_train_clean, y_train_clean)
knn_metrics = apply_model(knn_grid, X_test_std, y_test)
fig, axes = plt.subplots(2, 2, figsize=(17, 10))

title = "Learning Curves (Random Forest) - ROS "

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
plot_learning_curve(tree_grid_ros.best_estimator_, title, X_train_red, y_train_clean, axes = axes[0][0], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)

title = r"Learning Curves (SVM) - ROS"
plot_learning_curve(svm_grid_ros.best_estimator_, title,  X_train_red, y_train_clean, axes = axes[0][1], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)

title = r"Learning Curves (Logistic Regression) - ROS"
plot_learning_curve(lr_grid_ros.best_estimator_, title,  X_train_red, y_train_clean, axes = axes[1][0], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)

title = r"Learning Curves (KNN) - ROS"

plot_learning_curve(knn_grid_ros, title,  X_train_clean, y_train_clean, axes = axes[1][1], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)
plt.show()
fig, axes = plt.subplots(2, 2, figsize=(17,10))

title = "Learning Curves (Random Forest) "

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=rand)
plot_learning_curve(tree_grid.best_estimator_, title, X_train_red, y_train_clean, axes = axes[0][0], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)

title = r"Learning Curves (SVM) "
plot_learning_curve(svm_grid.best_estimator_, title,  X_train_red, y_train_clean, axes = axes[0][1], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)

title = r"Learning Curves (Logistic Regression)"
plot_learning_curve(lr_grid.best_estimator_, title,  X_train_red, y_train_clean, axes = axes[1][0], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)
title = r"Learning Curves (KNN)"
estimator = knn_grid.best_estimator_
plot_learning_curve(knn_grid.best_estimator_, title,  X_train_clean, y_train_clean, axes = axes[1][1], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=4)
plt.show()
explainer = shap.KernelExplainer(lr_grid_ros.best_estimator_[1].predict_proba, data = X_train, link = 'logit')
shap_values = explainer.shap_values(X_test, n_samples = 50)
print(f'length of SHAP values: {len(shap_values)} as the number of classes') #as the number of classes
shap.initjs()
shap.force_plot(explainer.expected_value[0], shap_values[0][12,:], features = X_test.iloc[12,:], link = 'logit')
shap.force_plot(explainer.expected_value[1], shap_values[1][12,:], features = X_test.iloc[12,:], link = 'logit')
explainer = shap.KernelExplainer(lr_grid_ros.best_estimator_.predict_proba, data = X_train_clean, link = 'logit')
shap_values = explainer.shap_values(X_test_std, n_samples = 50)
shap.summary_plot(shap_values[0], X_test_std)
metrics = pd.DataFrame(np.array([tree_pca_ros_metrics, tree_pca_metrics, tree_ros_metrics, tree_metrics,
                                svm_pca_ros_metrics, svm_pca_metrics, svm_ros_metrics, svm_metrics,
                                lr_pca_ros_metrics, lr_pca_metrics, lr_ros_metrics, lr_metrics,
                                knn_pca_ros_metrics, knn_pca_metrics, knn_ros_metrics, knn_metrics]).reshape(4, 16),
                       index = ['Accuracy', 'F1_score', 'Specifivity', 'Sensitivity'],
                       columns = ['RandomForest PCA ROS', 'RandomForest PCA', 'RandomForest ROS', 'RandomForest',
                                'SVM PCA ROS', 'SVM PCA','SVM ROS','SVM',
                                'LogisticRegression PCA ROS','LogisticRegression PCA',
                                'LogisticRegression ROS','LogisticRegression',
                                 'KNN PCA ROS', 'KNN PCA', 'KNN ROS','KNN']
                      )
px.bar(metrics, barmode = 'group', height = 450, width = 1000)