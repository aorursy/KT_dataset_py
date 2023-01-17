# Import needed libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Import statements required for Plotly 
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff

# Import Models
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_validate, cross_val_score, StratifiedKFold
from sklearn import svm 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Define constants/variables to use
seed_Num = 8 # aka Random State
num_folds = 5
# Read in the data into pandas dataframe
data = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')
pd.DataFrame(data.columns, columns=['Column Names'])
data.sample(5, random_state=8)
# Display Number of Uniques, If Column has NA, number of NAs, and data types for each column
pd.DataFrame(data = {'Data Type': data.dtypes,
                     'Number of Unique Values': data.nunique().sort_values(),
                     'Contains NAs': data.isnull().any(),
                     'Number of NAs': data.isnull().sum()}).sort_values('Number of Unique Values')
cols_to_drop_for_modeling = ['Over18','StandardHours','EmployeeCount','EmployeeNumber']
data.drop(cols_to_drop_for_modeling, axis = 1,inplace=True)
cat_cols = list(data.select_dtypes(exclude=np.number).columns)
#cat_cols.remove('Attrition') 

fig_rows = 2
fig_cols = 4
fig = tls.make_subplots(rows=fig_rows, cols=fig_cols, 
                          subplot_titles=tuple(cat_cols));
curr_row = 1
curr_col = 1
for i, col in enumerate(cat_cols):
    trace = go.Bar(name=col,
                x= data[col].value_counts().index.values,
                y= data[col].value_counts().values,
                marker=dict(line=dict(color='black',width=1)))
    fig.append_trace(trace, curr_row, curr_col)
    curr_col+=1
    if curr_col >= fig_cols+1: # Zero Indexing for '-1'
        curr_row += 1
        curr_col = 1

fig['layout'].update(title =  'Count of Categorical Variables in Dataset', width = 900, height = 600,
                    showlegend=False)

py.iplot(fig)
attritionBarPlot = go.Bar(
            x= data["Attrition"].value_counts().index.values,
            y= data["Attrition"].value_counts().values,
            marker=dict( color=['Orange', 'steelblue'],line=dict(color='black',width=1)))
layout = dict(title =  'Count of Attrition in Dataset', width = 800, height = 400)
fig = dict(data = [attritionBarPlot], layout=layout)
py.iplot(fig)
#  Plot areas are called axes
import warnings    # We want to suppress warnings
warnings.filterwarnings("ignore")    # Ignore warnings

fig_rows = 5
fig_cols = 5
fig,ax = plt.subplots(fig_rows,fig_cols, figsize=(16,20)) 
fake_numeric_cols = ['Education','EnvironmentSatisfaction', 'JobInvolvement' , 'JobSatisfaction' , 
                     'PerformanceRating', 'RelationshipSatisfaction', 'WorkLifeBalance', 'JobLevel',
                    'StockOptionLevel']
#numericCols = [x for x in list(data.select_dtypes(include=np.number).columns) if x not in fake_numeric_cols]
numericCols = list(data.select_dtypes(include=np.number).columns)
curr_row = 0
curr_col = 0
for i, col in enumerate(numericCols):
    sns.distplot(data[col], ax = ax[(curr_row,curr_col)],rug=True, kde =False) 
    curr_col+=1
    if curr_col >= fig_cols: # Zero Indexing for '-1'
        curr_row += 1
        curr_col = 0

plt.show()
data[numericCols].describe().T
heatmapCols = numericCols + ['Attrition']
temp = data.copy()
temp['Attrition'] = data['Attrition'].replace('Yes',1).replace('No',0)
heatmapGo = [go.Heatmap(
        z= temp[heatmapCols].astype(float).corr().values, # Generating the Pearson correlation
        x= temp[heatmapCols].columns.values,
        y= temp[heatmapCols].columns.values,
        colorscale='Cividis',
        reversescale = False,
        opacity = 1.0)]

layout = go.Layout(
    title='Pearson Correlation Matrix Numerical Features',
    xaxis = dict(ticks='', tickfont = dict(size = 10)),
    yaxis = dict(ticks='', tickfont = dict(size = 7)),
    width = 900, height = 700)

fig = go.Figure(data=heatmapGo, layout=layout)
py.iplot(fig, filename='labelled-heatmap')
pseudo_numeric_cols = ['Education','EnvironmentSatisfaction', 'JobInvolvement' , 'JobSatisfaction' , 
                     'PerformanceRating', 'RelationshipSatisfaction', 'WorkLifeBalance', 'JobLevel',
                    'StockOptionLevel'] 
excludeCols = pseudo_numeric_cols + ['YearsWithCurrManager','YearsSinceLastPromotion','YearsInCurrentRole','YearsAtCompany']
pairplotCols = [x for x in list(data.select_dtypes(include=np.number).columns) if x not in excludeCols]+ ['Attrition']
sns.pairplot(data[pairplotCols], plot_kws={'scatter_kws': {'alpha': 0.1}},
             kind="reg", diag_kind = "kde"  , hue = 'Attrition' );
numeric_cols = list(data.select_dtypes(include=np.number).columns)

fig_rows = 6
fig_cols = 4
fig = tls.make_subplots(rows=fig_rows, cols=fig_cols, 
                          subplot_titles=tuple(numeric_cols));
curr_row = 1
curr_col = 1
for i, col in enumerate(numeric_cols):
    trace1 = go.Histogram(name = "No Attrition", 
                          marker=dict( line=dict(color='black',width=1)), #color=['steelblue']),
                          x = list(data[data['Attrition'] == 'No'][col]),
                          opacity=0.5)
    trace2 = go.Histogram(name = 'Yes Attrition', 
                          marker=dict(line=dict(color='black',width=1)),#,color=['Orange']),
                          x=data[data['Attrition'] == 'Yes'][col],
                          opacity=0.5)
    tmp3 = pd.DataFrame(pd.crosstab(data[col],
                    data['Attrition'].replace('Yes',1).replace('No',0)), )
    tmp3['Attr%'] = tmp3[1] / (tmp3[1] + tmp3[0]) * 100
    extra_yAxis = 'y' + str(fig_rows * fig_cols + i+1)
    trace3 =  go.Scatter(x=tmp3.index,y=tmp3['Attr%'],
        yaxis = extra_yAxis,name='% Attrition', opacity = .8, 
        marker=dict(color='black',line=dict(color='black',width=0.5)))
    fig.append_trace(trace1, curr_row, curr_col)
    fig.append_trace(trace2, curr_row, curr_col)
    if col not in ['MonthlyRate','DailyRate', 'MonthlyIncome']:
        fig.append_trace(trace3, curr_row, curr_col)
        fig['data'][-1].update(yaxis=extra_yAxis)
        yaxisStr = ''
        if curr_col == fig_cols:
            yaxisStr = '% Attrition'
        fig['layout']['yaxis' + str(fig_rows * fig_cols + i+1)] = dict(range= [0, max(tmp3['Attr%'])+10], 
                         showgrid=True,  overlaying= 'y'+str(i + 1), anchor= 'x'+str(i+1), side= 'right',
                         title= yaxisStr)
    curr_col+=1
    if curr_col >= fig_cols+1: # Zero Indexing for '-1'
        curr_row += 1
        curr_col = 1
fig['layout'].update(title =  'Numerical Distributions colored by Attrition', width = 900, height = 900,
                    barmode = 'overlay',showlegend=False, font=dict(size=10))#,
                    #yaxis2=dict(range= [0, 100], overlaying= 'y', anchor= 'x', 
                    #      side= 'right',zeroline=False,showgrid= False, title= '% Attrition'))

py.iplot(fig)
cat_cols = list(data.select_dtypes(exclude=np.number).columns)

fig_rows = 2
fig_cols = 4
fig = tls.make_subplots(rows=fig_rows, cols=fig_cols, 
                          subplot_titles=tuple(cat_cols));
curr_row = 1
curr_col = 1
for i, col in enumerate(cat_cols):
    yaxisStr = ''
    offset_val = -0.3
    if col not in ['Attrition']:
        offset_val = -0.2
    trace1 = go.Bar(name='No Attrition', opacity = .8, width= 0.6,#offset = -0.03,
                x= data[data['Attrition'] == 'No'][col].value_counts().index.values,
                y= data[data['Attrition'] == 'No'][col].value_counts().values,
                marker=dict(color = 'steelblue', line=dict(color='black',width=1)))
    trace2 = go.Bar(name='Yes Attrition', opacity = .8, width= 0.6,offset = offset_val,
                x= data[data['Attrition'] == 'Yes'][col].value_counts().index.values,
                y= data[data['Attrition'] == 'Yes'][col].value_counts().values,
                marker=dict(color = 'orange',line=dict(color='black',width=1)))
    tmp3 = pd.DataFrame(pd.crosstab(data[col],
                    data['Attrition'].replace('Yes',1).replace('No',0)), )
    tmp3['Attr%'] = tmp3[1] / (tmp3[1] + tmp3[0]) * 100
    extra_yAxis = 'y' + str(fig_rows * fig_cols + i+1)
    trace3 =  go.Scatter(x=tmp3.index,y=tmp3['Attr%'],mode = 'markers',
        yaxis = extra_yAxis,name='% Attrition', opacity = .8, 
        marker=dict(color='black',size= 10))
    fig.append_trace(trace1, curr_row, curr_col)
    fig.append_trace(trace2, curr_row, curr_col)
    if col not in ['Attrition']:
        fig.append_trace(trace3, curr_row, curr_col)
        fig['data'][-1].update(yaxis=extra_yAxis)
        if curr_col == fig_cols:
            yaxisStr = '% Attrition'
        fig['layout']['yaxis' + str(fig_rows * fig_cols + i+1)] = dict(range= [0, max(tmp3['Attr%'])+10], 
                         showgrid=True,  overlaying= 'y'+str(i + 1), anchor= 'x'+str(i+1), side= 'right',
                         title= yaxisStr)
    
    curr_col+=1
    if curr_col >= fig_cols+1: # Zero Indexing for '-1'
        curr_row += 1
        curr_col = 1

fig['layout'].update(title =  'Count of Categorical Variables in Dataset Colored by Attrition', 
                     width = 900, height = 600,
                     showlegend=False, barmode = 'overlay', font=dict(size=10))

py.iplot(fig)
numericCols = list(data.select_dtypes(include=np.number).columns)

cat_cols = list(data.select_dtypes(exclude=np.number).columns)
if 'Attrition' in cat_cols:
    cat_cols.remove('Attrition')
X_cat = pd.get_dummies(data[cat_cols])
X_cat.head()
y = data['Attrition'].replace('Yes',1).replace('No',0)
X = pd.concat([data[numericCols], X_cat], axis=1, sort=False)
X.head()
def get_cv_results(model, X, y):
    pipe = make_pipeline(StandardScaler(),model)
    cv_scores_dict = cross_validate(pipe, X, y, cv=num_folds, 
                                    scoring= ['roc_auc', 'accuracy', 'f1', 'precision','recall'],
                                   return_train_score = False)
    cv_scores_df = pd.DataFrame(cv_scores_dict, 
                 index = ['Fold {}'.format(x) for x in range(1,len(cv_scores_dict['test_accuracy'])+1)])
    return pd.concat([cv_scores_df, pd.DataFrame(cv_scores_df.mean(), columns=['Avg']).T])
logRegScores = get_cv_results(LogisticRegression(random_state=seed_Num), X, y)
logRegScores
from sklearn.naive_bayes import GaussianNB
naiveBayesScores = get_cv_results(GaussianNB(), X, y)
naiveBayesScores
from sklearn.neighbors import KNeighborsClassifier
#knnNeighborComparison = pd.DataFrame()
#for numOfK in range(1,51,2):
#    knnScores = get_cv_results(KNeighborsClassifier(n_neighbors=numOfK), X, y)
#    df = pd.DataFrame(knnScores.loc['Avg']).T
#    df.index = ['K='+str(numOfK) + ' Avg']
#    knnNeighborComparison = pd.concat([knnNeighborComparison,df])
#knnNeighborComparison
## Notes on Iterating through neighbors:
### Seems to plateau around roc_auc of 0.78 and f1 score generally decrease as k increases
### K = 7 seems to be last significant bump in roc_auc with a relatively decent f1
knnScores = get_cv_results(KNeighborsClassifier(n_neighbors=7), X, y)
knnScores
import lightgbm as lgb
lgbmScores = get_cv_results(lgb.LGBMClassifier(random_state=1, n_jobs = -1), X, y)
lgbmScores
from sklearn.svm import SVC
svcScores = get_cv_results(SVC(), X, y)
svcScores
from sklearn.svm import SVC
svcLinearScores = get_cv_results(SVC(kernel='linear'), X, y)
svcLinearScores
from sklearn.neural_network import MLPClassifier
mlpScores = get_cv_results(MLPClassifier(hidden_layer_sizes = (2,2),
                                         random_state=seed_Num), X, y)
mlpScores
cvAvgResultsCombined = pd.DataFrame()
# Get variables ending with Scores
varNames = [s for s in list(locals().keys()) if s.endswith('Scores')]
local_var_dict = locals()
for varName in varNames:
    df = pd.DataFrame(local_var_dict[varName].loc['Avg']).T
    df.index = [varName+' Avg']
    cvAvgResultsCombined = pd.concat([cvAvgResultsCombined,df])
cvAvgResultsCombined.sort_values(by='test_roc_auc', ascending=False)