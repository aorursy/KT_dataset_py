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
data = pd.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
data.drop(['EmployeeCount'], axis=1, inplace= True) ### It is just a counter
data.drop(['StandardHours'], axis=1, inplace= True) ### It is the same value for all(80)
data.drop(['Over18'], axis=1, inplace= True) ### It is the same value for all(Yes)
##### Importing the packages

import pandas as pd
from pandas import DataFrame
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, learning_curve, train_test_split
from sklearn.metrics import precision_score, roc_auc_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve, accuracy_score
import xgboost as xgb
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
data.Attrition.replace(to_replace = dict(Yes = 1, No = 0), inplace = True)
data['Generation'] = data['Age']
data['Generation'] = data['Generation'].replace([18,19,20,21],'Z_18_to_21')
data['Generation'] = data['Generation'].replace([22,23,24,25,26,27,28,29,30,31,32,33,34,35,36],'Y_22_to_36')
data['Generation'] = data['Generation'].replace([37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56], 'X_37_to_56')
data['Generation'] = data['Generation'].replace([57,58,59,60],'Boomers_57_77')
attrition = data[(data['Attrition'] != 0)]
no_attrition = data[(data['Attrition'] == 0)]
def plot_distribution(var_select, bin_size) : 
# Calculate the correlation coefficient between the new variable and the target
    corr = data['Attrition'].corr(data[var_select])
    corr = np.round(corr,3)
    tmp1 = attrition[var_select]
    tmp2 = no_attrition[var_select]
    hist_data = [tmp1, tmp2]
    
    group_labels = ['Yes_attrition', 'No_attrition']
    colors = ['#00264d', '#00cc99']

    fig = ff.create_distplot(hist_data, group_labels, colors = colors, show_hist = True, curve_type='kde', bin_size = bin_size)
    
    fig['layout'].update(title = var_select+' '+'(corr target ='+ str(corr)+')')

    py.iplot(fig, filename = 'Density plot')
    
def barplot(var_select, x_no_numeric) :
    tmp1 = data[(data['Attrition'] != 0)]
    tmp2 = data[(data['Attrition'] == 0)]
    tmp3 = pd.DataFrame(pd.crosstab(data[var_select],data['Attrition']), )
    tmp3['Attr%'] = tmp3[1] / (tmp3[1] + tmp3[0]) * 100
    if x_no_numeric == True  : 
        tmp3 = tmp3.sort_values(1, ascending = False)

    color=['#00264d', '#00cc99' ]
    trace1 = go.Bar(
        x=tmp1[var_select].value_counts().keys().tolist(),
        y=tmp1[var_select].value_counts().values.tolist(),
        name='Yes_Attrition',opacity = 0.8, marker=dict(
        color='#00264d',
        line=dict(color='#000000',width=1)))

    
    trace2 = go.Bar(
        x=tmp2[var_select].value_counts().keys().tolist(),
        y=tmp2[var_select].value_counts().values.tolist(),
        name='No_Attrition', opacity = 0.8, marker=dict(
        color='#00cc99',
        line=dict(color='#000000',width=1)))
    
    trace3 =  go.Scatter(   
        x=tmp3.index,
        y=tmp3['Attr%'],
        yaxis = 'y2',
        name='% Attrition', opacity = 0.6, marker=dict(
        color='black',
        line=dict(color='#000000',width=0.5
        )))

    layout = dict(title =  str(var_select),
              xaxis=dict(), 
              yaxis=dict(title= 'Count'), 
              yaxis2=dict(range= [-0, 75], 
                          overlaying= 'y', 
                          anchor= 'x', 
                          side= 'right',
                          zeroline=False,
                          showgrid= False, 
                          title= '% Attrition'
                         ))

    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    py.iplot(fig)
def High_Distance(data) :
    if data['DistanceFromHome'] > 10:
        return 1
    else:
        return 0
data['High_Distance'] = data.apply(lambda data:High_Distance(data) ,axis = 1)

def Young_Talents(data) :
    if data['Age'] < 25 :
        return 1
    else:
        return 0
data['Young_Talents'] = data.apply(lambda data:Young_Talents(data) ,axis = 1)

def Women_High_Traveller(data) : 
    if  data['Gender'] == 'Female' and data['BusinessTravel'] == 'Travel_Frequently':
        return 1
    else : 
        return 0
data['Women_High_Traveller'] = data.apply(lambda data:Women_High_Traveller(data) ,axis = 1)

def SamePositionforlongtime(data) :
    if data['YearsSinceLastPromotion'] > 5 :
        return 1
    else:
        return 0
data['SamePositionforlongtime'] = data.apply(lambda data:SamePositionforlongtime(data) ,axis = 1)

data['TotalSatisfaction_mean'] = (data['RelationshipSatisfaction']  + data['EnvironmentSatisfaction'] + data['JobSatisfaction'] + data['JobInvolvement'] + data['WorkLifeBalance'])/5

def NotSatif(data) : 
    if  data['TotalSatisfaction_mean'] < 2.5 :
        return 1
    else : 
        return 0
data['NotSatif'] = data.apply(lambda data:NotSatif(data) ,axis = 1)

def NotSatisfied_LongDistance(data) : 
    if  data['DistanceFromHome'] > 10 and data['NotSatif'] < 2.5 :
        return 1
    else : 
        return 0
data['NotSatisfied_LongDistance'] = data.apply(lambda data:NotSatisfied_LongDistance(data) ,axis = 1)

def OverTime_Sales(data) : 
    if  data['OverTime'] == 'Yes' and data['Department'] == 'Sales' :
        return 1
    else : 
        return 0
data['OverTime_Sales'] = data.apply(lambda data:OverTime_Sales(data) ,axis = 1)

def OverTime_HR(data) : 
    if  data['OverTime'] == 'Yes' and data['Department'] == 'Human Resources' :
        return 1
    else : 
        return 0
data['OverTime_HR'] = data.apply(lambda data:OverTime_HR(data) ,axis = 1)

def NotSatisfied_Overtime(data) : 
    if  data['OverTime'] =='Yes' and data['NotSatif'] < 2.5 :
        return 1
    else : 
        return 0
data['NotSatisfied_Overtime'] = data.apply(lambda data:NotSatisfied_Overtime(data) ,axis = 1)
barplot('Gender', False)
barplot('Department', False)
barplot('JobLevel', False)
plot_distribution('DistanceFromHome', False)
barplot('BusinessTravel', False)
barplot('EnvironmentSatisfaction', False)
# Boxplot
plt.title ('MonthlyIncome')
y = data.MonthlyIncome
x = data.JobLevel
fig = sns.boxplot(x,y,palette="Blues_d")
fig.axis(ymin=900, ymax=21999);
barplot('Generation', False)
barplot('OverTime', False)
barplot('DistanceFromHome', False)
barplot('BusinessTravel', False)
barplot('OverTime_Sales', False)
barplot('OverTime_HR', False)
barplot('JobLevel', False)
barplot('NotSatif', False)
barplot('Generation', False)
barplot('OverTime', False)
barplot('YearsAtCompany', False)
barplot('High_Distance', False)
barplot('Women_High_Traveller', False)
barplot('NotSatif', False)
# Boxplot
plt.title ('MonthlyIncome')
y = data.MonthlyIncome
x = data.JobLevel
fig = sns.boxplot(x,y,palette="Blues_d")
fig.axis(ymin=900, ymax=21999);
data[(data.Department=='Sales')].MonthlyIncome.describe()
data[(data.Department=='Human Resources')].MonthlyIncome.describe()
data[(data.Department=='Research & Development')].MonthlyIncome.describe()
barplot('High_Distance', False)
barplot('Young_Talents', False)
barplot('NotSatif', False)
barplot('OverTime_Sales', False)
barplot('OverTime_HR', False)
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(
ascending=False)
missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', 'Percent'])
missing_data.head(20)
data.Age.describe()
data_personal = DataFrame(data, columns = ['Age'])
g = sns.pairplot(data_personal, diag_kind="kde", markers="+",
                 plot_kws=dict(s=50, edgecolor="b", linewidth=1),
                 diag_kws=dict(shade=True))

# Curtosis
scipy.stats.kurtosis(data.Age)
#### Skewness
scipy.stats.skew(data.Age)
#### Below we can see that there is a slight difference between the Age boxplots considering the attritions(yes or no).
#### Maybe younger people are having more attrition cases.
y = data.Age
x = data.Attrition
fig = sns.boxplot(x,y,palette="Blues_d")
fig.axis(ymin=15, ymax=65);
data.Generation.unique()
grafico = sns.countplot(data = data , x = 'Generation', hue = 'Generation', palette="Blues_d")

for p in grafico.patches:
	b=p.get_bbox()
	grafico.annotate("{:.0f}".format(b.y1 + b.y0), ((b.x0 + b.x1)/2 - 0.03, b.y1 + 15))

plt.show()
data.groupby('Attrition')[['Age']].describe()
barplot('Age', False)
plot_distribution('Age', False)
barplot('Generation', False)
grafico = sns.countplot(data = data , x = 'Gender', hue = 'Gender',palette="Blues_d")

for p in grafico.patches:
	b=p.get_bbox()
	grafico.annotate("{:.0f}".format(b.y1 + b.y0), ((b.x0 + b.x1)/2 - 0.03, b.y1 + 15))

plt.show()
barplot('Gender', False)
data[(data.Gender=='Male')].groupby('Attrition')[['Age']].describe()
data[(data.Gender=='Female')].groupby('Attrition')[['Age']].describe()
grafico = sns.countplot(data = data , x = 'OverTime', hue = 'OverTime',palette="Blues_d")

for p in grafico.patches:
	b=p.get_bbox()
	grafico.annotate("{:.0f}".format(b.y1 + b.y0), ((b.x0 + b.x1)/2 - 0.03, b.y1 + 15))

plt.show()
barplot('OverTime', False)
data_timings1 = DataFrame(data, columns = ['YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion'])
g = sns.pairplot(data_timings1, diag_kind="kde", markers="+",
                 plot_kws=dict(s=50, edgecolor="b", linewidth=1),
                 diag_kws=dict(shade=True))
data_timings1.describe()
# Boxplot

plt.title ('YearsAtCompany')
y = data.YearsAtCompany
x = data.Attrition
fig = sns.boxplot(x,y,palette="Blues_d")
fig.axis(ymin=0, ymax=40);
plt.title ('YearsInCurrentRole')
y = data.YearsInCurrentRole
x = data.Attrition
fig = sns.boxplot(x,y,palette="Blues_d")
fig.axis(ymin=0, ymax=18);
plt.title ('YearsSinceLastPromotion')
y = data.YearsSinceLastPromotion
x = data.Attrition
fig = sns.boxplot(x,y,palette="Blues_d")
fig.axis(ymin=-2, ymax=15);
barplot('YearsAtCompany', False)
barplot('YearsInCurrentRole', False)
barplot('YearsSinceLastPromotion', False)
plot_distribution('YearsAtCompany', False)
plot_distribution('YearsInCurrentRole', False)
plot_distribution('YearsSinceLastPromotion', False)
data_distances = DataFrame(data, columns = ['DistanceFromHome'])
g = sns.pairplot(data_distances, diag_kind="kde", markers="+",
                 plot_kws=dict(s=50, edgecolor="b", linewidth=1),
                 diag_kws=dict(shade=True))
data_distances.describe()
# Boxplot

plt.title ('Distances')
y = data.DistanceFromHome
x = data.Attrition
fig = sns.boxplot(x,y,palette="Blues_d")
fig.axis(ymin=1, ymax=30);
barplot('DistanceFromHome', False)
plot_distribution('DistanceFromHome', False)
grafico = sns.countplot(data = data , x = 'BusinessTravel', hue = 'BusinessTravel',palette="Blues_d")

for p in grafico.patches:
	b=p.get_bbox()
	grafico.annotate("{:.0f}".format(b.y1 + b.y0), ((b.x0 + b.x1)/2 - 0.03, b.y1 + 15))

plt.show()
barplot('BusinessTravel', False)
grafico = sns.countplot(data = data , x = 'Department', hue = 'Department',palette="Blues_d")

for p in grafico.patches:
	b=p.get_bbox()
	grafico.annotate("{:.0f}".format(b.y1 + b.y0), ((b.x0 + b.x1)/2 - 0.03, b.y1 + 15))

plt.show()
grafico = sns.countplot(data = data , x = 'EnvironmentSatisfaction', hue = 'EnvironmentSatisfaction',palette="Blues_d")

for p in grafico.patches:
	b=p.get_bbox()
	grafico.annotate("{:.0f}".format(b.y1 + b.y0), ((b.x0 + b.x1)/2 - 0.03, b.y1 + 15))

plt.show()
grafico = sns.countplot(data = data , x = 'WorkLifeBalance', hue = 'WorkLifeBalance',palette="Blues_d")

for p in grafico.patches:
	b=p.get_bbox()
	grafico.annotate("{:.0f}".format(b.y1 + b.y0), ((b.x0 + b.x1)/2 - 0.03, b.y1 + 15))

plt.show()
grafico = sns.countplot(data = data , x = 'JobSatisfaction', hue = 'JobSatisfaction',palette="Blues_d")

for p in grafico.patches:
	b=p.get_bbox()
	grafico.annotate("{:.0f}".format(b.y1 + b.y0), ((b.x0 + b.x1)/2 - 0.03, b.y1 + 15))

plt.show()
grafico = sns.countplot(data = data , x = 'JobLevel', hue = 'JobLevel',palette="Blues_d")

for p in grafico.patches:
	b=p.get_bbox()
	grafico.annotate("{:.0f}".format(b.y1 + b.y0), ((b.x0 + b.x1)/2 - 0.03, b.y1 + 15))

plt.show()
grafico = sns.countplot(data = data , x = 'PerformanceRating', hue = 'PerformanceRating', palette="Blues_d")

for p in grafico.patches:
	b=p.get_bbox()
	grafico.annotate("{:.0f}".format(b.y1 + b.y0), ((b.x0 + b.x1)/2 - 0.03, b.y1 + 15))

plt.show()
grafico = sns.countplot(data = data , x = 'RelationshipSatisfaction', hue = 'RelationshipSatisfaction',palette="Blues_d")

for p in grafico.patches:
	b=p.get_bbox()
	grafico.annotate("{:.0f}".format(b.y1 + b.y0), ((b.x0 + b.x1)/2 - 0.03, b.y1 + 15))

plt.show()
barplot('Department', False)
barplot('EnvironmentSatisfaction', False)
barplot('WorkLifeBalance', False)
barplot('JobSatisfaction', False)
barplot('JobLevel', False)
barplot('PerformanceRating', False)
barplot('PerformanceRating', False)
data_income = DataFrame(data, columns = ['MonthlyIncome'])
g = sns.pairplot(data_income, diag_kind="kde", markers="+",
                 plot_kws=dict(s=50, edgecolor="b", linewidth=1),
                 diag_kws=dict(shade=True))
data_income.describe()
# Boxplot

plt.title ('MonthlyIncome')
y = data.MonthlyIncome
x = data.Attrition
fig = sns.boxplot(x,y,palette="Blues_d")
fig.axis(ymin=900, ymax=21999);
# Boxplot
plt.title ('MonthlyIncome')
y = data.MonthlyIncome
x = data.JobLevel
fig = sns.boxplot(x,y,palette="Blues_d")
fig.axis(ymin=900, ymax=21999);
data[(data.Department=='Sales')].MonthlyIncome.describe()
data[(data.Department=='Human Resources')].MonthlyIncome.describe()
data[(data.Department=='Research & Development')].MonthlyIncome.describe()
plot_distribution('MonthlyIncome', False)
barplot('StockOptionLevel', False)

barplot('High_Distance', False)
barplot('Young_Talents', False)
barplot('Women_High_Traveller', False)
barplot('SamePositionforlongtime', False)
barplot('NotSatif', False)
barplot('NotSatisfied_Overtime', False)
barplot('NotSatisfied_LongDistance', False)
barplot('OverTime_Sales', False)
barplot('OverTime_HR', False)
#customer id col
Id_col     = ['EmployeeNumber']
#Target columns
target_col = ["Attrition"]
#categorical columns
cat_cols   = data.nunique()[data.nunique() < 10].keys().tolist()
cat_cols   = [x for x in cat_cols if x not in target_col]
#numerical columns
num_cols   = [x for x in data.columns if x not in cat_cols + target_col + Id_col]
#Binary columns with 2 values
bin_cols   = data.nunique()[data.nunique() == 2].keys().tolist()
#Columns more than 2 values
multi_cols = [i for i in cat_cols if i not in bin_cols]

#Label encoding Binary columns
le = LabelEncoder()
for i in bin_cols :
    data[i] = le.fit_transform(data[i])
    
#Duplicating columns for multi value columns
data = pd.get_dummies(data = data,columns = multi_cols )

#Scaling Numerical columns
std = StandardScaler()
scaled = std.fit_transform(data[num_cols])
scaled = pd.DataFrame(scaled,columns=num_cols)

#dropping original values merging scaled values for numerical columns
df_data_og = data.copy()
data = data.drop(columns = num_cols,axis = 1)
data = data.merge(scaled,left_index=True,right_index=True,how = "left")
data = data.drop(['EmployeeNumber'],axis = 1)
#correlation
correlation = data.corr()
#tick labels
matrix_cols = correlation.columns.tolist()
#convert to array
corr_array  = np.array(correlation)

#Plotting
trace = go.Heatmap(z = corr_array,
                   x = matrix_cols,
                   y = matrix_cols,
                   colorscale='Cividis_r',
                   colorbar   = dict() ,
                  )
layout = go.Layout(dict(title = 'Correlation Matrix for variables',
                        autosize = False,
                        #height  = 1400,
                        #width   = 1600,
                        margin  = dict(r = 0 ,l = 210,
                                       t = 30,b = 210,
                                     ),
                        yaxis   = dict(tickfont = dict(size = 9)),
                        xaxis   = dict(tickfont = dict(size = 9)),
                       )
                  )
fig = go.Figure(data = [trace],layout = layout)
py.iplot(fig)
# Threshold for removing correlated variables
threshold = 0.8

# Absolute value correlation matrix
corr_matrix = data.corr().abs()
corr_matrix.head()

# Upper triangle of correlations
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
upper.head()

# Select columns with correlations above threshold
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

print('There are %d columns to remove :' % (len(to_drop)))

data = data.drop(columns = to_drop)

to_drop
# Def X and Y
y = np.array(data.Attrition.tolist())
data = data.drop('Attrition', 1)
X = data.to_numpy()

# Train_test split
random_state = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = random_state)
#### Defining the Parameters
param_grid = {'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.25],
              'max_depth': [3,4, 5, 6,10],
              'min_child_weight': [1, 5, 7, 10],
              'gamma': [0.1, 0.5, 1, 1.5, 5],
              'min_samples_leaf': [3, 4, 5],
              'subsample': [0.3, 0.5, 0.7],
              'colsample_bytree': [0.6, 0.8, 1.0],
              'n_estimators': [100, 300, 750, 1000]
              }

# Defining the Xtreme Gradiente Boosting Classifier as 'est'
est = xgb.XGBClassifier()

# GridSearchCV model
# gs_cv = RandomizedSearchCV(est, param_distributions=param_grid, n_iter=800, scoring='accuracy', n_jobs=-1, cv=5, verbose=3, random_state=42).fit(X_train, y_train)


# Printing the Best Parameters
#print('Melhores Hiperparâmetros: %r' % gs_cv.best_params_)
#### Informing the parameters
xgb_clf = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                           colsample_bytree=0.6, gamma=1, learning_rate=0.1,
                           max_delta_step=0, max_depth=3, min_samples_leaf=5, min_child_weight=7, missing=None,
                           n_estimators=300, n_jobs=-1, nthread=None,
                           objective='binary:logistic', random_state=0, reg_alpha=0,
                           reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
                           subsample=0.3)

xgb_clf.fit(X_train, y_train)
y_pred = xgb_clf.predict(X_test)
y_score = xgb_clf.predict_proba(X_test)[:,1]
#Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

#Metrics
tp = conf_matrix[1,1]
fn = conf_matrix[1,0]
fp = conf_matrix[0,1]
tn = conf_matrix[0,0]
Accuracy  =  ((tp+tn)/(tp+tn+fp+fn))
Precision =  (tp/(tp+fp))
Recall    =  (tp/(tp+fn))
F1_score  =  (2*(((tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn)))))
show_metrics = pd.DataFrame(data=[[Accuracy , Precision, Recall, F1_score]])
show_metrics = show_metrics.T

conf_matrix
colors = ['#00264d', '#00cc99' , '#ccfff2', '#e6f2ff']
Metrics = go.Bar(x = (show_metrics[0].values), 
                   y = ['Accuracy', 'Precision', 'Recall', 'F1_score'], text = np.round_(show_metrics[0].values,4),
                    textposition = 'auto',
                   orientation = 'h', opacity = 0.8,marker=dict(
            color=colors,
            line=dict(color='#000000',width=1.5)))
fig1 = go.Figure(data = [Metrics])
fig1.show()
#ROC Curve
model_roc_auc = round(roc_auc_score(y_test, y_score) , 3)
fpr, tpr, t = roc_curve(y_test, y_score)
ROC = go.Scatter(x = fpr,y = tpr,
                        name = "Roc : ",
                        line = dict(color = '#00264d',width = 2), fill='tozeroy')
ROC1 = go.Scatter(x = [0,1],y = [0,1],
                        line = dict(color = ('black'),width = 1.5,
                        dash = 'dot'))
fig2 = go.Figure(data = [ROC,ROC1])
fig2.show()
 #feature importance plot TOP 10
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

tmp = pd.DataFrame({'Feature': list(data), 'Feature importance': xgb_clf.feature_importances_})
tmp = tmp.sort_values(by='Feature importance',ascending=False).head(10)
plt.figure(figsize = (10,12))
plt.title('Top 10 - Features importance of the Predictive Model',fontsize=20)
s = sns.barplot(y='Feature',x='Feature importance',data=tmp, orient='h', palette="RdBu"  )
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show()
