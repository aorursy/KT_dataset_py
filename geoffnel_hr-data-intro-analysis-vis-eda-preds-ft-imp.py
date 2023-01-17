#whyempterminated.show()
py.offline.iplot(whyempterminated)
#org_position_proportion.show()
py.offline.iplot(org_position_proportion)
# import main libs
import os 
import pandas as pd
import numpy as np
from pandas import DataFrame as DFM
import warnings
warnings.filterwarnings('ignore')
# import viz libs
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import matplotlib.style
import matplotlib as mpl
mpl.style.use('bmh')
mpl.rcParams['figure.figsize'] = [12.0, 6.0]
mpl.rcParams['figure.dpi'] = 96
plotCOLOR = 'slategrey'
mpl.rcParams['text.color'] = plotCOLOR
mpl.rcParams['axes.labelcolor'] = plotCOLOR
mpl.rcParams['xtick.color'] = plotCOLOR
mpl.rcParams['ytick.color'] = plotCOLOR
mpl.rcParams['axes.labelsize'] = 15
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.express as px
import plotly.graph_objects as go

color1 = px.colors.qualitative.Prism[0]
color2 = px.colors.qualitative.Prism[7]
color3 = px.colors.qualitative.Prism[10]
color4 = px.colors.qualitative.Set1[8]

init_notebook_mode(connected=True) 
# enlarge the notebook
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; } </style>"))
#------------------------------------------------------------------------
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
# import file into a dataframe
filedir = '/kaggle/input/hr-data-set-based-on-human-resources-data-set/HR DATA.txt'
rawdata = pd.read_csv(filedir, sep='\t', parse_dates=True)
def dispdfm(x):
    x = display(pd.DataFrame(x))
    return x
def plotol(x):
    py.offline.iplot(x)
# print initial rows and cols of dataset
dispdfm(rawdata.sample(3))
# Review basic information on the dataset
rawdata.info()
# WORKOUT NULL PERCENTAGES
cols = ['Attr', 'Len', 'Nullcnt', 'Nullpct']
data = []
for i in rawdata.columns:
    attr        = i
    length      = int(len(rawdata[i]))
    attrnulls   = rawdata[i].isna().sum()
    attrnullpct = round((rawdata[i].isna().sum())/len(rawdata[i])*100,2)
    values = [
      attr, 
      length,
      attrnulls,
      attrnullpct
    ]
    zipped = zip(cols, values)
    nulldict = dict(zipped)
    data.append(nulldict)

#     print(values)
#     nulldf.append(pd.Series(data), ignore_index=True)

nulldf = pd.DataFrame(columns=cols).append(data)
display(nulldf[nulldf['Nullpct'] != 0].sort_values('Nullpct', ascending=False))
# Graph missing values
plotdf = nulldf[nulldf['Nullpct'] != 0].sort_values('Nullpct', ascending=False)
fig = px.histogram(
    plotdf,
    y='Nullpct',
    x='Attr',
    color='Nullpct',
    color_discrete_sequence=px.colors.qualitative.Prism,
    title='Null Values'
)

plotol(fig)
# Reasons for leaving based on the employmentstatus column
plotdf = DFM(rawdata[(rawdata['EmploymentStatus'] == 'Terminated for Cause') | 
        (rawdata['EmploymentStatus'] == 'Voluntarily Terminated')]['TermReason'].value_counts())
fig, axs = plt.subplots(figsize=(2, 6))
#------------------------------------------------------------------------
fig = sns.heatmap(
    plotdf,
    cbar=True,
    #square= True,
    fmt='.0f',
    annot=True,
    annot_kws={
        'size': 10,
        'rotation': 0
    },
    cmap='Purples',
    ax=axs)
whyemployeesleave = fig
#------------------------------------------------------------------------
axs.set(ylabel="Reasons ",xlabel="Count")
plt.title('Chart 1 - Reasons for Leaving \n')
plt.tight_layout()
# Reasons for leaving based on the termd attribute
plotdf = DFM(rawdata[rawdata['Termd'] == 1]['TermReason'].value_counts())
fig, axs = plt.subplots(figsize=(2, 6))
#------------------------------------------------------------------------
fig = sns.heatmap(
    plotdf,
    cbar=True,
    #square= True,
    fmt='.0f',
    annot=True,
    annot_kws={
        'size': 10,
        'rotation': 0
    },
    cmap='Purples',
    ax=axs)
#------------------------------------------------------------------------
axs.set(ylabel="Reasons ",xlabel="Count")
plt.title('Chart 2 - Reasons for Leaving \n')
plt.tight_layout()
termcolrev = rawdata[(rawdata['Termd'] == 1)
        & ~(rawdata['TermReason'] == 'N/A - still employed')
        & (rawdata['DateofTermination'].isnull()
           )][[#'TermReason','DateofTermination',
               'Termd', 'EmploymentStatus']]#['EmploymentStatus'].value_counts()

dispdfm(termcolrev.tail(5))
from datetime import datetime
now = datetime.now()
# Setting a list of categorial features/attributes
catcols = []
for x in rawdata.columns:
    if rawdata[x].dtype == 'object':
        #print("%s %s %s " % (x,'\t', rawdata[x].dtype))
        rawdata[x] = rawdata[x].astype('category')
        #print("%s %s %s " % (x,'\t', rawdata[x].dtype))
        #display(pd.DataFrame(rawdata[x].value_counts()))
        #print('\n'*2)
        catcols.append(x)
print("Each attribute has the following amount of unique values:  ")
for i, v in enumerate(catcols):
    print('  ',str(i).ljust(2) , v.ljust(28),"Values:", len(set(rawdata[v])))
#print(catcols, sep='\t')
# Replacing Null Values
rawdata['TermReason'] = rawdata['TermReason'].cat.add_categories('Unknown')
rawdata['TermReason'].fillna('Unknown', inplace=True)
rawdata['ManagerName'] = rawdata['ManagerName'].cat.add_categories('Unknown')
rawdata['ManagerName'].fillna('Unknown', inplace=True)
# Transform dates
dates = ['DOB', 'DateofHire', 'DateofTermination', 'LastPerformanceReview_Date']
display(rawdata[dates].sample(3),
        rawdata[dates].dtypes)
# Transform date types
for i in dates:
    rawdata[i] = pd.to_datetime(rawdata[i], 
                                errors = 'coerce',
                                infer_datetime_format=False,
                                format='%d-%m-%y'
                               )
# setting dob attribute
currentyear = now.year
currentyear
rawdata['doby'] = ''
#------------------------------------------------------------------------
for idx, date in enumerate(rawdata['DOB']):
#     print(type(date))
    if date.year >= currentyear:
        rawdata['doby'][idx] = rawdata['DOB'][idx] - pd.DateOffset(years=100)
        #print('greater',idx, date)
    else:
        rawdata['doby'][idx] = rawdata['DOB'][idx]
        #print('normal',idx, date)
#------------------------------------------------------------------------
print('Before:', rawdata['DOB'][138])        
rawdata['DOB'] = pd.to_datetime(rawdata['doby'], format='%y-%m-%d')
print('After:', rawdata['DOB'][138])
#------------------------------------------------------------------------
dupnames = ['Warner, Larissa', 'Young, Darien']
dispdfm(rawdata[rawdata['Employee_Name'].isin(dupnames)][['Employee_Name', 'EmpID','DOB', ]])
# Creating a binary churn column
rawdata['Churn'] = np.where(rawdata['DateofTermination'].notna() == True,1,0)
#------------------------------------------------------------------------
# Cross checking churn against Employment Status
pd.crosstab(rawdata['Churn'], rawdata['EmploymentStatus'])
#------------------------------------------------------------------------
# Creating an AGE column
rawdata['Age'] = (now.year - rawdata['DOB'].dt.year)
dispdfm(rawdata[['Churn', 'Age']].sample(3))
rawdata['Churn-Yes/No'] = rawdata['Churn'].astype('category')
rawdata['Churn-Yes/No'].replace({1:'Yes', 0:'No'}, inplace=True)
# Additional dropped columns
rawdata.drop(['Employee_Name', 'EmpID',], inplace=True, axis=1)
rawdata.drop(['Termd'], inplace=True, axis=1)
# Dropping certain columns with high null count
rawdata.drop(['doby'], inplace=True, axis=1)
rawdata.drop(['Zip', 'LastPerformanceReview_Date','DaysLateLast30'], inplace=True, axis=1)
# Cross checking churn against Employment Status - BEFORE
dispdfm(pd.crosstab(rawdata['Churn'], rawdata['EmploymentStatus']))
#------------------------------------------------------------------------
# Dropping employees that have not started working yet
rawdata.drop(rawdata[rawdata['EmploymentStatus']=='Future Start'].index, axis=0, inplace=True)
#
rawdata_copy1 = rawdata.copy()
dropcols = ['DateofTermination','ManagerID', 'DOB']
rawdata_copy1['Churn-Yes/No'] = rawdata_copy1['Churn-Yes/No'].astype('category')
rawdata_copy1.drop(dropcols,
              axis=1, inplace=True)
first_col = rawdata_copy1.pop('Churn')
rawdata_copy1.insert(len(rawdata_copy1.columns), 'Churn', first_col)
# Imbalance correction - Synthetic Minority Oversampling 
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.model_selection import train_test_split
# Resample the minority class. You can change the strategy to 'auto' if you are not sure.
test = DFM([rawdata_copy1.dtypes == 'category']).T.reset_index()
catlistsmotenc = list(test[test[0]==True]['index'])
smnc = SMOTENC(categorical_features=[rawdata_copy1.dtypes == 'category'],
               sampling_strategy='minority',
               random_state=10)
' - >  -  -  -  -  -  -  -  -  -  -  -  -  < - '
X, y = rawdata_copy1.drop('Churn', axis=1), rawdata_copy1['Churn']
' - >  -  -  -  -  -  -  -  -  -  -  -  -  < - '
# run the train/test split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    stratify=y)
' - >  -  -  -  -  -  -  -  -  -  -  -  -  < - '
# Fit the model to generate the data.
oversampled_trainX, oversampled_trainY = smnc.fit_sample(X_train, y_train)
' - >  -  -  -  -  -  -  -  -  -  -  -  -  < - '
os_rawdata = pd.concat(
    [pd.DataFrame(oversampled_trainX),
    pd.DataFrame(oversampled_trainY)],
    axis=1)
' - >  -  -  -  -  -  -  -  -  -  -  -  -  < - '
#os_modeldata.columns = modeldata.columns
os_rawdata.columns = rawdata_copy1.copy().columns
# create modeldata dataframe
modeldata = os_rawdata.copy()
#------------------------------------------------------------------------
# drop na values
dropNaValCols = [
    'ManagerID'
]
#------------------------------------------------------------------------
for col in dropNaValCols:
    if col in modeldata.columns:
        print('Dropped NA Values:', col)
        modeldata.dropna(subset=dropNaValCols, inplace=True)
# any additional cols that need to be dropped
modelcolstodrop = [
    'EmpStatusID',
    'EmploymentStatus',
    'TermReason',
    'ManagerID',
    'PositionID',
    'ManagerID',
    'DOB',
    'DateofHire',
    'DateofTermination',
    'DeptID',
    'PerfScoreID',
    'Churn-Yes/No',
    'GenderID',
]

for col in modelcolstodrop:
    if col in modeldata.columns:
        print('Dropped Column:', col)
        modeldata.drop(col, axis=1, inplace=True)
#print(modeldata.columns)
# Setting a list of categorial features/attributes
modelcatcols = []
for x in modeldata.columns:
    if str(modeldata[x].dtype) == 'category':
        #modeldata[x].astype('category')
        modelcatcols.append(x)
for i, v in enumerate(modelcatcols):
    print(i, v)
# Apply label encoder to each column with categorical data
from sklearn.preprocessing import LabelEncoder
# Apply label encoder to each column with categorical data
label_encoder = LabelEncoder()
#print(' - - Encoded Values - - ')
for col in modelcatcols:
    print("Encoded: ", col)
    modeldata[col] = label_encoder.fit_transform(modeldata[col])
# Review Correlation of covariates/explanatory variables
cordf = modeldata
cordf = cordf.corr()
#------------------------------------------------------------------------
fig, axs = plt.subplots(figsize=(10,7))
cordf_subset = pd.DataFrame(cordf).drop('Churn', axis=0)
cordf_subset = cordf_subset.drop('Churn', axis=1)
mask = np.triu(np.ones_like(cordf_subset, dtype=np.bool))
#------------------------------------------------------------------------
fig = sns.heatmap(
    cordf_subset,
    cbar=False,
    square= True,
    fmt='.2f',
    annot=True,
        annot_kws={
        'size': 5},
    cmap='Purples',
    mask=mask,
    #vmin=-1, vmax=1, center= 0,
    ax=axs)

plt.title('Correlations - Final Set of Model Data \n',
         size=20)
plt.tight_layout()
# Review Correlation of covariates/explanatory variables
cordf = modeldata
cordf = cordf.corr()
cordf_subset = pd.DataFrame(cordf.loc[:, 'Churn']).sort_values('Churn')
cordf_subset.drop('Churn', axis=0, inplace=True)
#------------------------------------------------------------------------
fig, axs = plt.subplots(figsize=(1, 6))
fig = sns.heatmap(
    cordf_subset,
    cbar=True,
    #square= True,
    fmt='.2f',
    annot=True,
    annot_kws={
        'size': 8,
        'rotation': 0,
        #'color': 'slategrey'
    },
    cmap='Purples',
    ax=axs)
plt.title('Correlations - Explanitory v Response \n')
plt.tight_layout()
# 
plotdata = pd.pivot_table(
	rawdata,
    values = ['PayRate'],
    index = 'Position',
    aggfunc = np.median
).sort_values('PayRate')

fig, axs = plt.subplots(figsize=(1, 12))
fig = sns.heatmap(
    plotdata,
    cbar=True,
    #square= True,
    fmt='.2f',
    annot=True,
    annot_kws={
        'size': 8,
        'rotation': 0,
        #'color': 'slategrey'
    },
    cmap='Purples',
    ax=axs)

plt.title('Payrate (Median) - by Position \n')
plt.tight_layout()
#
plotdata = pd.pivot_table(rawdata,
                          values=['PayRate'],
                          index='PerformanceScore',
                          aggfunc=np.median)

fig, (axs1) = plt.subplots(1, figsize=(1, 6), dpi=96)
g1 = sns.heatmap(
    plotdata,
    cbar=True,
    #square= True,
    fmt='.2f',
    annot=True,
    annot_kws={
        'size': 8,
        'rotation': 0,
        #'color': 'slategrey'
    },
    cmap='Purples',
    ax=axs1)
plt.title('Payrate (Median) - by PerformanceScore \n')
plt.tight_layout()
#
plotdata = pd.pivot_table(rawdata,
                          values=['PayRate'],
                          index='Department',
                          aggfunc=np.median)

fig, axs1 = plt.subplots(figsize=(1, 6),dpi=96)
g2 = sns.heatmap(
    plotdata,
    cbar=True,
    #square= True,
    fmt='.2f',
    annot=True,
    annot_kws={
        'size': 8,
        'rotation': 0,
        #'color': 'slategrey'
    },
    cmap='Purples',
    ax=axs1)

plt.title('Payrate (Median) - by Department \n')

plt.tight_layout()
# correlation matrix of initial numerical values
cols = [
    'FromDiversityJobFairID', 'PayRate', 'EmpSatisfaction',
    'SpecialProjectsCount', 'Age', 'Churn'
]
#------------------------------------------------------------------------
cordf = rawdata[cols]
cordf = cordf.corr()
#------------------------------------------------------------------------
mask = np.triu(np.ones_like(cordf, dtype=np.bool))
#------------------------------------------------------------------------
plt.figure(figsize=(8, 8))
sns.heatmap(cordf,
            cbar=True,
            square=True,
            mask=mask,
            fmt='.1f',
            annot=True,
            annot_kws={'size': 15},
            cmap='Purples')
plt.title('Correlation of Initial Numerical Values')
plt.tight_layout()
# 
print(
    'Meta Stats:')
display(rawdata.describe().T)
dispdfm(rawdata.var().sort_values())
# why do employees get terminated
termreason = rawdata[rawdata['TermReason'] != 'N/A - still employed' ]
termreason = termreason.groupby('TermReason')['Churn'].count().reset_index()
termreason.sort_values('Churn', ascending=False, inplace=True)

whyempterminated = px.histogram(
    termreason.iloc[:10],
    x='TermReason',
    y='Churn',
    color='TermReason',
    opacity=0.8,
    color_discrete_sequence=px.colors.qualitative.Prism,
    #color_discrete_sequence=px.colors.diverging.PuOr,
    #color_discrete_sequence=px.colors.sequential.Plasma,
    title='Why do employees Churn? (Top 10)',
)  #.update_xaxes(        categoryorder='total descending')

plotol(whyempterminated)

# Plotting currently 'active' employee position numbers - top 5 by percent of staff
plotdata = rawdata[rawdata['EmploymentStatus'] == 'Active']['Position'].value_counts(ascending=False)
plotdata = plotdata.reset_index().rename(columns={'index':'Position', 'Position': 'Position Count'})
#plotdata = rawdata[rawdata['EmploymentStatus'] == 'Active']['Position'].value_counts(ascending=False).reset_index()
fig = px.bar(plotdata,
             x='Position',
             y='Position Count',
             color='Position',
             opacity=0.8,
             #histfunc='sum',
             title='What is the company makeup by position?',
             color_discrete_sequence=px.colors.qualitative.Prism
            )#.update_xaxes(        categoryorder='total descending')

plotol(fig)
# Proportion of Organization by Positions Filled
plotdata = rawdata['Position'].value_counts().reset_index()
plotdata['Pct'] = ((plotdata['Position'] / plotdata['Position'].sum()) *
                   100).round(2)
fig = px.treemap(
    plotdata,
    values='Position',
    path=['Pct', 'index'],
    color_discrete_sequence=px.colors.qualitative.Prism,
    title='What Percent of Organization by Position Groups'
)

org_position_proportion = fig

#fig.data[0].textinfo = 'label+text+value+current path'
plotol(fig)
# Department
plotdata = rawdata[rawdata['EmploymentStatus'] == 'Active']['Department'].value_counts(ascending=False).reset_index()
px.pie(plotdata.loc[:4], values='Department', names='index', opacity=0.8,
       title='Proportion of Top 5 by Department',
       color_discrete_sequence=px.colors.qualitative.Prism, hole=.5)
# plot the state 
plotdata = rawdata[rawdata['State']!='MA']['State'].value_counts()
plotdata = plotdata.iloc[:9]
px.bar(plotdata, color=plotdata.index, 
       color_discrete_sequence=px.colors.qualitative.Prism, 
       opacity=0.8,
       title='Top 10 Hiring States (Not Including MA)')

#
plotdata = rawdata['State'].value_counts().reset_index()
plotdata['Pct'] = ((plotdata['State'] / plotdata['State'].sum()) *
                   100).round(2)

px.treemap(plotdata,
           path=['index', 'Pct'],
           values='State',
           title='Top 10 Hiring States (Does Include MA)')
plotdata = rawdata.groupby('ManagerName')['Position'].count().reset_index()
plotdata.sort_values('Position', ascending=False, inplace=True)
px.histogram(plotdata.iloc[0:20], x='ManagerName', y='Position', color='ManagerName',
       color_discrete_sequence=px.colors.qualitative.Prism,
       opacity=0.8,
       title='Top 20 Managers by Number of People Managed')
#fig.data[0].textinfo = 'label+text+value+current path'
# Percentages
dispdfm((rawdata['Churn-Yes/No'].value_counts(normalize=True)*100).round(2))
#
plotdata = rawdata.groupby('Churn-Yes/No')['Churn'].count()
px.bar(
    plotdata.reset_index(),
    x='Churn-Yes/No',
    y='Churn',
    title='Number of Churn v No Churn',
    color='Churn-Yes/No',
    color_discrete_sequence=[color4, color1],
    opacity=0.8
)


#
plotdata = os_rawdata.groupby('Churn-Yes/No')['Churn'].count()
px.bar(
    plotdata.reset_index(),
    x='Churn-Yes/No',
    y='Churn',
    title='Number of Churn v No Churn - Oversampled',
    color='Churn-Yes/No',
    color_discrete_sequence=[color4, color1],
    opacity=0.8
)


# 
plotdata = os_rawdata[os_rawdata['State'] != 'MA']

px.bar(plotdata,
             x='State',
             color='Churn-Yes/No',
             color_discrete_sequence=[color4, color1],
             barmode='stack',
             title='Churn by State (Not in MA)',
             opacity=0.8).update_xaxes(categoryorder='total descending')
#
plotdata = os_rawdata.copy()

bin_labels = ['0-24 ', '25-29', '30-34', '35-39', '40-44','45-49', '50-54', '55+']
plotdata['Ages'] = pd.cut(plotdata['Age'],
                              bins=[0, 24,30, 34,40, 44,50, 54, 100],
                              labels=bin_labels)

px.histogram(plotdata,
             y="Ages",
             color='Churn-Yes/No',
             title='Churn by Age Group',
             barnorm  ='percent', 
    color_discrete_sequence=[color4, color1],
    opacity=0.8)\
    .update_xaxes(title='Percent')\
    #.update_yaxes(categoryorder='total descending')
#
px.histogram(os_rawdata,
             y='Department',
             color='Churn-Yes/No',
             title='Churn by Department',
             barmode='stack' ,
             barnorm  ='percent',
             color_discrete_sequence=[color4, color1],
             opacity=0.8)\
    .update_xaxes(title='Percent')\
    .update_yaxes(categoryorder='total descending')
#
px.histogram(os_rawdata,
             y='Position',
             color='Churn-Yes/No',
             title='Churn by Position',
             barnorm  ='percent',
             height=750,
             color_discrete_sequence=[color4, color1],
             opacity=0.8)\
    .update_xaxes(title='Percent')\
    .update_yaxes(categoryorder='total descending')
px.histogram(os_rawdata,
             y='PayRate',
             color='Churn-Yes/No',
             title='Churn by PayRate (Ungrouped)',
             barmode='stack',
             barnorm='percent',
             height=750,
             color_discrete_sequence=[color4, color1],
             opacity=0.8)\
    .update_xaxes(title='Percent')
#
listed = []
for idx, val in enumerate(range(15, 85, 5)):
    #for idx2, val2 in enumerate(range(15,90,5)):
    #print(str(val)+","+str(val+4))
    listed.append(str(val) + "-" + str(val + 4))

    bin_labels = [
        '15-19', '0-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54',
        '55-59', '60-64', '65-69', '70-74', '75-79', '80+'
    ]

    bin_values = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 100]
plotdata = rawdata.copy()
plotdata['Pay Range'] = pd.cut(plotdata['PayRate'],
                               bins=bin_values,
                               labels=bin_labels)

px.histogram(plotdata,
             y='Pay Range',
             color='Churn-Yes/No',
             title='Churn by PayRate (Grouped)',
             barmode='stack',
             barnorm='percent',
             color_discrete_sequence=[color4, color1],
             opacity=0.8)\
    .update_xaxes(title='Percent')\
    .update_yaxes(categoryorder='max descending')
#
px.histogram(os_rawdata,
             y='ManagerName',
             color='Churn-Yes/No',
             title='Churn by Manager',
             
             barnorm='percent',
             color_discrete_sequence=[color4, color1],
             opacity=0.8)\
    .update_xaxes(title='Percent')\
    .update_yaxes(categoryorder='total descending')
#### Model Libs
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import LogisticRegression
from sklearn.pipeline        import Pipeline
from lightgbm                import LGBMClassifier
# IMPORTING MODEL METRICS
from sklearn.metrics         import confusion_matrix, roc_auc_score
print("Reviewing summary info: \n")
modeldata.info()
print("Reviewing null values:")
modeldata.isna().sum()
logpipe = Pipeline([
    ('normalizer', StandardScaler()),  #Step1 - normalize data
    ('clf', LogisticRegression(solver='sag'))  #step2 - classifier
])
#------------------------------------------------------------------------
lgbmpipe = Pipeline([
    ('normalizer', StandardScaler()),  #Step1 - normalize data
    ('clf', LGBMClassifier(eval_metric='auc'))  #step2 - classifier
])
# Set train test function and initiate train and test data
from sklearn.model_selection import train_test_split
# setting x and y dataset
X, y = modeldata.drop('Churn', axis=1), modeldata['Churn']

#------------------------------------------------------------------------
# run the train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
#------------------------------------------------------------------------
lenx_train = len(X_train)
leny_train = len(y_train)
lenx_test = len(X_test)
leny_test = len(y_test)
print('Sanity check on train/test:')
print('\n Train x length: ', lenx_train,
      '\n Train y length: ',leny_train,
      '\n Test x  length: ',lenx_test,
      '\n Test y  length: ',leny_test, '\n',
      '\n Split pct     : ',round((lenx_train / (lenx_test + lenx_train))*100, 2))
#print(X_train.columns, sep='\n')
# Fitting pipeline and predictions
logfitted = logpipe.fit(X_train, y_train)
lgbmfitted = lgbmpipe.fit(X_train, y_train)
#------------------------------------------------------------------------
y_pred_vals_log = logfitted.predict(X_test)
y_pred_probs_log = logfitted.predict_proba(X_test)[:, 1]
#------------------------------------------------------------------------
y_pred_vals_lgbm = lgbmfitted.predict(X_test)
y_pred_probs_lgbm = lgbmfitted.predict_proba(X_test)[:, 1]
# collecting the auc score
log_roc_auc      = roc_auc_score(y_test, y_pred_probs_log)
log_roc_auc_pct  = (log_roc_auc*100).round(1)
lgbm_roc_auc     = roc_auc_score(y_test, y_pred_probs_lgbm)
lgbm_roc_auc_pct = (lgbm_roc_auc*100).round(1)
#------------------------------------------------------------------------
# creating confusion matrices for each model
log_cnfmtx = confusion_matrix(y_test, y_pred_vals_log)
#------------------------------------------------------------------------
lgbm_cnfmtx = confusion_matrix(y_test, y_pred_vals_lgbm)
fig, (axs1, axs2) = plt.subplots(1,2, figsize=(10, 5))
g1 = sns.heatmap(
    log_cnfmtx,
    cbar=False,
    square= True,
    fmt='.0f',
    annot=True,
    #annot_kws={'size': 15},
    cmap='Purples', 
    ax=axs1)
g1.set_title('Confusion Matrix - LogReg \n AUC:{}% \n'.format(log_roc_auc_pct))

g2 = sns.heatmap(
    lgbm_cnfmtx,
    cbar=False,
    
    square= True,
    fmt='.0f',
    annot=True,
    #annot_kws={'size': 15},
    cmap='Purples',
    ax=axs2)
g2.set_title('Confusion Matrix - Light GBM \n AUC:{}% \n'.format(lgbm_roc_auc_pct))

plt.show()
import shap
shap.initjs()
# lgbm features
print('Light GBM Model - Attributes by Importance')
shap_values_lgbm = shap.TreeExplainer(lgbmfitted[1]).shap_values(X_test)
shap.summary_plot(shap_values_lgbm[1], X_test, plot_type="bar")
# logreg features
print('Log Reg Model - Attributes by Importance')
shap_values_log = shap.LinearExplainer(logfitted[1], X_test).shap_values(X_test)
shap.summary_plot(shap_values_log, X_test, plot_type="bar")
