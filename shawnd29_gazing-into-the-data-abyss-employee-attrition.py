# Import the necessary modules

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as matplot

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import plotly.figure_factory as ff

import seaborn as sns



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
# Read the analytics csv file and store our dataset into a dataframe called "df"

#os.listdir("../input")

df = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv', index_col=None)
# Check to seeh if there are any missing values in our data set

df.isnull().any()
# Get a quick overview of what we are dealing with in our dataset

df.head()
# Move the reponse variable "Attrition" to the front of the table

front = df['Attrition']

df.drop(labels=['Attrition'], axis=1,inplace = True)

df.insert(0, 'Attrition', front)

df.head()
# The dataset contains 35 columns and 1470 observations

df.shape
df.dtypes
# Looks like about 84% of employees stayed and 16% of employees left. 

# NOTE: When performing cross validation, its important to maintain this turnover ratio

AttritionRate=df.Attrition.value_counts()/len(df)

AttritionRate
# Display the statistical overview of the employees

df.describe()
# Overview of summary (Attrition V.S. Non-Attrition)

Attrition_Summary = df.groupby('Attrition')

Attrition_Summary.mean()
# Reassign target

df.Attrition.replace(to_replace = dict(Yes = 1, No = 0), inplace = True)

attrition = df[(df['Attrition'] != 0)]

no_attrition = df[(df['Attrition'] == 0)]
df.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
# Plotting the KDEplots

f, axes = plt.subplots(3, 3, figsize=(12, 8), 

                       sharex=False, sharey=False)



# Defining our colormap scheme

s = np.linspace(0, 3, 10)

cmap = sns.cubehelix_palette(start=0.0, light=1, as_cmap=True)



# Generate and plot

x = df['Age'].values

y = df['TotalWorkingYears'].values

sns.kdeplot(x, y, cmap=cmap, shade=True, cut=5, ax=axes[0,0])

axes[0,0].set( title = 'Age against Total working years')



cmap = sns.cubehelix_palette(start=0.333333333333, light=1, as_cmap=True)

# Generate and plot

x = df['Age'].values

y = df['DailyRate'].values

sns.kdeplot(x, y, cmap=cmap, shade=True, ax=axes[0,1])

axes[0,1].set( title = 'Age against Daily Rate')



cmap = sns.cubehelix_palette(start=0.666666666667, light=1, as_cmap=True)

# Generate and plot

x = df['YearsInCurrentRole'].values

y = df['Age'].values

sns.kdeplot(x, y, cmap=cmap, shade=True, ax=axes[0,2])

axes[0,2].set( title = 'Years in role against Age')



cmap = sns.cubehelix_palette(start=1.0, light=1, as_cmap=True)

# Generate and plot

x = df['DailyRate'].values

y = df['DistanceFromHome'].values

sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[1,0])

axes[1,0].set( title = 'Daily Rate against DistancefromHome')



cmap = sns.cubehelix_palette(start=1.333333333333, light=1, as_cmap=True)

# Generate and plot

x = df['DailyRate'].values

y = df['JobSatisfaction'].values

sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[1,1])

axes[1,1].set( title = 'Daily Rate against Job satisfaction')



cmap = sns.cubehelix_palette(start=1.666666666667, light=1, as_cmap=True)

# Generate and plot

x = df['YearsAtCompany'].values

y = df['JobSatisfaction'].values

sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[1,2])

axes[1,2].set( title = 'Daily Rate against distance')



cmap = sns.cubehelix_palette(start=2.0, light=1, as_cmap=True)

# Generate and plot

x = df['YearsAtCompany'].values

y = df['DailyRate'].values

sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[2,0])

axes[2,0].set( title = 'Years at company against Daily Rate')



cmap = sns.cubehelix_palette(start=2.333333333333, light=1, as_cmap=True)

# Generate and plot

x = df['RelationshipSatisfaction'].values

y = df['YearsWithCurrManager'].values

sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[2,1])

axes[2,1].set( title = 'Relationship Satisfaction vs years with manager')



cmap = sns.cubehelix_palette(start=2.666666666667, light=1, as_cmap=True)

# Generate and plot

x = df['WorkLifeBalance'].values

y = df['JobSatisfaction'].values

sns.kdeplot(x, y, cmap=cmap, shade=True,  ax=axes[2,2])

axes[2,2].set( title = 'WorklifeBalance against Satisfaction')



f.tight_layout()

def plot_distribution(var_select, bin_size) : 

# Calculate the correlation coefficient between the new variable and the target

    corr = df['Attrition'].corr(df[var_select])

    corr = np.round(corr,3)

    tmp1 = attrition[var_select]

    tmp2 = no_attrition[var_select]

    hist_data = [tmp1, tmp2]

    

    group_labels = ['Yes_attrition', 'No_attrition']

    colors = ['#FFD700', '#7EC0EE']



    fig = ff.create_distplot(hist_data, group_labels, colors = colors, show_hist = True, curve_type='kde', bin_size = bin_size)

    

    fig['layout'].update(title = var_select+' '+'(corr target ='+ str(corr)+')')



    py.iplot(fig, filename = 'Density plot')
def barplot(var_select, x_no_numeric) :

    tmp1 = df[(df['Attrition'] != 0)]

    tmp2 = df[(df['Attrition'] == 0)]

    tmp3 = pd.DataFrame(pd.crosstab(df[var_select],df['Attrition']), )

    tmp3['Attr%'] = tmp3[1] / (tmp3[1] + tmp3[0]) * 100

    if x_no_numeric == True  : 

        tmp3 = tmp3.sort_values(1, ascending = False)



    color=['lightskyblue','gold' ]

    trace1 = go.Bar(

        x=tmp1[var_select].value_counts().keys().tolist(),

        y=tmp1[var_select].value_counts().values.tolist(),

        name='Yes_Attrition',opacity = 0.8, marker=dict(

        color='gold',

        line=dict(color='#000000',width=1)))



    

    trace2 = go.Bar(

        x=tmp2[var_select].value_counts().keys().tolist(),

        y=tmp2[var_select].value_counts().values.tolist(),

        name='No_Attrition', opacity = 0.8, marker=dict(

        color='lightskyblue',

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
plot_distribution('Age', False)

barplot('Age', False)

plot_distribution('DailyRate', 100)

plot_distribution('DistanceFromHome', False)

barplot('DistanceFromHome', False)

plot_distribution('HourlyRate', False)

plot_distribution('MonthlyIncome', 100)

plot_distribution('MonthlyRate', 100)

plot_distribution('NumCompaniesWorked', False)

barplot('NumCompaniesWorked',False)

plot_distribution('PercentSalaryHike', False)

barplot('PercentSalaryHike', False) 

plot_distribution('TotalWorkingYears', False)

barplot('TotalWorkingYears', False)

plot_distribution('TrainingTimesLastYear', False)

barplot('TrainingTimesLastYear',False)

plot_distribution('YearsAtCompany', False)

barplot('YearsAtCompany', False)

plot_distribution('YearsInCurrentRole', False)

barplot('YearsInCurrentRole', False)

plot_distribution('YearsSinceLastPromotion', False)

barplot('YearsSinceLastPromotion', False)

plot_distribution('YearsWithCurrManager', False)

barplot('YearsWithCurrManager', False)
def plot_pie(var_select) :

    

    colors = ['gold', 'lightgreen', 'lightcoral', 'lightskyblue', 'lightgrey', 'orange', 'white', 'lightpink']

    trace1 = go.Pie(values  = attrition[var_select].value_counts().values.tolist(),

                    labels  = attrition[var_select].value_counts().keys().tolist(),

                    textfont=dict(size=15), opacity = 0.8,

                    hoverinfo = "label+percent+name",

                    domain  = dict(x = [0,.48]),

                    name    = "attrition employes",

                    marker  = dict(colors = colors, line = dict(width = 1.5)))

    trace2 = go.Pie(values  = no_attrition[var_select].value_counts().values.tolist(),

                    labels  = no_attrition[var_select].value_counts().keys().tolist(),

                    textfont=dict(size=15), opacity = 0.8,

                    hoverinfo = "label+percent+name",

                    marker  = dict(colors = colors, line = dict(width = 1.5)),

                    domain  = dict(x = [.52,1]),

                    name    = "Non attrition employes" )



    layout = go.Layout(dict(title = var_select + " distribution in employes attrition ",

                            annotations = [dict(text = "Yes_attrition",

                                                font = dict(size = 13),

                                                showarrow = False,

                                                x = .22, y = -0.1),

                                            dict(text = "No_attrition",

                                                font = dict(size = 13),

                                                showarrow = False,

                                                x = .8,y = -.1)]))

                                          



    fig  = go.Figure(data = [trace1,trace2],layout = layout)

    py.iplot(fig)
plot_pie("Gender")

barplot('Gender',True)

plot_pie('OverTime')

barplot('OverTime',True)

plot_pie('BusinessTravel')

barplot('BusinessTravel',True)

plot_pie('JobRole')

barplot('JobRole',True)

plot_pie('Department') 

barplot('Department',True)

plot_pie('MaritalStatus') 

barplot('MaritalStatus',True)

plot_pie('EducationField') 

barplot('EducationField',True)

plot_pie('Education') 

barplot('Education',False)

plot_pie('EnvironmentSatisfaction')

barplot('EnvironmentSatisfaction',False)

plot_pie('JobInvolvement')

barplot('JobInvolvement', False)

plot_pie('JobLevel')

barplot('JobLevel',False)

plot_pie('JobSatisfaction')

barplot('JobSatisfaction',False)

plot_pie('PerformanceRating')

barplot('PerformanceRating',False)

plot_pie('RelationshipSatisfaction')

barplot('RelationshipSatisfaction', False)

plot_pie('StockOptionLevel')

barplot('StockOptionLevel', False)

plot_pie('WorkLifeBalance')

barplot('WorkLifeBalance', False)
df_categorical = df[['Attrition', 'BusinessTravel','Department',

                       'EducationField','Gender','JobRole',

                       'MaritalStatus',

                       'OverTime']].copy()

df_categorical.head()
fig, axes = plt.subplots(round(len(df_categorical.columns) / 3), 3, figsize=(12, 20))



for i, ax in enumerate(fig.axes):

    if i < len(df_categorical.columns):

        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)

        sns.countplot(x=df_categorical.columns[i], alpha=0.7, data=df_categorical, ax=ax)



fig.tight_layout()
# We create dummies for the remaining categorical variables



df_categorical = pd.get_dummies(df_categorical)

df_categorical.head()
#Correlation Matrix

corr = df.corr()

corr = (corr)

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)



corr
#After removing the strongly correlated variables

df_numerical = df[['Age','DailyRate','DistanceFromHome','Education',

                       'EnvironmentSatisfaction', 'HourlyRate',                     

                       'JobInvolvement', 'JobLevel','MonthlyRate',

                       'JobSatisfaction',

                       'RelationshipSatisfaction', 

                       'StockOptionLevel',

                        'TrainingTimesLastYear','WorkLifeBalance']].copy()

df_numerical.head()
#Standardizing the numerical values

df_numerical = abs(df_numerical - df_numerical.mean())/df_numerical.std()  

df_numerical.head()
final_df = pd.concat([df_numerical,df_categorical], axis= 1)

final_df.head()
X = final_df.drop(['Attrition'],axis= 1)

y = final_df["Attrition"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.4,random_state = 4)
lr = LogisticRegression(solver = 'liblinear',random_state = 0) #Since this a small dataset, we use liblinear solver and Regularization strength as

# default i.e C = 1.0

lr.fit(X_train,y_train)
y_pred_lr = lr.predict(X_test)

accuracy_score_lr = accuracy_score(y_pred_lr,y_test)

accuracy_score_lr 

#Logistic Regression shows 85.7 percent accuracy
dtree = DecisionTreeClassifier(criterion='entropy',max_depth = 4,random_state = 0)

dtree.fit(X_train,y_train)
y_pred_dtree = dtree.predict(X_test)

accuracy_score_dtree = accuracy_score(y_pred_dtree,y_test)

accuracy_score_dtree
rf = RandomForestClassifier(criterion = 'gini',random_state = 0)

rf.fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)

accuracy_score_rf = accuracy_score(y_pred_rf,y_test)

accuracy_score_rf
sv = svm.SVC(kernel= 'linear',gamma =2)

sv.fit(X_train,y_train)
y_pred_svm = sv.predict(X_test)

accuracy_score_svm = accuracy_score(y_pred_svm,y_test)

accuracy_score_svm
knn = KNeighborsClassifier(n_neighbors = 2)

knn.fit(X_train,y_train)
y_pred_knn = knn.predict(X_test)

accuracy_score_knn = accuracy_score(y_pred_knn,y_test)

accuracy_score_knn
scores = [accuracy_score_lr,accuracy_score_dtree,accuracy_score_rf,accuracy_score_svm,accuracy_score_knn]

scores = [i*100 for i in scores]

algorithm  = ['Logistic Regression','Decision Tree','Random Forest','SVM', 'K-Means']

index = np.arange(len(algorithm))

plt.bar(index, scores)

plt.xlabel('Algorithm', fontsize=10)

plt.ylabel('Accuracy Score', fontsize=5)

plt.xticks(index, algorithm, fontsize=10, rotation=30)

plt.title('Accuracy scores for each classification algorithm')

plt.ylim(80,100)

plt.show()    
feat_importances = pd.Series(rf.feature_importances_, index=X.columns)

feat_importances= feat_importances.nlargest(5)

feat_importances.plot(kind='barh')

plt.show()