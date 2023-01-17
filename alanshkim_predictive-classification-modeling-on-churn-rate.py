import numpy as np

import pandas as pd



import os



import matplotlib.pyplot as plt#visualization

import seaborn as sns#visualization

%matplotlib inline



from PIL import  Image



import itertools

import warnings

warnings.filterwarnings("ignore")



import io

import plotly.offline as py #visualization

py.init_notebook_mode(connected=True) #visualization

import plotly.graph_objs as go #visualization

import plotly.tools as tls #visualization

import plotly.figure_factory as ff #visualization

from plotly.subplots import make_subplots #visualization



pd.set_option('display.max_columns', 100)
df = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head()
print(df.shape)

summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

summary['Missing'] = df.isnull().sum().values    

summary['Uniques'] = df.nunique().values

summary
df.describe()
for column in df.columns:

    if '' in df[column].values or ' ' in df[column].values:

        print(column)
df['TotalCharges'] = df['TotalCharges'].replace('',np.nan)

df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
totalinstances = df.shape[0]

totalnull = df['TotalCharges'].isnull().sum()

print(round((totalnull/totalinstances*100),2),'%')
df.dropna(axis=0, inplace = True)
df['TotalCharges'] = df['TotalCharges'].astype('float')
df["SeniorCitizen"] = df["SeniorCitizen"].replace({1:"Yes",0:"No"})
for column in df.columns:

    print(column +':','\n',np.unique(df[column])[:5],'\n')
nointernet = []

for column in df.columns:

    if "No internet service" in df[column].unique():

        nointernet.append(column)



for col in nointernet : 

    df[col]  = df[col].replace('No internet service', 'No')
label = list(df['Churn'].unique())

value = df['Churn'].value_counts()

value_percent = list(round(value/df.shape[0],2))
t1 = go.Bar(

    x=label,

    y=value_percent,

    width = [0.5, 0.5],

    marker=dict(

        color=['green', 'blue'])

    )





layout = go.Layout(dict(

    title='Overall Customer Churn Rate',

    plot_bgcolor  = "rgb(243,243,243)",

    paper_bgcolor = "rgb(243,243,243)",

    xaxis = dict(

        gridcolor = 'rgb(255, 255, 255)',

        title = "Churn",

        zerolinewidth=1,

        ticklen=5,

        gridwidth=2

        ),

    yaxis = dict(

        gridcolor = 'rgb(255, 255, 255)',

        title = "Percent",

        zerolinewidth=1,

        ticklen=5,

        gridwidth=2

        ),

    )

)



fig = go.Figure(data=t1, layout=layout)

fig.update_layout(title_x=0.5)

py.iplot(fig)





# Separating the churn rates for comparison of various categorical features.

churn = df[df['Churn'] == 'Yes']

retention = df[df['Churn'] == 'No']
# Generally, we round up sample sizes when estimating population mean/proportion.



import math



def round_decimals_up(og_list, decimals:int=2):

    """

    Returns rounded up list to a specific number of decimal places.

    """

    

    rounded_list = []

    if not isinstance(decimals, int):

        raise TypeError("decimal places must be an integer")

    elif decimals < 0:

        raise ValueError("decimal places has to be 0 or more")

    elif decimals == 0:

        return math.ceil(number)



    factor = 10 ** decimals

    

    for number in og_list:

        

        rounded_list.append((math.ceil(number * factor) / factor))

    

    return rounded_list
def barplot_rounded(col):

    

    rounded_churn = round_decimals_up((churn[col].value_counts()/churn.shape[0]),3)

    rounded_retention = round_decimals_up((retention[col].value_counts()/retention.shape[0]),3)

    

    t1 = go.Bar(

        x = list(churn[col].value_counts().keys()),

        y = rounded_churn, 

        name = 'Churn',

        marker_color = 'rgb(55, 83, 109)'

    )

    

    t2 = go.Bar(

        x = list(retention[col].value_counts().keys()),

        y = rounded_retention,

        name = 'Retention',

        marker_color = 'rgb(26, 118, 255)'

    )

    

    data = [t1,t2]

    

    layout = go.Layout(dict(

        title = "Churn Rate by " + col,

        plot_bgcolor  = "rgb(243,243,243)",

        paper_bgcolor = "rgb(243,243,243)",

        xaxis = dict(

            gridcolor = 'rgb(255, 255, 255)',

            title = col,

            zerolinewidth=1,

            ticklen=5,

            gridwidth=2

            ),

        yaxis = dict(

            gridcolor = 'rgb(255, 255, 255)',

            title = "Percent",

            zerolinewidth=1,

            ticklen=5,

            gridwidth=2

            ),

        )

    )

    

    fig  = go.Figure(data=data,layout=layout)

    fig.update_layout(title_x=0.5)

    py.iplot(fig)
df.nunique()[df.nunique() <= 4]
categorical = df.nunique()[df.nunique() <= 4].index[:-1]
for col in categorical:

    barplot_rounded(col)
tenure = df.groupby('Churn')['tenure'].value_counts().unstack(0)



def tenure_scatter(df):

    

    t1 = go.Scatter(

            x = list(df.index),

            y = df.loc[:,'Yes'],

            mode = 'markers',

            name = 'Churn',

            marker = dict(

                line = dict(

                    color = "black",

                    width = .5),

                color = 'blue',

                opacity = 0.8

               ),

        )



    t2 = go.Scatter(

            x = list(df.index),

            y = df.loc[:,'No'],

            mode = 'markers',

            name = 'Retention',

            marker = dict(

                line = dict(

                    color = "black",

                    width = .5),

                color = 'green',

                opacity = 0.8

               ),

        )



    layout = go.Layout(dict(

        title='Churn based on Tenure',

        plot_bgcolor  = "rgb(243,243,243)",

        paper_bgcolor = "rgb(243,243,243)",

        xaxis = dict(

            gridcolor = 'rgb(255, 255, 255)',

            title = "Tenure",

            zerolinewidth=1,

            ticklen=5,

            gridwidth=2

            ),

        yaxis = dict(

            gridcolor = 'rgb(255, 255, 255)',

            title = "Churn/Retention",

            zerolinewidth=1,

            ticklen=5,

            gridwidth=2

            ),

        )

    )

    

    data = [t1,t2]



    fig  = go.Figure(data=data,layout=layout)

    fig.update_layout(title_x=0.5)

    py.iplot(fig)



tenure_scatter(tenure)
def bin_tenure(df) :

    

    if df["tenure"] <= 12 :

        return "Tenure 0-12"

    elif (df["tenure"] > 12) & (df["tenure"] <= 24 ):

        return "Tenure 12-24"

    elif (df["tenure"] > 24) & (df["tenure"] <= 48) :

        return "Tenure 24-48"

    elif (df["tenure"] > 48) & (df["tenure"] <= 60) :

        return "Tenure 48-60"

    elif df["tenure"] > 60 :

        return "Tenure 60+"

    

df["bin_tenure"] = df.apply(lambda df:bin_tenure(df),

                                      axis = 1)



tenure_binned = df.groupby('Churn')['bin_tenure'].value_counts().unstack(0)



tenure_scatter(tenure_binned)
# Feature engineering for creating scatterplots on Monthly/Total Charges.



df_copy = df.copy()



df_copy.loc[df_copy.Churn=='No','Churn'] = 0 

df_copy.loc[df_copy.Churn=='Yes','Churn'] = 1



df_copy.head()
print(df_copy['Churn'].dtype)
# Change dtype.

df_copy['Churn'] = df_copy['Churn'].astype(int)
df_mc = df_copy.groupby('MonthlyCharges').Churn.mean().reset_index()
def charges_scatter(df,col,title,x_title):

    

    t1 = go.Scatter(

                x = df[col],

                y = df['Churn'],

                mode = 'markers',

                name = 'Churn',

                marker = dict(

                    line = dict(

                        color = "black",

                        width = .5),

                    color = 'red',

                    opacity = 0.8

                   ),

            )



    layout = go.Layout(dict(

        title = title,

        plot_bgcolor  = "rgb(243,243,243)",

        paper_bgcolor = "rgb(243,243,243)",

        xaxis = dict(

            gridcolor = 'rgb(255, 255, 255)',

            title = x_title,

            zerolinewidth=1,

            ticklen=5,

            gridwidth=2

            ),

        yaxis = dict(

            gridcolor = 'rgb(255, 255, 255)',

            title = "Churn Rate",

            zerolinewidth=1,

            ticklen=5,

            gridwidth=2

            ),

        )

    )



    data = [t1]



    fig = go.Figure(data=data,layout=layout)

    fig.update_layout(title_x=0.5)

    py.iplot(fig)



charges_scatter(

    df_mc,

    'MonthlyCharges',

    'Churn Rate by Monthly Charges',

    'Monthly Charges'

)

print(df_copy['MonthlyCharges'].dtype)
df_copy['MonthlyCharges'].value_counts().head()
df_copy = df.copy()



df_copy.loc[df_copy.Churn=='No','Churn'] = 0 

df_copy.loc[df_copy.Churn=='Yes','Churn'] = 1



df_copy['Churn'] = df_copy['Churn'].astype(int)

df_copy.MonthlyCharges = df_copy.MonthlyCharges.round()

df_copy.TotalCharges = df_copy.TotalCharges.round()



df_mc = df_copy.groupby('MonthlyCharges').Churn.mean().reset_index()

df_tc = df_copy.groupby('TotalCharges').Churn.mean().reset_index()



charges_scatter(

    df_mc,

    'MonthlyCharges',

    'Churn Rate by Monthly Charges',

    'Monthly Charges'

)
charges_scatter(

    df_tc,

    'TotalCharges',

    'Churn Rate by Total Charges',

    'Total Charges'

)
# Average monthly charges for each tenure group.

df_num = df.groupby(['Churn','bin_tenure'])[['MonthlyCharges','TotalCharges']].mean().reset_index()



df_num_churn = df_num[df_num.Churn == 'Yes']

df_num_retention = df_num[df_num.Churn == 'No']
t1 = go.Bar(

    x = df_num_churn.bin_tenure,

    y = df_num_churn.MonthlyCharges,

    name = 'Churn',

    marker = dict(

        line = dict(

            width = 1

        )

    ),

    text = 'Churn'

)



t2 = go.Bar(

    x = df_num_retention.bin_tenure,

    y = df_num_retention.MonthlyCharges,

    name = 'Retention',

    marker = dict(

        line = dict(

            width =1

        )

    ),

    text = 'Retention'

)



layout = go.Layout(

    dict(

        title = 'Average Monthly Charge by Tenure',

        plot_bgcolor  = "rgb(243,243,243)",

        paper_bgcolor = "rgb(243,243,243)",

        xaxis = dict(

            gridcolor = 'rgb(255, 255, 255)',

            title = 'Tenure Group',

            zerolinewidth=1,

            ticklen=5,

            gridwidth=2

            ),

        yaxis = dict(

            gridcolor = 'rgb(255, 255, 255)',

            title = "Average Monthly Charge",

            zerolinewidth=1,

            ticklen=5,

            gridwidth=2

            ),

        )

    )



data = [t1,t2]



fig = go.Figure(data=data, layout=layout)

fig.update_layout(title_x=0.5)

py.iplot(fig)
t1 = go.Bar(

    x = df_num_churn.bin_tenure,

    y = df_num_churn.TotalCharges,

    name = 'Churn',

    marker = dict(

        line = dict(

            width = 1

        )

    ),

    text = 'Churn'

)



t2 = go.Bar(

    x = df_num_retention.bin_tenure,

    y = df_num_retention.TotalCharges,

    name = 'Retention',

    marker = dict(

        line = dict(

            width =1

        )

    ),

    text = 'Retention'

)



layout = go.Layout(

    dict(

        title = 'Average Total Charge by Tenure',

        plot_bgcolor  = "rgb(243,243,243)",

        paper_bgcolor = "rgb(243,243,243)",

        xaxis = dict(

            gridcolor = 'rgb(255, 255, 255)',

            title = 'Tenure Group',

            zerolinewidth=1,

            ticklen=5,

            gridwidth=2

            ),

        yaxis = dict(

            gridcolor = 'rgb(255, 255, 255)',

            title = "Average Total Charge",

            zerolinewidth=1,

            ticklen=5,

            gridwidth=2

            ),

        )

    )



data = [t1,t2]



fig = go.Figure(data=data, layout=layout)

fig.update_layout(title_x=0.5)

py.iplot(fig)
from sklearn.preprocessing import LabelEncoder, StandardScaler



#Encoded dataframe

df_enc = df.copy()

df_enc.drop(columns=["bin_tenure"], inplace=True)

customerid = ['customerID']

target = ["Churn"]



#categorical columns

cat_cols = df_enc.nunique()[df_enc.nunique() < 6].keys().tolist()

cat_cols = [x for x in cat_cols if x not in target]

#numerical columns

num_cols = [x for x in df_enc.columns if x not in cat_cols + target + customerid]

#Binary columns with 2 values

bin_cols = df_enc.nunique()[df_enc.nunique() == 2].keys().tolist()

#Columns more than 2 values

multi_cols = [i for i in cat_cols if i not in bin_cols]



#Label encoding Binary columns

le = LabelEncoder()

for i in bin_cols :

    df_enc[i] = le.fit_transform(df_enc[i])

    

#Duplicating columns for multi value columns

df_enc = pd.get_dummies(data = df_enc,columns = multi_cols)



#Scaling Numerical columns

std = StandardScaler()

scaled = std.fit_transform(df_enc[num_cols])

scaled = pd.DataFrame(scaled,columns=num_cols)



#dropping original values merging scaled values for numerical columns

df_unscaled = df_enc.copy()



df_enc = df_enc.drop(columns = num_cols,axis = 1)

df_enc = df_enc.merge(scaled,left_index=True,right_index=True,how = "left").dropna()

df_enc.head(n=3)
#Check the values for each column for encoding and scaling.

for column in df_enc.columns:

    print(column +':','\n',np.unique(df_enc[column])[:5],'\n')
df_enc.describe()[num_cols]
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split



X = df_enc.drop(columns=['customerID','Churn'],axis=1)

y = df_enc.Churn



os = SMOTE(random_state=42)



sm_X_train, sm_X_test, sm_y_train, sm_y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

sm_X,sm_y = os.fit_sample(sm_X_train,sm_y_train)



# Overwriting as all classification models will use smoted data.

X = pd.DataFrame(data = sm_X,columns=sm_X_train.columns)

y = pd.DataFrame(data = sm_y,columns=['Churn'])



y.Churn.value_counts()
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

from sklearn.metrics import roc_auc_score,roc_curve,scorer

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

from yellowbrick.classifier import ClassificationReport, DiscriminationThreshold

from yellowbrick.model_selection import FeatureImportances
# X = df_enc.drop(columns=['customerID','Churn'],axis=1)

# y = df_enc.Churn

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3 ,random_state = 42)



log_reg = LogisticRegression(

    C=1,

    fit_intercept=True,

    penalty='l2',

    solver='liblinear'

)





def performance(model, title_text, importance=False):

    model.fit(X_train,y_train)



    prediction = model.predict(X_test)

    probability = model.predict_proba(X_test)



    auc = roc_auc_score(y_test,prediction)



    fpr,tpr,thresholds = roc_curve(y_test,probability[:,1])

    

    accuracy = accuracy_score(y_test,prediction)

    print ("Accuracy Score : ", accuracy,'\n')

    print ("Area under curve : ", auc, '\n')



    report = ClassificationReport(model, classes=['Retention','Churn'])



    report.score(X_test, y_test)

    c = report.poof()



    conf_matrix = confusion_matrix(y_test, prediction)



    t1 = go.Heatmap(

        z = conf_matrix ,

        x = ["Not churn","Churn"],

        y = ["Not churn","Churn"],

        showscale  = True,

        colorscale = "Portland",

        name = "Matrix"

    )



    t2 = go.Scatter(

        x=fpr, 

        y=tpr, 

        mode='lines', 

        line=dict(

            color='darkorange',

            width=2

        ),

        name= auc

    )



    t3 = go.Scatter(

        x=[0, 1], 

        y=[0, 1], 

        mode='lines', 

        line=dict(

            color='navy',

            width=2,

            dash='dash'

        ),

        showlegend=False

    )



    fig = make_subplots(

        rows=2, 

        cols=1, 

        subplot_titles=(

            'Confusion Matrix',

            'Receiver Operating Characteristic'

        )

    )

    fig.append_trace(t1,1,1)

    fig.append_trace(t2,2,1)

    fig.append_trace(t3,2,1)

    fig.update_layout(

        height=700, 

        width=600,

        plot_bgcolor = 'lightgrey',

        paper_bgcolor = 'lightgrey',

        title_text=title_text,

        title_x=0.5,

        showlegend=False,

    )



    fig.update_xaxes(

        range=[-0.05,1.1],

        title="False Positive Rate",

        row=2, col=1

    )

    fig.update_yaxes(

        range=[-0.05,1.1],

        title="True Positive Rate",

        row=2, col=1

    )



    annot = list(fig.layout.annotations)

    annot[0].y = 1.02

    annot[1].y = 0.4

    fig.layout.annotations = annot



    py.iplot(fig)



    # Fit and show the discrimination threshold

    visualizer = DiscriminationThreshold(model)

    visualizer.fit(X_train,y_train)

    v = visualizer.poof()

    

    if importance == False:

        pass

    else:

        feature_importance = FeatureImportances(model, relative=False)



        # Fit and show the feature importances

        feature_importance.fit(X_train,y_train)

        f = feature_importance.poof()

    

    return accuracy



log = performance(

    log_reg, 

    title_text="Logistic Regression Performance",

    importance=True)
from sklearn.feature_selection import RFECV

from sklearn.linear_model import LogisticRegression



rfe = RFE(log_reg)

rfe = rfe.fit(X_train,y_train.values.ravel())



print(rfe.support_)

print()

print(rfe.ranking_)
df_rfe = pd.DataFrame({

    "rfe_support":rfe.support_,

    "columns":X.columns,

    "ranking":rfe.ranking_

})



rfe_cols = df_rfe[df_rfe["rfe_support"] == True]["columns"].tolist()

rfe_cols
X_train, X_test, y_train, y_test = train_test_split(X[rfe_cols], y, test_size = 0.3 ,random_state = 42)



log_rfe = performance(

    log_reg,

    title_text="Logistic Regression Performance",

    importance=True)
from sklearn.feature_selection import SelectKBest, chi2



#Have to use unstandardized dataframe since chi2 does not take negative values.

cols = [i for i in df_unscaled.columns if i not in customerid + target ]



chi2_select = SelectKBest(score_func = chi2,k = 3)

chi2_fit = chi2_select.fit(df_unscaled[cols],df_unscaled.Churn)



score = pd.DataFrame({

    "features":cols,

    "scores":chi2_fit.scores_,

    "p_values":chi2_fit.pvalues_ 

})

score = score.sort_values(by = "scores" ,ascending =False)



#Label each columns as either numerical or categorical.

score["feature_type"] = np.where(score["features"].isin(num_cols),"Numerical","Categorical")

display(score)



t1 = go.Scatter(

    x = score[score["feature_type"] == "Categorical"]["features"],

    y = score[score["feature_type"] == "Categorical"]["scores"],

    name = "Categorial",mode = "lines+markers",

    marker = dict(

        color = "red",

        line = dict(width =1)

    )

)



t2 = go.Bar(

    x = score[score["feature_type"] == "Numerical"]["features"],

    y = score[score["feature_type"] == "Numerical"]["scores"],name = "Numerical",

    marker = dict(

        color = "blue",

        line = dict(width =1)

    ),

    xaxis = "x2",

    yaxis = "y2"

)



layout = go.Layout(dict(

    title = "Chi-Squared Scores",

    plot_bgcolor  = "rgb(243,243,243)",

    paper_bgcolor = "rgb(243,243,243)",

    xaxis = dict(

        gridcolor = 'rgb(255, 255, 255)',

        tickfont = dict(size =10),

        domain=[0, 0.7],

        tickangle = 90,zerolinewidth=1,

        ticklen=5,gridwidth=2),

    yaxis = dict(

        gridcolor = 'rgb(255, 255, 255)',

        title = "scores",

        zerolinewidth=1,

        ticklen=5,gridwidth=2),

    margin = dict(b=200),

    xaxis2=dict(

        domain=[0.8, 1],tickangle = 90,

        gridcolor = 'rgb(255, 255, 255)'),

    yaxis2=dict(anchor='x2',gridcolor = 'rgb(255, 255, 255)')

        )

)



data=[t1,t2]

fig = go.Figure(data=data,layout=layout)

fig.update_layout(title_x=0.5)

py.iplot(fig)
score_10 = score.nlargest(10,'scores')['features']

X_train, X_test, y_train, y_test = train_test_split(X[score_10], y, test_size = 0.3 ,random_state = 42)



log_chi = performance(

    log_reg,

    title_text="Logistic Regression Performance",

    importance=False)
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import export_graphviz

from sklearn import tree

from graphviz import Source

from IPython.display import SVG,display



cols = [i for i in df_enc.columns if i not in customerid + target ]    



rf_x = X[cols]

rf_y = y[target]



X_train, X_test, y_train, y_test = train_test_split(rf_x, rf_y, test_size = 0.3 ,random_state = 42)



rfc = RandomForestClassifier()



forest = performance(

    rfc,

    title_text="Random Forest Performance",

    importance=True

)

from sklearn import svm

from sklearn.model_selection import GridSearchCV 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3 ,random_state = 42)



clf = svm.SVC()



# Finding the optimal parameters.

param_grid = {'C': [0.1, 1, 10],  

              'gamma': [1, 0.1, 0.01], 

              'kernel': ['rbf','linear']}  

  

grid = GridSearchCV(clf, param_grid, refit = True) 

  

# Fitting the model for grid search 

grid.fit(X_train, y_train) 



print(grid.best_params_) 

print(grid.best_estimator_)

clf = svm.SVC(

    C=1,

    gamma=1,

    kernel='rbf'

)



clf.fit(X_train, y_train)

prediction = clf.predict(X_test)



svm_acc = accuracy_score(y_test,prediction)

print ("Accuracy Score : ", svm_acc,'\n')



report = ClassificationReport(clf, classes=['Retention','Churn'])



report.score(X_test, y_test)

c = report.poof()



conf_matrix = confusion_matrix(y_test, prediction)



t1 = go.Heatmap(

    z = conf_matrix ,

    x = ["Not churn","Churn"],

    y = ["Not churn","Churn"],

    showscale  = True,

    colorscale = "Portland",

    name = "Matrix"

)



layout = go.Layout(

    dict(

        title = 'Confusion Matrix SVM',

        plot_bgcolor  = "rgb(243,243,243)",

        paper_bgcolor = "rgb(243,243,243)",

        xaxis = dict(

            gridcolor = 'lightgrey',

            zerolinewidth=1,

            ticklen=5,

            gridwidth=2

            ),

        yaxis = dict(

            gridcolor = 'lightgrey',

            zerolinewidth=1,

            ticklen=5,

            gridwidth=2

            ),

        )

    )



data = [t1]



fig = go.Figure(data=data, layout=layout)

fig.update_layout(title_x=0.5)

py.iplot(fig)
models = pd.DataFrame({

    'Models':[

        'Logistic Regression',

        'Logistic Regression + RFE',

        'Logistic Regression + Chi^2',

        'Random Forest Classifier',

        'Support Vector Machine'

    ],

    'Scores':[

        log, 

        log_rfe,  

        log_chi, 

        forest, 

        svm_acc, 

    ]

})



print("*** Accuracy Scores ***")

models.sort_values(by='Scores', ascending=False)