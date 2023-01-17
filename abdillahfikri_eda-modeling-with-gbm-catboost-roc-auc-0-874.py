import pandas as pd

import numpy as np

import plotly.graph_objects as go

import plotly.express as px

import plotly.figure_factory as ff

import missingno as msno

import category_encoders as ce

import optuna

import warnings

warnings.filterwarnings('ignore')



from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, StratifiedKFold

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics

from sklearn.pipeline import Pipeline
df = pd.read_csv('../input/churn-modelling/Churn_Modelling.csv')

df.head()
df.shape
df.info()
msno.matrix(df)
df = df.drop(columns=['RowNumber', 'CustomerId'])
df['Surname'].nunique()
df.describe()
df.describe(include='object')
fig = px.histogram(df, x='Exited',

                   height=400, width=500,

                   title='Target Feature Distribution')

fig.update_xaxes(type='category')

fig.show()
df['Exited'].value_counts(normalize=True)*100
fig = make_subplots(rows=2, cols=3)



fig.append_trace(go.Histogram(

    x=df['CreditScore'], name='Credit Score', nbinsx=50

), row=1, col=1)

fig.update_xaxes(title_text='Credit Score', row=1, col=1)



fig.append_trace(go.Histogram(

    x=df['Age'], name='Age', nbinsx=30

), row=1, col=2)

fig.update_xaxes(title_text='Age', row=1, col=2)



fig.append_trace(go.Histogram(

    x=df['Balance'], name='Balance', nbinsx=20

), row=1, col=3)

fig.update_xaxes(title_text='Balance', row=1, col=3)



for col, feature in enumerate(['Tenure', 'EstimatedSalary']):

    fig.append_trace(go.Histogram(

        x=df[feature], name=feature,

        nbinsx=20

    ), row=2, col=col+1)

    fig.update_xaxes(title_text=feature, row=2, col=col+1)



fig.update_layout(

    height=700, width=1200, 

    title_text='Features Distribution with Histogram'

)

fig.show()
print('Customer dengan Balance 0:',len(df[df['Balance']==0]))
fig = px.histogram(

    df, x='CreditScore', color='Exited',

    marginal='box', nbins=50,

    color_discrete_map={0: '#636EFA', 1: '#EF553B'},

    barmode='overlay'

)



fig.update_layout(

    height=500, width=800, 

    title_text='Credit Score Feature in Detail'

)

fig.show()
fig = px.histogram(

    df, x='Age', color='Exited',

    marginal='box', nbins=30,

    color_discrete_map={0: '#636EFA', 1: '#EF553B'},

    barmode='overlay'

)



fig.update_layout(height=500, width=800, 

                  title_text='Age Feature in Detail')

fig.show()
fig = px.histogram(

    df, x='Balance', color='Exited',

    marginal='box', nbins=20,

    color_discrete_map={0: '#636EFA', 1: '#EF553B'},

    barmode='overlay'

)



fig.update_layout(height=500, width=800, 

                  title_text='Balance Feature in Detail')

fig.show()
fig = px.histogram(

    df, x='Tenure', color='Exited', marginal='box',

    color_discrete_map={0: '#636EFA', 1: '#EF553B'},

    barmode='overlay'

)

fig.update_layout(height=500, width=800, 

                  title_text='Tenure Feature in Detail')

fig.show()
fig = px.histogram(

    df, x='Tenure', color='Exited',

    color_discrete_map={0: '#636EFA', 1: '#EF553B'},

    category_orders={'Tenure': [0,1,2,3,4,5,6,7,8,9,10]},

    barnorm='percent'

)



fig.update_layout(

    height=500, width=800, 

    title_text='Tenure Feature in Detail',

    yaxis_title='Percentage of Churn',

    yaxis={'ticksuffix':'%'}

)

fig.update_xaxes(type='category')

fig.show()
fig = px.histogram(

    df, x='EstimatedSalary', color='Exited', marginal='box',

    color_discrete_map={0: '#636EFA', 1: '#EF553B'},

    barmode='overlay', nbins=20

)



fig.update_layout(height=500, width=800, 

                  title_text='EstimatedSalary Feature in Detail')

fig.show()
fig = make_subplots(rows=2, cols=3)



# For loop for the first row

for col, feature in enumerate(['NumOfProducts', 'HasCrCard', 'IsActiveMember']):

    fig.append_trace(go.Histogram(

        x=df[feature], name=feature,

    ), row=1, col=col+1)

    fig.update_xaxes(title_text=feature, row=1, col=col+1)



# For loop for the second row

for col, feature in enumerate(['Geography', 'Gender']):

    fig.append_trace(go.Histogram(

        x=df[feature], name=feature,

    ), row=2, col=col+1)

    fig.update_xaxes(title_text=feature, row=2, col=col+1)



fig.update_xaxes(type='category', 

                 categoryorder='category ascending')

fig.update_layout(height=700, width=1200, 

                  title_text='Categorical Features Distribution')

fig.show()
fig = px.histogram(

    df, x='NumOfProducts', color='Exited',

    color_discrete_map={0: '#636EFA', 1: '#EF553B'},

    barnorm='percent'

)

fig.update_layout(

    height=500, width=800, 

    title_text='NumOfProducts Feature in Detail',

    yaxis_title='Percentage of Churn',

    yaxis={'ticksuffix':'%'}

)

fig.update_xaxes(

    type='category',

    categoryorder='category ascending'

)

fig.show()
fig = px.histogram(

    df, x='HasCrCard', color='Exited',

    color_discrete_map={0: '#636EFA', 1: '#EF553B'},

    barnorm='percent'

)

fig.update_layout(height=500, width=800, 

                  title_text='HasCrCard Feature in Detail',

                  yaxis_title='Percentage of Churn',

                  yaxis={'ticksuffix':'%'})

fig.update_xaxes(

    type='category',

    categoryorder='category ascending'

)

fig.show()
fig = px.histogram(

    df, x='IsActiveMember', color='Exited',

    color_discrete_map={0: '#636EFA', 1: '#EF553B'},

    barnorm='percent'

)

fig.update_layout(height=500, width=800, 

                  title_text='IsActiveMember Feature in Detail',

                  yaxis_title='Percentage of Churn',

                  yaxis={'ticksuffix':'%'})

fig.update_xaxes(

    type='category',

    categoryorder='category ascending'

)

fig.show()
fig = px.histogram(

    df, x='Geography', color='Exited',

    barnorm='percent',

    color_discrete_map={0: '#636EFA', 1: '#EF553B'}

)

fig.update_yaxes(title_text='Percentage of Churn')

fig.update_layout(height=500, width=800, 

                  title_text='Exited Percentage by Geography',

                  yaxis={'ticksuffix':'%'})

fig.show()
fig = px.histogram(

    df, x='Gender', color='Exited',

    barnorm='percent',

    color_discrete_map={0: '#636EFA', 1: '#EF553B'}

)

fig.update_yaxes(title_text='Percent')



fig.update_layout(height=500, width=800, 

                  title_text='Exited Percentage by Gender',

                  yaxis={'ticksuffix':'%'})

fig.show()
encoder = ce.TargetEncoder()

df_temp = encoder.fit_transform(df.drop(columns='Exited'), df['Exited'])

df_corr = df_temp.join(df['Exited']).corr()



fig = ff.create_annotated_heatmap(

    z=df_corr.values,

    x=list(df_corr.columns),

    y=list(df_corr.index),

    annotation_text=df_corr.round(2).values,

    showscale=True, colorscale='Viridis'

)

fig.update_layout(height=600, width=800, 

                  title_text='Feature Correlation')

fig.update_xaxes(side='bottom')

fig.show()
df['BalanceToSalaryRatio'] = df['Balance'] / df['EstimatedSalary']
from itertools import combinations

cat_cols = df.select_dtypes('object').columns



for col in combinations(cat_cols, 2):

    df[col[0]+'_'+col[1]] = df[col[0]] + "_" + df[col[1]]

    

df.head()
df.describe(include='object')
encoder = ce.TargetEncoder()

df_temp = encoder.fit_transform(df.drop(columns='Exited'), df['Exited'])

df_corr = df_temp.join(df['Exited']).corr()



fig = ff.create_annotated_heatmap(

    z=df_corr.values,

    x=list(df_corr.columns),

    y=list(df_corr.index),

    annotation_text=df_corr.round(2).values,

    showscale=True, colorscale='Viridis'

)

fig.update_layout(height=700, width=900, 

                  title_text='Feature Correlation')

fig.update_xaxes(side='bottom')

fig.show()
df.head()
df.describe(include='object')
X_train, X_test, y_train, y_test = train_test_split(

    df.drop(columns='Exited'), df['Exited'],

    test_size=0.2, random_state=0,

)
# Ratio using for scale_pos_weight to get better recall on imbalance class

ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
xgb_pipeline = Pipeline([

    ('one_hot', ce.OneHotEncoder(cols=['Geography', 'Gender', 'Geography_Gender'])),

    ('catboost', ce.CatBoostEncoder(cols=['Surname', 'Surname_Geography', 'Surname_Gender'])),

    ('xgb', XGBClassifier(scale_pos_weight=ratio))

])
lgb_pipeline = Pipeline([

    ('one_hot', ce.OneHotEncoder(cols=['Geography', 'Gender', 'Geography_Gender'])),

    ('catboost', ce.CatBoostEncoder(cols=['Surname', 'Surname_Geography', 'Surname_Gender'])),

    ('lgb', LGBMClassifier(scale_pos_weight=ratio))

])
cat_pipeline = Pipeline([

    ('one_hot', ce.OneHotEncoder(cols=['Geography', 'Gender', 'Geography_Gender'])),

    ('catboost', ce.CatBoostEncoder(cols=['Surname', 'Surname_Geography', 'Surname_Gender'])),

    ('cat', CatBoostClassifier(scale_pos_weight=ratio, verbose=0))

])
import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns

from sklearn.model_selection import cross_val_score

from sklearn import metrics



def make_confusion_matrix(cf,

                          group_names=None,

                          categories='auto',

                          count=True,

                          percent=True,

                          cbar=True,

                          xyticks=True,

                          xyplotlabels=True,

                          sum_stats=True,

                          figsize=None,

                          cmap='Blues',

                          title=None):

    '''

    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Arguments

    ---------

    cf:            confusion matrix to be passed in

    group_names:   List of strings that represent the labels row by row to be shown in each square.

    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    normalize:     If True, show the proportions for each category. Default is True.

    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.

                   Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'

                   See http://matplotlib.org/examples/color/colormaps_reference.html

                   

    title:         Title for the heatmap. Default is None.

    '''





    # CODE TO GENERATE TEXT INSIDE EACH SQUARE

    blanks = ['' for i in range(cf.size)]



    if group_names and len(group_names)==cf.size:

        group_labels = ["{}\n".format(value) for value in group_names]

    else:

        group_labels = blanks



    if count:

        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]

    else:

        group_counts = blanks



    if percent:

        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]

    else:

        group_percentages = blanks



    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]

    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])





    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS

    if sum_stats:

        #Accuracy is sum of diagonal divided by total observations

        accuracy  = np.trace(cf) / float(np.sum(cf))



        #if it is a binary confusion matrix, show some more stats

        if len(cf)==2:

            #Metrics for Binary Confusion Matrices

            precision = cf[1,1] / sum(cf[:,1])

            recall    = cf[1,1] / sum(cf[1,:])

            f1_score  = 2*precision*recall / (precision + recall)

            stats_text = "\n\nPrecision={:0.3f} | Recall={:0.3f}\nAccuracy={:0.3f} | F1 Score={:0.3f}".format(

                precision, recall, accuracy, f1_score)

        else:

            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)

    else:

        stats_text = ""





    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS

    if figsize==None:

        #Get default figure size if not set

        figsize = plt.rcParams.get('figure.figsize')



    if xyticks==False:

        #Do not show categories if xyticks is False

        categories=False





    # MAKE THE HEATMAP VISUALIZATION

    plt.figure(figsize=figsize)

    plt.subplot(1,2,1)

    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)



    if xyplotlabels:

        plt.ylabel('True label')

        plt.xlabel('Predicted label' + stats_text)

    else:

        plt.xlabel(stats_text)

    

    if title:

        plt.title(title)





def model_eval(model, X_train, y_train, 

               scoring_='roc_auc', cv_=5):

  

    model.fit(X_train, y_train)



    train_pred = model.predict(X_train)

    train_predprob = model.predict_proba(X_train)[:,1]

           

    cv_score = cross_val_score(model, X_train, y_train, cv=cv_, scoring=scoring_)

    print('Model Report on Train and CV Set:')

    print('--------')

    print('Train Accuracy: {:0.6f}'.format(metrics.accuracy_score(y_train, train_pred)))

    print('Train AUC Score: {:0.6f}'.format(metrics.roc_auc_score(y_train, train_predprob)))

    print('CV AUC Score: Mean - {:0.6f} | Std - {:0.6f} | Min - {:0.6f} | Max - {:0.6f} \n'.format(

        np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))







def test_eval(model, X_train, X_test, y_train, y_test):

    

    model.fit(X_train, y_train)



    pred = model.predict(X_test)

    predprob = model.predict_proba(X_test)[:,1]

    

    print('Model Report on Test Set:')

    print('--------')

    print('Classification Report \n', metrics.classification_report(y_test, pred))



    conf = metrics.confusion_matrix(y_test, pred)

    group_names = ['True Negative', 'False Positive', 'False Negtive', 'True Positive']

    make_confusion_matrix(conf, percent=False, group_names=group_names,

                          figsize=(14,5), title='Confusion Matrix')



    plt.subplot(1,2,2)

    fpr, tpr, _ = metrics.roc_curve(y_test, predprob)

    plt.plot(fpr, tpr, color='orange', label='ROC')

    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('ROC Curve\nAUC Score: {:0.3f}'.format(metrics.roc_auc_score(y_test, predprob)))

    plt.legend()
test_eval(xgb_pipeline, X_train, X_test, y_train, y_test)
test_eval(lgb_pipeline, X_train, X_test, y_train, y_test)
test_eval(cat_pipeline, X_train, X_test, y_train, y_test)
cat_features = df.select_dtypes('object').columns



cat = CatBoostClassifier(scale_pos_weight=ratio,

                         verbose=0, cat_features=cat_features)

test_eval(cat, X_train, X_test, y_train, y_test)