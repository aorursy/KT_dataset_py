# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import matplotlib.pyplot as plt

import re

import time

import warnings

import numpy as np

from nltk.corpus import stopwords

from sklearn.preprocessing import normalize

import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics.classification import accuracy_score, log_loss

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import SGDClassifier

from imblearn.over_sampling import SMOTE

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC

from sklearn.calibration import CalibratedClassifierCV

from sklearn.naive_bayes import MultinomialNB

from sklearn.naive_bayes import GaussianNB

import math

from sklearn.metrics import normalized_mutual_info_score

from sklearn.ensemble import RandomForestClassifier

from scipy import stats

from sklearn.decomposition import PCA



from scipy.stats import norm

from mlxtend.classifier import StackingClassifier

from IPython.display import Image

import pylab 



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

from sklearn.metrics import roc_auc_score,roc_curve,scorer

from sklearn.metrics import f1_score

import statsmodels.api as sm

from sklearn.metrics import precision_score,recall_score

from yellowbrick.classifier import DiscriminationThreshold



import plotly.offline as py#visualization

py.init_notebook_mode(connected=True)#visualization

import plotly.graph_objs as go#visualization

import plotly.tools as tls#visualization

import plotly.figure_factory as ff#visualization





from sklearn import model_selection

from sklearn.linear_model import Lasso

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression

from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler





from sklearn.metrics import recall_score





import plotly.offline as py#visualization

py.init_notebook_mode(connected=True)#visualization

import plotly.graph_objs as go#visualization

import plotly.tools as tls#visualization

import plotly.figure_factory as ff#visualization



%matplotlib inline

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
telcom_df =pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')



# Converting TotalChargees column to float

telcom_df['TotalCharges'] = pd.to_numeric(telcom_df['TotalCharges'],errors='coerce')

telcom_df['TotalCharges']= telcom_df['TotalCharges'].astype(float)



# Imputing missing values with mean should not produce significant shift to the overall distribution

telcom_df['TotalCharges']=telcom_df['TotalCharges'].fillna(telcom_df['TotalCharges'].mode()[0])





# Seperating Churn and No Churn

No_Churn = telcom_df[telcom_df['Churn']=='No']

Churn = telcom_df[telcom_df['Churn']=='Yes']



# Attributing No internet service to No

no_internet_feats = [ 'TechSupport','OnlineBackup', 'DeviceProtection','StreamingTV',

                 'OnlineSecurity','StreamingMovies']



for i in no_internet_feats:

    telcom_df[i] = telcom_df[i].replace({'No internet service':'No'})



# Attributing No phone service to No

telcom_df['MultipleLines']=telcom_df['MultipleLines'].replace({'No phone service':'No'})



# Attributing No phone service to No

telcom_df['SeniorCitizen']=telcom_df['SeniorCitizen'].replace({0:'No',

                                                              1:'Yes'})









# Feature seperation

Id_features = ['customerID']

target_feature = ['Churn']



num_features =  telcom_df.select_dtypes(include='number').columns.tolist()

cat_features = telcom_df.select_dtypes(include='object').columns.tolist()

cat_features   = [x for x in cat_features if x not in Id_features]

categ_count=[]

for i in cat_features:

    temp = telcom_df[i]

    temp = temp.value_counts()

    categ_count.append(temp)



for i in range(1,len(categ_count)+1):

    pylab.subplot(5,5,i)

    sns.barplot(categ_count[i-1].index, categ_count[i-1].values, alpha=0.8,label='big').set_title(categ_count[i-1].name)

    plt.gcf().set_size_inches(20, 15)

    plt.xticks(rotation=20)

    plt.tight_layout()
import numpy as np

import plotly.graph_objs as go

import plotly



x1 = telcom_df[telcom_df['Churn']=='Yes']['TotalCharges']

x2 = telcom_df[telcom_df['Churn']=='No']['TotalCharges']

x3 = telcom_df[telcom_df['Churn']=='Yes']['tenure']

x4 = telcom_df[telcom_df['Churn']=='No']['tenure']

x5 = telcom_df[telcom_df['Churn']=='Yes']['MonthlyCharges']

x6 = telcom_df[telcom_df['Churn']=='No']['MonthlyCharges']



fig = plotly.tools.make_subplots(rows=1,cols=3)



fig.append_trace(go.Histogram(x = x1, opacity = 0.75, name = 'Churn', histnorm= "percent" ,showlegend=False,marker=dict(color='#0000FF')),1,1)

fig.append_trace(go.Histogram(x = x2, opacity = 0.75, name = 'No Churn', histnorm= "percent",showlegend=False,marker=dict(color='#FFA500')),1,1)



fig.append_trace(go.Histogram(x = x3, opacity = 0.75, name = 'Churn', histnorm= "percent",showlegend=False,marker=dict(color='#0000FF')),1,2)

fig.append_trace(go.Histogram(x = x4, opacity = 0.75, name = 'No Churn', histnorm= "percent",showlegend=False,marker=dict(color='#FFA500')),1,2)



fig.append_trace(go.Histogram(x = x5, opacity = 0.75, name = 'Churn', histnorm= "percent",marker=dict(color='#0000FF')),1,3)

fig.append_trace(go.Histogram(x = x6, opacity = 0.75, name = 'No Churn', histnorm= "percent",marker=dict(color='#FFA500')),1,3)



fig['layout']['xaxis1'].update(title='TotalCharges($)')

fig['layout']['xaxis2'].update(title='Tenure(month)')

fig['layout']['xaxis3'].update(title='MonthlyCharges($)')



fig['layout']['yaxis1'].update(title='percent')



fig.layout.update(go.Layout(barmode = 'overlay',title='Numerical Features Distribution'))

py.iplot(fig)
# Feature seperation

Id_features = ['customerID']

target_feature = ['Churn']



num_features =  telcom_df.select_dtypes(include='number').columns.tolist()

cat_features = telcom_df.select_dtypes(include='object').columns.tolist()

cat_features   = [x for x in cat_features if x not in Id_features]



# Binary columns with 2 values

bin_features   = telcom_df.nunique()[telcom_df.nunique() == 2].keys().tolist()

#Columns more than 2 values

multi_features = [x for x in cat_features if x not in bin_features]

    

#Duplicating columns for multi value columns

telcom_df = pd.get_dummies(data = telcom_df,columns = multi_features )



#Label encoding Binary columns

le = LabelEncoder()

for i in bin_features :

    telcom_df[i] = le.fit_transform(telcom_df[i])



#Scaling Numerical columns

std = StandardScaler()

scaled = std.fit_transform(telcom_df[num_features])

scaled = pd.DataFrame(scaled,columns=num_features)



#dropping original values merging scaled values for numerical columns

telcom_df_og = telcom_df.copy()

telcom_df = telcom_df.drop(columns = num_features+Id_features,axis = 1)

telcom_df = telcom_df.merge(scaled,left_index=True,right_index=True,how = "left")



#correlation

correlations = telcom_df.corr().sort_values(by=['Churn'],ascending=False)

correlations = correlations.T.sort_values(by=['Churn'],ascending=False)

#tick labels

matrix_cols = correlations.columns.tolist()

#convert to array

corr_array  = np.array(correlations)



#Plotting

trace = go.Heatmap(z = corr_array,

                   x = matrix_cols,

                   y = matrix_cols,

                   colorscale = "Portland",

                   colorbar   = dict(title = "Pearson Correlation Coefficient",

                                     titleside = "right"

                                    ) ,

                  )



layout = go.Layout(dict(title = "Correlation Matrix for variables",

                        autosize = False,

                        height  = 720,

                        width   = 800,

                        margin  = dict(r = 0 ,l = 210,

                                       t = 25,b = 210,

                                      ),

                        yaxis   = dict(tickfont = dict(size = 9)),

                        xaxis   = dict(tickfont = dict(size = 9))

                       )

                  )



data = [trace]

fig = go.Figure(data=data,layout=layout)

py.iplot(fig)
pca = PCA(n_components = 2)



X = telcom_df[[i for i in telcom_df.columns if i not in Id_features + target_feature]]

Y = telcom_df[target_feature]



principal_components = pca.fit_transform(X)

pca_data = pd.DataFrame(principal_components,columns = ["PC1","PC2"])

pca_data = pca_data.merge(Y,left_index=True,right_index=True,how="left")

pca_data["Churn"] = pca_data["Churn"].replace({1:"Churn",0:"Not Churn"})



components_df = pd.DataFrame(pca.components_,columns=X.columns)



pc1 = components_df.loc[0].sort_values(ascending=True).to_dict()

pc2 = components_df.loc[1].sort_values(ascending=True).to_dict()

comp_1 = pd.DataFrame([pc1],columns=pc1.keys())

comp_2 = pd.DataFrame([pc2],columns=pc2.keys())

x1 = comp_1.T.reset_index()

x2 = comp_2.T.reset_index()

x1= x1.rename(columns={'index':'Feature',0:'Value'})

x2 = x2.rename(columns={'index':'Feature',0:'Value'})
trace1 = go.Scatter(x = pca_data[pca_data["Churn"] == 'Churn']["PC1"],

                    y = pca_data[pca_data["Churn"] == 'Churn']["PC2"], name = 'Churn',

                    mode = "markers", 

                    marker = dict(color = 'red',symbol =  "diamond-open"))



trace2 = go.Scatter(x = pca_data[pca_data["Churn"] == 'Not Churn']["PC1"],

                    y = pca_data[pca_data["Churn"] == 'Not Churn']["PC2"], name = 'Not Churn',

                    mode = "markers",

                    marker = dict(color = 'blue',symbol =  "diamond-open"))



trace4 = go.Bar(x = x1["Feature"],y = x1["Value"],name = "PC1")

trace3 = go.Bar(x = x2["Feature"],y = x2["Value"],name = "PC2")



fig = plotly.tools.make_subplots(rows=2,cols=2,horizontal_spacing = 0.1,vertical_spacing = 0.5,)



fig.append_trace(trace1,1,1)

fig.append_trace(trace2,1,1)

fig.append_trace(go.Bar(x = x1["Feature"],y = x1["Value"],name = "PC1",showlegend=False,marker=dict(color='#9400D3')),1,2)

fig.append_trace(go.Bar(x = x2["Feature"],y = x2["Value"],name = "PC2",showlegend=False,marker=dict(color='#008000')),2,2)



fig['layout'].update(showlegend=True, title="Model performance" ,

                     autosize = False,height = 800,width = 1200,

                     plot_bgcolor = 'rgba(240,240,240, 0.95)',

                     paper_bgcolor = 'rgba(240,240,240, 0.95)',

                     margin = dict(b = 250))



fig['layout']['xaxis1'].update(title='principal component 1',domain=[0.05, 0.5])

fig['layout']['yaxis1'].update(title='principal component 2',domain=[1, 0.3])

fig['layout']['xaxis2'].update(tickangle=60)

fig['layout']['yaxis2'].update(title='PC 1')

fig['layout']['xaxis4'].update(tickangle=60)

fig['layout']['yaxis4'].update(title='PC 2')



fig.layout.update(go.Layout(barmode = 'overlay',title='Principle Component Analysis and Feature Contribution' ))

py.iplot(fig)
telcom_df =pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')



# Converting TotalChargees column to float

telcom_df['TotalCharges'] = pd.to_numeric(telcom_df['TotalCharges'],errors='coerce')

telcom_df['TotalCharges']= telcom_df['TotalCharges'].astype(float)



# Imputing missing values with mean should not produce significant shift to the overall distribution

telcom_df['TotalCharges']=telcom_df['TotalCharges'].fillna(telcom_df['TotalCharges'].mode()[0])





# Seperating Churn and No Churn

No_Churn = telcom_df[telcom_df['Churn']=='No']

Churn = telcom_df[telcom_df['Churn']=='Yes']



# Attributing No internet service to No

no_internet_feats = [ 'TechSupport','OnlineBackup', 'DeviceProtection','StreamingTV',

                 'OnlineSecurity','StreamingMovies']



for i in no_internet_feats:

    telcom_df[i] = telcom_df[i].replace({'No internet service':'No'})



# Attributing No phone service to No

telcom_df['MultipleLines']=telcom_df['MultipleLines'].replace({'No phone service':'No'})



# Attributing No phone service to No

telcom_df['SeniorCitizen']=telcom_df['SeniorCitizen'].replace({0:'No',

                                                              1:'Yes'})
# Feature seperation

Id_features = ['customerID']

target_feature = ['Churn']



# All services 

telcom_df['TotalServices'] = (telcom_df[['PhoneService', 'InternetService', 'OnlineSecurity','OnlineBackup', 

                                         'DeviceProtection', 'TechSupport','StreamingTV', 'StreamingMovies']]== 'Yes').sum(axis=1)



# This feature tends to reflect the relationship between services and monthy charges. 

# This features assumes equal weight between services



# Engaged customer that subscribes to all services with contract greater than month-to-month 

telcom_df.loc[:,'EngagedCustomers'] = np.where((telcom_df['StreamingTV'] =='Yes') &

                               (telcom_df['PhoneService']=='Yes') &

                               (telcom_df['StreamingMovies']=='Yes') &

                               (telcom_df['Contract'] != 'Month-to-month'), 1,0)



# Not Senior, Single, No Dependents 

telcom_df.loc[:,'Young_NoDep'] = np.where((telcom_df['Dependents'] =='No') &

                                    (telcom_df['SeniorCitizen']=='No'), 1,0)



# Not Senior, No Dependents 

telcom_df.loc[:,'Single_NotSen'] = np.where((telcom_df['Partner'] =='No') &

                                    (telcom_df['SeniorCitizen']=='No'), 1,0)



#########################



num_features =  telcom_df.select_dtypes(include='number').columns.tolist()

cat_features = telcom_df.select_dtypes(include='object').columns.tolist()

cat_features   = [x for x in cat_features if x not in Id_features]



# Binary columns with 2 values

bin_features   = telcom_df.nunique()[telcom_df.nunique() == 2].keys().tolist()

#Columns more than 2 values

multi_features = [x for x in cat_features if x not in bin_features]



telcom_df[num_features]=telcom_df[num_features].astype(float)

telcom_df['EngagedCustomers']=telcom_df['EngagedCustomers'].astype(float)

######################



    

#Duplicating columns for multi value columns

telcom_df = pd.get_dummies(data = telcom_df,columns = multi_features )



#Label encoding Binary columns

le = LabelEncoder()

for i in bin_features :

    telcom_df[i] = le.fit_transform(telcom_df[i])



#Scaling Numerical columns

std = StandardScaler()

scaled = std.fit_transform(telcom_df[num_features])

scaled = pd.DataFrame(scaled,columns=num_features)



#dropping original values merging scaled values for numerical columns

telcom_df_og = telcom_df.copy()

telcom_df = telcom_df.drop(columns = num_features+Id_features,axis = 1)

telcom_df = telcom_df.merge(scaled,left_index=True,right_index=True,how = "left")
# Try Feature

telcom_df['AveServicePrice[$/mo]'] = telcom_df['MonthlyCharges']/telcom_df['TotalServices'] 

# Comb Engin_feat

engin_feat = telcom_df[['TotalServices','AveServicePrice[$/mo]','EngagedCustomers','Single_NotSen']]
train,test = train_test_split(telcom_df,test_size = .25 ,random_state = 111)

    

##seperating dependent and independent variables

features    = [i for i in telcom_df.columns if i not in Id_features + target_feature]

X_train = train[features]

Y_train = train[target_feature]

X_test  = test[features]

Y_test  = test[target_feature]
def model_performance_80_20(name,

                            clf,

                            X_train,

                            X_test,

                            Y_train,

                            Y_test):

    '''

    IN: Model name, Classifier, Best Alpha, and All 3 OneHotEncoded Sets 

    OUT: Log-Loss Report data frame

    '''

    # Model

    clf = clf

    clf.fit(X_train.values, Y_train.values)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(X_train.values, Y_train.values)

    

    train_predict_y = sig_clf.predict_proba(X_train.values)

    train_log_loss = np.round(log_loss(Y_train, train_predict_y, labels=clf.classes_, eps=1e-15),3)

    

    test_predict_y = sig_clf.predict_proba(X_test)

    test_log_loss = np.round(log_loss(Y_test, test_predict_y, labels=clf.classes_, eps=1e-15),3)

    

    pred_y = sig_clf.predict(X_test.values)

    miss_class = np.round(np.count_nonzero(100*(pred_y- Y_test['Churn']))/Y_test.shape[0],5)

    miss_class=miss_class*100

    

    test_recall = np.round(recall_score(Y_test['Churn'], pred_y, average='macro'),5)

    accuracy    = np.round(accuracy_score(Y_test['Churn'], pred_y,),5)

    precision    = np.round(precision_score(Y_test['Churn'], pred_y,),5)

    roc_auc      = np.round(roc_auc_score(Y_test['Churn'], pred_y,),5)

    f1score      = np.round(f1_score(Y_test['Churn'], pred_y,) ,5)



    

    report=[name,

            accuracy,

            test_recall,

            precision,

            roc_auc,

            f1score,

            train_log_loss,

            test_log_loss,

            miss_class]

    

    temp_df = pd.DataFrame([report],columns=['Model',

                                             'accuracy score',

                                             'recall score',

                                             'precision score',

                                             'roc_auc',

                                             'f1score',

                                             'train_log_loss',

                                             'test_log_loss',

                                             'miss_classified(%)' ])   

    return temp_df

# Logistic Regression

clf  = LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,

          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,

          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,

          verbose=0, warm_start=False)



LogisticReg= model_performance_80_20('LogisticReg',clf,X_train,X_test,Y_train,Y_test)



# XGB

clf = XGBClassifier(base_score=0.1, booster='gbtree', colsample_bylevel=1,

                    colsample_bytree=1, gamma=0, learning_rate=0.9, max_delta_step=0,

                    max_depth = 7, min_child_weight=1, missing=None, n_estimators=100,

                    n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,

                    reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

                    silent=True, subsample=1)



XGB= model_performance_80_20('XGB',clf,X_train,X_test,Y_train,Y_test)



# LightGBM

clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,

                        learning_rate=0.5, max_depth=3, min_child_samples=20,

                        min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,

                        n_jobs=-1, num_leaves=500, objective='binary', random_state=None,

                        reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,

                        subsample_for_bin=200000, subsample_freq=0)

LightGBM= model_performance_80_20('LightGBM',clf,X_train,X_test,Y_train,Y_test)



# Random Forest

clf = RandomForestClassifier(n_estimators=100,  max_depth=10, criterion='gini',random_state=42, n_jobs=-1)

RandomForest= model_performance_80_20('RandomForest',clf,X_train,X_test,Y_train,Y_test)





# Decision Tree

clf = DecisionTreeClassifier(max_depth = 5,

                                   random_state = 123,

                                   splitter  = "best",

                                   criterion = "gini",

                                  )

DecisionTree= model_performance_80_20('DecisionTree',clf,X_train,X_test,Y_train,Y_test)



clf = GaussianNB()

NaiveBase= model_performance_80_20('NaiveBase',clf,X_train,X_test,Y_train,Y_test)



# SVM

clf  = SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,

               decision_function_shape='ovr', degree=3, gamma=1.0, kernel='linear',

               max_iter=-1, probability=True, random_state=None, shrinking=True,

               tol=0.001, verbose=False)



SVM= model_performance_80_20('SVM',clf,X_train,X_test,Y_train,Y_test)



all_models_80_20 = pd.concat([DecisionTree,

                        LogisticReg,

                        XGB,

                        LightGBM,

                        RandomForest,

                        DecisionTree,

                             NaiveBase,

                             SVM])



all_models_80_20 = all_models_80_20.sort_values(by ='miss_classified(%)',ascending=True)
train_set = []

test_set = []



train_class_distribution = Y_train['Churn'].value_counts()

test_class_distribution = Y_test['Churn'].value_counts()



sorted_train = np.argsort(-train_class_distribution.values)

sorted_test = np.argsort(-test_class_distribution.values)



for i in sorted_train:

    train_set.append(np.round((train_class_distribution.values[i]/telcom_df.shape[0]*100), 3))

for i in sorted_test:

    test_set.append(np.round((test_class_distribution.values[i]/telcom_df.shape[0]*100),3))



distribution_per_set = pd.DataFrame(

    {'Train Set(%)': train_set,

     'Test Set(%)':test_set

    })

# Plotting Distribution per class 

distribution_per_set.index = distribution_per_set.index

distribution_per_set.plot.bar(figsize=(12,6))

plt.xticks(rotation=0)

plt.title('Distribution of data per set and Target Variable')

plt.xlabel('Churn')

plt.ylabel('% Of total data')
from imblearn.over_sampling import SMOTE



##seperating dependent and independent variables

features    = [i for i in telcom_df.columns if i not in Id_features + target_feature]





smote_X = telcom_df[features]

smote_Y = telcom_df[target_feature]



#Split train and test data

smote_train_X,smote_test_X,smote_train_Y,smote_test_Y = train_test_split(smote_X,smote_Y,

                                                                         test_size = .25 ,

                                                                         random_state = 111)



#oversampling minority class using smote

os = SMOTE(random_state = 0)

os_smote_X,os_smote_Y = os.fit_sample(smote_train_X,smote_train_Y)

os_smote_X = pd.DataFrame(data = os_smote_X,columns=features)

os_smote_Y = pd.DataFrame(data = os_smote_Y,columns=target_feature)

###
# Logistic Regression

logit  = LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,

          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,

          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,

          verbose=0, warm_start=False)



LogisticReg_SMOTE= model_performance_80_20('LogisticReg_SMOTE',logit,smote_train_X,smote_test_X,smote_train_Y,smote_test_Y)



# XGB

clf = XGBClassifier(base_score=0.1, booster='gbtree', colsample_bylevel=1,

                    colsample_bytree=1, gamma=0, learning_rate=0.9, max_delta_step=0,

                    max_depth = 7, min_child_weight=1, missing=None, n_estimators=100,

                    n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,

                    reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

                    silent=True, subsample=1)



XGB_SMOTE= model_performance_80_20('XGB_SMOTE',clf,smote_train_X,smote_test_X,smote_train_Y,smote_test_Y)



# LightGBM

clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,

                        learning_rate=0.5, max_depth=3, min_child_samples=20,

                        min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,

                        n_jobs=-1, num_leaves=500, objective='binary', random_state=None,

                        reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,

                        subsample_for_bin=200000, subsample_freq=0)

LightGBM_SMOTE= model_performance_80_20('LightGBM_SMOTE',clf,smote_train_X,smote_test_X,smote_train_Y,smote_test_Y)



# Random Forest

clf = RandomForestClassifier(n_estimators=100,  max_depth=10, criterion='gini',random_state=42, n_jobs=-1)

RandomForest_SMOTE= model_performance_80_20('RandomForest_SMOTE',clf,smote_train_X,smote_test_X,smote_train_Y,smote_test_Y)





# Decision Tree

clf = DecisionTreeClassifier(max_depth = 5,

                                   random_state = 123,

                                   splitter  = "best",

                                   criterion = "gini",

                                  )

DecisionTree_SMOTE= model_performance_80_20('DecisionTree_SMOTE',clf,smote_train_X,smote_test_X,smote_train_Y,smote_test_Y)



# Naive Base 

clf = GaussianNB()

NaiveBase_SMOTE= model_performance_80_20('NaiveBase_SMOTE',clf,smote_train_X,smote_test_X,smote_train_Y,smote_test_Y)





# SVM

clf  = SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,

               decision_function_shape='ovr', degree=3, gamma=1.0, kernel='linear',

               max_iter=-1, probability=True, random_state=None, shrinking=True,

               tol=0.001, verbose=False)



SVM_SMOTE= model_performance_80_20('SVM_SMOTE',clf,smote_train_X,smote_test_X,smote_train_Y,smote_test_Y)





all_models_SMOTE = pd.concat([DecisionTree_SMOTE,

                        LogisticReg_SMOTE,

                        XGB_SMOTE,

                        LightGBM_SMOTE,

                        RandomForest_SMOTE,

                        DecisionTree_SMOTE,

                             NaiveBase_SMOTE,

                             SVM_SMOTE])



all_models_SMOTE = all_models_SMOTE.sort_values(by ='miss_classified(%)',ascending=True)
comb_models = pd.concat([all_models_80_20,all_models_SMOTE ]) 



comb_models = comb_models.sort_values(by ='accuracy score',ascending=True)

x1 = comb_models['accuracy score']

y1 = comb_models['Model']

comb_models = comb_models.sort_values(by ='recall score',ascending=True)

x2 = comb_models['recall score']

y2 = comb_models['Model']

comb_models = comb_models.sort_values(by ='precision score',ascending=True)

x3 = comb_models['precision score']

y3 = comb_models['Model']



fig = plotly.tools.make_subplots(rows=1,cols=3,horizontal_spacing = 0.132)



fig.append_trace(go.Bar(

            x=x1,

            y=y1,

            orientation='h',

            name = 'Accuracy Score'),1,1)



fig.append_trace(go.Bar(

            x=x2,

            y=y2,

            orientation='h',

            name = 'Recall Score'),1,2)



fig.append_trace(go.Bar(

            x=x3,

            y=y3,

            orientation='h',

            name = 'Precision'),1,3)



fig['layout'].update(showlegend=True, title="Model performance" ,

                     autosize = False,height = 600,width = 1200,

                     plot_bgcolor = 'rgba(240,240,240, 0.95)',

                     paper_bgcolor = 'rgba(240,240,240, 0.95)',

                     )



fig['layout']['xaxis1'].update(title='Accuracy Score')

fig['layout']['xaxis2'].update(title='Recall Score')

fig['layout']['xaxis3'].update(title='Precision Score')



fig.layout.update(go.Layout(barmode = 'overlay',title='All Models Performance'))

py.iplot(fig)
def single_model_performance(name,model,X_training,X_testing,Y_training,Y_testing,features):

    

    model.fit(X_training.values,Y_training.values)

    Y_predicted = model.predict(X_testing.values)

    Y_prob_predicted = model.predict_proba(X_testing.values)

    

    # Performace matrices 

    conf_matrix = confusion_matrix(Y_testing,Y_predicted)     

    precision =(conf_matrix/conf_matrix.sum(axis=0))

    recall =(((conf_matrix.T)/(conf_matrix.sum(axis=1))).T)

    

    model_roc_auc = roc_auc_score(Y_testing,Y_predicted) 

    print ("Area under curve : ",model_roc_auc,"\n")

    fpr,tpr,thresholds = roc_curve(Y_testing,Y_prob_predicted[:,1])

    

    #coefficients  = pd.DataFrame(algorithm.feature_importances_)



    coeffs  = pd.DataFrame(model.coef_.ravel())    

    feature_df     = pd.DataFrame(features)

    coef_sumry    = (pd.merge(coeffs,feature_df,left_index= True,

                              right_index= True, how = "left"))

    

    coef_sumry.columns = ["coefficients","features"]

    coef_sumry    = coef_sumry.sort_values(by = "coefficients",ascending = False)

    



    print ("\n Classification report : \n",classification_report(Y_testing,Y_predicted))

    

    # Plotting 

    

    #Precision Matrix

    trace1 = go.Heatmap(z=precision,

                           x=['Not Churn','Churn'],

                           y=['Not Churn','Churn'], 

                           name = 'Precision',

                           colorscale = "Portland",showscale  = False

                           )



    #Recall Matrix

    trace2 = go.Heatmap(z=recall,

                           x=['Not Churn','Churn'],

                           y=['Not Churn','Churn'], 

                           name = 'Recall',

                           colorscale = "Portland",showscale  = False)

   # AUC  

    trace3 = go.Scatter(x = fpr,

                        y = tpr, 

                        name = 'ROC'+ str(model_roc_auc),

                        mode = "markers",

                        marker = dict(color = 'blue',symbol =  "diamond-open"))

    

    trace4 = go.Scatter(x = [0,1],

                        y = [0,1],

                        name=None,

                        line = dict(color = ('red'),width = 2,

                        dash = 'dot'))



    

    # Feature Importance

    

    trace5 = go.Bar(x = coef_sumry["features"],

                    y = coef_sumry["coefficients"],name = "Feature Importance")



    fig = plotly.tools.make_subplots(rows=5, cols=2,

        specs=[[{}, {"rowspan": 2}],

               [{}, None],

               [{"rowspan": 2, "colspan": 2}, None],

               [None, None],

               [{}, {}]],

        print_grid=True,subplot_titles=('Precision',

                                            'Receiver operating characteristic',

                                            'Recall','Feature Importances'))



    fig.add_trace(trace1,row=1, col=1)

    fig.add_trace(trace2, row=2, col=1)

    fig.add_trace(trace3, row=1, col=2)

    fig.add_trace(trace4, row=1, col=2)

    fig.add_trace(trace5, row=3, col=1)

    

    fig['layout']['xaxis2'].update(title='False PR')

    fig['layout']['yaxis2'].update(title='True PR')

    

    fig['layout'].update(go.Layout(height=800, width=1000, title_text="{} Model Performance".format(name),showlegend=False))

    py.iplot(fig)

    

features = [i for i in telcom_df.columns if i not in Id_features + target_feature]



logit  = LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,

          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,

          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,

          verbose=0, warm_start=False)



single_model_performance('Logistic Regression',logit,X_train,X_test,Y_train,Y_test,features)
features = [i for i in telcom_df.columns if i not in Id_features + target_feature]



logit  = LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,

          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,

          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,

          verbose=0, warm_start=False)



def get_features(model,features,X_train,Y_train):

    

    model.fit(X_train.values,Y_train.values)

    

    coeffs  = pd.DataFrame(model.coef_.ravel())    

    feature_df     = pd.DataFrame(features)

    coef_sumry    = (pd.merge(coeffs,

                              feature_df,

                              left_index= True,

                              right_index= True, how = "left"))

    

    coef_sumry.columns = ["coefficients","features"]

    coef_sumry    = coef_sumry.sort_values(by = "coefficients",ascending = False)

    

    return coef_sumry



get_features(logit,features,X_train,Y_train).head(10)
from sklearn.base import clone 



# Negative importance means that removing a given feature from the model actually improves the performance

def drop_col_feat_imp(model, X_train, y_train, random_state = 42):

    

    # clone the model to have the exact same specification as the one initially trained

    model_clone = clone(model)

    # set random_state for comparability

    model_clone.random_state = random_state

    # training and scoring the benchmark model

    model_clone.fit(X_train, y_train)

    benchmark_score = model_clone.score(X_train, y_train)

    # list for storing feature importances

    importances = []

    

    # iterating over all columns and storing feature importance (difference between benchmark and new model)

    for col in X_train.columns:

        model_clone = clone(model)

        model_clone.random_state = random_state

        model_clone.fit(X_train.drop(col, axis = 1), y_train)

        drop_col_score = model_clone.score(X_train.drop(col, axis = 1), y_train)

        importances.append(benchmark_score - drop_col_score)

    

    importances_df = pd.DataFrame({'Column':X_train.columns,

                                   'Importance':importances})

    return importances_df.sort_values(by = "Importance",ascending = False)



drop_col_feat_imp(logit,X_train,Y_train).head(10)