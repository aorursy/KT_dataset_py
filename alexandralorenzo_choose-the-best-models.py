# Import Packages

# data analysis and wrangling
import pandas as pd
import numpy as np

# visualization tool
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff

#independence test
import scipy

# Data Processing
from sklearn.preprocessing import MinMaxScaler

#imbalanced data
from imblearn.combine import SMOTETomek

# evaluation models
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.feature_selection import RFE
from yellowbrick.classifier import DiscriminationThreshold

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

#Load Data 
dataset = pd.read_csv('../input/data_v1.0.csv', delimiter=',')
dataset = dataset.drop("Unnamed: 0",1) #Not Useful Column
print ('\033[1m' +"Rows     : " +'\033[0m' ,dataset.shape[0])
print ('\033[1m' + "Columns  : "  +'\033[0m',dataset.shape[1])

print('\033[1m'+"Types of variables :\n" +'\033[0m',dataset.dtypes)
print( '\033[1m'+"Non Missing Values :\n" +'\033[0m',dataset.count())
dataset.head(3) # preview the data
dataset.isnull().sum() #Delete Missing Values (<1% of Total Obs.)
data_wrangling= dataset.dropna()
only_na = dataset[~dataset.index.isin(data_wrangling.index)] # Check Observations with Missing Values deleted 
data_wrangling.count()
only_na.head(4)
data_wrangling = data_wrangling.drop_duplicates() #No Duplicates
data_wrangling.count()
data_wrangling.describe()
# Delete if Age is less than 18, Experience < 0 , 
data_wrangling = data_wrangling[data_wrangling["age"] > 16] # 454 obs. deleted
data_wrangling = data_wrangling[data_wrangling["age"] <= 70] #  obs. deleted
data_wrangling = data_wrangling[data_wrangling["exp"] > 0] #2 obs. deleted
data_wrangling.count()
# If the Note is higher than 100 than 100
data_wrangling.loc[(data_wrangling['note'] > 100), 'note'] = 100
#Change Date Type
data_wrangling['date'] = pd.to_datetime(data_wrangling['date'],format="%Y-%m-%d")
data_wrangling['year_candidature']=  data_wrangling['date'].dt.year
data_wrangling['month_candidature']=  data_wrangling['date'].dt.month
data_wrangling['day_candidature']=  data_wrangling['date'].dt.day
data_wrangling['c_exp'] = pd.qcut(data_wrangling['exp'],3, precision =0)
data_wrangling['c_age'] = pd.qcut(data_wrangling['age'], 3, precision =0)
data_wrangling['c_note'] = pd.qcut(data_wrangling['note'],4, precision =0)
data_wrangling['c_salaire'] = pd.qcut(data_wrangling['salaire'],5, precision =0)
data_wrangling.tail(2)
# Copy the dataframe : keep indexes
scaled_features = data_wrangling.copy()
 
#Choose the variables to standardize
features = scaled_features[['age', 'exp','note','salaire']]
#Standardized
scaler = MinMaxScaler().fit(features.values)
features = scaler.transform(features.values)

scaled_features[['age', 'exp','note','salaire']] = features
scaled_features.rename(columns={'age':'norm_age', 'exp':'norm_exp', 'salaire':'norm_salaire', 'note':'norm_note' }, inplace=True)

data_cleaning = pd.merge(data_wrangling, scaled_features[['norm_age','norm_exp','norm_salaire','norm_note','index']],
                 left_on='index',
                 right_on='index',
                 how='inner')
#Pie Plot
#labels
lab = data_cleaning["embauche"].value_counts().keys().tolist()
#values
val = data_cleaning["embauche"].value_counts().values.tolist()

fig1, ax1 = plt.subplots()
ax1.pie(val, labels=lab, shadow=True, autopct='%1.0f%%', startangle=90, colors=['#00bf81','#d900b3'])
ax1.axis('equal')
plt.show()
# Unbalanced Data
data_cleaning[data_cleaning.embauche==1].describe()
data_cleaning[data_cleaning.embauche==0].describe()
# Draw a nested boxplot per embauche
import warnings; warnings.simplefilter('ignore')

numeric_features = ['age','exp','salaire','note']    
target = ['embauche']

def sns_boxplot(data,target,numeric_feature):  
    fig, ax = plt.subplots(figsize=(7,5))
    sns.boxplot(x=target, y=numeric_feature,
                hue=target, palette=["#00bf81", "#d900b3"],
                data=data, showmeans=True)
    plt.show()

for i in numeric_features:
    sns_boxplot(data_cleaning,"embauche",i)
# plot Histogram
data_embauche1= data_cleaning[data_cleaning["embauche"] == 1]
data_embauche0= data_cleaning[data_cleaning["embauche"] == 0]

def histogram(data_target,data_nntarget,numeric_feature):  
    f, axes = plt.subplots(1,2, figsize=(11, 5), sharex=True)
    sns.distplot( data_nntarget[numeric_feature] , color="#00bf81", ax=axes[0],bins=15,kde=True).set_title("Embauche = 0")
    sns.distplot( data_target[numeric_feature] , color="#d900b3", ax=axes[1],bins=15,kde=True).set_title("Embauche = 1")

for i in numeric_features:
    histogram(data_embauche1,data_embauche0,i)
categorical_features = ['c_age','c_exp','c_note','c_salaire'] 

def cat_distribution(data,target,categorical_features):
    x, y, hue = categorical_features, "prop", target


    prop_df = (data[categorical_features]
               .groupby(data_cleaning[hue])
               .value_counts(normalize=True)
               .rename(y)
               .reset_index())
    prop_df = prop_df.sort_values([categorical_features]).reset_index(drop=True)

    f, ax = plt.subplots(figsize=(8, 6))
    ax = sns.barplot(x=x, y=y, hue=target, data=prop_df,palette=['#4cd2a6','#e032c2']).set_title(x + " distribution")

for i in categorical_features:
    cat_distribution(data_cleaning,"embauche",i)
categorical_features = ['cheveux','sexe','diplome','specialite','dispo','year_candidature','month_candidature','day_candidature'] 

def cat_distribution(data,target,categorical_features):
    x, y, hue = categorical_features, "prop", target


    prop_df = (data[categorical_features]
               .groupby(data_cleaning[hue])
               .value_counts(normalize=True)
               .rename(y)
               .reset_index())

    f, ax = plt.subplots(figsize=(7, 5))
    ax = sns.barplot(x=x, y=y, hue=target, data=prop_df,palette=['#4cd2a6','#e032c2']).set_title(x + " distribution")

for i in categorical_features:
    cat_distribution(data_cleaning,"embauche",i)
categorical_variables = ['cheveux','sexe','diplome','specialite','dispo','c_age','c_note','c_salaire','c_exp']
final_result = pd.DataFrame(columns=['statchisq','pvalue','cat_variable'])

for i in categorical_variables:
    # Chi-square test
    ct1= pd.crosstab(data_cleaning['embauche'],data_cleaning[i])
    colsum=ct1.sum(axis=0)
    colpct =ct1/colsum
    cs1 =scipy.stats.chi2_contingency(ct1)

    # Add the statchisq and pvalue
    statchisq = cs1[0]
    pvalue = cs1[1]

    result = [statchisq,pvalue,i]
    final_result.loc[len(final_result)]=result
    

final_result.sort_values(by=['pvalue'])

categorical_variables = ['cheveux','sexe','diplome','specialite','dispo','c_age','c_note','c_salaire','c_exp']
mfinal_result = pd.DataFrame(columns=['statchisq','pvalue','variable1','variable2'])

for i in categorical_variables:
    for j in categorical_variables:
        # Chi-square test
        if i != j:
            ct1= pd.crosstab(data_cleaning[j],data_cleaning[i])
            colsum=ct1.sum(axis=0)
            colpct =ct1/colsum
            cs1 =scipy.stats.chi2_contingency(ct1)

            # Add the statchisq and pvalue
            statchisq = cs1[0]
            pvalue = cs1[1]

            result = [statchisq,pvalue,i,j]
            mfinal_result.loc[len(mfinal_result)]=result


mfinal_result.sort_values(by=['variable1']).head(30) # multicollinéarité sexe & cheveux / sexe& specialite / diplome & dispo / 
#The Kruskal-Wallis H-test tests the null hypothesis that the population median of all of the groups are equal. 
#It is a non-parametric version of ANOVA. Since ANOVA has strong assumptions
numerical_variables =['age','exp','note','salaire','day_candidature','month_candidature','year_candidature']

for i in numerical_variables:
    # compare samples
    stat, p = scipy.stats.kruskal(data_embauche0[i], data_embauche1[i])
    print('Statistics=%.3f, p=%.3f' % (stat, p) +", variable:" +i)
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distributions (fail to reject H0)')
    else:
        print('Different distributions (reject H0)')
data_blond =data_cleaning[data_cleaning["cheveux"]=="blond"]
data_brun = data_cleaning[data_cleaning["cheveux"]=="brun"]
data_roux =data_cleaning[data_cleaning["cheveux"]=="roux"]
data_chatain =data_cleaning[data_cleaning["cheveux"]=="chatain"]
stat, p = scipy.stats.kruskal(data_blond["salaire"], data_brun["salaire"],data_roux["salaire"] ,data_chatain["salaire"])

print('Statistics=%.3f, p=%.3f' % (stat, p) +" " +i)
# interpret
alpha = 0.001
if p > alpha:
    print('Same distributions (fail to reject H0)')
else:
    print('Different distributions (reject H0)')
# calculate the correlation matrix (Spearman-Correlation)
# Only on Numeric Features
data_numeric = data_cleaning[['exp','note','age','salaire','year_candidature','month_candidature','day_candidature']]

def heatMap(df):
   #Create Correlation df
    corr = df.corr(method='spearman')
    #Plot figsize
    fig, ax = plt.subplots(figsize=(7, 7))
    #Generate Color Map, red & blue
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    #Generate Heat Map, allow annotations and place floats in map
    sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
    #Apply xticks
    plt.xticks(range(len(corr.columns)), corr.columns);
    #Apply yticks
    plt.yticks(range(len(corr.columns)), corr.columns)
    #show plot
    plt.show()

heatMap(data_numeric)

#Spearman, Pearson, Kendall-tau correlation
variables =data_numeric.columns
method = ['pearson','spearman','kendalltau']
corr_result = pd.DataFrame(columns=['variable1','variable2','corr_spearman','pvalue_spearman','corr_pearson',
                                    'pvalue_pearson','corr_kendalltau','pvalue_kendalltau'])

for i in variables:
    for j in variables:
        if j!= i:
            corr1 =scipy.stats.spearmanr(data_numeric[i], data_numeric[j]) 
            corr2 =scipy.stats.pearsonr(data_numeric[i], data_numeric[j])
            corr3 =scipy.stats.kendalltau(data_numeric[i], data_numeric[j])
            # Add the corr and pvalue per method
            corrspearman = corr1[0]
            pvaluespearman = corr1[1]
            corrpearson = corr2[0]
            pvaluepearson = corr2[1]
            corrkendalltau = corr3[0]
            pvaluekendalltau = corr3[1]
            
            result = [i,j,corrspearman,pvaluespearman,corrpearson,pvaluepearson,corrkendalltau,pvaluekendalltau]
            corr_result.loc[len(corr_result)]=result
           
corr_result.drop_duplicates(subset=['pvalue_spearman'], keep='first').sort_values(by='pvalue_spearman')

data_modelling = pd.get_dummies(data_cleaning, prefix_sep='_', drop_first=True)
data_modelling.columns

#Splitting train and test data 
train,test = train_test_split(data_modelling,test_size = .25 ,random_state = 111)

##seperating dependent and independent variables
cols    = [i for i in data_modelling.columns if i not in 'index' + 'embauche' + 'date']
train_X = train[cols]
train_Y = train['embauche']
test_X  = test[cols]
test_Y  = test['embauche']
indices_train=train_Y.index.values
indices_test=test_Y.index.values

target_count = train.embauche.value_counts()
test_count=test.embauche.value_counts()
print('Class 0 (Training):', target_count[0],'/Class 0 (Test):',test_count[0])
print('Class 1 (Training):', target_count[1],'/Class 1 (Test):',test_count[1])

#Oversampling minority class using smote and Undersampling using Tomek links
smotetomek_X = train[cols]
smotetomek_Y = train['embauche']

smote_tomek = SMOTETomek(random_state=42, ratio=0.5) #Ratio !=1 the objective is to predict class 0 AND 1
X_resampled, y_resampled = smote_tomek.fit_resample(train_X, train_Y)

smotetomek_X = pd.DataFrame(data = X_resampled,columns=cols)
smotetomek_Y = pd.DataFrame(data = y_resampled,columns=['embauche'])
print ((smotetomek_Y['embauche'] == 1).sum())
print ((smotetomek_Y['embauche'] == 0).sum())

from sklearn import preprocessing

#Define Variables
col_num=['age','exp','note','salaire']
col_norm_num = ['norm_age','norm_exp', 'norm_salaire', 'norm_note']
col_cat_num = ['c_exp_(8.0, 11.0]', 'c_exp_(11.0, 23.0]', 'c_age_(32.0, 39.0]',
               'c_age_(39.0, 69.0]', 'c_note_(64.0, 75.0]', 'c_note_(75.0, 87.0]',
               'c_note_(87.0, 100.0]', 'c_salaire_(30757.0, 33701.0]',
               'c_salaire_(33701.0, 36220.0]', 'c_salaire_(36220.0, 39172.0]',
               'c_salaire_(39172.0, 53977.0]']
col_cat =['year_candidature', 'month_candidature', 'day_candidature','cheveux_brun', 'cheveux_chatain', 'cheveux_roux', 'sexe_M',
       'diplome_doctorat', 'diplome_licence', 'diplome_master','specialite_detective', 'specialite_forage', 'specialite_geologie',
       'dispo_oui']
col_index= ['index']
target = ['embauche']
# Function created to tune the parameters using AUC 
def hyperparameters_def(parameter_grid, kfold, algorithm,X,Y):   
    acc_scorer = metrics.make_scorer(metrics.accuracy_score) #Choose AUC score
    
    grid_search_algo = GridSearchCV(algorithm, param_grid = parameter_grid,
                              cv = kfold,scoring=acc_scorer) #K-Folds: 10

    grid_search_algo.fit(X, Y)

    print ("Best Score: {}".format(grid_search_algo.best_score_)) 
    print ("Best params: {}".format(grid_search_algo.best_params_)) 
def threshold_def(best_classifier,X,Y):
    visualizer = DiscriminationThreshold(best_classifier)

    visualizer.fit(X, Y)  # Fit the training data to the visualizer
    visualizer.poof()     # Draw/show/poof the data
py.init_notebook_mode(connected=True)

def evaluation(algorithm,cols,cf,X,Y):

    algorithm.fit(X,Y)
    predictions   = algorithm.predict(test_X[cols])
    probabilities = algorithm.predict_proba(test_X[cols])

    #confusion matrix
    conf_matrix = metrics.confusion_matrix(test_Y,predictions)
    #roc_auc_score
    model_roc_auc = metrics.roc_auc_score(test_Y,predictions) 

    print ("Area under curve on Test Set: ",model_roc_auc,"\n")
    print("F1 on Test Set", metrics.f1_score(test_Y, predictions),"\n")
    print ("Accuracy   Score on Test Set : ",metrics.accuracy_score(test_Y,predictions))
    
    #coeffs
    if   cf == "coefficients" :
        coefficients  = pd.DataFrame(algorithm.coef_.ravel())
    elif cf == "features" :
        coefficients  = pd.DataFrame(algorithm.feature_importances_)
    
    column_df     = pd.DataFrame(cols)
    coef_sumry    = (pd.merge(coefficients,column_df,left_index= True,
                              right_index= True, how = "left"))
    
    coef_sumry.columns = ["coefficients","features"]
    coef_sumry    = coef_sumry.sort_values(by = "coefficients",ascending = False)
    #FPR,TPR
    fpr,tpr,thresholds = metrics.roc_curve(test_Y,probabilities[:,1])
    
    #plot confusion matrix
    trace1 = go.Heatmap(z = conf_matrix ,
                        x = ["Not churn","Churn"],
                        y = ["Not churn","Churn"],
                        showscale  = False,colorscale = "Picnic",
                        name = "matrix")

    #plot roc curve
    trace2 = go.Scatter(x = fpr,y = tpr,
                        name = "Roc : " + str(model_roc_auc),
                        line = dict(color = ('rgb(22, 96, 167)'),width = 2))
    trace3 = go.Scatter(x = [0,1],y=[0,1],
                        line = dict(color = ('rgb(205, 12, 24)'),width = 2,
                        dash = 'dot'))

    #plot coeffs
    trace4 = go.Bar(x = coef_sumry["features"],y = coef_sumry["coefficients"],
                    name = "coefficients",
                    marker = dict(color = coef_sumry["coefficients"],
                                  colorscale = "Picnic",
                                  line = dict(width = .6,color = "black")))

    #subplots
    fig = tls.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],
                            subplot_titles=('Confusion Matrix',
                                            'Receiver operating characteristic',
                                            'Feature Importances'),print_grid=False)

    fig.append_trace(trace1,1,1)
    fig.append_trace(trace2,1,2)
    fig.append_trace(trace3,1,2)
    fig.append_trace(trace4,2,1)

    fig['layout'].update(showlegend=False, title="Model performance" ,
                         autosize = False,height = 900,width = 800,
                         plot_bgcolor = 'rgba(240,240,240, 0.95)',
                         paper_bgcolor = 'rgba(240,240,240, 0.95)',
                         margin = dict(b = 195))
    fig["layout"]["xaxis2"].update(dict(title = "false positive rate"))
    fig["layout"]["yaxis2"].update(dict(title = "true positive rate"))
    fig["layout"]["xaxis3"].update(dict(showgrid = True,tickfont = dict(size = 10),
                                        tickangle = 90))
    py.iplot(fig)
parameter_grid = {'Cs': [1, 2, 3, 4, 5, 6 ,7 ,8 ,9 ,10]
                  }
X_lr = smotetomek_X[['norm_age','norm_salaire','year_candidature', 'month_candidature', 'day_candidature','sexe_M','diplome_doctorat', 
                     'diplome_licence', 'diplome_master','specialite_detective', 'specialite_forage', 'specialite_geologie']]

logistic = LogisticRegressionCV(random_state= 0,cv=10)

hyperparameters_def(parameter_grid,10,logistic,X_lr,smotetomek_Y)
logit= LogisticRegressionCV(random_state=0 ,Cs=1)
threshold_def(logit,X_lr,smotetomek_Y)
cols = ['norm_age','norm_salaire','year_candidature', 'month_candidature', 'day_candidature','sexe_M','diplome_doctorat', 
                     'diplome_licence', 'diplome_master','specialite_detective', 'specialite_forage', 'specialite_geologie']

evaluation (logit, cols,"coefficients", X_lr,smotetomek_Y)
#Tuning the parameters using AUC

decision_tree_classifier = DecisionTreeClassifier(random_state=0)
X_decisiontree = smotetomek_X[col_num + col_cat]
parameter_grid = {'max_depth': [1, 2, 3, 4, 5, 6, 7], #Higher value will overfit the model
                  'max_features': [1, 2, 3, 4, 5],
                  'criterion': ['gini','entropy'],
                  'splitter': ['best','random'],
                  }
    
hyperparameters_def(parameter_grid,10,decision_tree_classifier,X_decisiontree,smotetomek_Y.values.ravel())
decision_tree= DecisionTreeClassifier(random_state=0 ,max_features= 5, criterion='gini',splitter= 'best', max_depth= 8)
threshold_def(decision_tree,X_decisiontree,smotetomek_Y.values.ravel())

cols= col_num + col_cat

evaluation (decision_tree, cols, "features", X_decisiontree,smotetomek_Y.values.ravel())
X_randomforest = smotetomek_X[col_num + col_cat]
param_grid = {
    'min_samples_split': [3, 5, 10], 
    #'max_depth': [2, 3, 5, 15, 25],
    'max_features': ['auto', 'sqrt'],
    'bootstrap': [True, False],
    'criterion': ['gini','entropy']   
}

# Create a based model
#To obtain a deterministic behaviour during fitting
rf = RandomForestClassifier(random_state=0, n_estimators=10)#n_estimators = 100 really slow

hyperparameters_def(param_grid,10,rf,X_randomforest,smotetomek_Y.values.ravel())
rf= RandomForestClassifier(random_state=0, n_estimators=10,bootstrap= False, max_features= 'auto', criterion='entropy',min_samples_split= 5)
threshold_def(rf,X_randomforest,smotetomek_Y.values.ravel())

cols= col_num + col_cat

evaluation (rf, cols, "features", X_randomforest,smotetomek_Y.values.ravel())
from sklearn.svm import SVC
X_svm = smotetomek_X[col_norm_num + col_cat]
svc_lin  = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
               decision_function_shape='ovr', degree=3, gamma=1.0, kernel='linear',
               max_iter=-1, probability=True, random_state=None, shrinking=True,
               tol=0.001, verbose=False)
svc_lin.fit(X_svm,smotetomek_Y) 
cols= col_norm_num + col_cat

evaluation(svc_lin, cols, "coefficients", X_svm,smotetomek_Y)
xgc = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                    colsample_bytree=1, gamma=0, learning_rate=0., max_delta_step=0,
                    max_depth = 8, min_child_weight=1, missing=None, n_estimators=1000,
                    n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
                    reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                    silent=True, subsample=1)

X_Xgboost = smotetomek_X[col_num + col_cat]

#hyperparameters_def(param_grid,10,xgc,X_Xgboost,smotetomek_Y)
#threshold_def(xgc,X_Xgboost,smotetomek_Y)
cols= col_num + col_cat

evaluation (xgc, cols, "features", X_Xgboost,smotetomek_Y)
cols= col_norm_num + col_cat
X_KNN = smotetomek_X[col_norm_num + col_cat]

param_grid = {
    'leaf_size':[5,10,20,30], 
    'n_neighbors':[3,4,5,6,7]
}

# Create a based model
#To obtain a deterministic behaviour during fitting
knn_test = KNeighborsClassifier(algorithm='auto')

hyperparameters_def(param_grid,10,knn_test,X_KNN,smotetomek_Y)
knn = KNeighborsClassifier(algorithm='auto', leaf_size=10, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=4, p=2,
           weights='uniform')
threshold_def(knn,X_KNN,smotetomek_Y)
cols= col_norm_num + col_cat
 #model
knn.fit(X_KNN,smotetomek_Y)
predictions   = knn.predict(test_X[cols])
probabilities = knn.predict_proba(test_X[cols])
    
print ("Accuracy Score   : ",metrics.accuracy_score(test_Y,predictions))
#confusion matrix
conf_matrix = metrics.confusion_matrix(test_Y,predictions)
#roc_auc_score
model_roc_auc = metrics.roc_auc_score(test_Y,predictions) 
print ("Area under curve : ",model_roc_auc)
fpr,tpr,thresholds = metrics.roc_curve(test_Y,probabilities[:,1])
   
#plot roc curve
trace1 = go.Scatter(x = fpr,y = tpr,
                    name = "Roc : " + str(model_roc_auc),
                    line = dict(color = ('rgb(22, 96, 167)'),width = 2),
                   )
trace2 = go.Scatter(x = [0,1],y=[0,1],
                    line = dict(color = ('rgb(205, 12, 24)'),width = 2,
                    dash = 'dot'))

#plot confusion matrix
trace3 = go.Heatmap(z = conf_matrix ,x = ["Not churn","Churn"],
                    y = ["Not churn","Churn"],
                    showscale  = False,colorscale = "Blues",name = "matrix",
                    xaxis = "x2",yaxis = "y2"
                   )

layout = go.Layout(dict(title="Model performance" ,
                        autosize = False,height = 500,width = 800,
                        showlegend = False,
                        plot_bgcolor  = "rgb(243,243,243)",
                        paper_bgcolor = "rgb(243,243,243)",
                        xaxis = dict(title = "false positive rate",
                                     gridcolor = 'rgb(255, 255, 255)',
                                     domain=[0, 0.6],
                                     ticklen=5,gridwidth=2),
                        yaxis = dict(title = "true positive rate",
                                     gridcolor = 'rgb(255, 255, 255)',
                                     zerolinewidth=1,
                                     ticklen=5,gridwidth=2),
                        margin = dict(b=200),
                        xaxis2=dict(domain=[0.7, 1],tickangle = 90,
                                    gridcolor = 'rgb(255, 255, 255)'),
                        yaxis2=dict(anchor='x2',gridcolor = 'rgb(255, 255, 255)')
                       )
              )
data = [trace1,trace2,trace3]
fig = go.Figure(data=data,layout=layout)

py.iplot(fig)