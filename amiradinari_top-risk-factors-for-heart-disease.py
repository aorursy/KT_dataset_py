!pip install git+https://github.com/fastai/fastai@2e1ccb58121dc648751e2109fc0fbf6925aa8887 2>/dev/null 1>/dev/null
from fastai.imports import *

from fastai.structured import *

from pandas_summary import DataFrameSummary

from sklearn.ensemble import RandomForestClassifier

from IPython.display import display

from sklearn import metrics

from sklearn.model_selection import train_test_split

import numpy as np 

import pandas as pd

from sklearn.ensemble import RandomForestClassifier #for the model

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz #plot tree

from sklearn.metrics import roc_curve, auc #for model evaluation

from sklearn.metrics import classification_report #for model evaluation

from sklearn.metrics import confusion_matrix #for model evaluation

from sklearn.model_selection import train_test_split #for data splitting

import eli5 #for purmutation importance

import xgboost as xgb

from xgboost import XGBClassifier

from eli5.sklearn import PermutationImportance

import shap #for SHAP values

from pdpbox import pdp, info_plots #for partial plots

%load_ext autoreload

%autoreload 2

%matplotlib inline 

pd.options.mode.chained_assignment = None
df = pd.read_csv("../input/heart.csv")
## from this kernel: https://www.kaggle.com/tentotheminus9/what-causes-heart-disease-explaining-the-model



df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',

       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']



df['sex'][df['sex'] == 0] = 'female'

df['sex'][df['sex'] == 1] = 'male'



df['chest_pain_type'][df['chest_pain_type'] == 1] = 'typical angina'

df['chest_pain_type'][df['chest_pain_type'] == 2] = 'atypical angina'

df['chest_pain_type'][df['chest_pain_type'] == 3] = 'non-anginal pain'

df['chest_pain_type'][df['chest_pain_type'] == 4] = 'asymptomatic'



df['fasting_blood_sugar'][df['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'

df['fasting_blood_sugar'][df['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'



df['rest_ecg'][df['rest_ecg'] == 0] = 'normal'

df['rest_ecg'][df['rest_ecg'] == 1] = 'ST-T wave abnormality'

df['rest_ecg'][df['rest_ecg'] == 2] = 'left ventricular hypertrophy'



df['exercise_induced_angina'][df['exercise_induced_angina'] == 0] = 'no'

df['exercise_induced_angina'][df['exercise_induced_angina'] == 1] = 'yes'



df['st_slope'][df['st_slope'] == 1] = 'upsloping'

df['st_slope'][df['st_slope'] == 2] = 'flat'

df['st_slope'][df['st_slope'] == 3] = 'downsloping'



df['thalassemia'][df['thalassemia'] == 1] = 'normal'

df['thalassemia'][df['thalassemia'] == 2] = 'fixed defect'

df['thalassemia'][df['thalassemia'] == 3] = 'reversable defect'
df

import pandas_profiling

df_original=df
profile = pandas_profiling.ProfileReport(df)
profile
df.columns
df.chest_pain_type = df.chest_pain_type.astype("category")

df.exercise_induced_angina = df.exercise_induced_angina.astype("category")

df.fasting_blood_sugar = df.fasting_blood_sugar.astype("category")

df.rest_ecg = df.rest_ecg.astype("category")

df.sex = df.sex.astype("category")

df.st_slope = df.st_slope.astype("category")

df.thalassemia = df.thalassemia.astype("category")
df = pd.get_dummies(df, drop_first=True)
df_p,y,_=proc_df(df,"target")
df_p.head()
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

# Make an instance of the Model

pca = PCA(3)

df_p,y,_=proc_df(df_original,"target")

pca.fit(df_p)

reduced_df= pca.transform(df_p)

plt.scatter(reduced_df[:, 0], reduced_df[:, 1], alpha=0.2)

plt.xlabel('component 1')

plt.ylabel('component 2')

from mpl_toolkits.mplot3d import Axes3D

plt.clf()

fig = plt.figure(1, figsize=(4, 3))

ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()

ax.scatter(reduced_df[:, 0], reduced_df[:, 1], reduced_df[:, 2], c=y, edgecolor='k')

plt.show()


from sklearn.manifold import TSNE

tsne =TSNE(n_components=2, init='pca', random_state=0)

x_tsne = tsne.fit_transform(reduced_df)

plt.scatter(x_tsne[:, 0], x_tsne[:, 1], alpha=0.2)

plt.xlabel('component 1')

plt.ylabel('component 2')



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(df.iloc[:,:-1])

x_changed=scaler.transform(df.drop(['target'],axis=1))

x_change_pd=pd.DataFrame(x_changed,columns=df.drop(['target'],axis=1).columns)

X_train, X_test, y_train, y_test = train_test_split(x_change_pd, df['target'], test_size = .2, random_state=10) #split the data

model = RandomForestClassifier(max_depth=5)

model.fit(X_train, y_train)

estimator = model.estimators_[1]

feature_names = [i for i in x_change_pd.columns]



y_train_str = y_train.astype('str')

y_train_str[y_train_str == '0'] = 'no disease'

y_train_str[y_train_str == '1'] = 'disease'

y_train_str = y_train_str.values

export_graphviz(estimator, out_file='tree.dot', 

                feature_names = feature_names,

                class_names = y_train_str,

                rounded = True, proportion = True, 

                label='root',

                precision = 2, filled = True)



from subprocess import call

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])



from IPython.display import Image

Image(filename = 'tree.png')
from sklearn.metrics import accuracy_score

pred_forest=model.predict(X_test)

accuracy_forest = accuracy_score(y_test, pred_forest)

print("Accuracy: %.2f%%" % (accuracy_forest * 100.0))
data_dmatrix = xgb.DMatrix(data=X_train.iloc[:,:5],label=y_train)

''''''

params = {'learning_rate': 0.2,

                'max_depth':2, 'alpha': 3}

xg_class = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)

xgb.plot_tree(xg_class,num_trees=0)

plt.rcParams['figure.figsize'] = [50, 10]

plt.show()
from sklearn.metrics import accuracy_score

xg_classification = XGBClassifier(max_depth=4, learning_rate=0.1,

             alpha =10,n_estimators = 20, random_state = 1,eval_metric='error')

xg_classification.fit(X_train, y_train)



preds = xg_classification.predict(X_test)

accuracy = accuracy_score(y_test, preds)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
xgb.plot_importance(xg_class)

plt.rcParams['figure.figsize'] = [5, 5]

plt.show()
df.head()



x_change_pd
y_test
from sklearn.model_selection import RandomizedSearchCV
rf_param_grid = {

                 'max_depth' : [4, 6, 8,10],

                 'n_estimators': range(1,30),

                 'max_features': ['sqrt', 'auto', 'log2'],

                 'min_samples_split': [2, 3, 10,20],

                 'min_samples_leaf': [1, 3, 10,18],

                 'bootstrap': [True, False],

                 

                 }
m = RandomForestClassifier()
m_r = RandomizedSearchCV(param_distributions=rf_param_grid, 

                                    estimator = m, scoring = "accuracy", 

                                    verbose = 0, n_iter = 100, cv = 5)
%time m_r.fit(df_p, y)
m_r.best_score_
m_r.best_params_
rf_bp = m_r.best_params_
rf_classifier=RandomForestClassifier(n_estimators=rf_bp["n_estimators"],

                                     min_samples_split=rf_bp['min_samples_split'],

                                     min_samples_leaf=rf_bp['min_samples_leaf'],

                                     max_features=rf_bp['max_features'],

                                     max_depth=rf_bp['max_depth'],

                                     bootstrap=rf_bp['bootstrap'])
rf_classifier.fit(df_p,y)
fi = rf_feat_importance(rf_classifier,df_p)
def plot_fi(fi):

    return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
plot_fi(fi)
import lightgbm as lgbm
lgbm_model = lgbm.LGBMClassifier()
lgbm_params = {

    "n_estimators":[10,100,1000,2000],

    'boosting_type': ['dart','gbdt'],          

    'learning_rate': [0.05,0.1,0.2],       

    'min_split_gain': [0.0,0.1,0.5,0.7],     

    'min_child_weight': [0.001,0.003,0.01],     

    'num_leaves': [10,21,41,61],            

    'min_child_samples': [10,20,30,60,100]

              }
lgbm_model = lgbm.LGBMClassifier()
lgbm_c = RandomizedSearchCV(param_distributions=lgbm_params, 

                                    estimator = lgbm_model, scoring = "accuracy", 

                                    verbose = 0, n_iter = 100, cv = 4)
lgbm_c.fit(df_p,y)
lgbm_bp =lgbm_c.best_params_
lgbm_model = lgbm.LGBMClassifier(num_leaves=lgbm_bp["num_leaves"],

                                 n_estimators=lgbm_bp["n_estimators"],

                                 min_split_gain=lgbm_bp["min_split_gain"],

                                 min_child_weight=lgbm_bp["min_child_weight"],

                                 min_child_samples=lgbm_bp["min_child_samples"],

                                 learning_rate=lgbm_bp["learning_rate"],

                                 boosting_type=lgbm_bp["boosting_type"])
lgbm_model.fit(df_p,y)
def feature_imp(df,model):

    fi = pd.DataFrame()

    fi["feature"] = df.columns

    fi["importance"] = model.feature_importances_

    return fi.sort_values(by="importance", ascending=False)
feature_imp(df_p,lgbm_model).plot('feature', 'importance', 'barh', figsize=(12,7), legend=False)
import xgboost as xgb
xgb_classifier = xgb.XGBClassifier()
gbm_param_grid = {

    'n_estimators': range(1,20),

    'max_depth': range(1, 10),

    'learning_rate': [.1,.4, .45, .5, .55, .6],

    'colsample_bytree': [.6, .7, .8, .9, 1],

    'booster':["gbtree"],

     'min_child_weight': [0.001,0.003,0.01],

}
xgb_random = RandomizedSearchCV(param_distributions=gbm_param_grid, 

                                    estimator = xgb_classifier, scoring = "accuracy", 

                                    verbose = 0, n_iter = 100, cv = 4)
xgb_random.fit(df_p,y)
xgb_bp = xgb_random.best_params_
xgb_model=xgb.XGBClassifier(n_estimators=xgb_bp["n_estimators"],

                            min_child_weight=xgb_bp["min_child_weight"],

                            max_depth=xgb_bp["max_depth"],

                            learning_rate=xgb_bp["learning_rate"],

                            colsample_bytree=xgb_bp["colsample_bytree"],

                            booster=xgb_bp["booster"])
xgb_model.fit(df_p,y)
feature_imp(df_p,xgb_model).plot('feature', 'importance', 'barh', figsize=(12,7), legend=False)
from IPython.core.display import HTML



def multi_table(table_list):

    ''' Acceps a list of IpyTable objects and returns a table which contains each IpyTable in a cell

    '''

    return HTML(

        '<table><tr style="background-color:white;">' + 

        ''.join(['<td>' + table._repr_html_() + '</td>' for table in table_list]) +

        '</tr></table>'

    )
rf_fm = rf_feat_importance(rf_classifier,df_p)

lgbm_fm = feature_imp(df_p,lgbm_model)

xgb_fm = feature_imp(df_p,xgb_model)
multi_table([rf_fm,lgbm_fm,xgb_fm])
import seaborn as sns

corr = df_p.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))



sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values,mask=mask,cmap='summer_r',vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
from scipy.cluster import hierarchy as hc
def hierarchy_tree(df):

    corr = np.round(scipy.stats.spearmanr(df).correlation, 4)

    corr_condensed = hc.distance.squareform(1-df.corr())

    z = hc.linkage(corr_condensed, method='average')

    fig = plt.figure(figsize=(16,10))

    dendrogram = hc.dendrogram(z, labels=df.columns, orientation='left', leaf_font_size=16)

    plt.show()
hierarchy_tree(df_p)
df_p["target"] = y  
df_p.columns
max_heart_rate_achieved = pd.cut(df_p.max_heart_rate_achieved,4,labels=["71-104","105-137","138-170","171-202"])
df_p.columns
cross1=pd.crosstab([df_p.st_slope_flat[df_p.st_slope_flat==1],df_p.target],max_heart_rate_achieved).style.background_gradient(cmap='summer_r')

cross1
cross1=pd.crosstab([df_p.st_slope_flat[df_p.st_slope_flat==1],df_p["thalassemia_fixed defect"][df_p["thalassemia_fixed defect"]==1],

                    df_p.target],max_heart_rate_achieved).style.background_gradient(cmap='summer_r')

cross1
cross1=pd.crosstab([df_p.st_slope_flat[df_p.st_slope_flat==1],df_p["thalassemia_fixed defect"][df_p["thalassemia_fixed defect"]==1],df_p["chest_pain_type_typical angina"][df_p["chest_pain_type_typical angina"]==1],

                    df_p.target],max_heart_rate_achieved).style.background_gradient(cmap='summer_r')

cross1
cross1=pd.crosstab([df_p["thalassemia_fixed defect"]],df_p.sex_male,margins=True).style.background_gradient(cmap='summer_r')

cross1
cross1=pd.crosstab([df_p["thalassemia_fixed defect"][df_p["thalassemia_fixed defect"]==1],df_p.target],df_p.sex_male).style.background_gradient(cmap='summer_r')

cross1
cross1=pd.crosstab([df_p.exercise_induced_angina_yes[df_p.exercise_induced_angina_yes==1],df_p.target],[df_p.st_slope_upsloping[df_p.st_slope_upsloping==1],df_p.st_depression]).style.background_gradient(cmap='summer_r')

cross1
age = pd.cut(df_p.age,6,labels=["(28.952, 37.0)", "(37.0, 45.0)", "(45.0, 53.0)","(53.0, 61.0)","(61.0, 69.0)", "(69.0, 77.0)"])
cross1=pd.crosstab([pd.cut(df_p.resting_blood_pressure,3),df_p.target],[age]).style.background_gradient(cmap='summer_r')

cross1