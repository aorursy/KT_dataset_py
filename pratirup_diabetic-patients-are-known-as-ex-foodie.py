import numpy as np

import pandas as pa

import matplotlib.pyplot as plt

import seaborn as sn



%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')
data = pa.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

data.info()
data.describe()
sn.countplot(data['Outcome'])

plt.title('Diabetes Count')

## It's quite imbalanced  ( U can apply SMOTE to resample it but here I will be not resampling it)
from sklearn.preprocessing import Imputer

outcome = data['Outcome']

fill_values = Imputer(missing_values=0,strategy="mean",axis=0) # Replace missing values with mean

data = fill_values.fit_transform(data)



data = pa.DataFrame({'Pregnancies': data[:, 0], 'Glucose': data[:, 1],'BloodPressure':data[:, 2],

                        'SkinThickness':data[:, 3],'Insulin':data[:, 4],'BMI':data[:, 5],'DiabetesPedigreeFunction':data[:, 6],

                        'Age':data[:, 7],'Outcome':outcome})
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))

sn.distplot(data['Glucose'],ax=axis1).set_title('Probabilty density Function')

sn.boxplot(x='Outcome',y='Glucose',data=data).set_title('Box Plot ')
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))

sn.distplot(data['BloodPressure'],ax=axis1).set_title('Probabilty density Function')

sn.boxplot(x='Outcome',y='BloodPressure',data=data).set_title('Box Plot ')
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))

sn.distplot(data['Insulin'],ax=axis1).set_title('Probabilty density Function')

sn.boxplot(x='Outcome',y='Insulin',data=data).set_title('Box Plot ')
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))

sn.distplot(data['Age'],ax=axis1).set_title('Probabilty density Function')

sn.boxplot(x='Outcome',y='Age',data=data).set_title('Box Plot ')
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))

sn.distplot(data['BMI'],ax=axis1).set_title('Probabilty density Function')

sn.boxplot(x='Outcome',y='BMI',data=data).set_title('Box Plot ')
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))

sn.distplot(data['Pregnancies'],ax=axis1).set_title('Probabilty density Function')

sn.boxplot(x='Outcome',y='Pregnancies',data=data).set_title('Box Plot ')
y = data['Outcome']

data_x = data

data_x=data_x.drop('Outcome',axis=1)



data_n_2 = (data_x - data_x.mean())/(data_x.std())

dataz = pa.concat([y,data_n_2.iloc[:,0:9]],axis=1)



dataz = pa.melt(dataz,id_vars="Outcome",

                    var_name="features",

                    value_name='value')

#dataz.drop(dataz[dataz.features == 0].index,inplace=True)



plt.figure(figsize=(10,10))

sn.violinplot(x="features", y="value", hue="Outcome", data=dataz,split=True, inner="quart")

plt.xticks(rotation=90)
plt.figure(figsize=(20,8))

sn.violinplot(x='Pregnancies',y='Age',hue='Outcome',data=data)

plt.xticks(rotation=90)

## Pregnencies + Age + Outcome

plt.figure(figsize=(20,8))

sn.boxplot(x='Pregnancies',y='Age',hue='Outcome',data=data)

plt.xticks(rotation=90)
plt.figure(figsize=(20,8))

sn.boxplot(x='Pregnancies',y='BloodPressure',data=data)

plt.xticks(rotation=90)
sn.jointplot(data['BloodPressure'],data['Age'],kind='regg',color="#ce1414")
fig = plt.figure(figsize=(20, 10), dpi= 80)

grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)



ax_main = fig.add_subplot(grid[:-1, :-1])

ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])

ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])



sn.boxplot(x='Age',y='BloodPressure',data=data,ax=ax_main)

sn.boxplot(data['BloodPressure'],ax=ax_right)

sn.boxplot(data['Age'],ax=ax_bottom)
fig = plt.figure(figsize=(20, 10), dpi= 80)

grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)



ax_main = fig.add_subplot(grid[:-1, :-1])

ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])

ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])



sn.boxplot(x='BloodPressure',y='Insulin',data=data,ax=ax_main)

sn.boxplot(data['Insulin'],ax=ax_right)

sn.boxplot(data['BloodPressure'],ax=ax_bottom)
sn.jointplot(data['BMI'],data['SkinThickness'],kind='regg',color="#ce1414")
features = list(data.columns)

features.remove('Outcome')
plt.figure(figsize=(18,10), dpi= 80)

sn.pairplot(data,kind='scatter',hue='Outcome')
sn.set(style='whitegrid',palette='muted')

y = data['Outcome']

data_x = data

data_x=data_x.drop('Outcome',axis=1)



data_n_2 = (data_x - data_x.mean())/(data_x.std())

dataz = pa.concat([y,data_n_2.iloc[:,0:9]],axis=1)



dataz = pa.melt(dataz,id_vars="Outcome",

                    var_name="features",

                    value_name='value')



plt.figure(figsize=(15,9))

sn.swarmplot(x='features',y='value',hue='Outcome',data=dataz)

plt.xticks(rotation=90)
plt.figure(figsize=(10,7))

sn.heatmap(data.corr(),annot=True,linewidths=.5, fmt= '.2f',cmap='BuPu')
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,confusion_matrix

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.model_selection import train_test_split
radm_state = 42
train = data[features][0:760]

y = data['Outcome'][0:760]

test = data[features][760:]

actual_test = data['Outcome'][760:]
X_train,X_test,Y_train,Y_test = train_test_split(train,y,random_state=radm_state,test_size=0.2)
radm_classifier= RandomForestClassifier(random_state=42)

radm_model = radm_classifier.fit(X_train,Y_train)



accuracy = accuracy_score(Y_test,radm_classifier.predict(X_test))

print("Accuracy{}".format(accuracy))



cm = confusion_matrix(Y_test,radm_classifier.predict(X_test))

sn.heatmap(cm,annot=True,fmt="d")
from pdpbox import pdp, info_plots

pdp_ = pdp.pdp_isolate(

    model=radm_model, dataset=X_train, model_features=features, feature='Glucose'

)

fig, axes = pdp.pdp_plot(

    pdp_isolate_out=pdp_, feature_name='Glucose', center=True, 

     plot_lines=True, frac_to_plot=100

)
import shap

X_train = X_train.reset_index(drop=True)

estimator = radm_model

shap_explain = shap.TreeExplainer(estimator)

shap_values = shap_explain.shap_values(X_train.iloc[589])



shap.initjs()

shap.force_plot(shap_explain.expected_value[1], shap_values[1],X_train.iloc[589])
select_feature = SelectKBest(chi2,k=5).fit(X_train,Y_train)
#print(select_feature.scores_)

#print(X_train.columns)

x_train_2 = select_feature.transform(X_train)

x_test_2 = select_feature.transform(X_test)



radm_classifier= RandomForestClassifier(random_state=42)

radm_model = radm_classifier.fit(x_train_2,Y_train)
accuracy = accuracy_score(Y_test,radm_classifier.predict(x_test_2))

print("Accuracy{}".format(accuracy))



cm = confusion_matrix(Y_test,radm_classifier.predict(x_test_2))

sn.heatmap(cm,annot=True,fmt="d")
from sklearn.feature_selection import RFECV
random_classifer = RandomForestClassifier(random_state=42)

rfecv = RFECV(estimator=random_classifer,step=1,cv=5,scoring='accuracy')



rfcev_model = rfecv.fit(X_train,Y_train)



print("Optimal No. of features:",rfcev_model.n_features_)

print("Best Features:",X_train.columns[rfcev_model.support_])
optimal_features =['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI',

       'DiabetesPedigreeFunction', 'Age']



radm_model_test = random_classifer.fit(X_train[optimal_features],Y_train)

accuracy = accuracy_score(Y_test,random_classifer.predict(X_test[optimal_features]))

print("Accuracy{}".format(accuracy))

cm = confusion_matrix(Y_test,random_classifer.predict(X_test[optimal_features]))

sn.heatmap(cm,annot=True,fmt="d")
# Plot number of features VS. cross-validation scores

import matplotlib.pyplot as plt

plt.figure()

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score of number of selected features")

plt.plot(range(1, len(rfcev_model.grid_scores_) + 1), rfcev_model.grid_scores_)

plt.show()
clf_rf_5 = RandomForestClassifier(random_state=42)      

clr_rf_5 = clf_rf_5.fit(X_train,Y_train)

importances = clr_rf_5.feature_importances_

std = np.std([tree.feature_importances_ for tree in clf_rf_5.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(X_train.shape[1]):

    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))



# Plot the feature importances of the forest



plt.figure(1, figsize=(14, 13))

plt.title("Feature importances")

plt.bar(range(X_train.shape[1]), importances[indices],

       color="g", yerr=std[indices], align="center")

plt.xticks(range(X_train.shape[1]), X_train.columns[indices],rotation=90)

plt.xlim([-1, X_train.shape[1]])

plt.show()
from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()

train = standard_scaler.fit_transform(train[optimal_features])

test = standard_scaler.transform(test[optimal_features])
from sklearn.model_selection import cross_val_score,StratifiedShuffleSplit

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

from sklearn import metrics
X_train,X_test,Y_train,Y_test = train_test_split(train,y,test_size=0.2,random_state=42)

cv = StratifiedShuffleSplit(n_splits=2,test_size=0.2,random_state=42)



scoring = 'roc_auc'
model_scores = pa.DataFrame(columns=['Name','Best Parameters','Best Score','Test Score','CV Mean','CV Std'])
color = sn.color_palette()

sn.set_style('darkgrid')
def helper_function(name,model):

    global model_scores

    

    model_lf = model.best_estimator_.fit(X_train,Y_train)

    scores = cross_val_score(model.best_estimator_,X_train,Y_train,cv=5,scoring=scoring,verbose=0)

    

    cross_mean  = scores.mean()

    cross_std = scores.std()

    

    test_score = model.score(X_test,Y_test)

    

    model_scores = model_scores.append({'Name':name,'Best Parameters':model.best_params_,

                                        'Best Score':model.best_score_,'Test Score':test_score,

                                        'CV Mean':cross_mean,'CV Std':cross_std },ignore_index=True)

    

    fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,6))

    

    

    ## Draw Confusion Matrix.

    

    

    predicted_value = model.best_estimator_.predict(X_test)

    cm = metrics.confusion_matrix(Y_test,predicted_value)

    sn.heatmap(cm,annot=True,fmt=".2f",cmap='BuPu',ax=axis1).set_title("Confusion Matrix")

    

    ## Draw Roc Curve

    

    test_results_df = pa.DataFrame({'actual':Y_test})

    test_results_df = test_results_df.reset_index()

    

    predict_probabilites = pa.DataFrame(model.best_estimator_.predict_proba(X_test))

    test_results_df['chd_1'] = predict_probabilites.iloc[:,1:2]

    

    fpr,tpr,thresholds = metrics.roc_curve(test_results_df.actual,test_results_df.chd_1,drop_intermediate=False)

    

    auc_score = metrics.roc_auc_score(test_results_df.actual,test_results_df.chd_1)

    

    plt.plot(fpr,tpr,label="ROC Curve (area = %.2f)"% auc_score)

    plt.plot([0,1],[0,1],'k--')

    plt.xlim([0.0,1.0])

    plt.ylim([0.0,1.05])

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.legend(loc='lower right')

    

    ## print classification rreport

    

    print(metrics.classification_report(Y_test,predicted_value))

    pass
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier,RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

import xgboost as xgb

import lightgbm as lgm
logit_model = LogisticRegression()



param_grid = {'C':[0.001,0.01,0.05,1,100],'penalty':['l1','l2']}

logit_grid = GridSearchCV(logit_model,param_grid,cv=cv,scoring=scoring)

logit_grid.fit(X_train,Y_train)

helper_function("Logistic Regression",logit_grid)
param_grid = {'n_neighbors':[x for x in range(1,40)],'weights':['uniform','distance']}



knn_model =  KNeighborsClassifier()



knn_grid  = RandomizedSearchCV(knn_model,param_grid,cv=cv,scoring=scoring,random_state=42)

knn_grid.fit(X_train,Y_train)

helper_function("K-Nearest-Neighbors",knn_grid)
n_estimators = [int(x) for x in np.linspace(start=200,stop=2000,num=10)] #Boosting parameters

max_features = ['auto', 'sqrt']# Boosting Parameters

max_depth = [int(x) for x in np.linspace(10,200,num=20)] #Max depth of the tree

max_depth.append(None)

bootstrap = [True,False] # Bootstrap here means how the samples will be chosen with or without replacement



# Total Combination 10*2*20*2 = 800 !



param_grid = {'n_estimators':n_estimators,

              'max_features':max_features,

              'max_depth':max_depth,

              'bootstrap':bootstrap}



random_model = RandomForestClassifier()

grid_random = RandomizedSearchCV(radm_model,param_grid,cv=cv,scoring=scoring,n_iter=100,random_state=42)



grid_random.fit(X_train,Y_train)

helper_function("RADNOM FOREST",grid_random)
ada_model = AdaBoostClassifier()



param_grid = {'n_estimators':[int(x) for x in np.linspace(start=20,stop=300,num=15)],

              'learning_rate':np.arange(.1,4,.3)}



ada_grid = RandomizedSearchCV(ada_model,param_grid,cv=cv,scoring=scoring,n_iter=100,random_state=42)



ada_grid.fit(X_train,Y_train)

helper_function("ADA Boost",ada_grid)
n_estimators = [int(x) for x in np.linspace(start=20,stop=120,num=6)]

learning_rate = [0.1,0.01,0.05,0.001]

max_depth= np.arange(2,5,1)





param_grid = {'n_estimators':n_estimators,'learning_rate':learning_rate,'max_depth':max_depth}



grad_model = GradientBoostingClassifier()



grid_grad = GridSearchCV(grad_model,param_grid,cv=cv,scoring=scoring)

grid_grad.fit(X_train,Y_train)



helper_function("Gradient Boosting",grid_grad)
from xgboost.sklearn import XGBClassifier



param_grid = {'max_depth':range(3,8,2),'min_child_weight':range(1,10,2),'gamma':[0.5,1,1.5,2,5],

              'subsample':[0.6,0.8,1.0],'colsample_bytree':[0.6,0.8,1.0]}



xgboost_model = XGBClassifier(learning_rate=0.025,n_estimators=600,objective='binary:logistic',silent=True,nthread=1)

xgboost_grid = RandomizedSearchCV(xgboost_model,param_grid,cv=cv,scoring=scoring,n_iter=100,random_state=42)

xgboost_grid.fit(X_train,Y_train)

helper_function("XGBOOST",xgboost_grid)
plt.figure(figsize=(12,6))

sn.barplot(x='Name',y='CV Mean',data=model_scores)
#### Traiing the XGBoost  Model



xtreme_gradient_boost_model =xgboost_grid.best_estimator_

xtreme_gradient_boost_model.fit(X_train,Y_train)



print(metrics.classification_report(Y_train,xtreme_gradient_boost_model.predict(X_train)))
print(metrics.classification_report(actual_test,xtreme_gradient_boost_model.predict(test)))