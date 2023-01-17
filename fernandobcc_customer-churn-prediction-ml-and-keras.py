import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np





telco = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')

telco.head()



telco.dtypes
#Chek if there is any column with null values

telco.info()
telco.isna().sum()
#Get a overview of each column

telco.describe()
#Get maximum values

telco.max()
#Get minimum values

telco.min()
#Checking unique values

telco.nunique()
print('Empty cells in TotalCharges: ', len(telco[telco['TotalCharges']==' ']))
#number of rows

nrows_before=len(telco) 

#removing empty cells from TotalCharges (remove associeted rows)

telco=telco[telco['TotalCharges']!=' ']

#Number of rows after remove emptys TotalCharges

nrows_after=len(telco)

#Reset inted

telco.reset_index(inplace=True)

telco.drop('index', axis=1, inplace=True)

#Lost data

print(("lost data: {0:.3f} %").format(100*(1-nrows_after/nrows_before)))
telco["TotalCharges"] = telco["TotalCharges"].astype(float)

print("TotalCharges type:", telco["TotalCharges"].dtypes)
#Checking columns with 3 different unique values 

check_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingMovies']

for c in check_cols:

    print( c + '=', telco[c].unique())

telco["SeniorCitizen"] = telco["SeniorCitizen"].replace(to_replace=[0, 1], value=['No', 'Yes'])
print("SeniorCitizen unquies: ", telco["SeniorCitizen"].unique())
#Replace 'No internet service','No phone service' to 'No'

telco.replace(['No internet service','No phone service'],'No', inplace=True)

for c in check_cols:

    print( c + '=', telco[c].unique())
for c in telco.columns:

    print( c + '=', telco[c].unique())
#Separation of categorical and numerical columns

cat_cols = telco.select_dtypes(include='object')

num_cols = telco.select_dtypes(exclude='object')
sns.clustermap(num_cols.corr(),linecolor='white',cmap='coolwarm',annot=True, figsize=(15,15))

sns.set_context("paper", font_scale=2)   
sns.pairplot(telco, hue='Churn', palette='coolwarm', size=5)

f, axes = plt.subplots(nrows=1,ncols=3, squeeze=True,figsize=(30, 8))



sns.distplot(telco['MonthlyCharges'], ax=axes[0])



sns.distplot(telco['TotalCharges'],  ax=axes[1])



sns.distplot(telco['tenure'], ax=axes[2])

#TotalCharges vS  Churn

g = sns.FacetGrid(telco, hue="Churn", size=8, aspect=2,legend_out=True)

g = (g.map(sns.distplot, "TotalCharges", kde=False).add_legend())

#MonthlyCharges vS  Churn

g = sns.FacetGrid(telco, hue="Churn", size=8, aspect=2,legend_out=True)

g = (g.map(sns.distplot, "MonthlyCharges", kde=False).add_legend())
#Tenure vS  Churn

g = sns.FacetGrid(telco, hue="Churn", size=8, aspect=2,legend_out=True)

g = (g.map(sns.distplot, "tenure", kde=False).add_legend())
import math

f, axes = plt.subplots(nrows=math.ceil(len(cat_cols.columns[1:])/2),ncols=2,figsize=(30, 40))

graf_count = []

i=0

j=0

for col in cat_cols.columns[1:]:

    graf_count.append(sns.countplot(x=col,data=telco, ax=axes[i,j],palette="Paired"))

    j=j+1

    if j==2:

        i=i+1

        j=0

graf_count.append(sns.boxplot(x="TotalCharges", y="Churn", data=telco, whis=np.inf))

sns.set_context("paper", font_scale=1.5)      

sns.set_style("white")

sns.despine()



total = float(len(telco)) 

plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9, hspace = 0.5)





for g in graf_count:

    for p in g.patches:

        height = p.get_height()

        g.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format(100*height/total),

            ha="center") 

telco.head()
#Tenure to Categorical

def tenure_cat(telco) :

    

    if telco["tenure"] <= 12 :

        return "0-1_year"

    elif (telco["tenure"] > 12) & (telco["tenure"] <= 24):

        return "1-2_year"

    elif (telco["tenure"] > 24) & (telco["tenure"] <= 36):

        return "2-3_year"

    elif (telco["tenure"] > 36) & (telco["tenure"] <= 48):

        return "3-4_year"

    elif (telco["tenure"] > 48) & (telco["tenure"] <= 60):

        return "4-5_year"

    elif telco["tenure"] > 60:

        return "5-6_year"



telco["tenure_years"] = telco.apply(lambda t:tenure_cat(t), axis = 1)
telco.head()
telco['tenure_years'].value_counts().sort_index().index

fig = plt.gcf()

fig.set_size_inches(16, 10)

sns.set_context(context='paper', font_scale=2)

g=sns.countplot(x="tenure_years", data=telco, palette="magma", order=telco['tenure_years'].value_counts().sort_index().index)

total = float(len(telco)) 



for p in g.patches:

    height = p.get_height()

    g.text(p.get_x()+p.get_width()/2.,

    height + 3,'{:1.2f}%'.format(100*height/total), ha="center") 
cat_cols = telco.select_dtypes(include='object')

num_cols = telco.select_dtypes(exclude='object')

cat_cols=cat_cols.drop("customerID", axis=1)

telco=pd.get_dummies(data = telco,columns =  cat_cols.columns, drop_first=True)

telco.head()
telco.describe()
sns.set_context("paper", font_scale=1)  

sns.clustermap(telco.corr(),linecolor='white',cmap='coolwarm', figsize=(20,15), annot=True)
from sklearn.preprocessing import StandardScaler



#Normalize the mesure (numerical columns)

scale = StandardScaler()

scale.fit_transform(num_cols)
#scale

scaled_features = scale.transform(num_cols)

#Create a panda DF with scaled features

telco_feat = pd.DataFrame(scaled_features, columns= num_cols.columns)

#Concat this new DF with Telco DF

telco_feat = pd.concat([telco_feat, telco.drop(['tenure','MonthlyCharges','TotalCharges'], axis=1)], axis=1)

#Move Target columns to the last column position

telco_feat['Churn'] = telco_feat['Churn_Yes']

telco_feat.drop('Churn_Yes', axis=1, inplace=True)



telco_feat.rename(columns={'InternetService_Fiber optic':'InternetService_Fiber_Optic',

                          'PaymentMethod_Credit card (automatic)':'PaymentMethod_Credit_card_Auto',

                          'PaymentMethod_Electronic check':'PaymentMethod_Electronic_Check',

                          'PaymentMethod_Mailed check':'PaymentMethod_Mailed_check',

                           'Contract_One year':'Contract_One_year',

                           'Contract_Two year':'Contract_Two_year'

                          }, 

                 inplace=True)
telco_feat.info()
telco_feat.describe()
fig = plt.figure(figsize=(20,10))

ax = fig.add_axes([0,0,1,1])



#Bart plot

telco_feat.corr()['Churn'].sort_values(ascending = False).plot(kind='bar', cmap='RdGy', )



#Change font size

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +

             ax.get_xticklabels() + ax.get_yticklabels()):

    item.set_fontsize(12)



ax.axhline(0, color='black')

    

total=len(telco_feat.corr()['Churn'])
telco=telco_feat
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report,confusion_matrix 

%matplotlib inline
X_train, X_test, y_train, y_test = train_test_split(telco.drop(['customerID','Churn'],axis=1),telco['Churn'],

                                                    test_size=0.30)
max_depth=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
#GridSearch 

parameters = {

    'n_estimators'      : [100,150,200,250,500],

    'max_depth'         : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],

    'random_state'      : [0],

}

grid_rfc = GridSearchCV(RandomForestClassifier(), parameters, refit=True, verbose=3, scoring='accuracy', cv=5, n_jobs=4) 
grid_rfc.fit(X_train, y_train)
#RESULTS

churn = {'rfc':[grid_rfc.best_params_,grid_rfc.best_score_]}

print('Parametros', grid_rfc.best_params_)

print('Accuracy', grid_rfc.best_score_)

print('Estimator:', grid_rfc.best_estimator_)
rfc = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

            max_depth=9, max_features='auto', max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=2,

            min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=1,

            oob_score=False, random_state=0, verbose=0, warm_start=False)
rfc.fit(X_train,y_train)
pred = rfc.predict(X_test)
print("Churn")

print("Classification Report")

print(classification_report(y_test,pred))

print("Confusion Matrix")

print(confusion_matrix(y_test,pred))

print('Accuracy', churn['rfc'][1])
cm=confusion_matrix(y_test,pred)
df_cm = pd.DataFrame(cm, index = ['Churn','All'],

                  columns = ['Churn','All'])

plt.figure(figsize = (10,7))

sns.heatmap(df_cm, annot=True, cmap='Blues')
cv_scores = []

for i in max_depth:

    

    rfc_e = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

            max_depth=i, max_features='auto', max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=2,

            min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=1,

            oob_score=False, random_state=0, verbose=0, warm_start=False)

    scores = cross_val_score(rfc_e, X_train, y_train, cv=5, scoring='accuracy')

    cv_scores.append(scores.mean())
# changing to misclassification error

MSE = [1 - x for x in cv_scores]



# determining best k

optimal_k = max_depth[MSE.index(min(MSE))]

print("The optimal number of Max Depth is:", optimal_k)



# plot misclassification error vs k

plt.figure(figsize=(15,10))

plt.plot(max_depth, MSE,color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)

plt.xlabel('Number of Max Depth')

plt.ylabel('Misclassification Error')

plt.show()

from sklearn.neighbors import KNeighborsClassifier

k_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
parameters = {

    'n_neighbors'      : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],

    'weights'         : ['uniform','distance'],

    'metric'      : ['euclidean', 'manhattan'],

}
grid_knn = GridSearchCV(KNeighborsClassifier(), parameters, refit=True, verbose=3, scoring='accuracy', cv=5, n_jobs=4) 
grid_knn.fit(X_train,y_train)
#RESULTS

churn = {'Knn':[grid_knn.best_params_,grid_knn.best_score_]}

print('Parametros', grid_knn.best_params_)

print('Accuracy', grid_knn.best_score_)

print('Estimator:', grid_knn.best_estimator_)
knn=KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='manhattan',

           metric_params=None, n_jobs=1, n_neighbors=24, p=2,

           weights='uniform')
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print("Churn")

print("Classification Report")

print(classification_report(y_test,pred))

print("Confusion Matrix")

print(confusion_matrix(y_test,pred))

print('Accuracy', churn['Knn'][1])
cm=confusion_matrix(y_test,pred)

df_cm = pd.DataFrame(cm, index = ['Churn','All'],

                  columns = ['Churn','All'])

plt.figure(figsize = (10,7))

sns.heatmap(df_cm, annot=True, cmap='Blues')
cv_scores = []

for i in k_range:

    

    knn_e = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='manhattan',

           metric_params=None, n_jobs=1, n_neighbors=i, p=2,

           weights='uniform')

    scores = cross_val_score(knn_e, X_train, y_train, cv=5, scoring='accuracy')

    cv_scores.append(scores.mean())
# changing to misclassification error

MSE = [1 - x for x in cv_scores]



# determining best k

optimal_k = k_range[MSE.index(min(MSE))]

print("The optimal number of neighbors is:", optimal_k)



# plot misclassification error vs k

plt.figure(figsize=(15,10))

plt.plot(k_range, MSE,color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)

plt.xlabel('Number of Neighbors K')

plt.ylabel('Misclassification Error')

plt.show()

#XGBoosts

from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV
gamma = [0, 0.001, 0.01, 0.1,1 , 2, 5]
param_grid = {

   'max_depth': [3,4],

   'n_estimators': [100,150],

   'nthread': [8],

   'subsample': [0,0.9, 1.0],

   'gamma': [0, 0.001, 0.01, 0.1,1 , 2, 5],

   'min_child_weight': [1, 5, 10]

}

grid_gb = GridSearchCV(XGBClassifier(), param_grid=param_grid, refit=True, verbose=3, scoring='accuracy', n_jobs=4)
grid_gb.fit(X_train, y_train)
#RESULTS

if 'churn' in globals():

    churn['xgb'] = [grid_gb.best_params_,grid_gb.best_score_]

else:

    churn = {'xgb':[grid_gb.best_params_,grid_gb.best_score_]}



print('Parametros', grid_gb.best_params_)

print('Accuracy', grid_gb.best_score_)

print('Estimator:', grid_gb.best_estimator_)
xgb = XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,

       gamma=2, learning_rate=0.1, max_delta_step=0, max_depth=4,

       min_child_weight=1, missing=None, n_estimators=100, nthread=8,

       objective='binary:logistic', reg_alpha=0, reg_lambda=1,

       scale_pos_weight=1, seed=0, silent=True, subsample=1.0)
xgb.fit(X_train,y_train)
pred = xgb.predict(X_test)
print("Churn")

print("Classification Report")

print(classification_report(y_test,pred))

print("Confusion Matrix")

print(confusion_matrix(y_test,pred))

print('Accuracy', churn['xgb'][1])
cm=confusion_matrix(y_test,pred)

df_cm = pd.DataFrame(cm, index = ['Churn','All'],

                  columns = ['Churn','All'])

plt.figure(figsize = (10,7))

sns.heatmap(df_cm, annot=True, cmap='Blues')
cv_scores = []

for i in gamma:

    

    xgb_e = XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,

       gamma=i, learning_rate=0.1, max_delta_step=0, max_depth=4,

       min_child_weight=1, missing=None, n_estimators=100, nthread=8,

       objective='binary:logistic', reg_alpha=0, reg_lambda=1,

       scale_pos_weight=1, seed=0, silent=True, subsample=1.0)

    scores = cross_val_score(xgb_e, X_train, y_train, cv=5, scoring='accuracy')

    cv_scores.append(scores.mean())
# changing to misclassification error

MSE = [1 - x for x in cv_scores]



# determining best k

optimal = gamma[MSE.index(min(MSE))]

print("The optimal gamma is:", optimal)



# plot misclassification error vs k

plt.figure(figsize=(15,10))

plt.plot(gamma, MSE,color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)

plt.xlabel('Gamma number')

plt.ylabel('Misclassification Error')

plt.show()
from sklearn.svm import LinearSVC

C = [0.1, 1, 10, 100,500,1000]
param_grid = {'C': [0.1, 1, 10, 100,500,1000]}

grid_lsvm = GridSearchCV(LinearSVC(),param_grid,refit=True,verbose=3,scoring='accuracy', n_jobs=4)
grid_lsvm.fit(X_train,y_train)
churn['LinearSVC'] = [grid_lsvm.best_params_,grid_lsvm.best_score_]

print('Parametros', grid_lsvm.best_params_)

print('Accuarcy', grid_lsvm.best_score_)

print('Estimator:', grid_lsvm.best_estimator_)
lsvc=LinearSVC(C=0.1, class_weight=None, dual=True, fit_intercept=True,

     intercept_scaling=1, loss='squared_hinge', max_iter=1000,

     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,

     verbose=0)
lsvc.fit(X_train,y_train)
pred = lsvc.predict(X_test)
print("Churn")

print("Classification Report")

print(classification_report(y_test,pred))

print("Confusion Matrix")

print(confusion_matrix(y_test,pred))

print('Accuracy', churn['LinearSVC'][1])
cm=confusion_matrix(y_test,pred)

sns.set_context(font_scale=2)

df_cm = pd.DataFrame(cm, index = ['Churn','No Churn'],

                  columns = ['No Churn','Churn'])

plt.figure(figsize = (15,10))

sns.heatmap(df_cm, annot=True, cmap='Blues')
cv_scores = []

for i in C:

    

    lsvc_e = LinearSVC(C=i, class_weight=None, dual=True, fit_intercept=True,

     intercept_scaling=1, loss='squared_hinge', max_iter=1000,

     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,

     verbose=0)

    scores = cross_val_score(lsvc_e, X_train, y_train, cv=5, scoring='accuracy')

    cv_scores.append(scores.mean())
# changing to misclassification error

MSE = [1 - x for x in cv_scores]



# determining best k

optimal = C[MSE.index(min(MSE))]

print("The optimal C is:", optimal)



# plot misclassification error vs k

plt.figure(figsize=(15,10))

plt.plot(C, MSE,color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)

plt.xlabel('Number os Cs')

plt.ylabel('Misclassification Error')

plt.show()
from sklearn.svm import SVC
Csvc = [1,10,100,1000]
param_grid = {'C':[1,10,100,1000],'gamma':[1,0.1,0.001,0.0001], 'kernel':['rbf']}
grid_svc = GridSearchCV(SVC(),param_grid,refit = True, verbose=3)
grid_svc.fit(X_train,y_train)
if 'churn' in globals():

    churn['SVC'] = [grid_svc.best_params_,grid_svc.best_score_]

else:

    churn = {'SVC':[grid_svc.best_params_,grid_svc.best_score_]}

 

print('Parametros', grid_svc.best_params_)

print('Accuarcy', grid_svc.best_score_)

print('Estimator:', grid_svc.best_estimator_)
svm = SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,

  decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',

  max_iter=-1, probability=False, random_state=None, shrinking=True,

  tol=0.001, verbose=False)
svm.fit(X_train,y_train)
pred = svm.predict(X_test)
print("Churn")

print("Classification Report")

print(classification_report(y_test,pred))

print("Confusion Matrix")

print(confusion_matrix(y_test,pred))

print('Accuracy', churn['SVC'][1])
cm=confusion_matrix(y_test,pred)

plt.figure(figsize = (15,10))

sns.set_context(font_scale=2)

df_cm = pd.DataFrame(cm, index = ['Churn','No Churn'],

                  columns = ['No Churn','Churn'])



sns.heatmap(df_cm, annot=True, cmap='Blues')
cv_scores = []

for i in Csvc:

    

    svc_e = SVC(C=i, cache_size=200, class_weight=None, coef0=0.0,

  decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',

  max_iter=-1, probability=False, random_state=None, shrinking=True,

  tol=0.001, verbose=False)

    scores = cross_val_score(svc_e, X_train, y_train, cv=5, scoring='accuracy')

    cv_scores.append(scores.mean())
# changing to misclassification error

MSE = [1 - x for x in cv_scores]



# determining best k

optimal = Csvc[MSE.index(min(MSE))]

print("The optimal C is:", optimal)



# plot misclassification error vs k

plt.figure(figsize=(15,10))

plt.plot(Csvc, MSE,color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)

plt.xlabel('Number os Cs')

plt.ylabel('Misclassification Error')

plt.show()
from sklearn.linear_model import LogisticRegression
Clog = [0.001,0.01,0.1,1,10,100,1000]
grid_values = {'penalty': ['l1','l2'], 'C': [0.001,0.01,0.1,1,10,100,1000]}
grid_log = GridSearchCV(LogisticRegression(),param_grid=grid_values,refit = True, verbose=3)
grid_log.fit(X_train, y_train)
if 'churn' in globals():

    churn['log'] = [grid_log.best_params_,grid_log.best_score_]

else:

    churn = {'log':[grid_log.best_params_,grid_log.best_score_]}

 

print('Parametros', grid_log.best_params_)

print('Accuarcy', grid_log.best_score_)

print('Estimator:', grid_log.best_estimator_)
lr = LogisticRegression(C=1000, class_weight=None, dual=False, fit_intercept=True,

          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,

          penalty='l1', random_state=None, solver='liblinear', tol=0.0001,

          verbose=0, warm_start=False)
lr.fit(X_train,y_train)
pred=lr.predict(X_test)
print("Churn")

print("Classification Report")

print(classification_report(y_test,pred))

print("Confusion Matrix")

print(confusion_matrix(y_test,pred))

print('Accuracy', churn['log'][1])
cm=confusion_matrix(y_test,pred)

plt.figure(figsize = (15,10))

sns.set_context(font_scale=2)

df_cm = pd.DataFrame(cm, index = ['Churn','No Churn'],

                  columns = ['No Churn','Churn'])



sns.heatmap(df_cm, annot=True, cmap='Blues')
cv_scores = []

for i in Clog:

    

    log_e = LogisticRegression(C=i, class_weight=None, dual=False, fit_intercept=True,

          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,

          penalty='l1', random_state=None, solver='liblinear', tol=0.0001,

          verbose=0, warm_start=False)

    scores = cross_val_score(log_e, X_train, y_train, cv=5, scoring='accuracy')

    cv_scores.append(scores.mean())
# changing to misclassification error

MSE = [1 - x for x in cv_scores]



# determining best k

optimal = Clog[MSE.index(min(MSE))]

print("The optimal C is:", optimal)



# plot misclassification error vs k

plt.figure(figsize=(15,10))

plt.plot(Clog, MSE,color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)

plt.xlabel('Number os Cs')

plt.ylabel('Misclassification Error')

plt.show()
import keras

from keras.layers import Dense, Dropout

from keras.models import Sequential

from IPython.display import SVG

from keras.optimizers import Adam

from keras import regularizers

from keras.utils.vis_utils import model_to_dot

from keras import losses

from keras.layers import Embedding

from keras.layers import LSTM

#import pydot

from imblearn.over_sampling import SMOTE
os = SMOTE(random_state = 0,  k_neighbors=10)



#Train set

X_smote_os,y_smote_os = os.fit_sample(X_train,y_train)

X_smote_os = pd.DataFrame(data = X_smote_os,columns= telco.drop(['customerID','Churn'], axis=1).columns)

y_smote_os  = pd.DataFrame(data = y_smote_os,columns=["Churn"])



X_matrix_smote = X_smote_os.as_matrix()

y_matrix_smote =y_smote_os.as_matrix()



#Test set

X_smote_os_test,y_smote_os_test = os.fit_sample(X_test,y_test)

X_smote_os_test = pd.DataFrame(data = X_smote_os_test,columns= telco.drop(['customerID','Churn'], axis=1).columns)

y_smote_os_test  = pd.DataFrame(data = y_smote_os_test,columns=["Churn"])



X_matrix_smote_test = X_smote_os.as_matrix()

y_matrix_smote_test =y_smote_os.as_matrix()
n_cols=X_matrix_smote.shape[1]

model = Sequential()

model.add(Dense(32, input_shape=(n_cols,), activation='relu'))

model.add(Dense(16, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(4, activation='softmax'))

model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

history = model.fit(X_matrix_smote, y_matrix_smote, epochs=150, batch_size=10)

predictions   = model.predict(X_matrix_smote_test)

score = model.evaluate(X_matrix_smote_test, y_matrix_smote_test, verbose=0)
print('Test loss:', score[0])

print('Test accuracy:', score[1])
y_pred = (predictions > 0.5)

print("Churn")

print("Classification Report")

print(classification_report(y_matrix_smote_test, y_pred))

print("Confusion Matrix")

print(confusion_matrix(y_matrix_smote_test, y_pred))
cm = confusion_matrix(y_matrix_smote_test, y_pred)

plt.figure(figsize = (15,10))

sns.set_context(font_scale=2)

df_cm = pd.DataFrame(cm, index = ['Churn','No Churn'],

                  columns = ['No Churn','Churn'])



sns.heatmap(df_cm, annot=True, cmap='Blues')
plt.figure(figsize=(15,10))

plt.subplot(211)

plt.plot(history.history['acc'], marker='o', markerfacecolor='black', markersize=2)

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')





# Plot training & validation loss values

plt.subplot(212)

plt.plot(history.history['loss'], marker='o', markerfacecolor='black', markersize=2)

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score

from sklearn.model_selection import train_test_split

from xgboost import plot_importance
def model_evaluate(model,train_x,test_x,train_y,test_y,name) :

    model.fit(train_x,train_y)

    predictions  = model.predict(test_x)

    

    predictions  = model.predict(test_x)

    accuracy     = accuracy_score(test_y,predictions)

    recallscore  = recall_score(test_y,predictions)

    precision    = precision_score(test_y,predictions)

    auc          = cross_val_score(model,test_x,test_y, scoring='roc_auc').mean()

    f1score      = f1_score(test_y,predictions) 

        

    df = pd.DataFrame({"0_Model"           :  name,

                       "1_Accuracy_score"  : [accuracy],

                       "2_Recall_score"    : [recallscore],

                       "3_Precision"       : [precision],

                       "4_f1_score"        : [f1score],

                       "5_Area_under_curve": [auc]                  

                      })

    

    

    return df
ml_models = np.array([rfc, knn, xgb, lr, lsvc, svm])
names=np.array(['RandomForestClassifier', 'KNeighborsClassifier', 'XGBClassifier','LogisticRegression','LinearSVC', 'SVC'])
results = pd.DataFrame()

i=0
recallscore  = recall_score(y_matrix_smote_test,y_pred)

precision    = precision_score(y_matrix_smote_test,y_pred)

f1score      = f1_score(y_matrix_smote_test,y_pred) 

auc = roc_auc_score(y_matrix_smote_test,y_pred)
keras_model_result=pd.DataFrame({"0_Model"           : 'KerasSequencial',

                       "1_Accuracy_score"  : [score[1]],

                       "2_Recall_score"    : [recallscore],

                       "3_Precision"       : [precision],

                       "4_f1_score"        : [f1score],

                       "5_Area_under_curve": [auc]                  

                      })
for l in  ml_models:

    results=results.append(model_evaluate(l ,X_train,X_test,y_train,y_test,names[i]))

    i = i+1
results=results.append(keras_model_result).reset_index().drop('index', axis=1)
def highlight_max(s):

    '''

    highlight the maximum in a Series yellow.

    '''

    is_max = s == s.max()

    if s.dtype == 'float64':

        return ['background-color: yellow' if v else '' for v in is_max]

    else:

        return ['background-color: white' if v else '' for v in is_max]
results.style.apply(highlight_max)