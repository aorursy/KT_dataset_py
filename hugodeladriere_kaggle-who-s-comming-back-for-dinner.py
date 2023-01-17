# Main packages

import numpy as np

import pandas as pd

import xgboost as xgb



# Graph package 

import matplotlib.pyplot as plt

import seaborn as sns





# Classifiers

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingRegressor

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

#from sklearn.neighbors import KNeighborsClassifier

#from sklearn.svm import SVC

#from sklearn.tree import DecisionTreeClassifier



from xgboost import XGBClassifier

#from sklearn.feature_selection import SelectKBest ,chi2,RFE





# Tools

from mlxtend.classifier import StackingClassifier

from sklearn.model_selection import cross_val_score







#Yes pourquoi pas 

from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV

from sklearn.model_selection import train_test_split



from sklearn.metrics import roc_curve

from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score





#Noramlize your Data bro

from sklearn.preprocessing import StandardScaler



# Importing data

test = pd.read_csv('../input/data-kaggle/Test.csv')

train = pd.read_csv('../input/data-kaggle/Train.csv')



#Add extra data on your Data Set 

restaurant = pd.read_csv('../input/data-kaggle/restaurant.csv', sep=';', encoding='utf8')

#members = pd.read_csv('Data/member.csv', sep=';', encoding='utf8')





#preview test data

test.head()
print('Le nombre de ligne dans le tableau test est :{}.'.format(test.shape[0]))
print('Train\n',train.isnull().sum())

print('.................')

print('.................')

print('Test\n',test.isnull().sum())
train['purpose'].value_counts().plot(kind='barh', figsize=(6,6))

print('Le pourcentage de valeurs nulles dans la colonne "purpose"  est  %.2f%% ' %((train['purpose'].isnull().sum()/train.shape[0])*100))
def fill_na(df): 

    df['purpose'] = df['purpose'].fillna('Aucun')
train["datetime"].head(2)
#Clean la colone datetime en 4 colonnes, month, heure du déjeuner, heure du souper, nombre de jours réservation avant 

def date(df):

    df['d1'] = pd.to_datetime(df['datetime'])

    df['month']=pd.DatetimeIndex(df['d1']).month

    df['dejeuner']= pd.DatetimeIndex(df['d1']).hour< 17

   

    df['diner']= (pd.DatetimeIndex(df['d1']).hour >=17)



    df['nb_jours_avance'] = ((df["d1"]) - pd.to_datetime(df['cdate_x'])).dt.days

    

    return df
train['status'].value_counts().plot(kind='barh', figsize=(5,5))
pd.get_dummies( train['status']).head(5)
def  status(df):



    dummy = pd.get_dummies( df['status'])

    #dummy ['present'] = dummy['changed']+dummy['ok']+ dummy['new']

    #dummy['absent'] = dummy['no-show']+dummy['canceled']



    #dummy.drop('canceled',inplace=True, axis=1)

    #dummy.drop('changed',inplace=True, axis=1)

    #dummy.drop('new',inplace=True, axis=1)

    #dummy.drop('no-show',inplace=True, axis=1)

    #dummy.drop('ok',inplace=True, axis=1)

    

    return dummy





test['purpose'].value_counts().head(20).plot(kind='barh')

def purpose(df):

    df.drop((df[~df['purpose'].str.contains('anniversaire')].index) & 

            (df[~df['purpose'].str.contains('amis')].index) & 

            (df[~df['purpose'].str.contains('famille')].index) & 

            (df[~df['purpose'].str.contains('Sweet')].index) & 

            (df[~df['purpose'].str.contains('affaires')].index) & 

            (df[~df['purpose'].str.contains('importante')].index) & 

            (df[~df['purpose'].str.contains('Aucun')].index), inplace = True)



    return df
def dummy_purpose_train(df):

    dummy = pd.get_dummies(df['purpose'])

    df['anniversaire']= (dummy['Birthday']== 1) | (dummy['Birthday,Celebration']== 1 ) | (dummy["Célébration d'anniversaire"]==1) | (dummy["Fête d'anniversaire"]==1)

    df['amis']=(dummy['Friends,Reunion']== 1) | (dummy['Dîner entre amis']== 1 )

    df['famille'] = (dummy['Family,Gathering']== 1) | (dummy['Repas en famille']== 1 )| (dummy['Repas en famille,盡量靠W']== 1 )| (dummy['Repas en famille-生日']== 1 ) 

    df['business'] = (dummy["Dîner d'affaires" ]== 1)

    df['Date_tinder'] = (dummy['Sweet day' ]== 1)

    df['importante'] = (dummy['Date importante' ]== 1)

    df['autre']=(dummy['Aucun']== 1) |(dummy['Veuillez sélectionner']== 1) | (dummy['其他']== 1) |(dummy['Please,Select']== 1) | (dummy['&#65533;&#40115;&#65533;&#65533;&#26813;&#65533;']== 1) | (dummy['家人&#65533;']== 1)

    return df

def dummy_purpose_test(df):

    dummy = pd.get_dummies(df['purpose'])

    df['anniversaire']= (dummy["Fête d'anniversaire"]== 1) | (dummy["Célébration d'anniversaire"]== 1)

    df['amis'] = (dummy["Dîner entre amis"]== 1) | (dummy["Friends,Reunion"]== 1)| (dummy["Friends"]== 1)

    df['famille'] = (dummy["Repas en famille"]== 1) | (dummy["Family,Gathering"]== 1)

    df['business'] = (dummy["Dîner d'affaires" ]== 1)| (dummy["Business,meeting"]== 1)

    df['Date_tinder'] =(dummy['Sweet day' ]== 1)

    df['importante'] = (dummy['Date importante' ]== 1)

    df['autre']=(dummy['Aucun']== 1) |(dummy['Veuillez sélectionner']== 1) | (dummy['其他']== 1) |(dummy['Please,Select']== 1) |(dummy['慶生']== 1) | (dummy["&#65533;&#40115;&#65533;&#65533;&#26813;&#65533;"]== 1)

    return df

def gender(df):

    dummy = pd.get_dummies( df['gender'])

    df['Femme']=(dummy['F']==1)

    return df

def join(df1,df2):

    df3  = df1.join(df2, how='inner')

    return df3
def present(df):

    df['Present']= (df['present']==1)

    return df
# Add csv Restaurant on your Train and Test data set

resultat_train = pd.merge(train, restaurant, left_on= 'restaurant_id', right_on='id',how='left')

resultat_test = pd.merge(test, restaurant, left_on= 'restaurant_id', right_on='id',how='left')



resultat_test=resultat_test.set_index('booking_id')

resultat_train=resultat_train.set_index('booking_id')
resultat_train.isnull().sum()

def fill_resultat(df):

    df['is_hotel'] = df['is_hotel'].fillna(0)

    df['good_for_family'] = df['good_for_family'].fillna(0)

    df['accept_credit_card'] = df['accept_credit_card'].fillna(0)

    df['parking'] = df['parking'].fillna(0)

    df['outdoor_seating'] = df['outdoor_seating'].fillna(0)

    df['wifi'] = df['wifi'].fillna(0)

    df['wheelchair_accessible'] = df['wheelchair_accessible'].fillna(0)





    df['price1'] = df['price1'].fillna(df.price1.mean())

    df['price2'] = df['price2'].fillna(df.price2.mean())

    df['cdate_y'] = df['cdate_y'].fillna('Aucun')





    df['id'] = df['id'].fillna('Pasid')

    

    

    df['purpose'] = df['purpose'].fillna('Aucun')

    

    return df

restaurant.head(5)
def modif_restaurant(df):



    df['accesible']= df["parking"] + df["wheelchair_accessible"]

    df['agréable']= df["accept_credit_card"] + df["wifi"]+ df["outdoor_seating"]





    df['prix'] = (df['price1']+df['price2'])/2

    

    

    df['prix_haut']=df['prix']>=800

    df['prix_moyen']=(df['prix']>=488) & (df['prix']<800)

    df['prix_bas']=df['prix']<488

    

    

    df["date"] = 2020 - (pd.to_datetime(df["cdate_y"],errors='coerce')).dt.year

    

    df['date'] = df['date'].fillna(df.date.mean())

    

    df['vieux_resto']=df['date']>=5

    df['jeune_resto']=df['date']<5

    return df





def prepare_data(df):

    

    gender(df)

    

    

    df4 = status(df)

    df5 = join(df,df4)

    

    df6 = date(df5)

    df7=modif_restaurant(df6)

    

    

    colums = ['member_id', 'cdate_x', 'restaurant_id', 'datetime','purpose','gender',

              'status','is_required_prepay_satisfied', 'd1','cdate_y','id',

              'parking','wheelchair_accessible','accept_credit_card','wifi','outdoor_seating',

              'price1','price2','prix','date']

    

    df7.drop(colums,inplace=True, axis=1)

    

    return df7
fill_resultat(resultat_train)

fill_resultat(resultat_test)





test = dummy_purpose_test(resultat_test)

train = dummy_purpose_train(resultat_train)



df_test=prepare_data(test)

df_train=prepare_data(train)

df_test.head()
#creer une base X_train y_train et X_test et y_test (divise en 2/3 apprentissage, 1/3 test)

X_all = df_train.drop(['return90'], axis=1)

y_all = df_train['return90']



num_test = 0.33

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=2)





# build

linreg = LinearRegression()



# train

linreg.fit(X_train, y_train)



# predict

y1_pred = linreg.predict(df_test)



# Choose the type of classifier. 

clf = RandomForestClassifier()



# Choose some parameter combinations to try

parameters = {'n_estimators': [ 300,400,500], 

              'max_features': ['auto' ], 

              'criterion': ['gini', 'entropy'],

              'max_depth': [10],

              'min_samples_split': [500],

              'min_samples_leaf': [2]

             }







# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(roc_auc_score)



# Run the grid search

grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer, n_jobs=-1)

grid_obj = grid_obj.fit(X_train, y_train)



# Set the clf to the best combination of parameters

clf = grid_obj.best_estimator_

print(clf)



# Fit the best algorithm to the data. 

best_clf = clf.fit(X_train, y_train)

print('Train score', clf.score(X_train, y_train))

print('Test score', clf.score(X_test, y_test))





#plus proche voisin,  svm

#regarder la mtrice de confusion avec train test split

#AUC detection 
y_pred = best_clf.predict_proba(X_test)[:,1]





fpr, tpr, _ = roc_curve(y_test,y_pred)

plt.plot(fpr, tpr, marker='.', label='LogistiIIc')





print("Le score AUC est: {} ".format(roc_auc_score(y_test, y_pred)))
regl = LogisticRegression()



parameters = {'penalty': ['l2'],

              'max_iter': [1000],

              'C': [3.0],

              'solver' :[ 'newton-cg','sag', 'lbfgs'],

              'multi_class':['auto', 'ovr', 'multinomial']

             }





# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(roc_auc_score)









# Run the grid search

grid_obj = GridSearchCV(regl, parameters, scoring=acc_scorer, n_jobs=-1, cv=None)

grid_obj = grid_obj.fit(X_train, y_train)



# Set the clf to the best combination of parameters

regl = grid_obj.best_estimator_

print(regl)



# Fit the best algorithm to the data. 



regl.fit(X_train, y_train)

print('Train score', regl.score(X_train, y_train))

print('Test score', regl.score(X_test, y_test))





y1_pred = regl.predict_proba(X_test)[:,1]





fpr, tpr, _ = roc_curve(y_test,y_pred)

plt.plot(fpr, tpr, marker='.', label='LogistiIIc')





print("Le score AUC est: {}".format(roc_auc_score(y_test, y1_pred)))
xgb_model = XGBClassifier()







parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower

              'booster':['gbtree'],

              'learning_rate': [0.02], #so called eta value

              'max_depth': [18],

              'min_child_weight': [13],

              'silent': [1],

              'subsample': [0.7],

              'colsample_bytree': [0.5],

              'n_estimators': [100], #number of trees, change it to 1000 for better results

              'missing':[-999],

              'seed': [1337]}



xgb= RandomizedSearchCV(estimator=xgb_model, param_distributions=parameters, n_jobs=-1, cv=3, verbose=2, random_state=1,n_iter=7,scoring='roc_auc')





#xgb = GridSearchCV(xgb_model, parameters,scoring='roc_auc',n_jobs=-1)





xgb.fit(X_train, y_train)



xgb.best_params_











xgb = grid_obj.best_params_

print(xgb)
estimators= LogisticRegression(C=3.0, class_weight=None, dual=False, fit_intercept=True,

                   intercept_scaling=1, l1_ratio=None, max_iter=5,

                   multi_class='ovr', n_jobs=None, penalty='l2',

                   random_state=None, solver='sag', tol=0.0001, verbose=2,

                   warm_start=False)











xgb_clas=XGBClassifier(estimator=estimators, nthread=4,booster='gbtree',learning_rate=0.01,max_depth=14,

                      min_child_weight=11,silent=1,subsample=0.5,colsample_bytree=0.5,

                      n_estimators=500,missing=-999,seed=1337)



# Fit the best algorithm to the data. 



xgb_clas.fit(X_train, y_train)

print('Train score', xgb_clas.score(X_train, y_train))

print('Test score', xgb_clas.score(X_test, y_test))

y2_pred = xgb_clas.predict_proba(X_test)[:,1]





fpr, tpr, _ = roc_curve(y_test,y_pred)

plt.plot(fpr, tpr, marker='.', label='LogistiIIc')





print("Le score AUC est: {} ".format(roc_auc_score(y_test, y2_pred)))
resultat = xgb_clas.predict_proba(df_test)[:, 1]

print(resultat)
df_res = pd.DataFrame({'booking_id':test.index, 'return90':resultat})

df_res.to_csv('resultat_hugo.csv', index=False)

resu = pd.read_csv('resultat_hugo.csv', index_col='booking_id')

resu