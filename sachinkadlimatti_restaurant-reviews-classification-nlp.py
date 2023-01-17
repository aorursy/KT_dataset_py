import pandas as pd
import numpy as np
df=pd.read_csv("../input/restaurant/Rest.csv")
df.head()
import re
import nltk                                 ## Importing natural language tool kit
nltk.download('stopwords')                  ## downloading Stop words from NLTK
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer  ## Importing stemmmer to find the root word
corp = []

for i in range(0, 1000):
  review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')                 ## We have to keep 'NOT'as it plays a vital in classyfing negative review
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corp.append(review)
print(corp)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = pd.DataFrame(cv.fit_transform(corp).toarray())
y = df.iloc[:, -1].values
X.head(5)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.naive_bayes import GaussianNB,MultinomialNB
nb = GaussianNB()
mnb=MultinomialNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
acc_nb=accuracy_score(y_test, y_pred)
acc_nb
from sklearn.model_selection import cross_val_score
nb_Kfold_accu = cross_val_score(estimator = nb, X = X_train, y = y_train, cv = 10)
nb_Kfold_accu=nb_Kfold_accu.mean()
print("Accuracy: {:.2f} %".format(nb_Kfold_accu*100))
mnb.fit(X_train, y_train)
y_pred = mnb.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
acc_mnb=accuracy_score(y_test, y_pred)
acc_mnb
cm
from sklearn.model_selection import cross_val_score
mnb_Kfold_accu = cross_val_score(estimator = mnb, X = X_train, y = y_train, cv = 10)
mnb_Kfold_accu=mnb_Kfold_accu.mean()
print("Accuracy: {:.2f} %".format(mnb_Kfold_accu*100))

model_compare=pd.DataFrame({'Models':['GuassianNB','Multinomial'],
               'Accuracy Score':[acc_nb,acc_mnb],
            'K Fold Accuracy':[nb_Kfold_accu,mnb_Kfold_accu]})
print(model_compare)
from sklearn.model_selection import GridSearchCV
parameters = [{'alpha': [1,10,50,100], 'fit_prior': [True,False]}]       
grid_search = GridSearchCV(estimator = mnb,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)
parameters = [{'alpha': [10,11,12,13], 'fit_prior': [True]}]   
grid_search = GridSearchCV(estimator = mnb,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)
parameters = [{'alpha': [10.90,10.95,11,11.05,11.1,11.15,11.20], 'fit_prior': [True]}]   
mnb_grid_search = GridSearchCV(estimator = mnb,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
mnb_grid_search = mnb_grid_search.fit(X_train, y_train)
mnb_best_accuracy = mnb_grid_search.best_score_
mnb_best_parameters = mnb_grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)
mnb_y_pred=mnb_grid_search.predict(X_test)
mnb_y_pred
mnb_cm = confusion_matrix(y_test, mnb_y_pred)
print(classification_report(y_test, mnb_y_pred))
acc_mnb=accuracy_score(y_test, mnb_y_pred)
acc_mnb
from sklearn.metrics import roc_auc_score
mnb_roc=roc_auc_score(y_test,mnb_y_pred)
mnb_roc
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression(random_state = 0)
lg.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score
lg_Kfold_accu = cross_val_score(estimator = lg, X = X_train, y = y_train, cv = 5)
lg_Kfold_accu=lg_Kfold_accu.mean()
print("Accuracy: {:.2f} %".format(lg_Kfold_accu*100))
lg_y_pred = lg.predict(X_test)
print(np.concatenate((lg_y_pred.reshape(len(lg_y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
lg_cm = confusion_matrix(y_test, lg_y_pred)
print(classification_report(y_test, lg_y_pred))
acc_lg=accuracy_score(y_test, lg_y_pred)
acc_lg

parameters = [{'penalty': [11,12,'elasticnet'], 'C': [1,10,50,100,200]},
              {'tol': [0.001,0.0001,0.00001]}]
lg_grid_search = GridSearchCV(estimator = lg,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
lg_grid_search = lg_grid_search.fit(X_train, y_train)
lg_best_accuracy = lg_grid_search.best_score_
best_parameters = lg_grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)
parameters = [{'tol': [0.01,0.001,0.0012,0.0013,0.0014]}]
lg_grid_search = GridSearchCV(estimator = lg,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
lg_grid_search = lg_grid_search.fit(X_train, y_train)
lg_best_accuracy = lg_grid_search.best_score_
best_parameters = lg_grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)
parameters = [{'tol': [0.1,0.01,0.001]}]
lg_grid_search = GridSearchCV(estimator = lg,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
lg_grid_search = lg_grid_search.fit(X_train, y_train)
lg_best_accuracy = lg_grid_search.best_score_
best_parameters = lg_grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)
paramters=[int(x) for x in np.linspace(start = 0.0001, stop = 0.001, num = 1)]
lg_grid_search = GridSearchCV(estimator = lg,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
lg_grid_search = lg_grid_search.fit(X_train, y_train)
lg_best_accuracy = lg_grid_search.best_score_
best_parameters = lg_grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)
lg_y_pred = lg_grid_search.predict(X_test)
lg_cm = confusion_matrix(y_test, lg_y_pred)
print(classification_report(y_test, lg_y_pred))
acc_lg=accuracy_score(y_test, lg_y_pred)
acc_lg*100

lg_roc=roc_auc_score(y_test,lg_y_pred)
lg_roc
from sklearn.ensemble import RandomForestClassifier
rc = RandomForestClassifier(random_state = 0)
rc.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score
rc_Kfold_accu = cross_val_score(estimator = rc, X = X_train, y = y_train, cv = 10)
rc_Kfold_accu=rc_Kfold_accu.mean()
print("Accuracy: {:.2f} %".format(rc_Kfold_accu*100))
rc_y_pred = rc.predict(X_test)
print(np.concatenate((rc_y_pred.reshape(len(rc_y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
cm = confusion_matrix(y_test, rc_y_pred)
print(classification_report(y_test, rc_y_pred))
rc_acc=accuracy_score(y_test, rc_y_pred)
rc_acc
params = [{'n_estimators':[100,200,300,400,500], 'criterion':['entropy','gini'],
              'max_depth':[3,4,5]}]
rc_grid_search = GridSearchCV(estimator = rc,
                           param_grid = params,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
rc_grid_search = rc_grid_search.fit(X_train, y_train)
rc_best_accuracy = rc_grid_search.best_score_
best_parameters = rc_grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)
params = [{'n_estimators':[300,310,320,330,340,350,360,370,380,390], 'criterion':['entropy'],
              'max_depth':[5,5.5,6]}]
rc_grid_search = GridSearchCV(estimator = rc,
                           param_grid = params,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
rc_grid_search = rc_grid_search.fit(X_train, y_train)
rc_best_accuracy = rc_grid_search.best_score_
best_parameters = rc_grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)
params = [{'n_estimators':[390,400,450,500], 'criterion':['entropy'],
              'max_depth':[6,6.5,7,7.5,8]}]
rc_grid_search = GridSearchCV(estimator = rc,
                           param_grid = params,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
rc_grid_search = rc_grid_search.fit(X_train, y_train)
rc_best_accuracy = rc_grid_search.best_score_
best_parameters = rc_grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)
params = [{'n_estimators':[390,391,392,393,394,395,396,397,398,399], 'criterion':['entropy'],
              'max_depth':[8,9,10,11]}]
rc_grid_search = GridSearchCV(estimator = rc,
                           param_grid = params,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
rc_grid_search = rc_grid_search.fit(X_train, y_train)
rc_best_accuracy = rc_grid_search.best_score_
best_parameters = rc_grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)
rc_y_pred = rc_grid_search.predict(X_test)
rc_cm = confusion_matrix(y_test, rc_y_pred)
print(classification_report(y_test, rc_y_pred))
acc_rc=accuracy_score(y_test, rc_y_pred)
acc_rc*100

rc_roc=roc_auc_score(y_test,rc_y_pred)
rc_roc
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state = 0)
dtc.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score
dtc_Kfold_accu = cross_val_score(estimator = dtc, X = X_train, y = y_train, cv = 10)
dtc_Kfold_accu=dtc_Kfold_accu.mean()
print("Accuracy: {:.2f} %".format(dtc_Kfold_accu*100))
dtc_y_pred = dtc.predict(X_test)
dtc_y_pred
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
cm = confusion_matrix(y_test, dtc_y_pred)
print(classification_report(y_test, dtc_y_pred))
dtc_acc=accuracy_score(y_test, dtc_y_pred)
dtc_acc
params = [{ 'criterion':['entropy','gini'],
            'max_depth':[3,4,5],'splitter':["best","random"],
           'max_features' :["auto", "sqrt", "log2"],
          'min_samples_split':[2,3,4],
          'ccp_alpha':[0.1,0.01,0.001,0.0001]}]
dtc_grid_search = GridSearchCV(estimator = dtc,
                           param_grid = params,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
dtc_grid_search = dtc_grid_search.fit(X_train, y_train)
dtc_best_accuracy = dtc_grid_search.best_score_
best_parameters = dtc_grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)
params = [{ 'criterion':['gini'],
            'max_depth':[4.5,5,5.5,6,6.5],'splitter':["best"],
           'max_features' :["auto"],
          'min_samples_split':[1,2],
          'ccp_alpha':[0.001,0.0012,0.0013,0.0014,0.0015,0.0016]}]
dtc_grid_search = GridSearchCV(estimator = dtc,
                           param_grid = params,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
dtc_grid_search = dtc_grid_search.fit(X_train, y_train)
dtc_best_accuracy = dtc_grid_search.best_score_
best_parameters = dtc_grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)
dtc_y_pred = dtc_grid_search.predict(X_test)
dtc_cm = confusion_matrix(y_test, dtc_y_pred)
print(classification_report(y_test, dtc_y_pred))
acc_dtc=accuracy_score(y_test, dtc_y_pred)
acc_dtc*100

dtc_roc=roc_auc_score(y_test,dtc_y_pred)
dtc_roc
from sklearn.svm import SVC
svc = SVC(random_state = 0)
svc.fit(X_train, y_train)
svc_Kfold_accu = cross_val_score(estimator = svc, X = X_train, y = y_train, cv = 10)
svc_Kfold_accu=svc_Kfold_accu.mean()
print("Accuracy: {:.2f} %".format(svc_Kfold_accu*100))
svc_y_pred = svc.predict(X_test)
svc_y_pred
cm = confusion_matrix(y_test, svc_y_pred)
print(classification_report(y_test, svc_y_pred))
svc_acc=accuracy_score(y_test, svc_y_pred)
svc_acc
params = [{ 'C':[1,10,50,100,150,200],
            'kernel':['rbf','linear'],'gamma':["scale","auto"],
           'tol' :[0.001,0.0001,0.00001]}]
svc_grid_search = GridSearchCV(estimator = svc,
                           param_grid = params,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
svc_grid_search = svc_grid_search.fit(X_train, y_train)
svc_best_accuracy = svc_grid_search.best_score_
best_parameters = svc_grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)
params = [{ 'C':[150,160,170,180,190],
            'kernel':['rbf'],'gamma':["auto"],
           'tol' :[0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009]}]
svc_grid_search = GridSearchCV(estimator = svc,
                           param_grid = params,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
svc_grid_search = svc_grid_search.fit(X_train, y_train)
svc_best_accuracy = svc_grid_search.best_score_
best_parameters = svc_grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)
params = [{ 'C':[150,151,152,153,154,155,156,157,158,159],
            'kernel':['rbf'],'gamma':["auto"],
           'tol' :[0.002,0.0021,0.0022,0.0023,0.0024,0.0025,0.0026,0.0027,0.0028,0.0029]}]
svc_grid_search = GridSearchCV(estimator = svc,
                           param_grid = params,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
svc_grid_search = svc_grid_search.fit(X_train, y_train)
svc_best_accuracy = svc_grid_search.best_score_
best_parameters = svc_grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)
params = [{ 'C':[148,149,150,150.1,150.2,150.3,150.4,150.5],
            'kernel':['rbf'],'gamma':["auto"],
           'tol' :[0.015,0.016,0.017,0.018,0.019,0.002]}]
svc_grid_search = GridSearchCV(estimator = svc,
                           param_grid = params,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
svc_grid_search = svc_grid_search.fit(X_train, y_train)
svc_best_accuracy = svc_grid_search.best_score_
best_parameters = svc_grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)
svc_y_pred = svc_grid_search.predict(X_test)
svc_cm = confusion_matrix(y_test, svc_y_pred)
print(classification_report(y_test, svc_y_pred))
acc_svc=accuracy_score(y_test, svc_y_pred)
acc_svc*100

svc_roc=roc_auc_score(y_test,svc_y_pred)
svc_roc
model_acc_comp=pd.DataFrame({'Models':['Multinomial_NB','Logistic Regression','Random Forest','Decision Tree',
                                       'Support Vector Machin'],
               'Accuracy Score':[acc_mnb,acc_lg,acc_rc,acc_dtc,acc_svc]})
print(model_acc_comp)
