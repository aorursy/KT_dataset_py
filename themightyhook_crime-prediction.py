import pandas as pd

import numpy as np
import pyreadr



FRAUD1 = pyreadr.read_r('all_feature_set_FRAUD_interval_1.RData')# 9 - 12 

FRAUD2 = pyreadr.read_r('all_feature_set_FRAUD_interval_2.RData')# 12 - 15 

FRAUD3 = pyreadr.read_r('all_feature_set_FRAUD_interval_3.RData')# 15 - 18

FRAUD4 = pyreadr.read_r('all_feature_set_FRAUD_interval_4.RData')#18 - 21 

FRAUD5 = pyreadr.read_r('all_feature_set_FRAUD_interval_5.RData')#21 - 24
FRAUD1 = FRAUD1["all_feature_set"]

FRAUD2 = FRAUD2["all_feature_set"]

FRAUD3 = FRAUD3["all_feature_set"]

FRAUD4 = FRAUD4["all_feature_set"]

FRAUD5 = FRAUD5["all_feature_set"]

# With Dynamic Features

feature_cols = ['seasonal_density','crime_density_30','crime_density_population_30','crime_density_7',

                'crime_density_population_7','neighbourhood_crime_density_30','neighbouhood_crime_density_population_30',

               'Neighbourhood_7_days_density','Neighbourhood_7_days_density_population','gender_ratio',

               'ethenic_diversity','Hispanic_percentage','black_percentage','economic_diversity','median_income',

               'Rented_house','median_age','Shop and Service_freq','Outdoor and Recreation_freq','Home_freq',

                'Professional and Other Places_freq','Food_freq','Travel and Transport_freq','Arts and Entertainment_freq',

                'College and University_freq','NightLife Spot_freq','total_density','location_entropy','similarity',

                'Entropy','user_count','Visitor_ratio','area_popularity','observation_freq']



#Without Dynamic Features

WOfeature_cols = ['seasonal_density','crime_density_30','crime_density_population_30','crime_density_7',

                'crime_density_population_7','neighbourhood_crime_density_30','neighbouhood_crime_density_population_30',

               'Neighbourhood_7_days_density','Neighbourhood_7_days_density_population','gender_ratio',

               'ethenic_diversity','Hispanic_percentage','black_percentage','economic_diversity','median_income',

               'Rented_house','median_age','Shop and Service_freq','Outdoor and Recreation_freq','Home_freq',

                'Professional and Other Places_freq','Food_freq','Travel and Transport_freq','Arts and Entertainment_freq',

                'College and University_freq','NightLife Spot_freq','total_density','location_entropy']

#Change DataFrame name and features everytime, thus reducing redundant code 

Fraud1[WOfeature_cols] = np.nan_to_num(Fraud1[WOfeature_cols])

from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=0)

X = Traffic1[WOfeature_cols]

y = Traffic1['response']

X_resampled, y_resampled = rus.fit_resample(X, y)

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics



X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled,test_size=0.50)





logreg = LogisticRegression()





logreg.fit(X_train, y_train)







import numpy as np

from sklearn import metrics

from sklearn.metrics import roc_auc_score





print(roc_auc_score(y_test, y_pred))
from sklearn.svm import SVC, LinearSVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, y_train)



Y_pred = linear_svc.predict(X_test)



acc_linear_svc = linear_svc.score(X_train, y_train)



print(roc_auc_score(y_test, Y_pred))
from sklearn.ensemble import RandomForestClassifier





clf=RandomForestClassifier(n_estimators=100)





clf.fit(X_train,y_train)



y_pred=clf.predict(X_test)
from sklearn.metrics import accuracy_score,classification_report

from sklearn.metrics import f1_score

print(roc_auc_score(y_test, y_pred))

print(accuracy_score(y_test, y_pred))

f1_score(y_test, y_pred)
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15,), random_state=1)

clf.fit(X_train, y_train)                         

MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',

              beta_1=0.9, beta_2=0.999, early_stopping=False,

              epsilon=1e-08, hidden_layer_sizes=(15,),

              learning_rate='constant', learning_rate_init=0.001,

              max_iter=200, momentum=0.9, n_iter_no_change=10,

              nesterovs_momentum=True, power_t=0.5,  random_state=1,

              shuffle=True, solver='lbfgs', tol=0.0001,

              validation_fraction=0.1, verbose=False, warm_start=False)

p = clf.predict(X_test)

print(roc_auc_score(y_test, p))
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

knn = KNeighborsClassifier()

params_knn = {'n_neighbors': np.arange(1, 25)}

knn_gs = GridSearchCV(knn, params_knn, cv=5)

knn_gs.fit(X_train, y_train)

knn_best = knn_gs.best_estimator_

print(knn_gs.best_params_)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

params_rf = {'n_estimators': [50, 100, 200]}

rf_gs = GridSearchCV(rf, params_rf, cv=5)

rf_gs.fit(X_train, y_train)

rf_best = rf_gs.best_estimator_

print(rf_gs.best_params_)
print('knn: {}'.format(knn_best.score(X_test, y_test)))

print('rf: {}'.format(rf_best.score(X_test, y_test)))

print('log_reg: {}'.format(logreg.score(X_test, y_test)))
from sklearn.ensemble import VotingClassifier

estimators=[('knn', knn_best), ('rf', rf_best), ('log_reg', logreg)]

ensemble = VotingClassifier(estimators, voting='hard')

ensemble.fit(X_train, y_train)

ensemble.score(X_test, y_test)
import pyreadr

Traffic1 = pyreadr.read_r('all_feature_set_TRAFFIC_interval_1.RData') 

Traffic2 = pyreadr.read_r('all_feature_set_TRAFFIC_interval_2.RData')

Traffic3 = pyreadr.read_r('all_feature_set_TRAFFIC_interval_3.RData')

Traffic4 = pyreadr.read_r('all_feature_set_TRAFFIC_interval_4.RData')

Traffic5 = pyreadr.read_r('all_feature_set_TRAFFIC_interval_5.RData')
Traffic1 = Traffic1["all_feature_set"]

Traffic2 = Traffic2["all_feature_set"]

Traffic3 = Traffic3["all_feature_set"]

Traffic4 = Traffic4["all_feature_set"]

Traffic5 = Traffic5["all_feature_set"]
# With Dynamic Features

feature_cols = ['seasonal_density','crime_density_30','crime_density_population_30','crime_density_7',

                'crime_density_population_7','neighbourhood_crime_density_30','neighbouhood_crime_density_population_30',

               'Neighbourhood_7_days_density','Neighbourhood_7_days_density_population','gender_ratio',

               'ethenic_diversity','Hispanic_percentage','black_percentage','economic_diversity','median_income',

               'Rented_house','median_age','Shop and Service_freq','Outdoor and Recreation_freq','Home_freq',

                'Professional and Other Places_freq','Food_freq','Travel and Transport_freq','Arts and Entertainment_freq',

                'College and University_freq','NightLife Spot_freq','total_density','location_entropy','similarity',

                'Entropy','user_count','observation_freq','Visitor_ratio','area_popularity']



#Without Dynamic Features

WOfeature_cols = ['seasonal_density','crime_density_30','crime_density_population_30','crime_density_7',

                'crime_density_population_7','neighbourhood_crime_density_30','neighbouhood_crime_density_population_30',

               'Neighbourhood_7_days_density','Neighbourhood_7_days_density_population','gender_ratio',

               'ethenic_diversity','Hispanic_percentage','black_percentage','economic_diversity','median_income',

               'Rented_house','median_age','Shop and Service_freq','Outdoor and Recreation_freq','Home_freq',

                'Professional and Other Places_freq','Food_freq','Travel and Transport_freq','Arts and Entertainment_freq',

                'College and University_freq','NightLife Spot_freq','total_density','location_entropy']

#Change Variable name and features everytime, thus reducing redundant code 

Traffic5[feature_cols] = np.nan_to_num(Traffic5[feature_cols])
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=0)

X = Traffic5[feature_cols]

y = Traffic5['response']

X_resampled, y_resampled = rus.fit_resample(X, y)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics





X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled,test_size=0.50)





logreg = LogisticRegression()





logreg.fit(X_train, y_train)









import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score



print(roc_auc_score(y_test, y_pred))
from sklearn.svm import SVC, LinearSVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, y_train)



Y_pred = linear_svc.predict(X_test)



acc_linear_svc = linear_svc.score(X_train, y_train)



print(roc_auc_score(y_test, Y_pred))
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,classification_report

from sklearn.metrics import f1_score



clf=RandomForestClassifier(n_estimators=100)





clf.fit(X_train,y_train)



y_pred=clf.predict(X_test)



print(roc_auc_score(y_test, y_pred))

print(accuracy_score(y_test, y_pred))

f1_score(y_test, y_pred)

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15,), random_state=1)

clf.fit(X_train, y_train)                         

MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',

              beta_1=0.9, beta_2=0.999, early_stopping=False,

              epsilon=1e-08, hidden_layer_sizes=(15,),

              learning_rate='constant', learning_rate_init=0.001,

              max_iter=200, momentum=0.9, n_iter_no_change=10,

              nesterovs_momentum=True, power_t=0.5,  random_state=1,

              shuffle=True, solver='lbfgs', tol=0.0001,

              validation_fraction=0.1, verbose=False, warm_start=False)

p = clf.predict(X_test)

print(roc_auc_score(y_test, p))
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

knn = KNeighborsClassifier()

params_knn = {'n_neighbors': np.arange(1, 25)}

knn_gs = GridSearchCV(knn, params_knn, cv=5)

knn_gs.fit(X_train, y_train)

knn_best = knn_gs.best_estimator_

print(knn_gs.best_params_)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

params_rf = {'n_estimators': [50, 100, 200]}

rf_gs = GridSearchCV(rf, params_rf, cv=5)

rf_gs.fit(X_train, y_train)

rf_best = rf_gs.best_estimator_

print(rf_gs.best_params_)
print('knn: {}'.format(knn_best.score(X_test, y_test)))

print('rf: {}'.format(rf_best.score(X_test, y_test)))

print('log_reg: {}'.format(logreg.score(X_test, y_test)))
from sklearn.ensemble import VotingClassifier

estimators=[('knn', knn_best), ('rf', rf_best), ('log_reg', logreg)]

ensemble = VotingClassifier(estimators, voting='hard')

ensemble.fit(X_train, y_train)

ensemble.score(X_test, y_test)
Burglary1 = pyreadr.read_r('all_feature_set_BURGLARY_interval_1.RData') 

Burglary2 = pyreadr.read_r('all_feature_set_BURGLARY_interval_2.RData')

Burglary3 = pyreadr.read_r('all_feature_set_BURGLARY_interval_3.RData')

Burglary4 = pyreadr.read_r('all_feature_set_BURGLARY_interval_4.RData')

Burglary5 = pyreadr.read_r('all_feature_set_BURGLARY_interval_5.RData')

Burglary1 = Burglary1["all_feature_set"]

Burglary2 = Burglary2["all_feature_set"]

Burglary3 = Burglary3["all_feature_set"]

Burglary4 = Burglary4["all_feature_set"]

Burglary5 = Burglary5["all_feature_set"]
import sklearn.preprocessing

# With Dynamic Features

feature_cols = ['seasonal_density','crime_density_30','crime_density_population_30','crime_density_7',

                'crime_density_population_7','neighbourhood_crime_density_30','neighbouhood_crime_density_population_30',

               'Neighbourhood_7_days_density','Neighbourhood_7_days_density_population','gender_ratio',

               'ethenic_diversity','Hispanic_percentage','black_percentage','economic_diversity','median_income',

               'Rented_house','median_age','Shop and Service_freq','Outdoor and Recreation_freq','Home_freq',

                'Professional and Other Places_freq','Food_freq','Travel and Transport_freq','Arts and Entertainment_freq',

                'College and University_freq','NightLife Spot_freq','total_density','location_entropy','similarity',

                'Entropy','user_count','observation_freq','Visitor_ratio','area_popularity']



#Without Dynamic Features

WOfeature_cols = ['seasonal_density','crime_density_30','crime_density_population_30','crime_density_7',

                'crime_density_population_7','neighbourhood_crime_density_30','neighbouhood_crime_density_population_30',

               'Neighbourhood_7_days_density','Neighbourhood_7_days_density_population','gender_ratio',

               'ethenic_diversity','Hispanic_percentage','black_percentage','economic_diversity','median_income',

               'Rented_house','median_age','Shop and Service_freq','Outdoor and Recreation_freq','Home_freq',

                'Professional and Other Places_freq','Food_freq','Travel and Transport_freq','Arts and Entertainment_freq',

                'College and University_freq','NightLife Spot_freq','total_density','location_entropy']

#Change DataFrame name and features everytime, thus reducing redundant code 



Burglary5[WOfeature_cols] = np.nan_to_num(Burglary5[WOfeature_cols])



Burglary5[WOfeature_cols] = sklearn.preprocessing.normalize(Burglary5[WOfeature_cols], norm='l1',

                                                         axis=1, copy=True, return_norm=False)
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=0)

X = Burglary5[WOfeature_cols]

y = Burglary5['response']

X_resampled, y_resampled = rus.fit_resample(X, y)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics



X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled,test_size=0.50)





logreg = LogisticRegression()





logreg.fit(X_train, y_train)





import numpy as np

from sklearn import metrics

from sklearn.metrics import roc_auc_score

print(roc_auc_score(y_test, Y_pred))
from sklearn.svm import SVC, LinearSVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, y_train)



Y_pred = linear_svc.predict(X_test)



acc_linear_svc = linear_svc.score(X_train, y_train)



print(roc_auc_score(y_test, Y_pred))
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,classification_report

from sklearn.metrics import f1_score



clf=RandomForestClassifier(n_estimators=100)





clf.fit(X_train,y_train)



y_pred=clf.predict(X_test)



print(roc_auc_score(y_test, y_pred))

print(accuracy_score(y_test, y_pred))

f1_score(y_test, y_pred)
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15,), random_state=1)

clf.fit(X_train, y_train)                         

MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',

              beta_1=0.9, beta_2=0.999, early_stopping=False,

              epsilon=1e-08, hidden_layer_sizes=(15,),

              learning_rate='constant', learning_rate_init=0.001,

              max_iter=200, momentum=0.9, n_iter_no_change=10,

              nesterovs_momentum=True, power_t=0.5,  random_state=1,

              shuffle=True, solver='lbfgs', tol=0.0001,

              validation_fraction=0.1, verbose=False, warm_start=False)

p = clf.predict(X_test)

print(roc_auc_score(y_test, p))
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

knn = KNeighborsClassifier()

params_knn = {'n_neighbors': np.arange(1, 25)}

knn_gs = GridSearchCV(knn, params_knn, cv=5)

knn_gs.fit(X_train, y_train)

knn_best = knn_gs.best_estimator_

print(knn_gs.best_params_)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

params_rf = {'n_estimators': [50, 100, 200]}

rf_gs = GridSearchCV(rf, params_rf, cv=5)

rf_gs.fit(X_train, y_train)

rf_best = rf_gs.best_estimator_

print(rf_gs.best_params_)
print('knn: {}'.format(knn_best.score(X_test, y_test)))

print('rf: {}'.format(rf_best.score(X_test, y_test)))

print('log_reg: {}'.format(logreg.score(X_test, y_test)))
from sklearn.ensemble import VotingClassifier

estimators=[('knn', knn_best), ('rf', rf_best), ('log_reg', logreg)]

ensemble = VotingClassifier(estimators, voting='hard')

ensemble.fit(X_train, y_train)

ensemble.score(X_test, y_test)

p = ensemble.predict(X_test)

print(roc_auc_score(y_test, p))


Drug1 = pyreadr.read_r('all_feature_set_DRUG_interval_1.RData') 

Drug2 = pyreadr.read_r('all_feature_set_DRUG_interval_2.RData')

Drug3 = pyreadr.read_r('all_feature_set_DRUG_interval_3.RData')

Drug4 = pyreadr.read_r('all_feature_set_DRUG_interval_4.RData')

Drug5 = pyreadr.read_r('all_feature_set_DRUG_interval_5.RData')
Drug1 = Drug1["all_feature_set"]

Drug2 = Drug2["all_feature_set"]

Drug3 = Drug3["all_feature_set"]

Drug4 = Drug4["all_feature_set"]

Drug5 = Drug5["all_feature_set"]
import sklearn.preprocessing

# With Dynamic Features

feature_cols = ['seasonal_density','crime_density_30','crime_density_population_30','crime_density_7',

                'crime_density_population_7','neighbourhood_crime_density_30','neighbouhood_crime_density_population_30',

               'Neighbourhood_7_days_density','Neighbourhood_7_days_density_population','gender_ratio',

               'ethenic_diversity','Hispanic_percentage','black_percentage','economic_diversity','median_income',

               'Rented_house','median_age','Shop and Service_freq','Outdoor and Recreation_freq','Home_freq',

                'Professional and Other Places_freq','Food_freq','Travel and Transport_freq','Arts and Entertainment_freq',

                'College and University_freq','NightLife Spot_freq','total_density','location_entropy','similarity',

                'Entropy','user_count','observation_freq','Visitor_ratio','area_popularity']



#Without Dynamic Features

WOfeature_cols = ['seasonal_density','crime_density_30','crime_density_population_30','crime_density_7',

                'crime_density_population_7','neighbourhood_crime_density_30','neighbouhood_crime_density_population_30',

               'Neighbourhood_7_days_density','Neighbourhood_7_days_density_population','gender_ratio',

               'ethenic_diversity','Hispanic_percentage','black_percentage','economic_diversity','median_income',

               'Rented_house','median_age','Shop and Service_freq','Outdoor and Recreation_freq','Home_freq',

                'Professional and Other Places_freq','Food_freq','Travel and Transport_freq','Arts and Entertainment_freq',

                'College and University_freq','NightLife Spot_freq','total_density','location_entropy']

#Change DataFrame name and features everytime, thus reducing redundant code 

Drug1[feature_cols] = np.nan_to_num(Drug1[feature_cols])

Drug1[feature_cols] = sklearn.preprocessing.normalize(Drug1[feature_cols], norm='l1',

                                                      axis=1, copy=True, return_norm=False)
from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import train_test_split

rus = RandomUnderSampler(random_state=0)

X = Drug1[feature_cols]

y = Drug1['response']

X_resampled, y_resampled = rus.fit_resample(X, y)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import roc_auc_score





X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled,test_size=0.50)





logreg = LogisticRegression()





logreg.fit(X_train, y_train)









Y_pred = logreg.predict(X_test)

print(roc_auc_score(y_test, Y_pred))
from sklearn.svm import SVC, LinearSVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, y_train)



Y_pred = linear_svc.predict(X_test)



acc_linear_svc = linear_svc.score(X_train, y_train)



print(roc_auc_score(y_test, Y_pred))
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,classification_report

from sklearn.metrics import f1_score



clf=RandomForestClassifier(n_estimators=100)





clf.fit(X_train,y_train)



y_pred=clf.predict(X_test)



print(roc_auc_score(y_test, y_pred))

print(accuracy_score(y_test, y_pred))

f1_score(y_test, y_pred)
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15,), random_state=1)

clf.fit(X_train, y_train)                         

MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',

              beta_1=0.9, beta_2=0.999, early_stopping=False,

              epsilon=1e-08, hidden_layer_sizes=(15,),

              learning_rate='constant', learning_rate_init=0.001,

              max_iter=200, momentum=0.9, n_iter_no_change=10,

              nesterovs_momentum=True, power_t=0.5,  random_state=1,

              shuffle=True, solver='lbfgs', tol=0.0001,

              validation_fraction=0.1, verbose=False, warm_start=False)

p = clf.predict(X_test)

print(roc_auc_score(y_test, p))
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

knn = KNeighborsClassifier()

params_knn = {'n_neighbors': np.arange(1, 25)}

knn_gs = GridSearchCV(knn, params_knn, cv=5)

knn_gs.fit(X_train, y_train)

knn_best = knn_gs.best_estimator_

print(knn_gs.best_params_)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

params_rf = {'n_estimators': [50, 100, 200]}

rf_gs = GridSearchCV(rf, params_rf, cv=5)

rf_gs.fit(X_train, y_train)

rf_best = rf_gs.best_estimator_

print(rf_gs.best_params_)
print('knn: {}'.format(knn_best.score(X_test, y_test)))

print('rf: {}'.format(rf_best.score(X_test, y_test)))

print('log_reg: {}'.format(logreg.score(X_test, y_test)))
from sklearn.ensemble import VotingClassifier

estimators=[('knn', knn_best), ('rf', rf_best), ('log_reg', logreg)]

ensemble = VotingClassifier(estimators, voting='hard')

ensemble.fit(X_train, y_train)

ensemble.score(X_test, y_test)

p = ensemble.predict(X_test)

print(roc_auc_score(y_test, p))
Theft1 = pyreadr.read_r(r'C:\Level 7\ML\Project\New folder\New folder\all_feature_set_LARCENY_interval_3.RData')# 15 - 18

Theft2 = pyreadr.read_r(r'C:\Level 7\ML\Project\New folder\New folder\all_feature_set_LARCENY_interval_5.RData')# 21 - 24 
Theft1 = Theft1["all_feature_set"]

Theft2 = Theft2["all_feature_set"]
feature_cols = ['seasonal_density','crime_density_30','crime_density_population_30','crime_density_7',

                'crime_density_population_7','neighbourhood_crime_density_30','neighbouhood_crime_density_population_30',

               'Neighbourhood_7_days_density','Neighbourhood_7_days_density_population','gender_ratio',

               'ethenic_diversity','Hispanic_percentage','black_percentage','economic_diversity','median_income',

               'Rented_house','median_age','Shop and Service_freq','Outdoor and Recreation_freq','Home_freq',

                'Professional and Other Places_freq','Food_freq','Travel and Transport_freq','Arts and Entertainment_freq',

                'College and University_freq','NightLife Spot_freq','total_density','location_entropy','similarity',

                'Entropy','user_count','observation_freq','Visitor_ratio','area_popularity']



Theft2[feature_cols] = np.nan_to_num(Theft2[feature_cols])

Theft2[feature_cols] = sklearn.preprocessing.normalize(Theft2[feature_cols], norm='l1',

                                                      axis=1, copy=True, return_norm=False)
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=0)

X = Theft2[feature_cols]

y = Theft2['response']

X_resampled, y_resampled = rus.fit_resample(X, y)
#normalize all the features between 0 and 1

from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()

minmax = min_max_scaler.fit_transform(Theft1[feature_cols])
#find correlation with LR model between the most important 12 features

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import roc_auc_score





X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled,test_size=0.50)





logreg = LogisticRegression()





logreg.fit(X_train, y_train)





#print(logreg.coef_)



coef_dict = {}

for coef, feat in zip(logreg.coef_[0,:],feature_cols):

    coef_dict[feat] = coef

    

print(coef_dict)
#error plot 

from sklearn.datasets import make_classification

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



from matplotlib import pyplot







pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Without Dynamic')

pyplot.plot(lr_fpr, lr_tpr, marker='.', label='With Dynamic')

pyplot.xlabel('False Positive Rate')

pyplot.ylabel('True Positive Rate')



pyplot.legend()



pyplot.show()