import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_curve, auc
data_hotel = pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')
data_hotel.head()
data_hotel.isnull().sum(axis = 0)
data_hotel.drop(['agent', 'company', 'arrival_date_week_number', 'arrival_date_day_of_month', 
                 'days_in_waiting_list', 'reservation_status_date', 'reservation_status_date', 
                 'reservation_status', ], inplace=True, axis=1)
data_hotel.dropna(subset=['country', 'children'], inplace=True)
data_hotel.isnull().sum(axis = 0)
class_month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8,
                   'September': 9, 'October': 10, 'November': 11, 'December': 12}
data_hotel['arrival_date_month'] = data_hotel['arrival_date_month'].map(class_month_map)

class_meal_map = {'BB': 1, 'HB': 2, 'FB': 3, 'SC': 0, 'Undefined': 0}
data_hotel['meal'] = data_hotel['meal'].map(class_meal_map)

data_hotel['adr'] = pd.cut(data_hotel['adr'], bins=3, labels=['low', 'median', 'high'])
dummy = pd.get_dummies(data_hotel[['hotel', 'distribution_channel', 'reserved_room_type',
                                  'assigned_room_type', 'deposit_type', 'customer_type', 'adr', 'country', 'market_segment']])
data_hotel = pd.concat([data_hotel, dummy],axis=1)
data_hotel.drop(['hotel', 'distribution_channel', 'reserved_room_type', 'assigned_room_type', 'deposit_type',
                 'customer_type', 'adr', 'country', 'market_segment'], inplace=True, axis=1)

hotel_labels = data_hotel['is_canceled']
hotel_features = data_hotel.drop(['is_canceled'], axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(hotel_features.values, hotel_labels.values, test_size=0.25)
#DecisionTree
from sklearn.tree import DecisionTreeClassifier

#Find the best parameters

# max_depth = range(2, 7)
# min_samples_split = range(2, 9, 2)
# min_samples_leaf = range(2, 11, 2)
# tree_parameters = {'max_depth': max_depth, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}
# grid_search = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=tree_parameters, cv=10)
# grid_search.fit(X_train, Y_train)
# grid_search.best_params_

decision_tree = DecisionTreeClassifier(max_depth=6, min_samples_leaf=6, min_samples_split=2)
decision_tree.fit(X_train, Y_train)
decision_tree_pre = decision_tree.predict(X_test)
print('test-ACC: ', accuracy_score(Y_test, decision_tree_pre))
print('test-REC ', recall_score(Y_test, decision_tree_pre))
print('test-F1: ', f1_score(Y_test, decision_tree_pre))
decision_tree_score = decision_tree.predict_proba(X_test)[:, 1]
tree_fpr, tree_tpr, tree_threshold = roc_curve(Y_test, decision_tree_score)
decision_tree_auc = auc(tree_fpr, tree_tpr)
plt.title('DecisionTree')
plt.stackplot(tree_fpr, tree_tpr, color='red', alpha=0.3)
plt.plot(tree_fpr, tree_tpr, color='black')
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.text(0.5, 0.25, 'AREA={:.2f}\nACC={:.2f}\nREC={:.2f}\nF1={:.2f}'
         .format(decision_tree_auc, accuracy_score(Y_test, decision_tree_pre), 
                 recall_score(Y_test, decision_tree_pre), f1_score(Y_test, decision_tree_pre)))
plt.show()
#-------------------------------------------------------------------------------------------
#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=150)
random_forest.fit(X_train, Y_train)
random_forest_pre = random_forest.predict(X_test)
print('test-ACC: ', accuracy_score(Y_test, random_forest_pre))
print('test-REC ', recall_score(Y_test, random_forest_pre))
print('test-F1: ', f1_score(Y_test, random_forest_pre))
random_forest_score = random_forest.predict_proba(X_test)[:, 1]
rf_fpr, rf_tpr, rf_threshold = roc_curve(Y_test, random_forest_score)
random_forest_auc = auc(rf_fpr, rf_tpr)
plt.title('RandomForest')
plt.stackplot(rf_fpr, rf_tpr, color='red', alpha=0.3)
plt.plot(rf_fpr, rf_tpr, color='black')
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.text(0.5, 0.25, 'AREA={:.2f}\nACC={:.2f}\nREC={:.2f}\nF1={:.2f}'
         .format(random_forest_auc, accuracy_score(Y_test, random_forest_pre), 
                 recall_score(Y_test, random_forest_pre), f1_score(Y_test, random_forest_pre)))
plt.show()
#-------------------------------------------------------------------------------------------
#KNN
from sklearn.neighbors import KNeighborsClassifier

#Find the best parameters

# k = np.arange(10, 21)
# accuracy_test = []
# for i in k:
#     cv_result = cross_val_score(KNeighborsClassifier(n_neighbors=i, weights='distance'), X_train, Y_train, cv=6, scoring='accuracy')
#     accuracy_test.append(cv_result.mean())
# arg_max = np.array(accuracy_test).argmax()
# plt.plot(k, accuracy_test, marker='o')
# plt.text(k[arg_max], accuracy_test[arg_max], 'The best k is {}'.format(k[arg_max]))
# plt.show()

knn_clf = KNeighborsClassifier(n_neighbors=18, weights='distance')
knn_clf.fit(X_train, Y_train)
knn_clf_pre = knn_clf.predict(X_test)
print('test-ACC: ', accuracy_score(Y_test, knn_clf_pre))
print('test-REC ', recall_score(Y_test, knn_clf_pre))
print('test-F1: ', f1_score(Y_test, knn_clf_pre))
knn_clf_score = knn_clf.predict_proba(X_test)[:, 1]
knn_fpr, knn_tpr, knn_threshold = roc_curve(Y_test, knn_clf_score)
knn_auc = auc(knn_fpr, knn_tpr)
plt.title('KNN')
plt.stackplot(knn_fpr, knn_tpr, color='red', alpha=0.3)
plt.plot(knn_fpr, knn_tpr, color='black')
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.text(0.5, 0.25, 'AREA={:.2f}\nACC={:.2f}\nREC={:.2f}\nF1={:.2f}'
         .format(knn_auc, accuracy_score(Y_test, knn_clf_pre), 
                 recall_score(Y_test, knn_clf_pre), f1_score(Y_test, knn_clf_pre)))
plt.show()
#-------------------------------------------------------------------------------------------
#Naive_Bayes
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

mnb_clf = MultinomialNB()
mnb_clf.fit(X_train, Y_train)
mnb_clf_pre = mnb_clf.predict(X_test)
print('mnb-test-ACC: ', accuracy_score(Y_test, mnb_clf_pre))
print('mnb-test-REC ', recall_score(Y_test, mnb_clf_pre))
print('mnb-test-F1: ', f1_score(Y_test, mnb_clf_pre))

print('--------------------------------------------------')

bnb_clf = BernoulliNB()
bnb_clf.fit(X_train, Y_train)
bnb_clf_pre = bnb_clf.predict(X_test)
print('bnb-test-ACC: ', accuracy_score(Y_test, bnb_clf_pre))
print('bnb-test-REC ', recall_score(Y_test, bnb_clf_pre))
print('bnb-test-F1: ', f1_score(Y_test, bnb_clf_pre))
mnb_clf_score = mnb_clf.predict_proba(X_test)[:, 1]
mnb_fpr, mnb_tpr, mnb_threshold = roc_curve(Y_test, mnb_clf_score)
mnb_auc = auc(mnb_fpr, mnb_tpr)
plt.figure('MultinomiaNB')
plt.title('MultinomiaNB')
plt.stackplot(mnb_fpr, mnb_tpr, color='red', alpha=0.3)
plt.plot(mnb_fpr, mnb_tpr, color='black')
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.text(0.5, 0.25, 'AREA={:.2f}\nACC={:.2f}\nREC={:.2f}\nF1={:.2f}'
         .format(mnb_auc, accuracy_score(Y_test, mnb_clf_pre), 
                 recall_score(Y_test, mnb_clf_pre), f1_score(Y_test, mnb_clf_pre)))

bnb_clf_score = bnb_clf.predict_proba(X_test)[:, 1]
bnb_fpr, bnb_tpr, bnb_threshold = roc_curve(Y_test, bnb_clf_score)
bnb_auc = auc(bnb_fpr, bnb_tpr)
plt.figure('BernoulliNB')
plt.title('BernoulliNB')
plt.stackplot(bnb_fpr, bnb_tpr, color='red', alpha=0.3)
plt.plot(bnb_fpr, bnb_tpr, color='black')
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.text(0.5, 0.25, 'AREA={:.2f}\nACC={:.2f}\nREC={:.2f}\nF1={:.2f}'
         .format(bnb_auc, accuracy_score(Y_test, bnb_clf_pre), 
                 recall_score(Y_test, bnb_clf_pre), f1_score(Y_test, bnb_clf_pre)))
plt.show()
#-------------------------------------------------------------------------------------------
#SVM
from sklearn.svm import SVC

# Find the best parameters

# kernel_para=['sigmoid', 'poly', 'linear', 'rbf']
# c_para = [5000, 10000, 15000, 20000, 25000]
# svm_parameters = {'C': c_para, 'kernel': kernel_para}
# grid_SVC = GridSearchCV(estimator=SVC(), param_grid=svm_parameters, scoring='accuracy', cv=6)
# grid_SVC.fit(X_train, Y_train)
# grid_SVC.best_params_, grid_SVC.best_score_

svc_clf = SVC(C=10000, kernel='rbf', probability=True)
svc_clf.fit(X_train, Y_train)
svc_clf_pre = svc_clf.predict(X_test)
svc_clf_train_pre = svc_clf.predict(X_train)
print('test-ACC: ', accuracy_score(Y_test, svc_clf_pre))
print('test-REC ', recall_score(Y_test, svc_clf_pre))
print('test-F1: ', f1_score(Y_test, svc_clf_pre))
print('-----------------------------------------------------')
print('train-ACC: ', accuracy_score(Y_train, svc_clf_train_pre))
print('train-REC ', recall_score(Y_train, svc_clf_train_pre))
print('train-F1: ', f1_score(Y_train, svc_clf_train_pre))
svc_clf_score = svc_clf.predict_proba(X_test)[:, 1]
svc_fpr, svc_tpr, svc_threshold = roc_curve(Y_test, svc_clf_score)
svc_auc = auc(svc_fpr, svc_tpr)
plt.title('SVC')
plt.stackplot(svc_fpr, svc_tpr, color='red', alpha=0.3)
plt.plot(svc_fpr, svc_tpr, color='black')
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.text(0.5, 0.25, 'AREA={:.2f}\nACC={:.2f}\nREC={:.2f}\nF1={:.2f}'
         .format(svc_auc, accuracy_score(Y_test, svc_clf_pre), 
                 recall_score(Y_test, svc_clf_pre), f1_score(Y_test, svc_clf_pre)))
plt.show()
#adaboost
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

#Find the best parameters

# max_depth = range(2, 11)
# min_samples_split = range(2, 9, 2)
# min_samples_leaf = range(2, 9, 2)
# ada_tree_parameters = {'base_estimator__max_depth': max_depth, 
#                        'base_estimator__min_samples_split': min_samples_split,
#                        'base_estimator__min_samples_leaf': min_samples_leaf}
# ada_grid_search = GridSearchCV(estimator=AdaBoostClassifier(base_estimator=DecisionTreeClassifier()), 
#                                param_grid=ada_tree_parameters, cv=5, scoring='roc_auc')
# ada_grid_search.fit(X_train, Y_train)
# ada_grid_search.best_params_

# n_estimators = range(100, 1600, 100)
# learning_rate = [0.01, 0.05, 0.1, 0.15, 0.2]
# ada_tree_params2 = {
#     'n_estimators':n_estimators, 'learning_rate':learning_rate
# }
# ada_grid_search2 = GridSearchCV(estimator=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=5,min_samples_leaf=6,min_samples_split=2)),
#                                  param_grid=gbdt_tree_params2,
#                                  scoring='roc_auc',
#                                  cv=5)
# ada_grid_search2.fit(X_train, Y_train)
# ada_grid_search2.best_params_

ada_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=5, min_samples_leaf=6, min_samples_split=2), n_estimators=1000)
ada_clf.fit(X_train, Y_train)
ada_clf_pre = ada_clf.predict(X_test)
print('test-ACC: ', accuracy_score(Y_test, ada_clf_pre))
print('test-REC ', recall_score(Y_test, ada_clf_pre))
print('test-F1: ', f1_score(Y_test, ada_clf_pre))
ada_clf_score = ada_clf.predict_proba(X_test)[:, 1]
ada_fpr, ada_tpr, ada_threshold = roc_curve(Y_test, ada_clf_score)
ada_auc = auc(ada_fpr, ada_tpr)
plt.title('Adaboost')
plt.stackplot(ada_fpr, ada_tpr, color='red', alpha=0.3)
plt.plot(ada_fpr, ada_tpr, color='black')
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.text(0.5, 0.25, 'AREA={:.2f}\nACC={:.2f}\nREC={:.2f}\nF1={:.2f}'
         .format(ada_auc, accuracy_score(Y_test, ada_clf_pre), 
                 recall_score(Y_test, ada_clf_pre), f1_score(Y_test, ada_clf_pre)))
plt.show()
#GBDT
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

#Find the best parameters

# max_depth = range(2, 11)
# min_samples_split = range(2, 9, 2)
# min_samples_leaf = range(2, 9, 2)
# n_estimators = range(100, 1600, 100)
# learning_rate = [0.01, 0.05, 0.1, 0.15, 0.2]
# gbdt_tree_parameters = {'max_depth': max_depth, 
#                        'min_samples_split': min_samples_split,
#                        'min_samples_leaf': min_samples_leaf,
#                        'n_estimators':n_estimators, 
#                         'learning_rate':learning_rate}
# gbdt_grid_search = GridSearchCV(estimator=GradientBoostingClassifier(), 
#                                param_grid=gbdt_tree_parameters, cv=5, scoring='roc_auc')
# gbdt_grid_search.fit(X_train, Y_train)
# gbdt_grid_search.best_params_

gbdt_clf = GradientBoostingClassifier(max_depth=5, min_samples_split=2, min_samples_leaf=6, learning_rate=0.05, n_estimators=1000)
gbdt_clf.fit(X_train, Y_train)
gbdt_clf_pre = gbdt_clf.predict(X_test)
print('test-ACC: ', accuracy_score(Y_test, gbdt_clf_pre))
print('test-REC ', recall_score(Y_test, gbdt_clf_pre))
print('test-F1: ', f1_score(Y_test, gbdt_clf_pre))
gbdt_clf_score = gbdt_clf.predict_proba(X_test)[:, 1]
gbdt_fpr, gbdt_tpr, gbdt_threshold = roc_curve(Y_test, gbdt_clf_score)
gbdt_auc = auc(gbdt_fpr, gbdt_tpr)
plt.title('GBDT')
plt.stackplot(gbdt_fpr, gbdt_tpr, color='red', alpha=0.3)
plt.plot(gbdt_fpr, gbdt_tpr, color='black')
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.text(0.5, 0.25, 'AREA={:.2f}\nACC={:.2f}\nREC={:.2f}\nF1={:.2f}'
         .format(gbdt_auc, accuracy_score(Y_test, gbdt_clf_pre), 
                 recall_score(Y_test, gbdt_clf_pre), f1_score(Y_test, gbdt_clf_pre)))
plt.show()
#XGBoost
import xgboost

#Find the best parameters

# max_depth = range(2, 11)
# learning_rate = [0.01, 0.05, 0.1, 0.15, 0.2]
# n_estimators = range(100, 1600, 100)
# xgboost_params = {'max_depth':max_depth, 'learning_rate':learning_rate, 'n_estimators':n_estimators}
# xgboost_grid = GridSearchCV(estimator=xgboost.XGBClassifier(), param_grid=xgboost_params, cv=5, scoring='roc_auc')
# xgboost_grid.fit(X_train, Y_train)
# xgboost_grid.best_params_

xgboost_clf = xgboost.XGBClassifier(max_depth=10, learning_rate=0.05, n_estimators=1000)
xgboost_clf.fit(X_train, Y_train)
xgboost_clf_pre = xgboost_clf.predict(X_test)
print('test-ACC: ', accuracy_score(Y_test, xgboost_clf_pre))
print('test-REC ', recall_score(Y_test, xgboost_clf_pre))
print('test-F1: ', f1_score(Y_test, xgboost_clf_pre))
xgboost_clf_score = xgboost_clf.predict_proba(X_test)[:, 1]
xgboost_fpr, xgboost_tpr, xgboost_threshold = roc_curve(Y_test, xgboost_clf_score)
xgboost_auc = auc(xgboost_fpr, xgboost_tpr)
plt.title('XGBoost')
plt.stackplot(xgboost_fpr, xgboost_tpr, color='red', alpha=0.3)
plt.plot(xgboost_fpr, xgboost_tpr, color='black')
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.text(0.5, 0.25, 'AREA={:.2f}\nACC={:.2f}\nREC={:.2f}\nF1={:.2f}'
         .format(xgboost_auc, accuracy_score(Y_test, xgboost_clf_pre), 
                 recall_score(Y_test, xgboost_clf_pre), f1_score(Y_test, xgboost_clf_pre)))
plt.show()
#-------------------------------------------------------------------------------------------
#ANNs(Artificial Neural Networks)
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD


mdl = Sequential()
mdl.add(Dense(500, input_dim=len(X_train[0])))
mdl.add(Activation('sigmoid'))
# mdl.add(Dense(500))
# mdl.add(Activation('sigmoid'))
mdl.add(Dense(2))
mdl.add(Activation('softmax'))
sgd = SGD(lr=0.01)
mdl.compile(loss='mean_squared_error', optimizer='sgd')
mdl.fit(X_train, Y_train, epochs=100, batch_size=10000)
mdl_pre = mdl.predict_classes(X_test)
print('test-ACC: ', accuracy_score(Y_test, mdl_pre))
print('test-REC ', recall_score(Y_test, mdl_pre))
print('test-F1: ', f1_score(Y_test, mdl_pre))
#LR
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, Y_train)
lr_pre = lr.predict(X_test)
print('test-ACC: ', accuracy_score(Y_test, lr_pre))
print('test-REC ', recall_score(Y_test, lr_pre))
print('test-F1: ', f1_score(Y_test, lr_pre))
lr_score = lr.predict_proba(X_test)[:, 1]
lr_fpr, lr_tpr, lr_threshold = roc_curve(Y_test, lr_score)
lr_auc = auc(lr_fpr, lr_tpr)
plt.title('LinearRegression')
plt.stackplot(lr_fpr, lr_tpr, color='red', alpha=0.3)
plt.plot(lr_fpr, lr_tpr, color='black')
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.text(0.5, 0.25, 'AREA={:.2f}\nACC={:.2f}\nREC={:.2f}\nF1={:.2f}'
         .format(lr_auc, accuracy_score(Y_test, lr_pre), 
                 recall_score(Y_test, lr_pre), f1_score(Y_test, lr_pre)))
plt.show()
# we can find that RandomForest model is the best one to predict the cancelling, 
# therefore we can see the importance of variable, and analyze the data.

# find the importance of variable
importance_variable = random_forest.feature_importances_
importance_series = pd.Series(importance_variable, index = hotel_features.columns)
importance_series.sort_values(ascending=False, inplace=True)
top_impt = importance_series.head(10)
plt.title('Importance_variable')
sns.barplot(y=top_impt.index, x=top_impt.values)
plt.show()
# analyze the lead_time

plt.figure()
plt.title('lead_time hist')
sns.distplot(data_hotel['lead_time'], bins=20) #in this figure,we know that almost time is the period [0, 100]

lead_time_part1 = data_hotel[data_hotel['lead_time'] >= 0][data_hotel['lead_time'] <= 100]
lead_time_part2 = data_hotel[data_hotel['lead_time'] > 100][data_hotel['lead_time'] <= 200]
lead_time_part3 = data_hotel[data_hotel['lead_time'] > 200][data_hotel['lead_time'] <= 300]
lead_time_part4 = data_hotel[data_hotel['lead_time'] > 300]

plt.figure()
plt.bar(x=['0-100', '101-200', '201-300', '300-'], height=[lead_time_part1.shape[0], 
                                                           lead_time_part2.shape[0],
                                                           lead_time_part3.shape[0], 
                                                           lead_time_part4.shape[0]], color='red', width=0.5, label='total')

plt.bar(x=['0-100', '101-200', '201-300', '300-'], height=[lead_time_part1[lead_time_part1['is_canceled']==1].shape[0], 
                                                           lead_time_part2[lead_time_part2['is_canceled']==1].shape[0],
                                                           lead_time_part3[lead_time_part3['is_canceled']==1].shape[0],
                                                           lead_time_part4[lead_time_part4['is_canceled']==1].shape[0]], 
                                                           color='#0000A3', width=0.5, label='canceled')
plt.ylabel('the number of samples')
plt.xlabel('lead_time /d')
plt.legend()
plt.tight_layout()
plt.show()
# analyze "arrival_date_month"

plt.figure()
plt.subplot(121)
plt.pie(x=hotel_features['arrival_date_month'].value_counts(normalize=True).values,
       labels=hotel_features['arrival_date_month'].value_counts(normalize=True).index,
       autopct='%.1f%%')

plt.subplot(122)
hotel_features['quarters'] = pd.cut(hotel_features['arrival_date_month'], bins=4, labels=['1st', '2nd', '3rd', '4th']) 
plt.pie(x=hotel_features['quarters'].value_counts(normalize=True).values,
       labels=hotel_features['quarters'].value_counts(normalize=True).index,
       autopct='%.1f%%')
plt.show()