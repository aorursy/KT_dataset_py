# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler,LabelEncoder

from xgboost import XGBClassifier



from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV,train_test_split

from sklearn.pipeline import Pipeline



from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.utils import class_weight

#performance metrics

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold, StratifiedKFold
train_fe_X = pd.read_feather('../input/surface-prediction-feature-engineering/Train_FE')

test_fe_X = pd.read_feather('../input/surface-prediction-feature-engineering/Test_FE')

train_df_Y = pd.read_feather('../input/surface-prediction-feature-engineering/Target')
train_fe_X.shape, train_df_Y.shape, test_fe_X.shape
# fig, ax = plt.subplots(figsize=(12, 10))

# corr=train.corr()



# sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values,ax=ax)

# plt.show()
# train_data = train_fe_X.drop(columns=['series_id','velocity_to_acceleration_norm_entropy'])

# test_data = test_fe_X.drop(columns=['series_id','velocity_to_acceleration_norm_entropy'])

le = LabelEncoder()

target =pd.DataFrame()

target['surface'] = le.fit_transform(train_df_Y['surface'])

classes = (train_df_Y['surface'].value_counts()).index

# train_data.shape, target.shape, test_data.shape

target.shape
vif = pd.DataFrame()

%time vif['vif_factor'] = [variance_inflation_factor(train_fe_X.values,i) for i in range(train_fe_X.shape[1])]

vif['features'] = train_fe_X.columns
vif.fillna(value=9999,inplace=True)

vif.isnull().sum()
vif.sort_values('vif_factor',axis=0,inplace=True, ascending=False)
features_to_remove = vif.loc[vif['vif_factor'] > 5,'features'].values

features_to_remove = list(features_to_remove)
features_to_remove.append('series_id')

features_to_remove.append('velocity_to_acceleration_norm_entropy')
#Drop these columns

train_data=train_fe_X.drop(columns=features_to_remove)

test_data = test_fe_X.drop(columns=features_to_remove)

train_data.shape,test_data.shape,target.shape
# pipe = Pipeline((

#     ('xgb', XGBClassifier(n_estimators=500 , random_state=22)),    

#     ))



# params = {        

#     'xgb__learning_rate':[1,0.001,0.05]

#     }





# model,name = (params,'XGBoost')

# print('-'*50)

# print( "Starting Randomized Search for %s" %name)                

# rs = RandomizedSearchCV(pipe, model, verbose=5, refit=False, n_jobs=3,cv=5,random_state=22,n_iter=20)

# rs = rs.fit(train_data, target)
# print("Finished Randomized Search for %s"%name)

# print('Best Score %.5f'%rs.best_score_)

# print('Best Param %s'%rs.best_params_)

# # print('-'*50)





# class_weights = class_weight.compute_class_weight('balanced', np.unique(target['surface'] ),target['surface'] )

# class_weights = dict(zip(np.unique(target['surface']),class_weights))

# class_weights



%%time

folds = StratifiedKFold(n_splits=50, shuffle=True, random_state=59)

predicted = np.zeros((test_data.shape[0],9))

measured= np.zeros((train_data.shape[0]))

score = 0

for times, (trn_idx, val_idx) in enumerate(folds.split(train_data.values,target['surface'].values)):

    model = RandomForestClassifier(n_estimators=1500,random_state=22, n_jobs = -1,max_features=30,class_weight='balanced_subsample')

#     model = XGBClassifier(n_estimators=75 , random_state=22,learning_rate=1,subsample=0.9,)

    model.fit(train_data.iloc[trn_idx],target['surface'][trn_idx])

    measured[val_idx] = model.predict(train_data.iloc[val_idx])

    predicted += model.predict_proba(test_data)/folds.n_splits

    score += model.score(train_data.iloc[val_idx],target['surface'][val_idx])

    print("Fold: {} score: {}".format(times,model.score(train_data.iloc[val_idx],target['surface'][val_idx])))

    gc.collect()

    
print('Avg Accuracy RF', score / folds.n_splits)
# list(le.inverse_transform(np.unique(target['surface'])))
#     importances = model.feature_importances_

#     indices = np.argsort(importances)

#     features = train_data.columns





#     hm = 30

#     plt.figure(figsize=(7, 10))

#     plt.title('Feature Importances')

#     plt.barh(range(len(indices[:hm])), importances[indices][:hm], color='b', align='center')

#     plt.yticks(range(len(indices[:hm])), [features[i] for i in indices])

#     plt.xlabel('Relative Importance')

#     plt.show()
# plt.plot(model.feature_importances_)

# plt.xticks(np.arange(train_data.shape[1]),train_data.columns.tolist(),rotation=90)

# plt.show()

# X_train, X_test, y_train, y_test = train_test_split(train_data, target['surface'], test_size=0.3, random_state=22,shuffle=True,stratify=target['surface'])

# X_train.shape, y_train.shape, X_test.shape, y_test.shape
# class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train),y_train)    

# class_weights = dict(zip(np.unique(y_train),class_weights))

# class_weights
# class_weights={0:0.06,

# 1:0.16, 

# 2:0.09, 

# 3:0.06, 

# 4:0.10, 

# 5:0.17, 

# 6:0.23, 

# 7:0.3, 

# 8:0.06} 

# model = RandomForestClassifier(n_estimators=1500,random_state=22, n_jobs = -1,max_features=30,class_weight=class_weights)

# %time model.fit(X_train,y_train)
# y_pred = model.predict(X_test)
# print('Accuracy: %.4f' %accuracy_score(y_pred=y_pred,y_true=y_test))

# print('Confusion Matrix: \n%s'%confusion_matrix(y_pred=y_pred,y_true=y_test))

# print('Classification report: \n %s'%classification_report(y_pred=y_pred,y_true=y_test))

# # print('AUC score: %.5f'%roc_auc_score(y_test,y_pred))
# plot_confusion_matrix(confusion_matrix(y_pred=y_pred,y_true=y_test), le.classes_,normalize=False)
# predicted = model.predict_proba(test_data)
# submission_rf = pd.DataFrame({"series_id": test_fe_X.series_id, "surface": le.inverse_transform(predicted.argmax(axis=1))})

# submission_rf.to_csv("submission_rf.csv", index = False)

# submission_rf.head(10)

# submission_rf.surface.value_counts(normalize=True)
# %%time

# folds = StratifiedKFold(n_splits=50, shuffle=True, random_state=59)

# predicted = np.zeros((test_data.shape[0],9))

# measured= np.zeros((train_data.shape[0]))

# score = 0



# for times, (trn_idx, val_idx) in enumerate(folds.split(train_data.values,target['surface'].values)):

#     model = RandomForestClassifier(n_estimators=750,random_state=22,max_features='sqrt', n_jobs = -1)

#     #model = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_split=5, n_jobs=-1)

#     model.fit(train_data.iloc[trn_idx],target['surface'][trn_idx])

#     measured[val_idx] = model.predict(train_data.iloc[val_idx])

#     predicted += model.predict_proba(test_data)/folds.n_splits

#     score += model.score(train_data.iloc[val_idx],target['surface'][val_idx])

#     print("Fold: {} score: {}".format(times,model.score(train_data.iloc[val_idx],target['surface'][val_idx])))

    

#     importances = model.feature_importances_

#     indices = np.argsort(importances)

#     features = train_data.columns



# #     if model.score(train_data.iloc[val_idx],target['surface'][val_idx]) > 0.92000:

# #         hm = 30

# #         plt.figure(figsize=(7, 10))

# #         plt.title('Feature Importances')

# #         plt.barh(range(len(indices[:hm])), importances[indices][:hm], color='b', align='center')

# #         plt.yticks(range(len(indices[:hm])), [features[i] for i in indices])

# #         plt.xlabel('Relative Importance')

# #         plt.show()



# #     gc.collect()



# print('Avg Accuracy RF', score / folds.n_splits)
def plot_confusion_matrix(cm,

                          target_names,

                          title='Confusion matrix',

                          cmap=None,

                          normalize=True):

    """

    given a sklearn confusion matrix (cm), make a nice plot



    Arguments

    ---------

    cm:           confusion matrix from sklearn.metrics.confusion_matrix



    target_names: given classification classes such as [0, 1, 2]

                  the class names, for example: ['high', 'medium', 'low']



    title:        the text to display at the top of the matrix



    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm

                  see http://matplotlib.org/examples/color/colormaps_reference.html

                  plt.get_cmap('jet') or plt.cm.Blues



    normalize:    If False, plot the raw numbers

                  If True, plot the proportions



    Usage

    -----

    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by

                                                              # sklearn.metrics.confusion_matrix

                          normalize    = True,                # show proportions

                          target_names = y_labels_vals,       # list of names of the classes

                          title        = best_estimator_name) # title of graph



    Citiation

    ---------

    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html



    """

    import matplotlib.pyplot as plt

    import numpy as np

    import itertools



    accuracy = np.trace(cm) / float(np.sum(cm))

    misclass = 1 - accuracy



    if cmap is None:

        cmap = plt.get_cmap('Blues')



    plt.figure(figsize=(10, 10))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()



    if target_names is not None:

        tick_marks = np.arange(len(target_names))

        plt.xticks(tick_marks, target_names, rotation=45)

        plt.yticks(tick_marks, target_names)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]





    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        if normalize:

            plt.text(j, i, "{:0.4f}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")

        else:

            plt.text(j, i, "{:,}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")





    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    plt.show()
plot_confusion_matrix(confusion_matrix(measured,target['surface']), le.classes_,normalize=False)
# soft tiles 0.23

# soft pvc 0.17

# concrete 0.16

# hard tiles large space 0.10

# fine concrete 0.09

# carpet 0.06

# hard tiles 0.06

# wood 0.06

# tiled 0.03





# concrete                  915

# wood                      781

# soft_pvc                  634

# soft_tiles                463

# tiled                     322

# hard_tiles_large_space    295

# fine_concrete             274

# carpet                    122

# hard_tiles                 10



# concrete                  912

# wood                      778

# soft_pvc                  644

# soft_tiles                458

# tiled                     318

# hard_tiles_large_space    295

# fine_concrete             278

# carpet                    124

# hard_tiles                  9

submission_xgb = pd.DataFrame({"series_id": test_fe_X.series_id, "surface": le.inverse_transform(predicted.argmax(axis=1))})

submission_xgb.to_csv("submission_xgb.csv", index = False)

submission_xgb.head(10)

submission_xgb.surface.value_counts()
# xgb = XGBClassifier(n_estimators=50 , random_state=22)

# %time xgb.fit(X_train.drop(columns=['surface','velocity_to_acceleration_norm_entropy']),X_train['surface'],eval_metric='map')

# y_pred_xgb = xgb.predict(X_val.drop(columns=['surface','velocity_to_acceleration_norm_entropy']))
# print('Accuracy: %.4f' %accuracy_score(y_pred=y_pred_xgb,y_true=X_val['surface']))

# print('Confusion Matrix: \n%s'%confusion_matrix(y_pred=y_pred_xgb,y_true=X_val['surface']))

# print('Classification report: \n %s'%classification_report(y_pred=y_pred_xgb,y_true=X_val['surface']))