import numpy as np, pandas as pd

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import roc_auc_score

from sklearn import svm, neighbors, linear_model

from sklearn.svm import NuSVC

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from tqdm import tqdm

from catboost import CatBoostClassifier

from catboost import Pool

import lightgbm as lgb

from sklearn.feature_selection import VarianceThreshold

import warnings

warnings.filterwarnings("ignore")



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



oof = np.zeros(len(train))

preds = np.zeros(len(test))

oof_2 = np.zeros(len(train))

preds_2 = np.zeros(len(test))

oof_3 = np.zeros(len(train))

preds_3 = np.zeros(len(test))

oof_4 = np.zeros(len(train))

preds_4 = np.zeros(len(test))

oof_5 = np.zeros(len(train))

preds_5 = np.zeros(len(test))

cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]



for i in tqdm(range(512)):

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)



    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

    data2 = StandardScaler().fit_transform(PCA(n_components=40, random_state=4).fit_transform(data[cols]))

    train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]

    

#     sel = VarianceThreshold(threshold=1.5).fit(data[cols])

#     data3 = sel.transform(data[cols])

#     train4 = data3[:train2.shape[0]]; test4 = data3[train2.shape[0]:]

    

    # GridSearch for LGB

    skf = StratifiedKFold(n_splits=5, random_state=42)

    param_grid = {

                'n_jobs':[-1],

                'n_estimators': [500],

                'learning_rate': [0.05,0.1],

                'max_depth': [6,7],

                'n_jobs':[-1],

                'subsample' :[0.5],

                'num_leaves': [16,31],

                'reg_alpha' : [0],

                'reg_lambda' : [0],

            }



    #gridsearch approx 27 sec

    model = GridSearchCV(lgb.LGBMClassifier(),param_grid, cv=skf.split(train3,train2['target']),verbose=0, scoring= 'roc_auc',iid=True,n_jobs=-1)

    model.fit(train3,train2['target'])

    #for the metric

    model.best_params_['metric']= 'auc'

    del model.best_params_['n_estimators']

    

    for train_index, test_index in skf.split(train2, train2['target']):



        #SVC

        clf = NuSVC(probability=True, kernel='poly', degree=4, gamma='auto', random_state=4, nu=0.59, coef0=0.053)

        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

        oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]

        preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

        

        #KNN

        clf = neighbors.KNeighborsClassifier(n_neighbors=17, p=2.9)

        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

        oof_2[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]

        preds_2[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

        

        

        #LGB

        train_dataset = lgb.Dataset(train3[train_index,:],train2.loc[train_index]['target'])

        val_dataset = lgb.Dataset(train3[test_index,:],train2.loc[test_index]['target'])

        

        clf = lgb.train(model.best_params_, train_dataset, valid_sets=[train_dataset, val_dataset],

                        verbose_eval=False,num_boost_round=5000,early_stopping_rounds=250)

        

        oof_3[idx1[test_index]] = clf.predict(train3[test_index,:], num_iteration=clf.best_iteration)

        preds_3[idx2] += clf.predict(test3, num_iteration=clf.best_iteration) / skf.n_splits

        

        #CatBoost (seems better than LGB here because there aren't a lot of data but it's too long to train)

#         train_dataset = Pool(train3[train_index,:],train2.loc[train_index]['target'])

#         eval_dataset = Pool(train3[test_index,:],train2.loc[test_index]['target'])

        

#         clf = CatBoostClassifier(iterations=800,random_state=1,task_type = "GPU",eval_metric='AUC')

#         clf.fit(train_dataset,use_best_model=True,eval_set=eval_dataset)

        

#         oof_4[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]

#         preds_4[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits



        #Log

        clf = linear_model.LogisticRegression(solver='lbfgs',penalty='l2',C=0.1)

        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

        oof_4[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]

        preds_4[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

        

        #MLP

        clf = MLPClassifier(random_state=3, activation='relu', solver='adam', tol=1e-06, hidden_layer_sizes=(250, ))

        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

        oof_5[idx1[test_index]] = clf.predict(train3[test_index,:])

        preds_5[idx2] += clf.predict(test3) / skf.n_splits
results = np.array([[1,0,0,0,0,'Nu_SVM'],

          [0,1,0,0,0,'KNN'],

          [0,0,1,0,0,'LGB'],

          [0,0,0,1,0,'LogR'],

          [0,0,0,0,1,'MLP'],

          [0.8,0.2,0,0,0,'0.8 SVM & 0.2 KNN'],

          [0.9,0.1,0,0,0,'0.9 SVM & 0.1 KNN'],

          [0.95,0.05,0,0,0,'0.95 SVM & 0.05 KNN'],

          [1.05,-0.05,0,0,0,'1.05 SVM & -0.05 KNN'],

                    

          [0.75,0.15,0.1,0,0,'0.75 SVM & 0.15 KNN & 0.1 LGB'],

          [0.8,0.1,0.1,0,0,'0.8 SVM & 0.1 KNN & 0.1 LGB'],

          [0.8,0.15,0.05,0,0,'0.8 SVM & 0.15 KNN & 0.05 LGB'],

          [0.9,0.05,0.05,0,0,'0.9 SVM & 0.05 KNN & 0.05 LGB'],

                    

          [0.75,0.15,0,0.1,0,'0.75 SVM & 0.15 KNN & 0.1 LogR'],

          [0.8,0.1,0,0.1,0,'0.8 SVM & 0.1 KNN & 0.1 LogR'],

          [0.8,0.15,0,0.05,0,'0.8 SVM & 0.15 KNN & 0.05 LogR'],

          [0.9,0.05,0,0.05,0,'0.9 SVM & 0.05 KNN & 0.05 LogR'],

                    

          [0.75,0.15,0,0,0.1,'0.75 SVM & 0.15 KNN & 0.1 MLP'],

          [0.8,0.1,0,0,0.1,'0.8 SVM & 0.1 KNN & 0.1 MLP'],

          [0.8,0.15,0,0,0.05,'0.8 SVM & 0.15 KNN & 0.05 MLP'],

          [0.9,0.05,0,0,0.05,'0.9 SVM & 0.05 KNN & 0.05 MLP'],

                    

          [0.8,0.1,0.05,0.05,0,'0.8 SVM & 0.1 KNN & 0.05 LGB & 0.05 LogR'],

          [0.8,0.15,0.025,0.025,0,'0.8 SVM & 0.15 KNN & 0.025 LGB & 0.025 LogR'],



          [0.8,0.1,0.05,0,0.05,'0.8 SVM & 0.1 KNN & 0.05 LGB & 0.05 MLP'],

          [0.8,0.15,0.025,0,0.025,'0.8 SVM & 0.15 KNN & 0.025 LGB & 0.025 MLP'],

        

          [0.6,0.2,0.1,0,0.1,'0.6 SVM & 0.2 KNN & 0.1 LGB & 0.1 MLP'],

          [0.7,0.1,0.1,0,0.1,'0.7 SVM & 0.1 KNN & 0.1 LGB & 0.1 MLP'],

          [0.7,0.2,0.05,0,0.05,'0.7 SVM & 0.1 KNN & 0.1 LGB & 0.1 MLP']])

            

print("So which is the best ?")



bestScore = 0

for i in range(results.shape[0]):

    score = roc_auc_score(train['target'], float(results[i][0])*oof+float(results[i][1])*oof_2+float(results[i][2])*oof_3

                          +float(results[i][3])*oof_4+float(results[i][4])*oof_5)

    if(score > bestScore):

        j = i

        bestScore = score



    print(results[i][-1],score)

    

print('----------------------------------------')

print('Best score',results[j][-1],bestScore)



sub = pd.read_csv('../input/sample_submission.csv')



sub['target'] = float(results[j][0])*preds+float(results[j][1])*preds_2+float(results[j][2])*preds_3+float(results[j][3])*preds_4+float(results[j][4])*preds_5

sub.to_csv('Best_submission.csv', index=False)
