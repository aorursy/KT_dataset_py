import pandas as pd

import numpy as np



from sklearn.model_selection import KFold, StratifiedKFold



from sklearn.preprocessing import LabelEncoder



from sklearn.metrics import log_loss, roc_auc_score



from datetime import date



from sklearn.pipeline import Pipeline



from xgboost import XGBClassifier, plot_importance



from category_encoders import CountEncoder



from matplotlib import pyplot
# settings



nfolds = 5

data_folder = '../input/lish-moa/'
# load the data

xtrain = pd.read_csv(data_folder + 'train_features.csv')

xtest = pd.read_csv(data_folder + 'test_features.csv')
# prepare split

kf = StratifiedKFold(n_splits = nfolds)



# separation

id_train = xtrain['sig_id']; id_test = xtest['sig_id']

xtrain.drop('sig_id', axis = 1, inplace = True)

xtest.drop('sig_id', axis = 1, inplace = True)



# add the differentiating column

xtrain['is_test'] = 0

xtest['is_test'] = 1



# combine the two datasets

xdat = pd.concat([xtrain, xtest], axis = 0)

del xtrain, xtest
# little bit of FE



enc = LabelEncoder()

enc_cnt = CountEncoder()

category_cols = ['cp_dose', 'cp_type']

print(category_cols)



for cols in category_cols:

    xdat[cols] = enc.fit_transform(xdat[cols])

# model



classifier = XGBClassifier(tree_method='gpu_hist')



params = {'colsample_bytree': 0.6522,

          'gamma': 3.6975,

          'learning_rate': 0.0503,

          'max_delta_step': 2.0706,

          'max_depth': 10,

          'min_child_weight': 31.5800,

          'n_estimators': 200,

          'subsample': 0.8639,

          'alpha': 0.05

         }



classifier.set_params(**params)
# separate the target

ydat = xdat['is_test']; xdat.drop('is_test', axis =1, inplace = True)



# storage structure for the predicted probabilities

prmat = np.zeros((xdat.shape[0],1))
for (ff, (id0, id1)) in enumerate(kf.split(xdat,ydat)):

     

    x0, x1 = xdat.iloc[id0], xdat.iloc[id1]

    y0, y1 = ydat.iloc[id0], ydat.iloc[id1]

    

    print(sum(y0))

    

    # fit model

    classifier.fit(x0, y0)

    

    # generate predictions

    vpreds = classifier.predict_proba(x1)

    prmat[id1,0] = vpreds[:,1]



    print(ff)

    

    print('--')

    
# Evaluate the performance 

print(roc_auc_score(ydat, prmat))
# plot the features relevant for distinction



plot_importance(classifier, max_num_features = 25)

pyplot.show()