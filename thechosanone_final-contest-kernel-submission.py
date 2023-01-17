import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn import tree
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV


test_feat = pd.read_csv('../input/testFeatures.csv')
train_feat = pd.read_csv('../input/trainFeatures.csv')
train_label = pd.read_csv('../input/trainLabels.csv')

#Filled the NaN values with the mean of the rest of the column
train_feat = train_feat.fillna(train_feat.mean())
test_feat = test_feat.fillna(test_feat.mean())
train_label = train_label.fillna(train_label.mean())

#Used for later assembling the submission csv
Y_id = test_feat['ids'].astype(int)
Y_train = train_label['OverallScore']

#Cleaned up the data
def clean(feats):
    del feats['ids']
    del feats['erkey']
    del feats['RatingID']
    del feats['AccountabilityID']
    del feats['RatingYear']
    del feats['BaseYear']
    del feats['previousratingid']
    del feats['Rpt_Comp_Emp']
    del feats['Rpt_Comp_Date']
    del feats['Rpt_Ap_Date']
    del feats['Rpt_Ver_Date']
    del feats['Reader2_Date']
    del feats['EmployeeID']
    del feats['Incomplete']
    del feats['StatusID']
    del feats['DonorAdvisoryDate']
    del feats['DonorAdvisoryText']
    del feats['IRSControlID']
    del feats['ResultsID']
    del feats['RatingTableID']
    del feats['CauseID']
    del feats['Direct_Support']
    del feats['Indirect_Support']
    del feats['Int_Expense']
    del feats['Depreciation']
    del feats['Assets_45']
    del feats['Assets_46']
    del feats['Assets_47c']
    del feats['Assets_48c']
    del feats['Assets_49']
    del feats['Assets_54']
    del feats['Liability_60']

clean(train_feat)
clean(test_feat)
# grid_params = {'alpha': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]}
# rid = linear_model.Ridge()
# opt_rid = GridSearchCV(rid, grid_params, cv=5, n_jobs=-1, verbose=5)
# opt_rid.fit(train_feat, Y_train)
# opt_rid.best_params_, opt_rid.best_score_

rid = linear_model.Ridge(alpha=0.5)
rid.fit(train_feat, Y_train)
Y_test = rid.predict(test_feat)
Y_test = np.column_stack((Y_id, Y_test))
Y_test_df = pd.DataFrame(data=Y_test[0:,1:],    # values
             index=Y_test[0:,0],    # 1st column as index
             columns= ['OverallScore'], dtype=object)
Y_test_df.index.name = 'Id'
Y_test_df.index = Y_test_df.index.astype(int)
Y_test_df.to_csv("testLabels_ridgeregression.csv", sep=',')
# grid_params = {'alpha': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]}
# las = linear_model.Lasso()
# opt_las = GridSearchCV(las, grid_params, cv=5, n_jobs=-1, verbose=5)
# opt_las.fit(train_feat, Y_train)
# opt_las.best_params_, opt_las.best_score_

las = linear_model.Lasso(alpha=0.01)
las.fit(train_feat, Y_train)
Y_test = las.predict(test_feat)
Y_test = np.column_stack((Y_id, Y_test))
Y_test_df = pd.DataFrame(data=Y_test[0:,1:],    # values
             index=Y_test[0:,0],    # 1st column as index
             columns= ['OverallScore'], dtype=object)
Y_test_df.index.name = 'Id'
Y_test_df.index = Y_test_df.index.astype(int)
Y_test_df.to_csv("testLabels_lassoregression.csv", sep=',')
# grid_params = {'n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}
# pls2 = PLSRegression()
# opt_pls2 = GridSearchCV(pls2, grid_params, cv=5, n_jobs=-1, verbose=5)
# opt_pls2.fit(train_feat, Y_train)
# opt_pls2.best_params_, opt_pls2.best_score_

pls2 = PLSRegression(n_components=9)
pls2.fit(train_feat, Y_train)
Y_test = pls2.predict(test_feat)
Y_test = np.column_stack((Y_id, Y_test))
Y_test_df = pd.DataFrame(data=Y_test[0:,1:],    # values
             index=Y_test[0:,0],    # 1st column as index
             columns= ['OverallScore'], dtype=object)
Y_test_df.index.name = 'Id'
Y_test_df.index = Y_test_df.index.astype(int)
Y_test_df.to_csv("testLabels_PLSregression.csv", sep=',')
# grid_params = {'max_depth': [2, 3, 4, 5], 'random_state': [0, 1], 'n_estimators': [50, 75, 100, 125, 150]}
# regr = RandomForestRegressor()
# opt_regr = GridSearchCV(regr, grid_params, cv=5, n_jobs=-1, verbose=5)
# opt_regr.fit(train_feat, Y_train)
# opt_regr.best_params_, opt_regr.best_score_

regr = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=50)
regr.fit(train_feat, Y_train)
Y_test = regr.predict(test_feat)
Y_test = np.column_stack((Y_id, Y_test))
Y_test_df = pd.DataFrame(data=Y_test[0:,1:],    # values
             index=Y_test[0:,0],    # 1st column as index
             columns= ['OverallScore'], dtype=object)
Y_test_df.index.name = 'Id'
Y_test_df.index = Y_test_df.index.astype(int)
Y_test_df.to_csv("testLabels_RandomForestregression.csv", sep=',')
#Credit to Aarshay Jain for his guide on how to tune XGBoost parameters
model_params = {'learning_rate': 0.05,
                'n_estimators': 2200,
                'max_depth': 5,
                'min_child_weight': 2,
                'gamma': 0.5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.01}
# grid_params = {'learning_rate':[0.01, 0.05, 0.1, 0.5],
#                'n_estimators': [2000, 2100, 2200, 2300, 2500],
#                'max_depth': [3, 4, 5, 6], 
#                'min_child_weight': [2, 4, 6, 8, 10],
#                'gamma': [0, 0.01, 0.05, 0.1, 0.5],
#                'subsample': [0.5, 0.6, 0.7, 0.8, 0.9],
#                'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
#                'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05]}
model = XGBRegressor(**model_params)
# opt_model = GridSearchCV(model, grid_params, cv=5, n_jobs=-1, verbose=5)
# opt_model.fit(train_feat, Y_train)
# opt_model.best_params_, opt_model.best_score_
model.fit(train_feat, Y_train)
Y_test = model.predict(test_feat)
Y_test = np.column_stack((Y_id, Y_test))
Y_test_df = pd.DataFrame(data=Y_test[0:,1:],    # values
             index=Y_test[0:,0],    # 1st column as index
             columns= ['OverallScore'], dtype=object)
Y_test_df.index.name = 'Id'
Y_test_df.index = Y_test_df.index.astype(int)
Y_test_df.to_csv("testLabels_XGBoost.csv", sep=',')