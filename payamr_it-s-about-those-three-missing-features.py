import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
df = pd.read_csv('../input/train.csv')
df.replace('yes',1,inplace=True)
df.replace('no', 0, inplace=True)
df.sort_values('Target', inplace=True)
cols = df.columns.values
y = df.iloc[:,-1].values
# Function that extracts the X matrix from dataframe 'data'. 
# It first attempts to fill the missing data using household averages.
# If not available, it will use population averages. 
# This is run after the prediction models for the three columns with many missing values.
def get_X_remove_nan(data, train = 1):
    cols = data.columns.values
    hog_ind=np.where(cols=='idhogar')[0][0]
    #X_raw keeps the idhorag 
    if train == 1:
        X_raw = data.iloc[:, :-1].values # exclude target for training data
    else:
        X_raw = data.values # no target for test data     
    # XX throws away the id (at index = 0) and idhorag
    XX = np.delete(X_raw, hog_ind, axis=1)[:,1:]
    XX = XX.astype(float)
    _ , n = XX.shape
    households = data['idhogar'].unique()
    # first attempt to replace nan's with values from same households, when at least one available
    for hh in households:
        aux = XX[X_raw[:,hog_ind] == hh]
        k, _ = aux.shape
        for i in range(n):
            if all(map(lambda x: x!=x, aux[:,i])):
                pass
            else:
                not_nan_ind = [m for m in range(k)]
                nan_ind = []
                for j in range(k):
                    if aux[j,i] != aux[j,i]:
                        nan_ind.append(j)
                        not_nan_ind.remove(j)
                aux_mean = np.average(aux[not_nan_ind, i])
                aux[nan_ind, i] = aux_mean
        XX[X_raw[:,hog_ind] == hh] = aux
    
    # one last pass to replace remaining nan's with means values with those in the same socio-economic class
    imputer = SimpleImputer(strategy = 'mean')
    imputer = imputer.fit(XX)
    XX = imputer.transform(XX)
    return XX
def collect_existing_values(colsX, s, df, train = 1): 
    # separates nan's and existing values in column s and retunrs two dataframes
    # also returns the X and y for rows where column s has data
    # and X for rows where column s is missing
    # colX is list of columns with two of the three problematic columns removed (s is kept)
    dfX = df[colsX]
    df_exist = dfX[dfX[s] == dfX[s]] # rows where value of the column s is not nan
    df_nans = dfX[dfX[s] != dfX[s]] # rows where the value of column s is nan
    y = df_exist[s].values
    colsX.remove(s)
    X = get_X_remove_nan(df_exist[colsX], train)
    X_nans = get_X_remove_nan(df_nans[colsX], train) # X for rows where value of column s is nan
    return X, y, df_exist, df_nans, X_nans
process_X = Pipeline([('sc', StandardScaler())]) 
reg1 = Pipeline([('forest', RandomForestRegressor())])
reg2 = Pipeline([('svr', SVR(kernel = 'rbf', gamma='scale'))])
reg3 = Pipeline([('linear', LinearRegression())])
regs = [reg1, reg2, reg3]
params = [{'forest__n_estimators': [20, 40, 80]}, {}, {}]
sweep_to_reg = {0:0, 1:0, 2:0, 3:1, 4:2}
model_name = {0: 'Random Forest', 1: 'SVR', 2: 'Linear Regression'}
from sklearn.model_selection import GridSearchCV
grids = []
for i in range(3):
    grids.append(GridSearchCV(regs[i], param_grid=params[i]))
cols_v2a1 = [i for i in cols if i not in ['v18q1','rez_esc']] # has 'v2a1' but not the other two
Xv2, yv2, dfv2_exist, dfv2_nans, Xv2_nans = collect_existing_values(cols_v2a1, 'v2a1', df)
Xv2_train, Xv2_val, yv2_train, yv2_val = train_test_split(Xv2, yv2, test_size = 0.2)
Xv2_train = process_X.fit_transform(Xv2_train)
Xv2_val = process_X.transform(Xv2_val)
Xv2_nans = process_X.transform(Xv2_nans)
scores = np.array([])
for i in range(3):
    grids[i].fit(Xv2_train, yv2_train)
    scores = np.concatenate((scores, grids[i].cv_results_['mean_test_score']))
model_selected = sweep_to_reg[np.argmax(scores)]
reg_v2a1 = grids[model_selected].best_estimator_
model_name[model_selected]
yv2_pred = reg_v2a1.predict(Xv2_val)
RMSE = np.sqrt(mean_squared_error(yv2_val, yv2_pred))
RMSE = RMSE / np.average(yv2_val)
print("Normalized RMSE is", RMSE)
yv2_nans = reg_v2a1.predict(Xv2_nans)
cols_v18 = [i for i in cols if i not in ['v2a1','rez_esc']] # has 'v18q1' but not the other two
Xv18, yv18, dfv18_exist, dfv18_nans, Xv18_nans = collect_existing_values(cols_v18, 'v18q1', df)
Xv18_train, Xv18_val, yv18_train, yv18_val = train_test_split(Xv18, yv18, test_size = 0.2)
Xv18_train = process_X.fit_transform(Xv18_train)
Xv18_val = process_X.transform(Xv18_val)
Xv18_nans = process_X.transform(Xv18_nans)
scores = np.array([])
for i in range(3):
    grids[i].fit(Xv18_train, yv18_train)
    scores = np.concatenate((scores, grids[i].cv_results_['mean_test_score']))
model_selected = sweep_to_reg[np.argmax(scores)]
reg_v18q1 = grids[model_selected].best_estimator_
model_name[model_selected]
yv18_pred = reg_v18q1.predict(Xv18_val)
RMSE = np.sqrt(mean_squared_error(yv18_val, yv18_pred))
RMSE = RMSE / np.average(yv18_val)
print("Normalized RMSE is", RMSE)
yv18_nans = reg_v18q1.predict(Xv18_nans)
cols_rez = [i for i in cols if i not in ['v2a1','v18q1']] # has 'rez_esc' but not the other two
Xrez, yrez, dfrez_exist, dfrez_nans, Xrez_nans = collect_existing_values(cols_rez, 'rez_esc', df)
Xrez_train, Xrez_val, yrez_train, yrez_val = train_test_split(Xrez, yrez, test_size = 0.2)
Xrez_train = process_X.fit_transform(Xrez_train)
Xrez_val = process_X.transform(Xrez_val)
Xrez_nans = process_X.transform(Xrez_nans)
scores = np.array([])
for i in range(3):
    grids[i].fit(Xrez_train, yrez_train)
    scores = np.concatenate((scores, grids[i].cv_results_['mean_test_score']))
model_selected = sweep_to_reg[np.argmax(scores)]
reg_rez_esc = grids[model_selected].best_estimator_
model_name[model_selected]
yrez_pred = reg_rez_esc.predict(Xrez_val)
RMSE = np.sqrt(mean_squared_error(yrez_val, yrez_pred))
RMSE = RMSE / np.average(yrez_val)
print("Normalized RMSE is", RMSE)
yrez_nans = reg_rez_esc.predict(Xrez_nans)
df2 = df.copy()
df2.loc[dfv2_nans.index, 'v2a1'] = yv2_nans
df2.loc[dfv18_nans.index, 'v18q1'] = yv18_nans
df2.loc[dfrez_nans.index, 'rez_esc'] = yrez_nans
y = df2.iloc[:,-1].values
X = get_X_remove_nan(df2)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)
X_train = process_X.fit_transform(X_train)
X_val = process_X.transform(X_val)
from sklearn.ensemble import RandomForestClassifier
classifiers = Pipeline([('forest', RandomForestClassifier(criterion = 'entropy'))])
params = [{'forest': [RandomForestClassifier(criterion= 'entropy')], 
          'forest__n_estimators' : [10, 20, 40, 80, 160]}]
grid = GridSearchCV(classifiers, cv = 5, param_grid= params)
grid.fit(X_train, y_train)
grid.cv_results_['mean_test_score']
selected_classifier = grid.best_estimator_
from sklearn.metrics import confusion_matrix
y_pred = selected_classifier.predict(X_val)
cm = confusion_matrix(y_val, y_pred)
cm
print("Classification rate is", np.sum(y_val==y_pred) / float(len(y_val)))
err_1 = (np.sum(cm[0])-cm[0,0])/float(np.sum(cm[0]))
print("Misclassification rate for group 1 is", err_1)
df_test = pd.read_csv('../input/test.csv')
df_test.replace('yes',1,inplace=True)
df_test.replace('no', 0, inplace=True)
cols_v2a1 = [i for i in cols if i not in ['v18q1','rez_esc']] # has 'v2a1' but not the other two
cols_v2a1.remove('Target')
_, _, _, dfv2_nans_test, Xv2_nans_test = collect_existing_values(cols_v2a1, 'v2a1', df_test, train=0)
yv2_nans_test = reg_v2a1.predict(Xv2_nans_test)
cols_v18 = [i for i in cols if i not in ['v2a1','rez_esc']] # has 'v18q1' but not the other two
cols_v18.remove('Target')
_, _, _, dfv18_nans_test, Xv18_nans_test = collect_existing_values(cols_v18, 'v18q1', df_test, train=0)
yv18_nans_test = reg_v18q1.predict(Xv18_nans_test)
cols_rez = [i for i in cols if i not in ['v2a1','v18q1']] # has 'rez_esc' but not the other two
cols_rez.remove('Target')
_, _, _, dfrez_nans_test, Xrez_nans_test = collect_existing_values(cols_rez, 'rez_esc', df_test, train=0)
yrez_nans_test = reg_rez_esc.predict(Xrez_nans_test)
df_test.loc[dfv2_nans_test.index, 'v2a1'] = yv2_nans_test
df_test.loc[dfv18_nans_test.index, 'v18q1'] = yv18_nans_test
df_test.loc[dfrez_nans_test.index, 'rez_esc'] = yrez_nans_test
X_test = get_X_remove_nan(df_test, train = 0)
X_test = process_X.transform(X_test)
y_test = selected_classifier.predict(X_test)
# putting results in a dataframe
df_result = pd.concat([df_test[['Id','idhogar']], pd.DataFrame({'Target': y_test})], axis = 1)
# household average
df_hh_ave = df_result.groupby('idhogar').mean() 
df_hh_ave = df_hh_ave.reset_index()
df_hh_ave['Target'] = np.round(df_hh_ave['Target'])
df_hh_ave = df_hh_ave.astype({'Target': int})
df_result = df_result.rename(columns={'Target': 'Target_individual'})
df_hh_ave = pd.merge(df_result, df_hh_ave, on='idhogar')
df_hh_ave.index = df_hh_ave['Id']
df_final = df_hh_ave[['Target']]
df_final.to_csv('submission_payamr.csv')
# This fills the missing values based on the values of 'Target'
targets = sorted(df['Target'].unique())
X = np.array([])
for t in targets:
    data = df[df['Target'] == t]
    XX = get_X_remove_nan(data)
    try:
        X = np.concatenate((X, XX), axis = 0)
    except:
        X = XX
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)
X_train = process_X.fit_transform(X_train)
X_val = process_X.transform(X_val)
# Random Forest with 40 trees seemed to work just fine. 
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 40, criterion = 'entropy')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_val)
cm = confusion_matrix(y_val, y_pred)
cm
print("Classification rate is", np.sum(y_val==y_pred) / float(len(y_val)))
err_1 = (np.sum(cm[0])-cm[0,0])/float(np.sum(cm[0]))
print("Misclassification rate for group 1 is", err_1)