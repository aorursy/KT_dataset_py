import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from gplearn.genetic import SymbolicRegressor,SymbolicTransformer

from gplearn.functions import make_function



from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import StandardScaler



from sklearn.ensemble import RandomForestRegressor

from sklearn.feature_selection import RFE

from sklearn.feature_selection import RFECV



import os



print(os.listdir("../input"))

print(os.listdir("../input/LANL-Earthquake-Prediction"))

print(os.listdir("../input/lanl-features"))
X = pd.read_csv('../input/lanl-features/train_features_denoised.csv')

X_test = pd.read_csv('../input/lanl-features/test_features_denoised.csv')

y = pd.read_csv('../input/lanl-features/y.csv')

submission = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv',index_col='seg_id')
X.drop('seg_id',axis=1,inplace=True)

X_test.drop('seg_id',axis=1,inplace=True)

X.drop('target',axis=1,inplace=True)

X_test.drop('target',axis=1,inplace=True)



alldata = pd.concat([X, X_test])



scaler = StandardScaler()



alldata = pd.DataFrame(scaler.fit_transform(alldata), columns=alldata.columns)



X = alldata[:X.shape[0]]

X_test = alldata[X.shape[0]:]
%%time

corr_matrix = X.corr()

corr_matrix = corr_matrix.abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]



X = X.drop(to_drop, axis=1)

X_test = X_test.drop(to_drop, axis=1)

print(X.shape)

print(X_test.shape)
%%time

rf = RandomForestRegressor(n_estimators=10)

rfecv = RFECV(estimator=rf, step=1, cv=5, scoring='neg_mean_absolute_error', verbose=0, n_jobs=4) #4-fold cross-validation with mae

rfecv = rfecv.fit(X, y.values)

print('Optimal number of features :', rfecv.n_features_)

print('Best features :', X.columns[rfecv.support_])



X = X[X.columns[rfecv.support_].values]

X_test = X_test[X_test.columns[rfecv.support_].values]

print(X.shape)

print(X_test.shape)
X["mean_y"] = np.full(len(y), y.values.mean())

X["max_y"] = np.full(len(y), y.values.max())

X["min_y"] = np.full(len(y), y.values.min())

X["std_y"] = np.full(len(y), y.values.std())



X_test["mean_y"] = np.full(len(X_test), y.values.mean())

X_test["max_y"] = np.full(len(X_test), y.values.max())

X_test["min_y"] = np.full(len(X_test), y.values.min())

X_test["std_y"] = np.full(len(X_test), y.values.std())



print(X.shape)

print(X_test.shape)
list(X.columns)
def tanh(x):

    return np.tanh(x)

def sinh(x):

    return np.sinh(x)

def cosh(x):

    return np.cosh(x)

def arctan(x):

    return np.arctan(x)

def arcsin(x):

    return np.arcsin(x)

def arccos(x):

    return np.arccos(x)

def arctanh(x):

    return np.arctan(x)

def arcsinh(x):

    return np.arcsin(x)

def arccosh(x):

    return np.arccos(x)

def exp(x):

    return np.exp(x)

def exp2(x):

    return np.exp2(x)

def expm1(x):

    return np.expm1(x)

def log2(x):

    return np.log2(x)

def log1p(x):

    return np.log1p(x)

 



gp_tanh = make_function(tanh,"tanh",1)

gp_sinh = make_function(sinh,"sinh",1)

gp_cosh = make_function(cosh,"cosh",1)



gp_arctan = make_function(arctan,"arctan",1)

gp_arcsin = make_function(arcsin,"arcsin",1)

gp_arccos = make_function(arccos,"arccos",1)



gp_arctanh = make_function(arctanh,"arctanh",1)

gp_arcsinh = make_function(arcsinh,"arcsinh",1)

gp_arccosh = make_function(arccosh,"arccosh",1)



gp_exp = make_function(exp,"exp",1)

gp_exp2 = make_function(exp2,"exp2",1)

gp_expm1 = make_function(expm1,"expm1",1)

#gp_log2 = make_function(log2,"log2",1)

#gp_log1p = make_function(log1p,"log1p",1)
%%time

est_gp = SymbolicRegressor(population_size=X.shape[1]*17*10,

                           tournament_size=X.shape[1]*17//1,

                           generations=50, stopping_criteria=1.79,

                           p_crossover=0.9, p_subtree_mutation=0.0001, p_hoist_mutation=0.0001, p_point_mutation=0.0001,

                           max_samples=0.8, verbose=1,

                           function_set = ('add', 'sub', 'mul', 'div', 

                                           'sqrt', 'log', 'abs', 'neg', 'inv','max', 'min', 

                                           'tan', 'cos', 'sin', 

                                           gp_tanh, #gp_sinh, gp_cosh,

                                           gp_arctan, #gp_arcsin, gp_arccos,

                                           gp_arctanh, #gp_arcsinh, gp_arccosh,

                                           #gp_exp,

                                           #gp_exp2,

                                           #gp_expm1,

                                           #gp_log1p,                                           

                                          ),

                           #function_set = (gp_tanh, 'add', 'sub', 'mul', 'div'),

                           metric = 'mean absolute error', warm_start=True,

                           n_jobs = 4, parsimony_coefficient=0.00001, random_state=11)

'''



est_gp = SymbolicTransformer(population_size=1000, 

                             hall_of_fame=100, 

                             n_components=10, 

                             generations=20, 

                             tournament_size=20, 

                             stopping_criteria=1.0, 

                             const_range=(-1.0, 1.0), 

                             init_depth=(2, 6), 

                             init_method='half and half', 

                             function_set=('add', 'sub', 'mul', 'div'), 

                             metric='pearson', 

                             parsimony_coefficient=0.001, p_crossover=0.9, p_subtree_mutation=0.01, p_hoist_mutation=0.01, p_point_mutation=0.01, p_point_replace=0.05, max_samples=1.0, 

                             feature_names=None, warm_start=False, low_memory=False, n_jobs=4, verbose=1, random_state=11)

 '''



est_gp.fit(X, y)
#print("gpLearn Program:", est_gp._program)

genetic_formula = str(est_gp._program)

for i in range(len(X.columns)):

    genetic_formula = genetic_formula.replace(f'X{i}', X.columns[i])

    

print("Genetic Formula: ", genetic_formula)
y_gp = est_gp.predict(X)

gpLearn_MAE = mean_absolute_error(y, y_gp)

print("gpLearn MAE:", gpLearn_MAE)
df_result = pd.DataFrame()

df_result["predict"] = y_gp

df_result["real"] = y
df_result[:1500].plot()
df_result[-1500:].plot()
df_result.plot()
submission.time_to_failure = est_gp.predict(X_test)

submission.to_csv('submission.csv', index=True)
submission.head(10)