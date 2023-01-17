import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import ElasticNet

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error

from sklearn import preprocessing

import itertools

from pprint import pprint

from sklearn.model_selection import RandomizedSearchCV

inp = pd.read_csv('/home/luv/Documents/4-2/ML/ML_LAB/train.csv')

test_data = pd.read_csv('/home/luv/Documents/4-2/ML/ML_LAB/test.csv')

#inp.drop(columns = ['id'], inplace = True)

inp_sel = []

inp_sel.append(inp[inp['a0'] == 1])

inp_sel.append(inp[inp['a1'] == 1])

inp_sel.append(inp[inp['a2'] == 1])

inp_sel.append(inp[inp['a3'] == 1])

inp_sel.append(inp[inp['a4'] == 1])

inp_sel.append(inp[inp['a5'] == 1])

inp_sel.append(inp[inp['a6'] == 1])

for i in range(0,7):

    inp_sel[i] = inp_sel[i].drop(columns = ['a0','a1','a2','a3','a4','a5','a6'])

    inp_sel[i].set_index('time',inplace = True)



inp_sel_test = []

inp_sel_test.append(test_data[test_data['a0'] == 1])

inp_sel_test.append(test_data[test_data['a1'] == 1])

inp_sel_test.append(test_data[test_data['a2'] == 1])

inp_sel_test.append(test_data[test_data['a3'] == 1])

inp_sel_test.append(test_data[test_data['a4'] == 1])

inp_sel_test.append(test_data[test_data['a5'] == 1])

inp_sel_test.append(test_data[test_data['a6'] == 1])

for i in range(0,7):

    inp_sel_test[i] = inp_sel_test[i].drop(columns = ['a0','a1','a2','a3','a4','a5','a6'])

    inp_sel_test[i].set_index('time',inplace = True)
fig = plt.subplots(figsize = (10,10))

sns.heatmap(inp_sel[1].corr())
plt.figure(figsize = (20,10))

sns.set_style("whitegrid") 

plt.boxplot(inp_sel[3]['label'])
inp_sel_refined = []



plt.figure(figsize = (20,10))

sns.set_style("whitegrid") 

plt.boxplot(inp_sel[0]['label'])
inp_sel_refined.append(inp_sel[0].drop(inp_sel[0][inp_sel[0]['label'] > 40].index))

plt.figure(figsize = (20,10))

sns.set_style("whitegrid") 

plt.boxplot(inp_sel_refined[0]['label'])
inp_sel_refined[0] = inp_sel_refined[0].drop(inp_sel_refined[0][inp_sel_refined[0]['label'] < 7].index)

plt.figure(figsize = (20,10))

sns.set_style("whitegrid") 

plt.boxplot(inp_sel_refined[0]['label'])
plt.figure(figsize = (20,10))

sns.set_style("whitegrid") 

plt.boxplot(inp_sel[1]['label'])
inp_sel_refined.append(inp_sel[1].drop(inp_sel[1][inp_sel[1]['label'] > 55].index))

plt.figure(figsize = (20,10))

sns.set_style("whitegrid") 

plt.boxplot(inp_sel_refined[1]['label'])
inp_sel_refined[1] = inp_sel_refined[1].drop(inp_sel_refined[1][inp_sel_refined[1]['label'] < 3].index)

plt.figure(figsize = (20,10))

sns.set_style("whitegrid") 

plt.boxplot(inp_sel_refined[1]['label'])
plt.figure(figsize = (20,10))

sns.set_style("whitegrid") 

plt.boxplot(inp_sel[2]['label'])
inp_sel_refined.append(inp_sel[2].drop(inp_sel[2][inp_sel[2]['label'] > 52].index))

plt.figure(figsize = (20,10))

sns.set_style("whitegrid") 

plt.boxplot(inp_sel_refined[2]['label'])
inp_sel_refined[2] = inp_sel_refined[2].drop(inp_sel_refined[2][inp_sel_refined[2]['label'] < 3].index)

plt.figure(figsize = (20,10))

sns.set_style("whitegrid") 

plt.boxplot(inp_sel_refined[2]['label'])
plt.figure(figsize = (20,10))

sns.set_style("whitegrid") 

plt.boxplot(inp_sel[3]['label'])
inp_sel_refined.append(inp_sel[3].drop(inp_sel[3][inp_sel[3]['label'] > 60].index))

plt.figure(figsize = (20,10))

sns.set_style("whitegrid") 

plt.boxplot(inp_sel_refined[3]['label'])
inp_sel_refined[3] = inp_sel_refined[3].drop(inp_sel_refined[3][inp_sel_refined[3]['label'] < 3].index)

plt.figure(figsize = (20,10))

sns.set_style("whitegrid") 

plt.boxplot(inp_sel_refined[3]['label'])
plt.figure(figsize = (20,10))

sns.set_style("whitegrid") 

plt.boxplot(inp_sel[4]['label'])
inp_sel_refined.append(inp_sel[4].drop(inp_sel[4][inp_sel[4]['label'] > 45].index))

plt.figure(figsize = (20,10))

sns.set_style("whitegrid") 

plt.boxplot(inp_sel_refined[4]['label'])
inp_sel_refined[4] = inp_sel_refined[4].drop(inp_sel_refined[4][inp_sel_refined[4]['label'] < 3].index)

plt.figure(figsize = (20,10))

sns.set_style("whitegrid") 

plt.boxplot(inp_sel_refined[4]['label'])
plt.figure(figsize = (20,10))

sns.set_style("whitegrid") 

plt.boxplot(inp_sel[5]['label'])
inp_sel_refined.append(inp_sel[5].drop(inp_sel[5][inp_sel[5]['label'] > 30].index))

plt.figure(figsize = (20,10))

sns.set_style("whitegrid") 

plt.boxplot(inp_sel_refined[5]['label'])
plt.figure(figsize = (20,10))

sns.set_style("whitegrid") 

plt.boxplot(inp_sel[6]['label'])
inp_sel_refined.append(inp_sel[6].drop(inp_sel[6][inp_sel[6]['label'] > 55].index))

plt.figure(figsize = (20,10))

sns.set_style("whitegrid") 

plt.boxplot(inp_sel_refined[6]['label'])
inp_sel_refined[6] = inp_sel_refined[6].drop(inp_sel_refined[6][inp_sel_refined[6]['label'] < 3].index)

plt.figure(figsize = (20,10))

sns.set_style("whitegrid") 

plt.boxplot(inp_sel_refined[6]['label'])
plt.figure(figsize = (20,10))

sns.set_style("whitegrid") 

plt.plot(inp_sel_refined[6]['label'])
def all_equal(iterable):

    "Returns True if all elements are equal to each other"

    g = itertools.groupby(iterable)

    return next(g, True) and not next(g, False)
for i in range(0,7):

    constant_columns = inp_sel_test[i].columns[inp_sel_test[i].apply(all_equal)]

    inp_sel_refined[i].drop(columns = constant_columns,inplace = True)

    inp_sel_test[i].drop(columns = constant_columns,inplace = True)    
X_refined = []

Y_refined = []

for i in range (0,7):

    Y_refined.append(inp_sel_refined[i]['label'])

    X_refined.append(inp_sel_refined[i].drop(columns = ['label']))
#Agent 5

X_train_rf,X_test_rf,Y_train_rf,Y_test_rf = train_test_split(X_refined[5],Y_refined[5],test_size = 0.4, random_state = 0)
rf_r = RandomForestRegressor()

rf_r.fit(X_train_rf.drop(columns = ['id']),Y_train_rf)

accuracy = evaluate(rf_r,X_test_rf.drop(columns = ['id']),Y_test_rf)
output_refined = []

for i in range (0,7):

    rf_r.fit(X_refined[i].drop(columns = ['id']),Y_refined[i])    

    pred_rf = rf_r.predict(inp_sel_test[i].drop(columns = ['id']))  

    output_refined.append(pd.DataFrame({'id' : inp_sel_test[i]['id'],'label' : pred_rf}))   
output_refined
frames_refined = []

for i in range(0,7):

    frames_refined.append(output_refined[i])
result_refined = pd.concat(frames_refined)

result_refined = result_refined.sort_values(by=['id'])
result_refined
filename_refined = 'Sumbission1_Latest121.csv'



result_refined.to_csv(filename_refined,index=False)
sub_refined = pd.read_csv('/home/luv/Documents/4-2/ML/ML_LAB/Sumbission1_Latest121.csv')
sub_refined.info
def all_equal(iterable):

    "Returns True if all elements are equal to each other"

    g = itertools.groupby(iterable)

    return next(g, True) and not next(g, False)
for i in range(0,7):

    constant_columns = inp_sel_test[i].columns[inp_sel_test[i].apply(all_equal)]

    inp_sel[i].drop(columns = constant_columns,inplace = True)

    inp_sel_test[i].drop(columns = constant_columns,inplace = True)    
for i in range (0,7):

    print(inp_sel[i].shape)

    print(inp_sel_test[i].shape)
X = []

Y = []

for i in range (0,7):

    Y.append(inp_sel[i]['label'])

    X.append(inp_sel[i].drop(columns = ['label']))
for i in range (0,7):

    print(X[i].shape)

    print(inp_sel_test[i].shape)
def evaluate(model, X_test, Y_test):

    predictions = model.predict(X_test)

    errors = np.sqrt(mean_squared_error(Y_test, predictions))

    print('Model Performance')

    print('RMSE of: ', errors)

    return errors
#Agent 5

X_train,X_test,Y_train,Y_test = train_test_split(X[5],Y[5],test_size = 0.4, random_state = 0)
rf_r = RandomForestRegressor()

rf_r.fit(X_train.drop(columns = ['id']),Y_train)

accuracy = evaluate(rf_r,X_test.drop(columns = ['id']),Y_test)
best_random_a5 = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=80,

           max_features='auto', max_leaf_nodes=None,

           min_impurity_decrease=0.0, min_impurity_split=None,

           min_samples_leaf=1, min_samples_split=5,

           min_weight_fraction_leaf=0.0, n_estimators=600, n_jobs=1,

           oob_score=False, random_state=None, verbose=0, warm_start=False)

best_random_a5.fit(X_train.drop(columns = ['id']) , Y_train)



random_accuracy = evaluate(best_random_a5, X_test.drop(columns = ['id']), Y_test)
(random_accuracy - accuracy)*100/accuracy
feature_weights = best_random_a5.feature_importances_
important_features = []

for idx, val in enumerate(feature_weights):

    if val > 0.001:

        important_features.append("b"+str(idx))
important_features
best_random_a5.fit(X[5].drop(columns = ['id']),Y[5])
output = []

for i in range (0,7):

    best_random_a5.fit(X[i].drop(columns = ['id']),Y[i])    

    pred = best_random_a5.predict(inp_sel_test[i].drop(columns = ['id']))  

    output.append(pd.DataFrame({'id' : inp_sel_test[i]['id'],'label' : pred}))   
frames = []

for i in range(0,7):

    frames.append(output[i])

        
result = pd.concat(frames)

result = result.sort_values(by=['id'])
filename = 'Sumbission1.csv'



result.to_csv(filename,index=False)
sub = pd.read_csv('/home/luv/Documents/4-2/ML/ML_LAB/Sumbission1.csv')
sample_sub = pd.read_csv('/home/luv/Documents/4-2/ML/ML_LAB/sampleSubmission.csv')
sample_sub
for i in range (0,7):

    print(X[i].shape)

    print(inp_sel_test[i].shape)