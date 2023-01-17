import csv

import time

import math

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from math import sqrt

from sklearn import tree

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV, StratifiedKFold

from sklearn.linear_model import  LogisticRegression, lars_path

from sklearn.preprocessing import StandardScaler, MinMaxScaler,power_transform

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SelectPercentile 

import warnings

warnings.filterwarnings("ignore",category=FutureWarning)

warnings.filterwarnings("ignore",category=DeprecationWarning)

from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning) 

from bokeh.io import output_notebook, show

from bokeh.plotting import figure

from bokeh.plotting import gmap

from bokeh.models import WMTSTileSource, LinearColorMapper, LogColorMapper,ColumnDataSource, HoverTool, CustomJS, Slider, ColorBar, FixedTicker

from bokeh.transform import linear_cmap, factor_cmap

from bokeh.layouts import row, column

from matplotlib import colors as mcolors
data = pd.read_csv("../input/sonar.all-data.csv",header = None,prefix='V')

data.head()
print ("Data size is: {}".format(data.shape))

print ("Variable types: \n{}".format(data.dtypes))
data.describe()  # does not 'describe' the last qualitative column
data[data.isnull().any(axis=1)]
data['V60'].value_counts()
def conv(x):

    if x == 'M':

        return 1

    if x == 'R':

        return 0

    

data['V60'] = data['V60'].apply(lambda x : conv(x))
num_cols = data.shape[1]

data_corr = data.corr(method = 'pearson') 

fig,axes = plt.subplots(figsize=(15,5))

for i in range (0,20):

    plt.plot(data_corr.iloc[i:num_cols,i], label=str(i))

plt.xticks(rotation='vertical')

plt.legend(bbox_to_anchor=(1.1, 1.05))

plt.axhline(y=0,color='k')

plt.ylabel("Pearson correlation coefficient")

plt.show()
fig = plt.figure(figsize=(12,10))

plt.pcolor(data_corr)

plt.colorbar()

plt.show()
data.hist(xlabelsize = 0, figsize=(20,12))

plt.show()
skewness = []

skewness_sqrt = []

skewness_yj = []



data_sqrt = data.apply(np.sqrt) 



temp = power_transform(data.iloc[:,:-1],method = "yeo-johnson")

data_yj = pd.DataFrame(temp,columns=data.iloc[:,:-1].columns.tolist())



for var in data.iloc[:,:-1].columns:

        skewness.append(data[var].skew())

for var in data_sqrt.iloc[:,:-1].columns:

        skewness_sqrt.append(data_sqrt[var].skew())

for var in data_yj.iloc[:,:-1].columns:

        skewness_yj.append(data_yj[var].skew())

        

fig,ax = plt.subplots(3,1, figsize=(30,6),sharex=True)

bins = 208

fontsize = 15

ax[0].hist(skewness,bins=bins)

ax[1].hist(skewness_sqrt,bins=bins)

ax[2].hist(skewness_yj,bins=bins)

ax[0].text(3,2.5,"no transform",fontsize = fontsize)

ax[1].text(3,2.5,"sqrt",fontsize = fontsize)

ax[2].text(3,2.5,"yeo-johnson",fontsize = fontsize)



fig.subplots_adjust(hspace=0)

for ax in ax:

    ax.label_outer()

    ax.axvline(x=0,color='r')

    

plt.show()
fig,ax = plt.subplots(figsize=(10,6))

plt.plot(skewness, label = "No transformation")

plt.plot(skewness_sqrt, color='r', label = "Sqrt transformation")

plt.plot(skewness_yj, color='g', label = "Yeo-Johnson transformation")

plt.axhline(y=0,color='k')

plt.xlabel("Variable/Column")

plt.ylabel("Skewness")

plt.legend(loc="best")

plt.show()
data_yj.hist(xlabelsize = 0, figsize=(20,12))

plt.show()
data.plot(kind = 'box',figsize=(20,10))

plt.xticks(rotation='vertical')

plt.show()
outlier_idx = []

for col in data.columns.tolist():

    Q1 = np.percentile(data[col],25)

    Q3 = np.percentile(data[col],75)

    outlier = 1.5*(Q3-Q1)

    outlier_list = data[(data[col]<Q1-outlier) | (data[col]>Q3+outlier)].index

    outlier_idx.extend(outlier_list)



amount_of_rows_with_outliers=[]

for i in range(1,data.shape[1]-1):

    idx_mult_outliers = set([x for x in outlier_idx if outlier_idx.count(x)>i])

    amount_of_rows_with_outliers.append(len(idx_mult_outliers))

    

output_notebook()

TOOLS = "pan,wheel_zoom,reset,hover,save"

my_dict = dict(amount_of_rows_with_outliers = amount_of_rows_with_outliers, number_of_outliers = list(range(1,data.shape[1]-1)))

source2 = ColumnDataSource(my_dict)

p = figure(plot_width=600, plot_height=400, tools = TOOLS, tooltips=[("Number of outliers","@number_of_outliers"),("Amount of rows with such amount of outliers","@amount_of_rows_with_outliers")])

p.line(x="number_of_outliers",y="amount_of_rows_with_outliers", source = source2, line_width  = 2)

p.xaxis.axis_label = 'Number of outliers'

p.yaxis.axis_label = 'Amount of rows with outliers'

show(p)
rows_to_remove = set(list([x for x in outlier_idx if outlier_idx.count(x)>7]))

data_new = data.drop(rows_to_remove,axis=0)

print("The expected number of rows in the new dataset is 208 - 13 = {}".format(data_new.shape[0]))
skewness_new = []



for var in data_new.iloc[:,:-1].columns:

        skewness_new.append(data_new[var].skew())



fig,ax = plt.subplots(2,1, figsize=(20,6),sharex=True)

bins = 208

fontsize = 15

ax[0].hist(skewness_yj,bins=bins)

ax[1].hist(skewness_new,bins=bins)

ax[0].text(3,2.5,"Yeo-Johnson",fontsize = fontsize)

ax[1].text(3,2.5,"Original with removal",fontsize = fontsize)



fig.subplots_adjust(hspace=0)

for ax in ax:

    ax.label_outer()

    ax.axvline(x=0,color='r')



plt.show()
fig,ax = plt.subplots(figsize=(10,6))

plt.plot(skewness_yj, label = "Yeo-Johnson removal")

plt.plot(skewness_new,color='r', label = "Original with removal")

plt.xlabel("Column")

plt.ylabel("Skewness")

plt.legend(loc="best")

ax.axhline(y=0,color='k')

plt.show()
transformer_data = StandardScaler().fit(data)

data_norm = pd.DataFrame(transformer_data.transform(data),columns = data.columns.tolist())

data_norm.head()
data_norm.plot(kind = 'box',figsize=(20,10))

plt.xticks(rotation='vertical')

plt.show()
X = data_norm.drop('V60',axis=1).values

y = data_norm['V60'].values



alphas, _, coefs = lars_path(X, y, verbose=True)

coefs_df = pd.DataFrame(coefs.T, columns =  data_norm.drop('V60',axis=1).columns.tolist())



colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

listy = []

for item in colors.keys():

    listy.append(item)

random_color = listy[0:data.shape[1]-1]



list_of_alphas = [alphas]*(data.shape[1]-1)

list_of_data = coefs_df.T.values.tolist()

my_dictionary = dict(alpha = list_of_alphas,dat = list_of_data,var_names=data.iloc[:,:-1].columns.tolist(),c=random_color)

source3=ColumnDataSource(my_dictionary)



p = figure(plot_width=800, plot_height=600, tools = TOOLS, tooltips=[("Variable","@var_names")])

p.multi_line(xs = 'alpha',ys ='dat',source=source3, line_width  = 2,line_color='c')



p.xaxis.axis_label="alpha"

show(p)
select  = SelectPercentile(percentile=10)

select.fit(X,y)

mask = select.get_support()

print (pd.DataFrame(mask))
# The orignal dataset

X = data.drop('V60',axis=1)

y = data['V60']



number_of_models = 6



result_matrix_grid = [[] for _ in range(number_of_models)] 

result_matrix_test = [[] for _ in range(number_of_models)]

names=[]



start = time.time()



for _ in range(0,10):

    X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,test_size=0.2,random_state=np.random.randint(low=0,high=1000))

    grid_search_pipelines=[]

    

    svm_param = {'svm__C': [0.001, 0.01, 0.1, 1, 10, 100], 'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

    grid_search_pipelines.append(("SVC",svm_param,Pipeline([("scaler",MinMaxScaler()),("svm",SVC())])))

    

    linsvc_param = {'linsvc__C': [0.001, 0.01, 0.1, 1, 10, 100], 'linsvc__penalty': ["l1","l2"]}

    grid_search_pipelines.append(("LINSVC",linsvc_param,Pipeline([("scaler",MinMaxScaler()),("linsvc",LinearSVC(dual=False,max_iter=3000))])))

    

    knn_param = {'knn__n_neighbors':[2,5,10]}

    grid_search_pipelines.append(("KNN",knn_param,Pipeline([("scaler",MinMaxScaler()),("knn",KNeighborsClassifier())])))

    

    lr_param = {'lr__C':[0.001, 0.01, 0.1, 1, 10, 100]}

    grid_search_pipelines.append(("LR",lr_param,Pipeline([("scaler",MinMaxScaler()),("lr",LogisticRegression())])))

    

    rfc_param = {'rfc__max_features':['auto',None]}

    grid_search_pipelines.append(("RFC",rfc_param,Pipeline([("rfc",RandomForestClassifier(n_estimators=2000))])))

    

    gbc_param = {'gbc__loss':["deviance","exponential"],'gbc__max_depth':[1,3],'gbc__learning_rate':[0.01,0.1,0.2,0.3]}

    grid_search_pipelines.append(("GBC",gbc_param,Pipeline([("gbc",GradientBoostingClassifier(n_estimators=2000))])))



    

    i = 0

    for name,param,model in grid_search_pipelines:

        names.append(name)

        grid = GridSearchCV(estimator=model,param_grid=param,cv=10,n_jobs=-1)

        grid.fit(X_train,y_train)

        result_matrix_grid[i].append(grid.best_score_)

        result_matrix_test[i].append(grid.score(X_test,y_test))

        i +=1



print ("Elapsed time is: ", time.time()-start, "seconds")



grid_results = pd.DataFrame(np.array(result_matrix_grid).T,columns=names[0:number_of_models])

print ("\nGrid results:")

for i in grid_results.columns:

    print ("{}: {:.2f} ± {:.2f}".format(i,grid_results[i].mean(),grid_results[i].std()))

    

test_results = pd.DataFrame(np.array(result_matrix_test).T,columns=names[0:number_of_models])

print("\nTest results:")

for i in test_results.columns:

    print ("{}: {:.2f} ± {:.2f}".format(i,test_results[i].mean(),test_results[i].std()))
# The yj dataset

X = data_yj#.drop('V60',axis=1)

y = data['V60']



number_of_models = 6



result_matrix_grid = [[] for _ in range(number_of_models)] 

result_matrix_test = [[] for _ in range(number_of_models)]

names=[]



start = time.time()



for _ in range(0,10):

    X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,test_size=0.2,random_state=np.random.randint(low=0,high=1000))

    grid_search_pipelines=[]

    

    svm_param = {'svm__C': [0.001, 0.01, 0.1, 1, 10, 100], 'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

    grid_search_pipelines.append(("SVC",svm_param,Pipeline([("scaler",MinMaxScaler()),("svm",SVC())])))

    

    linsvc_param = {'linsvc__C': [0.001, 0.01, 0.1, 1, 10, 100], 'linsvc__penalty': ["l1","l2"]}

    grid_search_pipelines.append(("LINSVC",linsvc_param,Pipeline([("scaler",MinMaxScaler()),("linsvc",LinearSVC(dual=False))])))

    

    knn_param = {'knn__n_neighbors':[5,10],'knn__algorithm':['auto','brute']}

    grid_search_pipelines.append(("KNN",knn_param,Pipeline([("scaler",MinMaxScaler()),("knn",KNeighborsClassifier())])))

    

    lr_param = {'lr__C':[0.001, 0.01, 0.1, 1, 10, 100]}

    grid_search_pipelines.append(("LR",lr_param,Pipeline([("scaler",MinMaxScaler()),("lr",LogisticRegression())])))

    

    rfc_param = {'rfc__max_features':['auto',None]}

    grid_search_pipelines.append(("RFC",rfc_param,Pipeline([("rfc",RandomForestClassifier(n_estimators=2000))])))

    

    gbc_param = {'gbc__loss':["deviance","exponential"],'gbc__max_depth':[1,3],'gbc__learning_rate':[0.01,0.1,0.2,0.3]}

    grid_search_pipelines.append(("GBC",gbc_param,Pipeline([("gbc",GradientBoostingClassifier(n_estimators=2000))])))



    

    i = 0

    for name,param,model in grid_search_pipelines:

        names.append(name)

        grid = GridSearchCV(estimator=model,param_grid=param,cv=10,n_jobs=-1)

        grid.fit(X_train,y_train)

        result_matrix_grid[i].append(grid.best_score_)

        #print(name, grid.best_score_)

        result_matrix_test[i].append(grid.score(X_test,y_test))

        i +=1



print ("Elapsed time is: ", time.time()-start, "sec")



grid_results = pd.DataFrame(np.array(result_matrix_grid).T,columns=names[0:number_of_models])

print ("\nGrid results:")

for i in grid_results.columns:

    print ("{}: {:.2f} ± {:.2f}".format(i,grid_results[i].mean(),grid_results[i].std()))

    

test_results = pd.DataFrame(np.array(result_matrix_test).T,columns=names[0:number_of_models])

print("\nTest results:")

for i in test_results.columns:

    print ("{}: {:.2f} ± {:.2f}".format(i,test_results[i].mean(),test_results[i].std()))
# Reduced dataset

data_red = data[["V9","V10","V11","V35","V44","V47","V48","V51","V60"]]



X = data_red.drop('V60',axis=1)

y = data_red['V60']



number_of_models = 6



result_matrix_grid = [[] for _ in range(number_of_models)] 

result_matrix_test = [[] for _ in range(number_of_models)]

names=[]



start = time.time()



for _ in range(0,10):

    X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,test_size=0.2,random_state=np.random.randint(low=0,high=1000))

    grid_search_pipelines=[]

    

    svm_param = {'svm__C': [0.001, 0.01, 0.1, 1, 10, 100], 'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

    grid_search_pipelines.append(("SVC",svm_param,Pipeline([("scaler",MinMaxScaler()),("svm",SVC())])))

    

    linsvc_param = {'linsvc__C': [0.001, 0.01, 0.1, 1, 10, 100], 'linsvc__penalty': ["l1","l2"]}

    grid_search_pipelines.append(("LINSVC",linsvc_param,Pipeline([("scaler",MinMaxScaler()),("linsvc",LinearSVC(dual=False))])))

    

    knn_param = {'knn__n_neighbors':[5,10],'knn__algorithm':['auto','brute']}

    grid_search_pipelines.append(("KNN",knn_param,Pipeline([("scaler",MinMaxScaler()),("knn",KNeighborsClassifier())])))

    

    lr_param = {'lr__C':[0.001, 0.01, 0.1, 1, 10, 100]}

    grid_search_pipelines.append(("LR",lr_param,Pipeline([("scaler",MinMaxScaler()),("lr",LogisticRegression())])))

    

    rfc_param = {'rfc__max_features':['auto',None]}

    grid_search_pipelines.append(("RFC",rfc_param,Pipeline([("rfc",RandomForestClassifier(n_estimators=2000))])))

    

    gbc_param = {'gbc__loss':["deviance","exponential"],'gbc__max_depth':[1,3],'gbc__learning_rate':[0.01,0.1,0.2,0.3]}

    grid_search_pipelines.append(("GBC",gbc_param,Pipeline([("gbc",GradientBoostingClassifier(n_estimators=2000))])))

    



        

    i = 0

    for name,param,model in grid_search_pipelines:

        names.append(name)

        grid = GridSearchCV(estimator=model,param_grid=param,cv=10,n_jobs=-1)

        grid.fit(X_train,y_train)

        result_matrix_grid[i].append(grid.best_score_)

        #print(name, grid.best_score_)

        result_matrix_test[i].append(grid.score(X_test,y_test))

        i +=1



print ("Elapsed time is: ", time.time()-start, "sec")



grid_results = pd.DataFrame(np.array(result_matrix_grid).T,columns=names[0:number_of_models])

print ("\nGrid results:")

for i in grid_results.columns:

    print ("{}: {:.2f} ± {:.2f}".format(i,grid_results[i].mean(),grid_results[i].std()))

    

test_results = pd.DataFrame(np.array(result_matrix_test).T,columns=names[0:number_of_models])

print("\nTest results:")

for i in test_results.columns:

    print ("{}: {:.2f} ± {:.2f}".format(i,test_results[i].mean(),test_results[i].std()))
X = data_yj

y = data['V60']



result_matrix_grid = [] 

result_matrix_test = []

best_param=[]



start = time.time()



for _ in range(0,10):

    X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,test_size=0.2,random_state=np.random.randint(low=0,high=1000))

    

    svm_param = {'svm__C': [0.001, 0.01, 0.1, 1, 10, 100], 'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100],'svm__kernel':['rbf','poly','sigmoid'],'svm__degree':[3,4,5]}

    #grid_search_pipelines.append(("SVC",svm_param,Pipeline([("scaler",MinMaxScaler()),("svm",SVC())])))

    pipe = Pipeline([("scaler",MinMaxScaler()),("svm",SVC())])

    grid = GridSearchCV(estimator=pipe,param_grid=svm_param,cv=10,n_jobs=-1)

    grid.fit(X_train,y_train)

    result_matrix_grid.append(grid.best_score_)

    result_matrix_test.append(grid.score(X_test,y_test))

    best_param.append(grid.best_params_)



    

print ("Elapsed time is: ", time.time()-start, "sec")



print ("\nGrid results: {:.2f} ± {:.2f}".format(np.array(result_matrix_grid).mean(),np.array(result_matrix_grid).std()))

print ("\nTest results: {:.2f} ± {:.2f}".format(np.array(result_matrix_test).mean(),np.array(result_matrix_test).std()))

print("\nBest parameters: {}".format(best_param))
pd.DataFrame.from_dict(best_param)
X = data_yj

y = data['V60']



result_matrix_grid = [] 

result_matrix_test = []

best_param=[]



start = time.time()



for _ in range(0,10):

    X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,test_size=0.2,random_state=np.random.randint(low=0,high=1000))

    

    svm_param = {'svm__C': [3,5,10,20,30,40,50,60,70], 'svm__gamma': [0.3,0.5,1,2,3,4,5,6,7]}

    #grid_search_pipelines.append(("SVC",svm_param,Pipeline([("scaler",MinMaxScaler()),("svm",SVC())])))

    pipe = Pipeline([("scaler",MinMaxScaler()),("svm",SVC())])

    grid = GridSearchCV(estimator=pipe,param_grid=svm_param,cv=10,n_jobs=-1)

    grid.fit(X_train,y_train)

    result_matrix_grid.append(grid.best_score_)

    result_matrix_test.append(grid.score(X_test,y_test))

    best_param.append(grid.best_params_)



    

print ("Elapsed time is: ", time.time()-start, "sec")



print ("Grid results: {:.2f} ± {:.2f}".format(np.array(result_matrix_grid).mean(),np.array(result_matrix_grid).std()))

print ("Test results: {:.2f} ± {:.2f}".format(np.array(result_matrix_test).mean(),np.array(result_matrix_test).std()))

print("Best parameters: {}".format(best_param))
pd.DataFrame.from_dict(best_param)
X = data_yj

y = data['V60']



result_matrix_grid = [] 

result_matrix_test = []

best_param=[]



start = time.time()



for _ in range(0,10):

    X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,test_size=0.2,random_state=np.random.randint(low=0,high=1000))

    

    svm_param = {'svm__C': [2,3,4,5,6,7], 'svm__gamma': [0,2,0.3,0.4,0.5,0.6,0.7]}

    #grid_search_pipelines.append(("SVC",svm_param,Pipeline([("scaler",MinMaxScaler()),("svm",SVC())])))

    pipe = Pipeline([("scaler",MinMaxScaler()),("svm",SVC())])

    grid = GridSearchCV(estimator=pipe,param_grid=svm_param,cv=10,n_jobs=-1)

    grid.fit(X_train,y_train)

    result_matrix_grid.append(grid.best_score_)

    result_matrix_test.append(grid.score(X_test,y_test))

    best_param.append(grid.best_params_)



    

print ("Elapsed time is: ", time.time()-start, "sec")



print ("Grid results: {:.2f} ± {:.2f}".format(np.array(result_matrix_grid).mean(),np.array(result_matrix_grid).std()))

print ("Test results: {:.2f} ± {:.2f}".format(np.array(result_matrix_test).mean(),np.array(result_matrix_test).std()))

print("Best parameters: {}".format(best_param))
pd.DataFrame.from_dict(best_param)
X = data.drop('V60',axis=1)

y = data['V60']



final_test = []



start = time.time()



for _ in range(0,100):

    X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,test_size=0.2,random_state=np.random.randint(low=0,high=1000))

    svm = SVC(C=3,gamma=0.5)

    svm.fit(X_train,y_train)

    final_test.append(svm.score(X_test,y_test))

    

    

print ("Elapsed time is: ", time.time()-start, "sec")

print ("\nTest results: {:.2f} ± {:.2f}".format(np.array(final_test).mean(),np.array(final_test).std())) 