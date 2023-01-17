import os,sys,time,random,math,time

import tarfile, zipfile



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit

from sklearn.linear_model import LinearRegression,Ridge



from sklearn import decomposition, datasets, ensemble

from sklearn.cluster import KMeans

from sklearn.pipeline import Pipeline

from sklearn.grid_search import GridSearchCV

from sklearn.metrics import make_scorer,precision_score, recall_score, f1_score, average_precision_score, accuracy_score, mean_absolute_error



from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR



import matplotlib.pyplot as plt

from IPython.display import display, Image



import xgboost as xgb





from subprocess import check_output

datadir="../input/"

print(check_output(["ls", datadir]).decode("utf8"))



%matplotlib inline  
def loadData(datadir,filename):

    # Load the wholesale customers dataset

    #data = pd.read_csv(filename)

    data = ''

    print ("loading: "+datadir+filename)

    try:

        if zipfile.is_zipfile(datadir+filename):

            z = zipfile.ZipFile(datadir+filename)

            filename = z.open(filename[:-4])

        else:

            filename=datadir+filename

        data = pd.read_csv(filename, parse_dates=True)  

        print ("Dataset has {} samples with {} features each.".format(*data.shape))

    except Exception as e:

        print ("Dataset could not be loaded. Is the dataset missing?")

        print(e)

    return data



def writeData(data,filename):

    # Load the wholesale customers dataset

    try:

        data.to_csv(filename, index=False)

    except Exception as e:

        print ("Dataset could not be written.")

        print(e)

    verify=[]

    try:

        with open(filename, 'r') as f:

            for line in f:

                verify.append(line)

        f.closed

        return verify[:5]

    except IOError:

        sys.std

        

def LabelEncoder(data):

    # lifted in parts from:

    #https://www.kaggle.com/mmueller/allstate-claims-severity/yet-another-xgb-starter/code

    features = data.columns

    cats = [feat for feat in features if 'cat' in feat]

    for feat in cats:

        data[feat] = pd.factorize(data[feat], sort=True)[0]

    return data



# XGB!



def xgbfit(X_train,y_train):

    dtrain = xgb.DMatrix(X_train, label=y_train)

    



    xgb_params = {

        'seed': 0,

        'colsample_bytree': 0.7,

        'silent': 1,

        'subsample': 0.7,

        'learning_rate': 0.075,

        'objective': 'reg:linear',

        'max_depth': 6,

        'num_parallel_tree': 1,

        'min_child_weight': 1,

        'eval_metric': 'mae',

    }



    start_time = time.time()

    res = xgb.cv(xgb_params, dtrain, num_boost_round=750, nfold=4, seed=42, stratified=False,

                 early_stopping_rounds=15, verbose_eval=100, show_stdv=True, maximize=False)

    print("fit time:{}s".format(round((time.time()-start_time), 3) ))



    best_nrounds = res.shape[0] - 1

    cv_mean = res.iloc[-1, 0]

    cv_std = res.iloc[-1, 1]

    print('CV-Mean: {0}+{1}'.format(cv_mean, cv_std))

    # XGB Train!

    start_time = time.time()

    gbdt = xgb.train(xgb_params, dtrain, best_nrounds)

    print("Train time:{}s".format(round((time.time()-start_time), 3) ))

    return gbdt
data = loadData(datadir,'train.csv')

display(data.info())

display(data.head(5))



test_data= loadData(datadir,'test.csv') 

display(test_data.info())

display(test_data.head(5))
# combine the two frames so we can encode the labels!

test_data['loss']=0



lengthofData=len(data)

lengthoftest_data=len(test_data)



print("data:",lengthofData)

print("test:",lengthoftest_data)



combineddata=pd.concat([data,test_data])

lengthofcombined=len(combineddata)

print("combined:",lengthofcombined)



# the categorical data that we need in a number format

combineddata=LabelEncoder(combineddata)



# time to split the data back apart!

data=combineddata.iloc[:lengthofData].copy()

test_data=combineddata.iloc[lengthofData:].copy()

test_data.drop(['loss'],1,inplace=True) # didn't have this column before, make it go away!





x_test = test_data.copy()

x_test.drop(['id'],1,inplace=True)



# we don't want the ID columns in X, and of course not loss either

x=data.drop(['id','loss'],1)

# loss is our label

y=data['loss']



#minmax scaler

scaler= MinMaxScaler() 

x = scaler.fit_transform(x)

x_test_data = scaler.fit_transform(x_test)



#display(x[:5])

#display(y.head(5))



print("Pre-Processing done")

print("data:",len(x))

print("labels:",len(y))

print("test:",len(x_test_data))
# OK let's actually do some ML

regrList=[] # a list of regressions to use

#regrList.append(LinearRegression())

regrList.append(ExtraTreesRegressor())

regrList.append(Ridge())

    

regrList.append(RandomForestRegressor(n_estimators=10,

                                      #criterion = 'mae',

                                      n_jobs =-1, 

                                      random_state=42))

print("number of scikitlearn regressors to use:",len(regrList))
#prepare the fold divisions



data_size=x.shape[0]

print("size of train data:",data_size)

folds=[]

num_folds=5

fold_start=0

for k in range(num_folds-1):

    fold_end=int(((data_size/num_folds)*(k+1)))

    folds.append((fold_start,fold_end))

    fold_start=fold_end

folds.append((fold_start,data_size))

print("folds at:",folds)

print("fold size:", (data_size/num_folds))

print("train size:",(data_size/num_folds)*(num_folds-1))



count=0

for i in folds:

    count+=i[1]-i[0]

print(count)
x_layer2=[]

start_time0 = time.time()



for fold_start,fold_end in folds:

    print("Fold:",fold_start,"to",fold_end,"of",data_size)

    start_time1 = time.time()

    fold_result=[]

    

    X_test = x[fold_start:fold_end].copy()

    y_test = y[fold_start:fold_end].copy()

    X_train=np.concatenate((x[:fold_start], x[fold_end:]), axis=0)

    y_train=np.concatenate((y[:fold_start], y[fold_end:]), axis=0)

    print("\nfolding! len test {}, len train {}".format(len(X_test),len(X_train)))

    

    for i in range(len(regrList)): # for each of the regressions we use, fit/predict the data

        start_time = time.time()

        regrList[i].fit(X_train,y_train)

        print("\nfit time:{}s".format(round((time.time()-start_time), 3) ))



        start_time = time.time()

        print(regrList[i])

        curr_predict=regrList[i].predict(X_test)

        if fold_result == []:

            fold_result = np.array(curr_predict.copy())

        else:

            fold_result = np.column_stack((fold_result,curr_predict))

        

        print("predict time:{}s".format(round((time.time()-start_time), 3) ))

        #show some stats on that last regressions run    

        print("Mean abs error: {:.2f}".format(np.mean(abs(curr_predict - y_test))))

        print("Score: {:.2f}".format(regrList[i].score(X_test, y_test)))

    

    #XGB -- it doesn't fit the pattern of scikit, so do it seperatly

    #dtest = xgb.DMatrix(X_test)

    #gbdt=xgbfit(X_train,y_train)



    # now do a prediction and spit out a score(MAE) that means something

    #start_time = time.time()

    #curr_predict=gbdt.predict(dtest)

    #fold_result = np.column_stack((fold_result,curr_predict))  

    #print("XGB Mean abs error: {:.2f}".format(np.mean(abs(curr_predict - y_test))))

    #print("XGB predict time:{}s".format(round((time.time()-start_time), 3) ))

    

    if x_layer2 == []:

        x_layer2=fold_result

    else:

        x_layer2=np.append(x_layer2,fold_result,axis=0)

        

    print("--layer2 length:",len(x_layer2))

    print("--layer2 shape:",np.shape(x_layer2))

    print("Fold run time:{}s".format(round((time.time()-start_time1), 3) ))   

print("Full run time:{}s".format(round((time.time()-start_time0), 3) ))   
print(len(x_layer2))

print(len(y))



#  train/validation split

X_layer2_train, X_layer2_validation, y_layer2_train, y_layer2_validation = train_test_split( x_layer2,

                                                                                y,

                                                                                test_size=0.25,

                                                                                random_state=42)

layer2_regr=LinearRegression()



layer2_regr.fit(X_layer2_train,y_layer2_train)



layer2_predict=layer2_regr.predict(X_layer2_validation)



#show some stats on that last regressions run    

print("Mean abs error: {:.2f}".format(np.mean(abs(layer2_predict - y_layer2_validation))))

print("Score: {:.2f}".format(layer2_regr.score(X_layer2_validation, y_layer2_validation)))





#with LinearReg: Mean abs error: 1238.52
# The XGB version of layer 2

print(len(x_layer2))

print(len(y))



#  train/validation split

X_layer2_train, X_layer2_validation, y_layer2_train, y_layer2_validation = train_test_split( x_layer2,

                                                                                y,

                                                                                test_size=0.25,

                                                                                random_state=42)

#XGB -- it doesn't fit the pattern of scikit, so do it seperatly

dtest = xgb.DMatrix(X_layer2_validation)

#layer2_gbdt=xgbfit(X_layer2_train,y_layer2_train)



# now do a prediction and spit out a score(MAE) that means something

start_time = time.time()

#print("XGB Mean abs error: {:.2f}".format(np.mean(abs(layer2_gbdt.predict(dtest) - y_layer2_validation))))

print("XGB predict time:{}s".format(round((time.time()-start_time), 3) ))

#with LinearReg: XGB Mean abs error: 1205.77
x_layer2_test = []

start_time1 = time.time()

for i in range(len(regrList)): # for each of the regressions we use, fit/predict the data

    start_time = time.time()

    print(regrList[i])

    curr_predict=regrList[i].predict(x_test_data)

    print("predict time:{}s".format(round((time.time()-start_time), 3) ))

    

    if x_layer2_test == []:

        x_layer2_test = np.array(curr_predict.copy())

    else:

        x_layer2_test = np.column_stack((x_layer2_test,curr_predict))

    print(curr_predict)



#XGB -- it doesn't fit the pattern of scikit, so do it seperatly

dtest = xgb.DMatrix(x_test_data)

# now do a prediction and spit out a score(MAE) that means something

start_time = time.time()

#curr_predict=gbdt.predict(dtest)

#x_layer2_test = np.column_stack((x_layer2_test,curr_predict))

#print("Mean abs error: {:.2f}".format(np.mean(abs(cache[i+1] - y_test))))

print("XGB predict time:{}s".format(round((time.time()-start_time), 3) ))



print("Fold run time:{}s".format(round((time.time()-start_time1), 3) ))   
# some problems noted---fact finding below!

display("size of original test data:",len(x_test_data))

display("Test shape:",np.shape(x_layer2_test))

display("train shape:",np.shape(x_layer2))



print("sample of layer2 test:\n",x_layer2_test[:4])



print("x_layer2_test mean:",x_layer2_test.mean( axis=0))

print("x_layer2 mean:",x_layer2.mean(axis=0))

train_layer2_col0_mean=x_layer2.mean(axis=0)[0]



print("x_layer2_test std:",x_layer2_test.std( axis=0)) 

print("x_layer2 std:",x_layer2.std(axis=0))



# notice that column 0(linregresion) has a significantly higher mean and std

# here's a hack to not fix that for now! 



# check which row in column 0 are significantly far from the mean

problem_column=x_layer2_test.T[0]

outliers=[]

for i in range(len(problem_column)):

    if problem_column[i]>30000:

        outliers.append((i,problem_column[i]))

print("num outliers:",len(outliers))



#for each problem child, set them to the average value from the train set, to null the affect some

for o in outliers:

    problem_column[o[0]]=train_layer2_col0_mean

    

print(problem_column[o[0]])



#check outliers again

problem_column=x_layer2_test.T[0]

outliers=[]

for i in range(len(problem_column)):

    if problem_column[i]>30000:

        outliers.append((i,problem_column[i]))

print("num outliers:",len(outliers))



print(x_layer2_test.T[0][o[0]]) # verify that the change made it all the way to the original
test_data['loss']=layer2_regr.predict(x_layer2_test)



result=test_data[['id','loss',]]

output_fname="result_submission_stack.csv"

display(writeData(result,output_fname))

#the XGB version:



dtest = xgb.DMatrix(x_layer2_test)

#test_data['loss']=layer2_gbdt.predict(dtest)



result=test_data[['id','loss',]]

output_fname="result_submission_stack_xgb.csv"

display(writeData(result,output_fname))

#let's have a look at the std of the result, as a cross check

print("result std:",result.std(axis=0))