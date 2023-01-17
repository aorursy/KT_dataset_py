import os,sys,time,random,math,time

import tarfile, zipfile



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit

from sklearn.linear_model import LinearRegression

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

print(check_output(["ls", "../input"]).decode("utf8"))
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
datadir="../input/"

data = loadData(datadir,'train.csv')

display(data.info())

display(data.head(5))



test_data= loadData(datadir,'test.csv') 

display(test_data.info())

display(test_data.head(5))
#lets take a moment for the data

features = data.columns

cats = [feat for feat in features if 'cat' in feat]

conts = [feat for feat in features if 'cont' in feat]

print("total features:",len(features),"categories:",len(cats)," continuous:", len(conts))

print("average loss:",data['loss'].mean())

for feat in conts:

    print("Avg value for:",feat,data[feat].mean()) 

print(data[conts].mean())
data['loss'].plot(kind='hist',title='Loss')
# hm....are all the values in the test_data seen in the train data?

for c in cats:

    values={}

    for v in data[c]:

        if v in values:

            values[v]+=1

        else:

            values[v]=1

    for v in test_data[c]:

        if v not in values:

            print(c,v,"not found in test!")

# hint...no they are not!
# combine the two frames so we can encode the labels!

test_data['loss']=0



lengthofData=len(data)

lengthoftest_data=len(test_data)



combineddata=pd.concat([data,test_data])

lengthofcombined=len(combineddata)



# lets have a look at the results to make sure everything matched back up!

display(combineddata.info())

display(combineddata.head())



print("data:",lengthofData)

print("test:",lengthoftest_data)

print("combined:",lengthofcombined)

print("next two frames should match")

display(test_data.head())

display(combineddata.iloc[lengthofData:].head())
# the categorical data that we need in a number format

combineddata=LabelEncoder(combineddata)

display(combineddata.info())

display(combineddata.head())

display(combineddata.iloc[lengthofData:].head())
display(combineddata.info())

display(combineddata.head())
# time to split the data back apart!



data=combineddata.iloc[:lengthofData].copy()

test_data=combineddata.iloc[lengthofData:].copy()

test_data.drop(['loss'],1,inplace=True) # didn't have this column before, make it go away!



print("origdata:",lengthofData)

print("origtest:",lengthoftest_data)

lengthofData=len(data)

lengthoftest_data=len(test_data)

print("newdata:",lengthofData)

print("newtest:",lengthoftest_data)
# we don't want the ID columns in X, and of course not loss either

x=data.drop(['id','loss'],1).fillna(value=0)

# loss is our label

y=data['loss']



display(x.head(5))

display(y.head(5))

display(x.head(5))
#minmax scaler

scaler= MinMaxScaler() 

x = scaler.fit_transform(x)



#  train/validation split

X_train, X_test, y_train, y_test = train_test_split( x, 

                                                    y.values, 

                                                    test_size=0.25, 

                                                    random_state=42)



dataSize=X_train.shape[0]



# subdivide the data size, in case we'd like to train on some smaller portion, for speed 

print ("size of train data",dataSize, )

test_sizes=[50]

for i in range(5):

    test_sizes.append(int(round(dataSize*(i+1)*.2)))



print ("run tests of size",test_sizes)
# OK let's actually do some ML

regrList=[] # a list of regressions to use

regrList.append(LinearRegression())

#regrList.append(SVR()) #long run time, high error!

#regrList.append(ExtraTreesRegressor())

regrList.append(RandomForestRegressor(n_estimators=10,

                                      #criterion = 'mae',

                                      n_jobs =-1, 

                                      random_state=42))

#regrList.append(ensemble.AdaBoostRegressor())  ## The error rate is the bad!



#below xgb seems to be broken in some non-obvious way!

#regrList.append(xgb.XGBClassifier(max_depth=6, learning_rate=0.075, n_estimators=15,

#                                objective="reg:linear", subsample=0.7,

#                                colsample_bytree=0.7, seed=42))







#pca = decomposition.PCA(n_components = 100)

#regr = Pipeline(steps=[('pca', pca), ('classifier', regr )]) # set up the clf as a pipeline so we can do randomized PCA



#params=dict(fit_intercept=[True,False], normalize  = [True,False])

#grid_search = GridSearchCV(regr, param_grid= params, n_jobs= 1, scoring=make_scorer(f1_score)) 

#grid_search.fit(X_train,y_train)



for i in range(len(regrList)): # for each of the regressions we use, fit the data

    start_time = time.time()

    regrList[i].fit(X_train[ :test_sizes[3]],y_train[ :test_sizes[3]] )

    print("fit time:{}s".format(round((time.time()-start_time), 3) ))
# ok, now lets predict with each regression, and spit out some score(MAE) data 

# so we know how it actually did! 

cache=[]

start_time0 = time.time()

for i in range(len(regrList)):

    start_time = time.time()

    print(regrList[i])

    cache.append(regrList[i].predict(X_test))

    print("Mean abs error: {:.2f}".format(np.mean(abs(cache[i] - y_test))))

    print("Score: {:.2f}".format(regrList[i].score(X_test, y_test)))

    print("predict time:{}s".format(round((time.time()-start_time), 3) ))

print("run time:{}s".format(round((time.time()-start_time0), 3) ))     
#XGB -- it doesn't fit the pattern of scikit, so do it seperatly

dtest = xgb.DMatrix(X_test)

gbdt=xgbfit(X_train,y_train)
# now do a prediction and spit out a score(MAE) that means something

start_time = time.time()

print("Mean abs error: {:.2f}".format(np.mean(abs(gbdt.predict(dtest) - y_test))))

print("predict time:{}s".format(round((time.time()-start_time), 3) ))
# refit the full train data!

for i in range(len(regrList)):

    start_time = time.time()

    print(regrList[i])

    regrList[i].fit(x ,y )

    print("re-fit time:{}s".format(round((time.time()-start_time), 3) ))
# My xgb is a diff format, do it here instead

start_time = time.time()

regrList.append(xgbfit(x,y))

print("re-fit time:{}s".format(round((time.time()-start_time), 3) ))
# predict the test data!

start_time0 = time.time()



# don't need the id 

test_X=test_data.drop(['id'],1).fillna(value=0)



#minmax scaler

scaler= MinMaxScaler() 

test_X = scaler.fit_transform(test_X)

#make our predictions and then average them

# we'll do one for each prediction, store/add them up in the loss colum

test_data['loss']=0

for i in range(len(regrList)-1): 

    start_time = time.time()

    test_data['loss'+str(i)]= regrList[i].predict(test_X) 

    print("final predict time:{}s".format(round((time.time()-start_time), 3) ))

    

num_learners=i+1



#xgb needs the data in it's format...

dtest = xgb.DMatrix(test_X)

start_time = time.time()

test_data['loss'+str(num_learners)]=regrList[num_learners].predict(dtest) 

print("final predict time:{}s".format(round((time.time()-start_time), 3) ))  



#test_data['loss']=test_data['loss']/(num_learners) # average the predictions 



for ii in range(i+1):

    test_data['loss']+=test_data['loss'+str(ii)]

test_data['loss']/num_learners



display(test_data.info())

display(test_data.head())



result=test_data[['id','loss']] # we just need these for the submission

display(result.info())

display(result.head())

print("run time:{}s".format(round((time.time()-start_time0), 3) ))
# spit this out to be used for the submission!

output_fname="result_submission.csv" 

writeData(result,output_fname)
# an xgb result on it's own! 

test_data['loss']= regrList[i+1].predict(dtest) 

resultxgb=test_data[['id','loss']]

output_fname="result_submission_xgb.csv"

writeData(resultxgb,output_fname)



# averaging didn't work well for XGB--it actually brought the score down overall, and

# it gets a higher score on it's own.
# I think that for stacking, I take my prediction for each method and retrain on that.

#cache=[np.array(len(X_train),1)] #make an empty array, to add each prediction set to

data_with_pred=data.copy()

display(data_with_pred.head(4))

start_time0 = time.time()



for i in range(len(regrList)-2): #don't use that last regression, it's xgb

    print(i,regrList[i])

    start_time = time.time()

    curr_predict = regrList[i].predict(x)

    data_with_pred['pred'+str(i)]=curr_predict

    #cache.append(regrList[i].predict(X_train), axis=1)



    print("Mean abs error: {:.2f}".format(np.mean(abs(curr_predict - y))))

    print("Score: {:.2f}".format(regrList[i].score(x, y)))

    print("predict time:{}s".format(round((time.time()-start_time), 3) ))

print("run time:{}s".format(round((time.time()-start_time0), 3) ))  





x_with_pred=data_with_pred.drop(['id','loss'],1).fillna(value=0).copy()

x_with_pred = scaler.fit_transform(x_with_pred)

display(data_with_pred.head(3))

del data_with_pred



#print(cache[1].shape)
# train XGB on data that included the predictions!

start_time = time.time()

gbdt=xgbfit(x_with_pred,y)

print("re-fit time:{}s".format(round((time.time()-start_time), 3) ))
#ok, we trained xgb with the predictions, now make predictions based on that, and save them

test_data_with_pred=test_data[['id','loss']].copy()

for ii in range(i+1):

    test_data_with_pred['pred'+str(ii)]=test_data['loss'+str(ii)]

test_data_with_pred.drop(['id','loss'],1,inplace=True)

display(test_data_with_pred.head(3))



dtest = xgb.DMatrix(test_data_with_pred.values)

test_data['loss']= gbdt.predict(dtest)

resultxgb=test_data[['id','loss']]

output_fname="result_submission_xgb.csv"

writeData(resultxgb,output_fname)