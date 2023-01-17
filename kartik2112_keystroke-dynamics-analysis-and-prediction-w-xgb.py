import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import StratifiedShuffleSplit
trainDF = pd.read_csv('/kaggle/input/keystroke-dynamics-challenge-1/train.csv')

testDF = pd.read_csv('/kaggle/input/keystroke-dynamics-challenge-1/test.csv')
trainDF.head()
testDF.head()
print('No. of rows in training dataset:',len(trainDF))

print('No. of users for which training data is present:',trainDF.user.nunique())
print('No. of rows in test dataset:',len(testDF))
trainDF1 = trainDF

for i in range(1,13):

    trainDF1['PPD-'+str(i)] = trainDF1['press-'+str(i)] - trainDF1['press-'+str(i-1)]

    trainDF1['RPD-'+str(i)] = trainDF1['release-'+str(i)] - trainDF1['press-'+str(i-1)]



for i in range(13):

    trainDF1['HD-'+str(i)] = trainDF1['release-'+str(i)] - trainDF1['press-'+str(i)]

    

testDF1 = testDF

for i in range(1,13):

    testDF1['PPD-'+str(i)] = testDF1['press-'+str(i)] - testDF1['press-'+str(i-1)]

    testDF1['RPD-'+str(i)] = testDF1['release-'+str(i)] - testDF1['press-'+str(i-1)]



for i in range(13):

    testDF1['HD-'+str(i)] = testDF1['release-'+str(i)] - testDF1['press-'+str(i)]
trainDF1.head()
# Check stats of first 5 users i.e. 5 x 8 typing patterns

noOfUsers = 5

if noOfUsers == -1:

    trainDF2 = trainDF1

else:

    trainDF2 = trainDF1[:noOfUsers*8]
temp1 = pd.DataFrame({'Min':trainDF2.min(),'Max':trainDF2.max()})

temp1.head()
for i in range(1,13):

    ax = sns.scatterplot(x='RPD-'+str(i),y='PPD-'+str(i),hue='user',data=trainDF2)



# Small trick to avoid repeating legends: https://stackoverflow.com/a/36268401/5370202    

handles, labels = ax.get_legend_handles_labels()

ax.legend(handles[:noOfUsers], labels[:trainDF2.user.nunique()])

ax.set_title('Scatterplot of PPD vs RPD')
plt.figure(figsize=(15,5))

for i in range(1,13):

    sns.swarmplot(y='RPD-'+str(i),x='user',data=trainDF2).set_title('Swarm of Release-Press Duration for users')
plt.figure(figsize=(15,5))

for i in range(1,13):

    sns.swarmplot(y='PPD-'+str(i),x='user',data=trainDF2).set_title('Swarm of Press-Press Duration for users')
plt.figure(figsize=(15,5))

for i in range(13):

    sns.swarmplot(y='HD-'+str(i),x='user',data=trainDF2).set_title('Swarm of Hold Duration for users')
# value_vars_cols = ['HD-'+str(i) for i in range(13)]



drop_cols_HD_analysis = ['PPD-'+str(i) for i in range(1,13)] + ['RPD-'+str(i) for i in range(1,13)] + ['release-'+str(i) for i in range(13)]



trainDF_HD_analysis = trainDF2.drop(columns=drop_cols_HD_analysis)

trainDF_HD_analysis['id'] = trainDF_HD_analysis.index

trainDF_HD_analysis = pd.wide_to_long(trainDF_HD_analysis,['press-','HD-'],i='id',j='key_no').sort_values(by=['user','id','key_no'])

trainDF_HD_analysis
plt.figure(figsize=(15,10))

sns.scatterplot(x='press-',y='HD-',hue='user',data=trainDF_HD_analysis,palette='deep')
plt.figure(figsize=(15,10))

# sns.load_dataset(trainDF_HD_analysis)

sns.lineplot(x='press-',y='HD-',hue='user',units='id',estimator=None,data=trainDF_HD_analysis.reset_index(),palette='deep').set_title('Line plots for each key sequence')
# value_vars_cols = ['HD-'+str(i) for i in range(13)]



drop_cols_PPD_analysis = ['HD-'+str(i) for i in range(13)] + ['RPD-'+str(i) for i in range(1,13)] + ['release-'+str(i) for i in range(13)] + ['press-0']



trainDF_PPD_analysis = trainDF2.drop(columns=drop_cols_PPD_analysis)

trainDF_PPD_analysis['id'] = trainDF_PPD_analysis.index

trainDF_PPD_analysis = pd.wide_to_long(trainDF_PPD_analysis,['press-','PPD-'],i='id',j='key_no').sort_values(by=['user','id','key_no'])

# trainDF_PPD_analysis
plt.figure(figsize=(15,10))

sns.scatterplot(x='press-',y='PPD-',hue='user',data=trainDF_PPD_analysis,palette='deep')
plt.figure(figsize=(15,10))

# sns.load_dataset(trainDF_HD_analysis)

sns.lineplot(x='press-',y='PPD-',hue='user',units='id',estimator=None,data=trainDF_PPD_analysis.reset_index(),palette='deep').set_title('Line plots for each key sequence')
# value_vars_cols = ['HD-'+str(i) for i in range(13)]



drop_cols_RPD_analysis = ['HD-'+str(i) for i in range(13)] + ['PPD-'+str(i) for i in range(1,13)] + ['release-'+str(i) for i in range(13)] + ['press-0']



trainDF_RPD_analysis = trainDF2.drop(columns=drop_cols_RPD_analysis)

trainDF_RPD_analysis['id'] = trainDF_RPD_analysis.index

trainDF_RPD_analysis = pd.wide_to_long(trainDF_RPD_analysis,['press-','RPD-'],i='id',j='key_no').sort_values(by=['user','id','key_no'])

# trainDF_RPD_analysis
plt.figure(figsize=(15,10))

sns.scatterplot(x='press-',y='RPD-',hue='user',data=trainDF_RPD_analysis,palette='deep')
plt.figure(figsize=(15,10))

# sns.load_dataset(trainDF_HD_analysis)

sns.lineplot(x='press-',y='RPD-',hue='user',units='id',estimator=None,data=trainDF_RPD_analysis.reset_index(),palette='deep').set_title('Line plots for each key sequence')
## Training Data

drop_cols_HD_analysis = ['PPD-'+str(i) for i in range(1,13)] + ['RPD-'+str(i) for i in range(1,13)] + ['release-'+str(i) for i in range(13)]



trainDF_HD_analysis = trainDF1.drop(columns=drop_cols_HD_analysis)

trainDF_HD_analysis['id'] = trainDF_HD_analysis.index

trainDF_HD_analysis = pd.wide_to_long(trainDF_HD_analysis,['press-','HD-'],i='id',j='key_no').sort_values(by=['user','id','key_no'])



drop_cols_PPD_analysis = ['HD-'+str(i) for i in range(13)] + ['RPD-'+str(i) for i in range(1,13)] + ['release-'+str(i) for i in range(13)] + ['press-0']



trainDF_PPD_analysis = trainDF1.drop(columns=drop_cols_PPD_analysis)

trainDF_PPD_analysis['id'] = trainDF_PPD_analysis.index

trainDF_PPD_analysis = pd.wide_to_long(trainDF_PPD_analysis,['press-','PPD-'],i='id',j='key_no').sort_values(by=['user','id','key_no'])



drop_cols_RPD_analysis = ['HD-'+str(i) for i in range(13)] + ['PPD-'+str(i) for i in range(1,13)] + ['release-'+str(i) for i in range(13)] + ['press-0']



trainDF_RPD_analysis = trainDF1.drop(columns=drop_cols_RPD_analysis)

trainDF_RPD_analysis['id'] = trainDF_RPD_analysis.index

trainDF_RPD_analysis = pd.wide_to_long(trainDF_RPD_analysis,['press-','RPD-'],i='id',j='key_no').sort_values(by=['user','id','key_no'])





## Test Data

testDF_HD_analysis = testDF1.drop(columns=drop_cols_HD_analysis)

testDF_HD_analysis['id'] = testDF_HD_analysis.index

testDF_HD_analysis = pd.wide_to_long(testDF_HD_analysis,['press-','HD-'],i='id',j='key_no').sort_values(by=['id','key_no'])



testDF_PPD_analysis = testDF1.drop(columns=drop_cols_PPD_analysis)

testDF_PPD_analysis['id'] = testDF_PPD_analysis.index

testDF_PPD_analysis = pd.wide_to_long(testDF_PPD_analysis,['press-','PPD-'],i='id',j='key_no').sort_values(by=['id','key_no'])



testDF_RPD_analysis = testDF1.drop(columns=drop_cols_RPD_analysis)

testDF_RPD_analysis['id'] = testDF_RPD_analysis.index

testDF_RPD_analysis = pd.wide_to_long(testDF_RPD_analysis,['press-','RPD-'],i='id',j='key_no').sort_values(by=['id','key_no'])
## Join these individual tables together

testDFCombined = testDF_HD_analysis.join(testDF_RPD_analysis.drop(columns=['press-']),rsuffix='RPD_').join(testDF_PPD_analysis.drop(columns=['press-']),rsuffix='PPD_')



trainDFCombined = trainDF_HD_analysis.join(trainDF_RPD_analysis.drop(columns=['user','press-']),rsuffix='RPD_').join(trainDF_PPD_analysis.drop(columns=['user','press-']),rsuffix='PPD_')

trainDFCombined
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.distplot(trainDFCombined['HD-']).set_title('Hist of Hold Duration')

plt.subplot(2,2,2)

sns.distplot(trainDFCombined['PPD-']).set_title('Hist of Press-Press Duration')

plt.subplot(2,2,3)

sns.distplot(trainDFCombined['RPD-']).set_title('Hist of Release-Press Duration')
noOfBins = 10



## Training Data

HDMax = trainDFCombined['HD-'].max()

RPDMax = trainDFCombined['RPD-'].max()

PPDMax = trainDFCombined['PPD-'].max()

print('Max values in train are: HDMax:',HDMax,'RPDMax:',RPDMax,'PPDMax:',PPDMax)

labels = [i for i in range(noOfBins)]



trainDFCombined['HDEnc'],HDBins = pd.qcut(trainDFCombined['HD-'],retbins=True,labels=labels,q=noOfBins)

trainDFCombined['PPDEnc'],RPDBins = pd.qcut(trainDFCombined['PPD-'],retbins=True,labels=labels,q=noOfBins)

trainDFCombined['RPDEnc'],PPDBins = pd.qcut(trainDFCombined['RPD-'],retbins=True,labels=labels,q=noOfBins)



trainDFCombined['HDEnc'] = trainDFCombined['HDEnc'].astype(str).replace('nan',-1).astype(int)

trainDFCombined['PPDEnc'] = trainDFCombined['PPDEnc'].astype(str).replace('nan',-1).astype(float)

trainDFCombined['RPDEnc'] = trainDFCombined['RPDEnc'].astype(str).replace('nan',-1).astype(float)





## Test Data

HDMax = testDFCombined['HD-'].max()

RPDMax = testDFCombined['RPD-'].max()

PPDMax = testDFCombined['PPD-'].max()

print('Max values in test data are: HDMax:',HDMax,'RPDMax:',RPDMax,'PPDMax:',PPDMax)

labels = [i for i in range(noOfBins)]



testDFCombined['HDEnc'] = pd.cut(testDFCombined['HD-'],labels=labels,bins=HDBins)

testDFCombined['PPDEnc'] = pd.cut(testDFCombined['PPD-'],labels=labels,bins=RPDBins)

testDFCombined['RPDEnc'] = pd.cut(testDFCombined['RPD-'],labels=labels,bins=PPDBins)



testDFCombined['HDEnc'] = testDFCombined['HDEnc'].astype(str).replace('nan',-1).astype(float)

testDFCombined['PPDEnc'] = testDFCombined['PPDEnc'].astype(str).replace('nan',-1).astype(float)

testDFCombined['RPDEnc'] = testDFCombined['RPDEnc'].astype(str).replace('nan',-1).astype(float)
trainDFCombined
testDFCombined
## Lower limit values of bins created

HDBins, RPDBins, PPDBins, 'No. of buckets: '+str(len(HDBins)-1)
plt.figure(figsize=(15,8))

noOfUsers = 5

plt.subplot(3,1,1)

sns.swarmplot(y='HDEnc',x='user',data=trainDFCombined[:8*12*noOfUsers],palette='deep').set_title('Swarm plot of binned hold duration')

plt.subplot(3,1,2)

sns.swarmplot(y='PPDEnc',x='user',data=trainDFCombined[:8*12*noOfUsers],palette='deep').set_title('Swarm plot of binned press-press duration')

plt.subplot(3,1,3)

sns.swarmplot(y='RPDEnc',x='user',data=trainDFCombined[:8*12*noOfUsers],palette='deep').set_title('Swarm plot of binned release-press duration')
trainDFCombinedHDAvg = trainDFCombined.reset_index().groupby(['user','key_no'])['HDEnc'].mean()

trainDFCombinedPPDAvg = trainDFCombined.reset_index().groupby(['user','key_no'])['PPDEnc'].mean()

trainDFCombinedRPDAvg = trainDFCombined.reset_index().groupby(['user','key_no'])['RPDEnc'].mean()

tempDF = pd.DataFrame({'HD':trainDFCombinedHDAvg,'PPD':trainDFCombinedPPDAvg,'RPD':trainDFCombinedRPDAvg})



trainDF_HDProperties = tempDF.reset_index().groupby('user')['HD'].apply(np.array)

trainDF_PPDProperties = tempDF.reset_index().groupby('user')['PPD'].apply(np.array)

trainDF_RPDProperties = tempDF.reset_index().groupby('user')['RPD'].apply(np.array)



trainDF_UserProps = pd.DataFrame({'HD':trainDF_HDProperties, 'PPD':trainDF_PPDProperties, 'RPD':trainDF_RPDProperties})



trainDF_UserProps = pd.DataFrame(trainDF_UserProps.HD.tolist(),index = trainDF_UserProps.index).add_prefix('HD_').join(

    pd.DataFrame(trainDF_UserProps.PPD.tolist(),index = trainDF_UserProps.index).add_prefix('PPD_')

).join(

    pd.DataFrame(trainDF_UserProps.RPD.tolist(),index = trainDF_UserProps.index).add_prefix('RPD_')

)



# Average bin keystrokes for each of the 110 users

trainDF_UserProps
trainDFCombinedHDAvg = testDFCombined.reset_index().groupby(['id','key_no'])['HDEnc'].mean()

trainDFCombinedPPDAvg = testDFCombined.reset_index().groupby(['id','key_no'])['PPDEnc'].mean()

trainDFCombinedRPDAvg = testDFCombined.reset_index().groupby(['id','key_no'])['RPDEnc'].mean()

tempDF = pd.DataFrame({'HD':trainDFCombinedHDAvg,'PPD':trainDFCombinedPPDAvg,'RPD':trainDFCombinedRPDAvg})



trainDF_HDProperties = tempDF.reset_index().groupby('id')['HD'].apply(np.array)

trainDF_PPDProperties = tempDF.reset_index().groupby('id')['PPD'].apply(np.array)

trainDF_RPDProperties = tempDF.reset_index().groupby('id')['RPD'].apply(np.array)



testDF_UserProps = pd.DataFrame({'HD':trainDF_HDProperties, 'PPD':trainDF_PPDProperties, 'RPD':trainDF_RPDProperties})



testDF_UserProps = pd.DataFrame(testDF_UserProps.HD.tolist(),index = testDF_UserProps.index).add_prefix('HD_').join(

    pd.DataFrame(testDF_UserProps.PPD.tolist(),index = testDF_UserProps.index).add_prefix('PPD_')

).join(

    pd.DataFrame(testDF_UserProps.RPD.tolist(),index = testDF_UserProps.index).add_prefix('RPD_')

)



# Bin allocation 

testDF_UserProps
trainDF_HDTemp = trainDFCombined.reset_index().groupby(['user','id'])['HDEnc'].apply(np.array)

trainDF_PPDTemp = trainDFCombined.reset_index().groupby(['user','id'])['PPDEnc'].apply(np.array)

trainDF_RPDTemp = trainDFCombined.reset_index().groupby(['user','id'])['RPDEnc'].apply(np.array)



trainDF_User_AllSampleProps = pd.DataFrame({'HD':trainDF_HDTemp, 'PPD':trainDF_PPDTemp, 'RPD':trainDF_RPDTemp})



trainDF_User_AllSampleProps = pd.DataFrame(trainDF_User_AllSampleProps.HD.tolist(),index = trainDF_User_AllSampleProps.index).add_prefix('HD_').join(

    pd.DataFrame(trainDF_User_AllSampleProps.PPD.tolist(),index = trainDF_User_AllSampleProps.index).add_prefix('PPD_')

).join(

    pd.DataFrame(trainDF_User_AllSampleProps.RPD.tolist(),index = trainDF_User_AllSampleProps.index).add_prefix('RPD_')

).reset_index().set_index('user').drop(columns=['id'])



trainDF_User_AllSampleProps
trainDF_HDTemp = testDFCombined.reset_index().groupby(['id'])['HDEnc'].apply(np.array)

trainDF_PPDTemp = testDFCombined.reset_index().groupby(['id'])['PPDEnc'].apply(np.array)

trainDF_RPDTemp = testDFCombined.reset_index().groupby(['id'])['RPDEnc'].apply(np.array)



testDF_User_AllSampleProps = pd.DataFrame({'HD':trainDF_HDTemp, 'PPD':trainDF_PPDTemp, 'RPD':trainDF_RPDTemp})



testDF_User_AllSampleProps = pd.DataFrame(testDF_User_AllSampleProps.HD.tolist(),index = testDF_User_AllSampleProps.index).add_prefix('HD_').join(

    pd.DataFrame(testDF_User_AllSampleProps.PPD.tolist(),index = testDF_User_AllSampleProps.index).add_prefix('PPD_')

).join(

    pd.DataFrame(testDF_User_AllSampleProps.RPD.tolist(),index = testDF_User_AllSampleProps.index).add_prefix('RPD_')

)



testDF_User_AllSampleProps
knn_summary = KNeighborsClassifier(1)

trainX_summary = trainDF_UserProps.reset_index().drop(columns=['user'])

trainY_summary = trainDF_UserProps.index



# testX_summary = testDF_UserProps.reset_index().drop(columns=['id'])



knn_summary.fit(trainX_summary,trainY_summary)



accuracy_score(knn_summary.predict(trainX_summary),trainY_summary)
trainX_allSamples = trainDF_User_AllSampleProps.reset_index().drop(columns=['user'])

trainY_allSamples = trainDF_User_AllSampleProps.index



def getCrossValidationAccuracy(n_neighbours):

    knn_allSamples = KNeighborsClassifier(n_neighbours)

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

    acc = []

    for train_index, test_index in sss.split(trainX_allSamples, trainY_allSamples):

        knn_allSamples.fit(trainX_allSamples.loc[train_index],trainY_allSamples[train_index])

        acc += [accuracy_score(knn_allSamples.predict(trainX_allSamples.loc[test_index]),trainY_allSamples[test_index])]

    return sum(acc) / len(acc)

allAttemptsAcc = [getCrossValidationAccuracy(i) for i in range(1,8)]

print('Accuracies:',allAttemptsAcc)

sns.lineplot(y=allAttemptsAcc,x=range(1,8)).set_title('Cross-Val Accuracy v/s no. of neighbours')
# knn_allSamples = KNeighborsClassifier(1)

# knn_allSamples.fit(trainX_allSamples,trainY_allSamples)



# testX_allSamples = testDF_User_AllSampleProps.reset_index().drop(columns=['id'])

# textPreds_allSamples = knn_allSamples.predict(testX_allSamples)

# pd.DataFrame({'user':textPreds_allSamples},index=testX_allSamples.index).to_csv('submission.csv',index=False)
%%time

from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

from xgboost.sklearn import XGBClassifier



xgb1 = XGBClassifier(

    learning_rate =0.1,

    n_estimators=10,

    max_depth=5,

    min_child_weight=3,

    gamma=0,

    subsample=0.8,

    colsample_bytree=0.8,

    objective= 'multi:softmax',

    num_class=trainY_allSamples.nunique(),

    nthread=4,

    seed=27)

param_search = {

    'learning_rate': [0.05, 0.1],

    'n_estimators': [100,200,210,230,250,270,290,310,330],

    'max_depth': range(4,10,1)

}

gsearch2b = GridSearchCV(estimator = xgb1,param_grid = param_search, scoring='accuracy',n_jobs=4,iid=False,cv=StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=0), verbose=1)

gsearch2b.fit(trainX_allSamples, trainY_allSamples)
print('Best Estimator:\n',gsearch2b.best_estimator_)
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

accs = []

for train_index, test_index in sss.split(trainX_allSamples, trainY_allSamples):

    gsearch2b.best_estimator_.fit(trainX_allSamples.loc[train_index],trainY_allSamples[train_index])

    acc = accuracy_score(gsearch2b.best_estimator_.predict(trainX_allSamples.loc[test_index]),trainY_allSamples[test_index])

    print('Accuracy Score:', acc)

    accs += [acc]

print('Average Accuracy:',sum(accs)/len(accs))
gsearch2b.best_estimator_.fit(trainX_allSamples,trainY_allSamples)



testX_allSamples = testDF_User_AllSampleProps.reset_index().drop(columns=['id'])

textPreds_allSamples = gsearch2b.best_estimator_.predict(testX_allSamples)

pd.DataFrame({'idx':testX_allSamples.index,'user':textPreds_allSamples},index=testX_allSamples.index).to_csv('submission.csv',index=False)