import time

import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import networkx as nx



from sklearn import model_selection

from sklearn import linear_model



from sklearn.metrics import accuracy_score

from sklearn.metrics import log_loss

from sklearn.metrics import roc_auc_score



from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer



matplotlib.style.use('fivethirtyeight')



import matplotlib

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline  



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
trainDF = pd.read_csv('../input/train.csv')

testDF = pd.read_csv('../input/test.csv')

trainDF = trainDF.dropna(how="any").reset_index(drop=True)

trainDF.head()
featureExtractionStartTime = time.time()





maxNumFeatures = 10000





q1_list = np.array(trainDF['question1']).tolist()

q2_list = np.array(trainDF['question2']).tolist()



cl = TfidfVectorizer(max_df=1000, min_df=1, max_features=maxNumFeatures, 

                                      analyzer='word', ngram_range=(1,4), stop_words = 'english', 

                                      binary=False, lowercase=True)



q1_matrix = cl.fit_transform(q1_list)

q2_matrix = cl.fit_transform(q2_list)



y = np.array(trainDF.ix[:,'is_duplicate'])



featureExtractionDurationInMinutes = (time.time()-featureExtractionStartTime)/60.0

print("feature extraction took %.2f minutes" % (featureExtractionDurationInMinutes))
from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import MultinomialNB

crossValidationStartTime = time.time()



numCVSplits = 8

numSplitsToBreakAfter = 4



X = (q1_matrix != q2_matrix).astype(int) + q1_matrix.multiply(q2_matrix)



logisticRegressor = linear_model.LogisticRegression(C= 0.1, solver='sag')

#RFC = RandomForestClassifier(n_estimators = 5)

#knn = KNeighborsClassifier(n_neighbors=4)





logRegAccuracy = []

logRegLogLoss = []

logRegAUC = []



print('---------------------------------------------')

stratifiedCV = model_selection.StratifiedKFold(n_splits=numCVSplits, random_state=2)

flag = 0

for k, (trainInds, validInds) in enumerate(stratifiedCV.split(X, y)):

    flag+=1

    print (flag)

    foldTrainingStartTime = time.time()



    X_train_cv = X[trainInds,:]

    X_valid_cv = X[validInds,:]



    y_train_cv = y[trainInds]

    y_valid_cv = y[validInds]

    

    logisticRegressor.fit(X_train_cv, y_train_cv)

   

    y_train_hat =  logisticRegressor.predict_proba(X_train_cv)[:,1]

    y_valid_hat =  logisticRegressor.predict_proba(X_valid_cv)[:,1]

    



    

    

    logRegAccuracy.append(accuracy_score(y_valid_cv, y_valid_hat > 0.5))

    logRegLogLoss.append(log_loss(y_valid_cv, y_valid_hat))

    logRegAUC.append(roc_auc_score(y_valid_cv, y_valid_hat))

    

    foldTrainingDurationInMinutes = (time.time()-foldTrainingStartTime)/60.0

    print('fold %d took %.2f minutes: accuracy = %.3f, log loss = %.4f, AUC = %.3f' % (k+1,

             foldTrainingDurationInMinutes, logRegAccuracy[-1],logRegLogLoss[-1],logRegAUC[-1]))



    if (k+1) >= numSplitsToBreakAfter:

        break





    crossValidationDurationInMinutes = (time.time()-crossValidationStartTime)/60.0

    

    print('---------------------------------------------')

    print('cross validation took %.2f minutes' % (crossValidationDurationInMinutes))

    print('mean CV: accuracy = %.3f, log loss = %.4f, AUC = %.3f' % (np.array(logRegAccuracy).mean(),

                                                                 np.array(logRegLogLoss).mean(),

                                                                 np.array(logRegAUC).mean()))

    print('---------------------------------------------')
#по аналогии сделали такую штучку))



matplotlib.rcParams['font.size'] = 14

matplotlib.rcParams['figure.figsize'] = (10,10)



plt.figure(); 

sns.kdeplot(y_valid_hat[y_valid_cv==0], shade=True, color="b", bw=0.01)

sns.kdeplot(y_valid_hat[y_valid_cv==1], shade=True, color="g", bw=0.01)

plt.legend(['non duplicate','duplicate'],fontsize=24)

plt.title('Validation Accuracy = %.3f, Log Loss = %.4f, AUC = %.3f' %(logRegAccuracy[-1],

                                                                      logRegLogLoss[-1],

                                                                      logRegAUC[-1]))



plt.xlabel('Prediction'); plt.ylabel('Probability Density'); plt.xlim(-0.01,1.01)



numFeaturesToShow = 30



sortedCoeffients = np.sort(logisticRegressor.coef_)[0]

featureNames = cl.get_feature_names()



sortedFeatureNames = [featureNames[x] for x in list(np.argsort(logisticRegressor.coef_)[0])]



matplotlib.rcParams['font.size'] = 14

matplotlib.rcParams['figure.figsize'] = (10,12)



plt.figure()

plt.suptitle('Feature Importance',fontsize=24)

ax = plt.subplot(1,2,1); plt.title('top non duplicate predictors'); 

plt.xlabel('minus logistic regression coefficient')

ax.barh(range(numFeaturesToShow), -sortedCoeffients[:numFeaturesToShow][::-1], align='center'); 



plt.ylim(-1,numFeaturesToShow); ax.set_yticks(range(numFeaturesToShow)); 

ax.set_yticklabels(sortedFeatureNames[:numFeaturesToShow][::-1],fontsize=20)



ax = plt.subplot(1,2,2); plt.title('top duplicate predictors'); 

plt.xlabel('logistic regression coefficient')

ax.barh(range(numFeaturesToShow), sortedCoeffients[-numFeaturesToShow:], align='center'); 

plt.ylim(-1,numFeaturesToShow); ax.set_yticks(range(numFeaturesToShow)); 

ax.set_yticklabels(sortedFeatureNames[-numFeaturesToShow:],fontsize=20)

logisticRegressor = linear_model.LogisticRegression(C=1, solver='sag', 

                                                    class_weight={1: 0.4, 0: 1.34})

                                                    

logisticRegressor.fit(X, y)
testPredictionStartTime = time.time()



testDF.ix[testDF['question1'].isnull(),['question1','question2']] = 'random empty question'

testDF.ix[testDF['question2'].isnull(),['question1','question2']] = 'random empty question'

q1_testmat  = cl.transform(testDF.ix[:,'question1'])   

q2_testmat  = cl.transform(testDF.ix[:,'question2'])   



X_test = (q1_testmat != q2_testmat).astype(int) + q1_testmat.multiply(q2_testmat)



seperators= [750000,1500000]

testPredictions1 = logisticRegressor.predict_proba(X_test[:seperators[0],:])[:,1]

testPredictions2 = logisticRegressor.predict_proba(X_test[seperators[0]:seperators[1],:])[:,1]

testPredictions3 = logisticRegressor.predict_proba(X_test[seperators[1]:,:])[:,1]

testPredictions = np.hstack((testPredictions1,testPredictions2,testPredictions3))



matplotlib.rcParams['font.size'] = 14

matplotlib.rcParams['figure.figsize'] = (9,9)



plt.figure(); 

plt.subplot(2,1,1); sns.kdeplot(y_valid_hat, shade=True, color="b", bw=0.01); 

plt.ylabel('Probability Density'); plt.xlim(-0.01,1.01)

plt.title('mean valid prediction = ' + str(np.mean(y_valid_hat)))

plt.subplot(2,1,2); sns.kdeplot(testPredictions, shade=True, color="b", bw=0.01);

plt.xlabel('Prediction'); plt.ylabel('Probability Density'); plt.xlim(-0.01,1.01)

plt.title('mean test prediction = ' + str(np.mean(testPredictions)))



testPredictionDurationInMinutes = (time.time()-testPredictionStartTime)/60.0

print('predicting on test took %.2f minutes' % (testPredictionDurationInMinutes))
submissionName = 'shallowBenchmark_'



submission = pd.DataFrame()

submission['test_id'] = testDF['test_id']

submission['is_duplicate'] = (testPredictions)# + prev['is_duplicate'])/2

submission.to_csv(submissionName + '.csv', index=False)