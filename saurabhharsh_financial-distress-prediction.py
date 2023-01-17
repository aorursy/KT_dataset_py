# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv("../input/Financial Distress.csv")

df.head()

# Any results you write to the current directory are saved as output.
print(df.Company.unique().shape) # 422 company numbers
print(df.x80.unique().shape)

corrDf = df.drop(labels = ['Time','Company'], axis = 1).corr().abs()

corrDf.sort_values(by = 'Financial Distress', inplace=True, ascending = False)

corrColumns = corrDf.drop(labels=['x80']).index.values #[corrDf['Financial Distress'] > 0.01]

corrDf.head(n = 10)
reducedDf = df[corrColumns]

reducedDf.head()
from sklearn.preprocessing import RobustScaler, StandardScaler

scaler = StandardScaler()

trainArray = reducedDf.as_matrix()

scaledData = trainArray

scaledData[:,1:] = scaler.fit_transform(trainArray[:,1:])

print(np.sum(scaledData[:,0] > -0.5)) # 3281 healthy

print(np.sum(scaledData[:,0] <= -0.5)) # 391 distressed cases

import seaborn as sns

sns.boxplot(data = scaledData[:,1:])
from sklearn.linear_model import LinearRegression

from sklearn import svm

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score, recall_score, f1_score

import matplotlib.pyplot as plt



uniformData = scaledData

X = uniformData[:,1:]

y = uniformData[:,0]

y_discrete = (uniformData[:,0] < -0.5).astype(int)



mdl = LinearRegression()



thresholds = np.arange(-1.5,-0.5,0.1) # Try some thresholds

precisions = np.zeros_like(thresholds)

recalls = np.zeros_like(thresholds)

f1_scores = np.zeros_like(thresholds)

predicted_metric = cross_val_predict(mdl, X, y, cv = 5)

fig, ax = plt.subplots()

for i in range(len(thresholds)):

    predicted = (predicted_metric < thresholds[i]).astype(int)

    precisions[i] = precision_score(y_discrete, predicted)

    recalls[i] = recall_score(y_discrete, predicted)

    f1_scores[i] = f1_score(y_discrete, predicted)

    plt.scatter(recalls[i], precisions[i])

    ax.annotate('%0.3f' % (f1_scores[i]),(recalls[i], precisions[i]))

plt.xlabel('Recall')    

plt.ylabel('Precision')



mdl = svm.SVR()

thresholds = np.arange(-0.5,0.5,0.1) # Try some thresholds

precisions = np.zeros_like(thresholds)

recalls = np.zeros_like(thresholds)

f1_scores = np.zeros_like(thresholds)

predicted_metric = cross_val_predict(mdl, X, y, cv = 5)

fig, ax = plt.subplots()

for i in range(len(thresholds)):

    predicted = (predicted_metric < thresholds[i]).astype(int)

    precisions[i] = precision_score(y_discrete, predicted)

    recalls[i] = recall_score(y_discrete, predicted)

    f1_scores[i] = f1_score(y_discrete, predicted)

    plt.scatter(recalls[i], precisions[i])

    ax.annotate('%0.3f' % (f1_scores[i]),(recalls[i], precisions[i]))

plt.xlabel('Recall')    

plt.ylabel('Precision')
from sklearn.model_selection import StratifiedKFold


from sklearn.metrics import precision_recall_curve

import matplotlib.pyplot as plt



def cvClassifier(mdl, X, y, color, name, confMat = False, confMatNormalize = True):

    skf = StratifiedKFold(n_splits = 5)

    predicted_prob = np.zeros_like(y, dtype = float)

    for train,test in skf.split(X, y):

        mdl.fit(X[train,:],y[train])

        y_prob = mdl.predict_proba(X[test,:])

        predicted_prob[test] = y_prob[:,1] #The second class 1 from 0,1 is the one to be predicted

    

    precision, recall, thresholds = precision_recall_curve(y, predicted_prob)

    plt.plot(recall, precision, color=color,label = name)

    plt.xlabel('Recall')

    plt.ylabel('Precision')

    plt.ylim([0.0, 1.05])

    plt.xlim([0.0, 1.0])

    plt.title('2-class Precision-Recall curve')

    plt.legend()

    

    fscore = 2*(precision*recall)/(precision + recall)

    maxFidx = np.nanargmax(fscore)

    selP = precision[maxFidx]

    selRecall = recall[maxFidx]

    selThreshold = thresholds[maxFidx]



    return predicted_prob, selP, selRecall, fscore[maxFidx], selThreshold
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y_discrete, test_size=0.3, stratify=y_discrete, random_state=42)



mdl = LogisticRegression(class_weight = 'balanced')

out1 = cvClassifier(mdl, X_train, y_train, 'y','Logit')



mdl = svm.SVC(kernel = 'linear', C=0.025, class_weight = 'balanced', probability = True)

out2 = cvClassifier(mdl, X_train, y_train, 'b','LinearSVC')



mdl = RandomForestClassifier(class_weight = 'balanced', n_estimators=1000)

out3 = cvClassifier(mdl, X_train, y_train, 'r','RandomForest')



mdl = svm.SVC(C=0.5, class_weight = 'balanced', probability = True)

out4 = cvClassifier(mdl, X_train, y_train, 'g','RBFSVC')
results = [out1, out2, out3, out4]

mdlNames = ['Logit','LinearSVC','RF','RBFSVC']

fig, ax = plt.subplots()

for i in range(len(results)):

    ax.scatter(results[i][2],results[i][1])

    ax.annotate('%s %0.4f' % (mdlNames[i], results[i][3]),(results[i][2],results[i][1]))

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.ylim([0.3, 0.5])

plt.xlim([0.35, 0.65])
threshold = out2[4]

y_pred = (out2[0] > threshold).astype(int)

from sklearn.metrics import accuracy_score

acc = accuracy_score(y_train, y_pred)

print('Accuracy %0.2f' % (acc))

print('Threshold %0.3f' % (threshold))
mdl = svm.SVC(kernel = 'linear', C=0.025, class_weight = 'balanced', probability = True)

out2 = cvClassifier(mdl, X_train, y_train, 'b','LinearSVC')



y_testp = (mdl.predict_proba(X_test)[:,1] > threshold).astype(int)

acc = accuracy_score(y_test, y_testp)

print('Accuracy %0.2f' % (acc))

print('Precision %0.2f' % (precision_score(y_test,y_testp)))

print('Recall %0.2f' % (recall_score(y_test,y_testp)))