import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # plot
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.
sp1 = pd.read_csv('../input/Raw_data.csv')
sp1.head()
seqLen =[len(s) for s in sp1['sequence']]
set(seqLen)
trainX,testX,trainY,testY = train_test_split(sp1['sequence'].tolist(),sp1['label'].tolist(),test_size=0.3,random_state=42)
print('Train data : ',len(trainX),len(trainY))
print('Test data : ',len(testX),len(testY))
cv2 = CountVectorizer(analyzer='char',ngram_range=(2,2),min_df=0.10,max_df=0.8)
# cv2 since bigram....
trainBiGram = cv2.fit_transform(trainX).toarray()
testBiGram = cv2.transform(testX).toarray()
countBiGram = pd.DataFrame(trainBiGram,columns=cv2.get_feature_names())
countBiGram['class'] = trainY
meltBiGram = pd.melt(countBiGram, id_vars='class')
fig, ax = plt.subplots(figsize=(12,9))
# for a bigger figure
sns.boxplot(x='variable', y='value', hue='class', data=meltBiGram)
trainBiGramBinary = trainBiGram.copy()
trainBiGramBinary[trainBiGramBinary>0]=1
testBiGramBinary = testBiGram.copy()
testBiGramBinary[testBiGramBinary>0]=1
print (trainBiGramBinary.shape)
print (testBiGramBinary.shape)
cvn = CountVectorizer(analyzer='char',ngram_range=(2,5),min_df=0.10,max_df=0.8)
# cvn for N-gram...
trainNGram = pd.DataFrame(cvn.fit_transform(trainX).toarray(),columns=cvn.get_feature_names())
testNGram = pd.DataFrame(cvn.transform(testX).toarray(),columns=cvn.get_feature_names())
trainNGram.describe()
nGramSum=trainNGram.sum()
nGramSum = nGramSum.sort_values( ascending=False)
nGramTop = pd.DataFrame(trainNGram[nGramSum.index[:20]])
nGramTop['class'] = trainY
meltNGram = pd.melt(nGramTop, id_vars='class')
fig, ax = plt.subplots(figsize=(12,9))
sns.boxplot(x='variable', y='value', hue='class', data=meltNGram)
trainNGramBinary = trainNGram.copy()
trainNGramBinary[trainNGramBinary>0]=1
testNGramBinary = testNGram.copy()
testNGramBinary[testNGramBinary>0]=1
print (trainNGramBinary.shape)
print (testNGramBinary.shape)
rfBi1 = RandomForestClassifier(n_jobs=1,random_state=42,n_estimators=2500,oob_score=True,min_samples_split=5,min_samples_leaf=2)
rfBi1.fit(trainBiGramBinary,trainY)
rfBi1Prob = rfBi1.predict_proba(testBiGramBinary)
rfBi1Acc = rfBi1.score(testBiGramBinary,testY)
print (rfBi1Acc)
rfBi2 = RandomForestClassifier(n_jobs=1,random_state=42,n_estimators=2500,oob_score=True,min_samples_split=5,min_samples_leaf=2)
rfBi2.fit(trainBiGram,trainY)
rfBi2Prob = rfBi2.predict_proba(testBiGram)
rfBi2Acc = rfBi2.score(testBiGram,testY)
print (rfBi2Acc)
rfN1 =RandomForestClassifier(n_jobs=1,random_state=42,n_estimators=2500,oob_score=True,min_samples_split=4,min_samples_leaf=1)
rfN1.fit(trainNGramBinary,trainY)
rfN1Prob = rfN1.predict_proba(testNGramBinary)
rfN1Acc = rfN1.score(testNGramBinary,testY)
print (rfN1Acc)
rfN2 =RandomForestClassifier(n_jobs=1,random_state=42,n_estimators=2500,oob_score=True,min_samples_split=2,min_samples_leaf=1)
rfN2.fit(trainNGram,trainY)
rfN2Prob = rfN2.predict_proba(testNGramBinary)
rfN2Acc = rfN2.score(testNGram,testY)
print (rfN2Acc)
accuracyTable = pd.DataFrame({'Mean Accuracy':[rfBi1Acc,rfBi2Acc,rfN1Acc,rfN2Acc]},index=['bigram Binary','bigram Count','ngram Binary','ngram Count'])
accuracyTable['Mean Accuracy'] = accuracyTable['Mean Accuracy']*100
# for easy reading
accuracyTable
probs = [rfBi1Prob,rfBi2Prob,rfN1Prob,rfN2Prob]
names = ['bigram Binary','bigram Count','ngram Binary','ngram Count']
fig, ax = plt.subplots(figsize=(15,12))
for i,cp in enumerate(probs):
    fpr, tpr, thresholds = roc_curve(testY, cp[:,0], pos_label='binding site')
    rocAuc = round(auc(fpr, tpr),4)
    plt.plot(fpr, tpr, lw=2,label= "{} (AUC = {})".format(names[i],rocAuc))
plt.plot([0, 1], [0, 1], color='black', lw=2.5, linestyle='--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show(block=False)