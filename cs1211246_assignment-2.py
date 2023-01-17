%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# The competition datafiles are in the directory ../input
# Read competition data files:
traindata = pd.read_csv("../input/train.csv")
# remove label row
train = traindata.iloc[:,1:]
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(traindata.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

# reshape and change type of training data
trainnp = np.array(train).reshape((-1,1,28,28)).astype(float)
testnp = np.array(test).reshape((-1,1,28,28)).astype(np.int)

trainn = trainnp.copy()
testn = testnp.copy()
# images
from skimage import data,io,filters,exposure
from skimage.morphology import skeletonize
from skimage.morphology import dilation
testi = 15
thresh = filters.threshold_otsu(trainnp[testi][0])
binary = trainnp[testi][0] > (thresh)
trainn[testi][0] = dilation(skeletonize(binary))
# plot the progress
ax1 = plt.subplot(1,3,1)
ax2 = plt.subplot(1,3,2)
ax3 = plt.subplot(1,3,3)
ax1.imshow(trainnp[testi][0], cmap=cm.binary)
ax1.set_title('Normal')
ax2.imshow(skeletonize(binary), cmap=cm.binary)
ax2.set_title('Skeleton')
ax3.imshow(trainn[testi][0], cmap=cm.binary)
ax3.set_title('Dilated')
plt.show()    
#reshape skeletondata
trainna = trainn.reshape(-1,784).astype(np.uint)
testna = testn.reshape(-1,784).astype(np.uint)    
# images
from skimage import data,io,filters,exposure
from skimage.morphology import skeletonize
from skimage.morphology import dilation

testi = 0

thresh = filters.threshold_otsu(trainnp[testi][0])
binary = trainnp[testi][0] > (thresh)
trainn[testi][0] = dilation(skeletonize(binary))

# plot the progress
ax1 = plt.subplot(1,3,1)
ax2 = plt.subplot(1,3,2)
ax3 = plt.subplot(1,3,3)
ax1.imshow(trainnp[testi][0], cmap=cm.binary)
ax1.set_title('Normal')
ax2.imshow(binary, cmap=cm.binary)
ax2.set_title('Altered')
ax3.imshow(trainn[testi][0], cmap=cm.binary)
ax3.set_title('Dilated')
plt.show()    

#reshape skeletondata
trainna = trainn.reshape(-1,784).astype(np.uint)
testna = testn.reshape(-1,784).astype(np.uint)

    
print('Done!')
# images
from skimage import data,io,filters
from skimage.morphology import skeletonize, dilation

# skeletonise all training data
#0 to 42000
for ind in range(0,28000):
    thresh = filters.threshold_otsu(trainnp[ind][0])
    binary = trainnp[ind][0] > (thresh)
    trainn[ind][0] = dilation(skeletonize(binary))
    thresh = filters.threshold_otsu(testnp[ind][0])
    binary = testnp[ind][0] > (thresh)
    testn[ind][0] = dilation(skeletonize(binary))
    
for ind in range(28001,42000):
    thresh = filters.threshold_otsu(trainnp[ind][0])
    binary = trainnp[ind][0] > (thresh)
    trainn[ind][0] = dilation(skeletonize(binary))
    
#reshape skeletondata
trainna = trainn.reshape(-1,784).astype(np.uint)
testna = testn.reshape(-1,784).astype(np.uint)
    
print('Done!')
# plot the progress
ax1 = plt.subplot(1,3,1)
ax2 = plt.subplot(1,3,2)
ax1.imshow(trainnp[231][0], cmap=cm.binary)
ax1.set_title('Default')
ax2.imshow(trainn[231][0], cmap=cm.binary)
ax2.set_title('Skeleton')
plt.show()
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.ensemble import AdaBoostClassifier as adaBoost
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.ensemble import BaggingClassifier as bagging
from sklearn.ensemble import VotingClassifier as vc
from sklearn import svm

X = pd.DataFrame(trainna)
y = traindata.loc[:,'label']
t = pd.DataFrame(testna)

print(y)

rfc = rf(n_estimators=200)
svmlin = svm.LinearSVC(loss = 'hinge', penalty = 'l2')
bagSVM = bagging(svmlin,n_estimators=500,bootstrap=True, max_samples = 0.6, max_features=0.6)
decisionStump = dt(criterion="entropy",max_depth=1)
boost = adaBoost(base_estimator = decisionStump,n_estimators = 1000)

vc1 = vc(estimators=[('RFC', rfc),('Bagged',bagSVM),('AdaBoost',boost)], voting='soft', weights=[1,2,2])#,('SVC',svc)])
vc1.fit(X,y)
print(vc1.predict(t))

vc1.fit(X.iloc[0:1000],y.iloc[0:1000])
print(vc1.score(X.iloc[1001:2000],y.iloc[1001:2000]))


#print(vc1.predict(t.iloc[0:1000]))

vc1 = vc(estimators=[('RFC', rfc),('Bagged',bagSVM),('AdaBoost',boost)], voting='soft', weights=[2,1,1])#,('SVC',svc)])
vc1.fit(X.iloc[0:1000],y.iloc[0:1000])
print(vc1.score(X.iloc[1001:2000],y.iloc[1001:2000]))
#print(vc1.predict(t.iloc[0:1000]))

#pd.Series(vc1.predict(t.iloc[0:1000])).to_csv("./skeletonbenchmark.csv")
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.ensemble import AdaBoostClassifier as adaBoost
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.ensemble import BaggingClassifier as bagging
from sklearn.ensemble import VotingClassifier as vc
from sklearn import svm

debug = 27999

# random forest
xvalues = [100,200,300,400,500,600,700,800]
yvalues = []

for a in xvalues:
    rfc = rf(n_estimators=100)
    rfc.fit(X,y)
    yvalues.append(rfc.score(X,y))
    
plt.plot(xvalues,yvalues)
    
#print(rfc.predict(t.iloc[0:debug]))
#rfc.fit(X,y)
#pd.Series(rfc.predict(t)).to_csv("./skeletonbenchmark.csv")

#svm
svmlin = svm.LinearSVC(loss = 'hinge', penalty = 'l2')
bagSVM = bagging(svmlin,n_estimators=10,bootstrap=True, max_samples = 0.6, max_features=0.6)
#bagSVM.fit(X,y)
#print(bagSVM.predict(t.iloc[0:debug]))

decisionStump = dt(criterion="entropy",max_depth=1)
boost = adaBoost(base_estimator = decisionStump,n_estimators = 80)
#boost.fit(X.iloc[0:debug],y.iloc[0:debug])
#print(boost.predict(t.iloc[0:debug]))


#print(ada.predict(t))
#svc = SVC()

vc1 = vc(estimators=[('RFC', rfc),('Bagged',bagSVM),('AdaBoost',boost)])#,('SVC',svc)])
#print('Voting classifier declared')

vc1.fit(X,y)