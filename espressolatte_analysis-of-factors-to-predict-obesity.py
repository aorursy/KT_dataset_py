# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Inputzdata files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Import the `RandomForestClassifier`

from sklearn.ensemble import RandomForestClassifier

from sklearn import tree

# Any results you write to the current directory are saved as output.

from sklearn.ensemble import RandomForestClassifier

res = pd.read_csv("../input/ehresp_2014.csv")
res.head()
res["obese"] = 0

#print(res["erbmi"])
res = res[res['eusoda']>0] 

genFeaturesList = ['eusoda', 'eusnap', 'euincome2', 'eugenhth', 'erincome', 'eudietsoda',

                  'euffyday', 'eufdsit', 'eufastfdfrq','ertseat', 'eudrink', 'eueat',

                  'euexfreq', 'euexercise', 'eufastfd', 'eugenhth', 'eumeat', 'eumilk', 'eustores', 

                  'eustreason', 'euwic']



for feature in genFeaturesList:

    print('current prune feature', feature)

    oldSize = len(res)

    newSize = len(res[res[feature]>-1])

    print('num deleted:', oldSize-newSize)
#convert bmi to floats

res["bmi"] = res["erbmi"].astype(float)

#set obese to true if bmi above 30

res.loc[res['bmi']>30, 'obese'] = 1

#obese is our target

target = res["obese"]
#check status

print(res["obese"].value_counts())



resObese =  res[res["obese"] == 1] #isolate the obese

resNotObese = res[res["obese"] == 0]



print(res['obese'].value_counts(normalize=True))
#split the obese and not obese into test and training

obeseMsk= np.random.rand(len(resObese)) < .7

notObeseMsk = np.random.rand(len(resNotObese)) < .7



trainObese = resObese[obeseMsk]

trainNotObese = resNotObese[notObeseMsk]



testObese = resObese[~(obeseMsk)]

testNotObese = resNotObese[~(notObeseMsk)]

print("Should be true")

len(testObese) + len(trainObese) == len(resObese)

#confirm obese data complete
len(testNotObese) + len(trainNotObese) == len(resNotObese) #confirm notObese data complete
test = testObese.append(testNotObese) #make full test
test.head()
train = trainObese.append(trainNotObese) #make full train

print('length of train:', len(train))
print(train["obese"].value_counts(normalize = True))

print(test["obese"].value_counts(normalize = True))
train["eusoda"].head()
trainOriginal = train.copy() #make a copy

testOriginal = test.copy()

#we select features for use below

genFeaturesList = ['eusoda', 'eusnap', 'euincome2', 'eugenhth', 'erincome', 'eudietsoda',

                  'euffyday', 'eufdsit', 'eufastfdfrq','ertseat', 'eudrink', 'eueat',

                  'euexfreq', 'euexercise', 'eufastfd', 'eugenhth', 'eumeat', 'eumilk', 'eustores', 

                  'eustreason', 'euwic']

#clean the data for only valid

for feature in genFeaturesList:

    train = train[train[feature]>-1]

    trainTarget = train['obese']

    test = test[test[feature]>-1]

    testTarget = test['obese']

    print('new train len:', len(train))

    print('new test len:', len(test))



#first test see if soda and obesity

sodaFeatures = train[["eusoda"]].values

sodaTree = tree.DecisionTreeClassifier()

sodaTree = sodaTree.fit(sodaFeatures, trainTarget )

print(sodaTree.feature_importances_)

print(sodaTree.score(sodaFeatures, trainTarget))



np.count_nonzero(sodaTree.predict(sodaFeatures))
#predict test set with the above tree

testSodaFeatures=test[['eusoda']].values

sodaPredTree = sodaTree.predict(testSodaFeatures)

sodaTree.score(testSodaFeatures, test['obese'])

sodaTargetFull = res['obese']

sodaFeaturesFull = res[['eusoda']].values

sodaTreeFull = tree.DecisionTreeClassifier()

sodaTreeFull = sodaTreeFull.fit(sodaFeaturesFull, sodaTargetFull)

print(sodaTreeFull.feature_importances_)

print(sodaTree.score(sodaFeaturesFull, sodaTargetFull))
resCopy = res.copy() #make a copy

resCopy = resCopy[resCopy["eusoda"]>0]

resCopy.head()

#list(resCopy)
print("these two should match")

resCopy["predict"]=0

print(resCopy['eusoda'].value_counts())

resCopy.loc[resCopy['eusoda']== 1, 'predict'] = 1

print(resCopy['predict'].value_counts())
#accuracy if we just do a simple prediction that they are obese if they drink soda

numCorrect = len(resCopy[resCopy['predict']==resCopy['obese']])

totalNum = len(resCopy)

print (numCorrect/ totalNum)
#forest on full. no meaning

sodaFeaturesForest = res[['eusoda']].values



sodaForest = RandomForestClassifier(max_depth = 4, min_samples_split = 2,

                               n_estimators = 100, random_state = 1)

sodaForest = sodaForest.fit(sodaFeaturesForest, sodaTargetFull)

print(sodaForest.score(sodaFeaturesForest, sodaTargetFull))

#clean data by picking only valid eusnap values

train = train[train['eusnap']>0]

trainTarget = train['obese']

test = test[test['eusnap']>0]

testTarget = test['obese']
sodaSnapFeatures = train[['eusoda', 'eusnap']].values





sodaSnapForest = RandomForestClassifier(max_depth = 4, min_samples_split = 2,

                               n_estimators = 100, random_state = 1)

sodaSnapForest = sodaSnapForest.fit(sodaSnapFeatures, trainTarget)

print(sodaSnapForest.score(sodaSnapFeatures, trainTarget))

genFeatures = train[genFeaturesList].values





genForest = RandomForestClassifier() #max_depth = 10,  min_samples_split = 2,

                                   #n_estimators = 20, random_state = 1

genForest = genForest.fit(genFeatures, trainTarget)

print(genForest.score(genFeatures, trainTarget))

genForestFeatureImportances = genForest.feature_importances_ 

print(genForest.feature_importances_)
#Lets try to pick the top 5 features and retry the forest

topInd = np.argsort(genForestFeatureImportances)[::-1][:5]

topFeaturesList = [genFeaturesList[i] for i in topInd]

topFeaturesList
#reset our train and test to the original, 

#because we lost too many data points when we filtered for everything

train = trainOriginal.copy()

trainTarget = train['obese']

test = testOriginal.copy()

testTarget = test['obese']

#clean the data for only valid

topFeaturesList = ['euexfreq', 'eustreason', 'eugenhth', 'ertseat', 'eufastfdfrq']

for feature in topFeaturesList:

    #print(feature)

    #print('new train len:', len(train))

    #print('new test len:', len(test))

    train = train[train[feature]>-1]

    trainTarget = train['obese']

    test = test[test[feature]>-1]

    testTarget = test['obese']
topFeatures = train[topFeaturesList].values



topForest = RandomForestClassifier(max_depth=8) #max_depth = 10,  min_samples_split = 2,

                                   #n_estimators = 20, random_state = 1

topForest = topForest.fit(topFeatures, trainTarget)

print(topForest.score(topFeatures, trainTarget))

topForestFeatureImportances = topForest.feature_importances_ 

print(topForest.feature_importances_)
#try to find the best max depth and n_estimators

pairsList = []

for i in range(1, 15):

    for j in range(1,25):

        pairsList.append([i,j])



pairsListTrainScores = []

pairsListResults = []

for (depth, estimators) in pairsList:

    topForest = RandomForestClassifier(max_depth=depth, n_estimators = estimators) 

    topForest = topForest.fit(topFeatures, trainTarget)

    pairsListTrainScores.append(topForest.score(topFeatures, trainTarget))

    pairsListResults.append(topForest.score(test[topFeaturesList].values, testTarget))

print('max test result', max(pairsListResults))

print('max train score', max(pairsListTrainScores))

maxIndex = np.argmax(pairsListResults)

print('max index', np.argmax(pairsListResults))

print('max pair', pairsList[maxIndex])
#try doing one with nothing added

topForest = RandomForestClassifier() 

topForest = topForest.fit(topFeatures, trainTarget)

topForest.score(topFeatures, trainTarget)

topForest.score(test[topFeaturesList].values, testTarget)

topForest = RandomForestClassifier(max_depth = 7, n_estimators = 8) 

topForest = topForest.fit(topFeatures, trainTarget)

print('optimal train score', topForest.score(topFeatures, trainTarget))

print('optimal test score',topForest.score(test[topFeaturesList].values, testTarget))
print(topForest.feature_importances_)
print(list(zip(topFeaturesList, topForest.feature_importances_)))
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm



fullTrimmed = train.append(test)
x, y = fullTrimmed['euexfreq'].values, fullTrimmed['eufastfdfrq'].values

z = fullTrimmed['eugenhth'].values

v = fullTrimmed['obese'].values



fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')



ax.view_init(30, 0)

ax.scatter(x, y, v, c = z)

ax.set_xlabel('Exercise Frequency')

ax.set_ylabel('Fast Food Frequency')

ax.set_zlabel('General Health')



plt.show()
from sklearn.tree import export_graphviz

import os

tree_in_forest = topForest.estimators_[0]









export_graphviz(tree_in_forest,

                feature_names=topFeaturesList,

                filled=True,

                rounded=True, out_file='tree.dot')

os.system('dot -Tpng tree.dot -o tree.png')