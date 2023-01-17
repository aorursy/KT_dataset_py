
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np 
import pandas as pd 


trainData = pd.read_csv('../input/dataset/train.csv')

trainData.head(5)
trainData['Open Date'] = pd.to_datetime(trainData['Open Date'], format='%m/%d/%Y')   
trainData['OpenDays']=""

dateLastTrain = pd.DataFrame({'Date':np.repeat(['01/01/2018'],[len(trainData)]) })
dateLastTrain['Date'] = pd.to_datetime(dateLastTrain['Date'], format='%m/%d/%Y')  

trainData['OpenDays'] = dateLastTrain['Date'] - trainData['Open Date']
trainData['OpenDays'] = trainData['OpenDays'].astype('timedelta64[D]').astype(int)

trainData = trainData.drop('Open Date', axis=1)
cityPerc = trainData[["City Group", "revenue"]].groupby(['City Group'],as_index=False).mean()

sns.barplot(x='City Group', y='revenue', data=cityPerc)
cityPerc = trainData[["City", "revenue"]].groupby(['City'],as_index=False).mean()

newDF = cityPerc.sort_values(["revenue"],ascending= False)
sns.barplot(x='City', y='revenue', data=newDF.head(10))
cityPerc = trainData[["City", "revenue"]].groupby(['City'],as_index=False).mean()
newDF = cityPerc.sort_values(["revenue"],ascending= True)
sns.barplot(x='City', y='revenue', data=newDF.head(10))
cityPerc = trainData[["Type", "revenue"]].groupby(['Type'],as_index=False).mean()
sns.barplot(x='Type', y='revenue', data=cityPerc)
cityPerc = trainData[["Type", "OpenDays"]].groupby(['Type'],as_index=False).mean()
sns.barplot(x='Type', y='OpenDays', data=cityPerc)
trainData = trainData.drop('Id', axis=1)
trainData = trainData.drop('Type', axis=1)
citygroupDummy = pd.get_dummies(trainData['City Group'])
trainData = trainData.join(citygroupDummy)


trainData = trainData.drop('City Group', axis=1)

trainData = trainData.drop('City', axis=1)

tempRev = trainData['revenue']
trainData = trainData.drop('revenue', axis=1)


trainData = trainData.join(tempRev)
trainData.head(10)
from sklearn.model_selection import train_test_split

X, y = trainData.iloc[:, 1:40].values, trainData.iloc[:, 40].values

X_trainForBestFeatures, X_testForBestFeatures, y_trainForBestFeatures, y_testForBestFeatures =\
    train_test_split(X, y, 
                     test_size=0.3, 
                     random_state=0, 
                )
    
X_trainForBestFeatures.shape, X_testForBestFeatures.shape, y_trainForBestFeatures.shape, y_testForBestFeatures.shape
y[:20]
y_trainForBestFeatures[:20]
from sklearn.ensemble import RandomForestClassifier

#To label our features form best to wors 
feat_labels = trainData.columns[1:40]

forest = RandomForestClassifier(n_estimators=500,
                                random_state=1)
forest.fit(X_trainForBestFeatures, y_trainForBestFeatures)



importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X_trainForBestFeatures.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))
    

plt.title('Feature Importance')
plt.bar(range(X_trainForBestFeatures.shape[1]), 
        importances[indices],
        align='center')

plt.xticks(range(X_trainForBestFeatures.shape[1]), 
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_trainForBestFeatures.shape[1]])
plt.tight_layout()

plt.show()

trainData[feat_labels[indices[0:39]]].head()
import numpy as numpy 
openDaysLog = trainData[feat_labels[indices[0:1]]].apply(numpy.log)
openDaysLog.head()
bestDataFeaturesTrain = trainData[feat_labels[indices[1:19]]]

#insert after takeing log of OpenDays feature.
bestDataFeaturesTrain.insert(loc=0, column='OpenDays', value=openDaysLog)

bestDataFeaturesTrain.head()
# take the natural logarithm of the 'revenue' column in order to make it more easy for model to predict
y = trainData['revenue'].apply(numpy.log)

from sklearn.model_selection import train_test_split

X, y = trainData.iloc[:, 1:40].values, trainData.iloc[:, 40].values

X_train, X_test, y_train, y_test =\
    train_test_split(bestDataFeaturesTrain, y, 
                     test_size=0.3, 
                     random_state=0, 
                )



    
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.preprocessing import StandardScaler

sc = StandardScaler(with_std  = True ,with_mean = True, copy = True)
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

X_train_std[:1]
from sklearn.decomposition import PCA,KernelPCA

pca = PCA(n_components=2,svd_solver='full')
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
pca.explained_variance_ratio_

kpca = KernelPCA(kernel="rbf", gamma=1)
X_kpca_train = kpca.fit_transform(X_train_pca)
X_kpca_test = kpca.transform(X_test_pca)



X_train_pca[:1]
fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(10,5))
ax[0].scatter(X_train_pca[:, 0], X_train_pca[:, 1],color='red',marker='o')
ax[1].scatter(X_kpca_train[:, 0], X_kpca_train[:, 1])
ax[0].set_xlabel('Before RBF')
ax[1].set_yticks([])
ax[1].set_xlabel('After RBF')
X_test_pca[:1]
X_train.head()
X_train_std[:1]
X_test.head()
X_test.head()
y_test[:5]

import numpy
from sklearn import linear_model
cls = RandomForestRegressor(n_estimators=250, criterion='mse', max_depth=30)#cls = RandomForestRegressor(n_estimators=150)

cls.fit(X_kpca_train, y_train)#We are training the model with RBF'ed data

scoreOfModel = cls.score(X_kpca_train, y_train)


print("Score is calculated as: ",scoreOfModel)
pred = cls.predict(X_kpca_test)

pred