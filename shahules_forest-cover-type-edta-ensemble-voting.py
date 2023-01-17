import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
warnings.filterwarnings(message='DeprecationWarning',action='ignore')
train=pd.read_csv('../input/train.csv').copy()
test=pd.read_csv('../input/test.csv').copy()
train.shape
train.info()
train.describe()
train.isnull().sum()
train=train.drop(['Id'],axis=1)

for i,col in enumerate(train.iloc[:,:9].columns):
    ax=plt.subplot(3,3,i+1)
    sns.boxplot(x='Cover_Type',y=col,data=train,ax=ax) 
plt.subplots_adjust(top = 1)
plt.gcf().set_size_inches(13,10)
plt.figure(figsize=(9,7))
sns.heatmap(train.iloc[:,:10].corr(),annot=True,linewidths=.5)
plt.show()

pca=PCA(n_components=3)
pca_results=pca.fit_transform(train.drop('Cover_Type',axis=1))
tp, ax = plt.subplots(figsize=(20,15))
temp = ax.scatter(pca_results[:,0], pca_results[:,1], c=train.Cover_Type,s=700/pca_results[:,2] ,cmap=plt.cm.get_cmap('rainbow', 8))
tp.colorbar(temp)
plt.show()
plt.cm.get_cmap('rainbow', 8)
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure(figsize=(10,7))

#pca_results=pca.fit_transform(train.iloc[:,:9])
ax=Axes3D(fig)
ax.scatter(pca_results[:,0],pca_results[:,1],pca_results[:,2],cmap=plt.cm.get_cmap('rainbow', 8),c=train.Cover_Type)

X_train,X_test,y_train,y_test=train_test_split(train.drop('Cover_Type',axis=1),train['Cover_Type'])
rf=RandomForestClassifier(random_state=0)
rf.fit(X_train,y_train)
y_pre=rf.predict(X_test)
acc=metrics.accuracy_score(y_test,y_pre)
cv=cross_val_score(rf,train.drop('Cover_Type',axis=1),train['Cover_Type'],cv=5)
print("Mean cross validation score = ",cv.mean())
print('accuracy is ',acc)
print(metrics.classification_report(y_test,y_pre))
y_train.value_counts()/len(y_train)*100-pd.Series(y_pre).value_counts()/len(y_pre)*100
gb=GradientBoostingClassifier()
gb.fit(X_train,y_train)
ypre=gb.predict(X_test)
acc=metrics.accuracy_score(y_test,ypre)
cv=cross_val_score(gb,train.drop('Cover_Type',axis=1),train['Cover_Type'],cv=5)
print("cross val score is ",cv.mean())
print('accuracy = ',acc)
y_train.value_counts()/len(y_train)*100-pd.Series(ypre).value_counts()/len(ypre)*100
print(metrics.classification_report(y_test,ypre))

train.head()


# train.head()
train['HorizontalHydrology_HorizontalFire'] = (train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Fire_Points'])
train['Neg_HorizontalHydrology_HorizontalFire'] = (train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Fire_Points'])
train['HorizontalHydrology_HorizontalRoadways'] = (train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Roadways'])
train['Neg_HorizontalHydrology_HorizontalRoadways'] = (train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Roadways'])
train['HorizontalFire_Points_HorizontalRoadways'] = (train['Horizontal_Distance_To_Fire_Points']+train['Horizontal_Distance_To_Roadways'])
train['Neg_HorizontalFire_Points_HorizontalRoadways'] = (train['Horizontal_Distance_To_Fire_Points']-train['Horizontal_Distance_To_Roadways'])

train['Neg_Elevation_Vertical'] = train['Elevation']-train['Vertical_Distance_To_Hydrology']
train['Elevation_Vertical'] = train['Elevation']+train['Vertical_Distance_To_Hydrology']

train['mean_hillshade'] =  (train['Hillshade_9am']  + train['Hillshade_Noon'] + train['Hillshade_3pm'] ) / 3

train['Mean_HorizontalHydrology_HorizontalFire'] = (train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Fire_Points'])/2
train['Mean_HorizontalHydrology_HorizontalRoadways'] = (train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Roadways'])/2
train['Mean_HorizontalFire_Points_HorizontalRoadways'] = (train['Horizontal_Distance_To_Fire_Points']+train['Horizontal_Distance_To_Roadways'])/2

train['MeanNeg_Mean_HorizontalHydrology_HorizontalFire'] = (train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Fire_Points'])/2
train['MeanNeg_HorizontalHydrology_HorizontalRoadways'] = (train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Roadways'])/2
train['MeanNeg_HorizontalFire_Points_HorizontalRoadways'] = (train['Horizontal_Distance_To_Fire_Points']-train['Horizontal_Distance_To_Roadways'])/2

train['Slope2'] = np.sqrt(train['Horizontal_Distance_To_Hydrology']**2+train['Vertical_Distance_To_Hydrology']**2)
train['Mean_Fire_Hydrology_Roadways']=(train['Horizontal_Distance_To_Fire_Points'] + train['Horizontal_Distance_To_Hydrology'] + train['Horizontal_Distance_To_Roadways']) / 3
train['Mean_Fire_Hyd']=(train['Horizontal_Distance_To_Fire_Points'] + train['Horizontal_Distance_To_Hydrology']) / 2 

train["Vertical_Distance_To_Hydrology"] = abs(train['Vertical_Distance_To_Hydrology'])

train['Neg_EHDtH'] = train.Elevation-train.Horizontal_Distance_To_Hydrology*0.2

# test.head()
test['HorizontalHydrology_HorizontalFire'] = (test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Fire_Points'])
test['Neg_HorizontalHydrology_HorizontalFire'] = (test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Fire_Points'])
test['HorizontalHydrology_HorizontalRoadways'] = (test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Roadways'])
test['Neg_HorizontalHydrology_HorizontalRoadways'] = (test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Roadways'])
test['HorizontalFire_Points_HorizontalRoadways'] = (test['Horizontal_Distance_To_Fire_Points']+test['Horizontal_Distance_To_Roadways'])
test['Neg_HorizontalFire_Points_HorizontalRoadways'] = (test['Horizontal_Distance_To_Fire_Points']-test['Horizontal_Distance_To_Roadways'])

test['Neg_Elevation_Vertical'] = test['Elevation']-test['Vertical_Distance_To_Hydrology']
test['Elevation_Vertical'] = test['Elevation']+test['Vertical_Distance_To_Hydrology']

test['mean_hillshade'] =  (test['Hillshade_9am']  + test['Hillshade_Noon'] + test['Hillshade_3pm'] ) / 3

test['Mean_HorizontalHydrology_HorizontalFire'] = (test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Fire_Points'])/2
test['Mean_HorizontalHydrology_HorizontalRoadways'] = (test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Roadways'])/2
test['Mean_HorizontalFire_Points_HorizontalRoadways'] = (test['Horizontal_Distance_To_Fire_Points']+test['Horizontal_Distance_To_Roadways'])/2

test['MeanNeg_Mean_HorizontalHydrology_HorizontalFire'] = (test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Fire_Points'])/2
test['MeanNeg_HorizontalHydrology_HorizontalRoadways'] = (test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Roadways'])/2
test['MeanNeg_HorizontalFire_Points_HorizontalRoadways'] = (test['Horizontal_Distance_To_Fire_Points']-test['Horizontal_Distance_To_Roadways'])/2

test['Slope2'] = np.sqrt(test['Horizontal_Distance_To_Hydrology']**2+test['Vertical_Distance_To_Hydrology']**2)
test['Mean_Fire_Hydrology_Roadways']=(test['Horizontal_Distance_To_Fire_Points'] + test['Horizontal_Distance_To_Hydrology'] + test['Horizontal_Distance_To_Roadways']) / 3
test['Mean_Fire_Hyd']=(test['Horizontal_Distance_To_Fire_Points'] + test['Horizontal_Distance_To_Hydrology']) / 2 

test["Vertical_Distance_To_Hydrology"] = abs(test['Vertical_Distance_To_Hydrology'])

test['Neg_EHDtH'] = test.Elevation-test.Horizontal_Distance_To_Hydrology*0.2

scaler=MinMaxScaler()
scaler.fit(train.drop('Cover_Type',axis=1))
scaled_train=scaler.transform(train.drop('Cover_Type',axis=1))
X_train,X_test,y_train,y_test=train_test_split(scaled_train,train['Cover_Type'])
rf=RandomForestClassifier(random_state=0)
rf.fit(X_train,y_train)
y_pre=rf.predict(X_test)
acc=metrics.accuracy_score(y_test,y_pre)
cv=cross_val_score(rf,train.drop('Cover_Type',axis=1),train['Cover_Type'],cv=5)
print("Mean cross validation score = ",cv.mean())
print('accuracy is ',acc)
plt.figure(figsize=(10,15))
sns.barplot(y=train.drop('Cover_Type',axis=1).columns,x=rf.feature_importances_)
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 200, num = 8,endpoint=False)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(20, 100, num = 8)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print(random_grid)
#rf1=RandomForestClassifier()
#rf1_rand=RandomizedSearchCV(rf1,param_distributions=random_grid,n_iter=100,cv=3,n_jobs=-1)
#rf1_rand.fit(X_train,y_train)
#print(rf1_rand.best_params_)
accuracy=metrics.make_scorer(metrics.accuracy_score)
#gcv=GridSearchCV(RandomForestClassifier(),param_grid=params,scoring=accuracy,n_jobs=1,cv=5)
#gcv.fit(X_train,y_train)
#gcv.best_score_
#gcv.best_params_
rf=RandomForestClassifier(bootstrap=False,
 max_depth= 30,
 max_features= 'sqrt',
 min_samples_split= 3,
 n_estimators= 550,
 criterion='gini')

rf.fit(X_train,y_train)
y_pre=rf.predict(X_test)
cv=cross_val_score(rf,X_test,y_test,cv=5)
print('accuracy score is ',metrics.accuracy_score(y_test,y_pre))
print('cv score is',cv.mean())
rf.score(X_train,y_train)
ex=ExtraTreesClassifier(n_estimators=950,random_state=0,max_features='sqrt',min_samples_split=3)
ex.fit(X_train,y_train)
y_pre=ex.predict(X_test)
cv=cross_val_score(ex,X_test,y_test,cv=5)
print('accuracy score is ',metrics.accuracy_score(y_test,y_pre))
print('cv score is',cv.mean())
ex.score(X_train,y_train)
ex.score(X_test,y_test)


train_acc=[]
test_acc=[]
for this_c in [.1,1,10,15,20]:
        clf = SVC(kernel = 'rbf', gamma = .5 ,C = this_c).fit(X_train, y_train)
        train_acc.append(clf.score(X_train,y_train))
        test_acc.append(clf.score(X_test,y_test))
        
        
c=[.1,1,10,15,20]
plt.figure()
plt.plot(c,train_acc,color='r')
plt.plot(c,test_acc,color='b')
plt.gca().set_xlabel('C')
plt.gca().set_ylabel('accuracy')
plt.gca().legend(['train','test'])
plt.show()
train_acc=[]
test_acc=[]
for this_g in [.01,.1,.5,1,5]:
        clf = SVC(kernel = 'rbf', gamma = this_g ,C =15 ).fit(X_train, y_train)
        train_acc.append(clf.score(X_train,y_train))
        test_acc.append(clf.score(X_test,y_test))
        
        
g=[.01,.1,.5,1,5]
plt.figure()
plt.plot(g,train_acc,color='r')
plt.plot(g,test_acc,color='b')
plt.gca().set_xlabel('gamma')
plt.gca().set_ylabel('accuracy')
plt.gca().legend(['train','test'])
plt.show()
        

clf=SVC(kernel='rbf',C=10,gamma=7).fit(X_train,y_train)
print('train set accuracy',clf.score(X_train,y_train))
print('test set accuracy',clf.score(X_test,y_test))
cross_val_score(clf,X_test,y_test,cv=5).mean()
vclf=VotingClassifier(estimators=[('svm',clf),('extra',ex),('rf',rf)],voting='hard')
vclf.fit(X_train,y_train)
print('test set accuracy',vclf.score(X_test,y_test))

scaler.fit(test.drop('Id',axis=1))
scaled=scaler.fit_transform(test.drop('Id',axis=1))
y_pre=vclf.predict(scaled)
df=pd.DataFrame({'Id':test['Id'],'Cover_Type':y_pre},columns=['Id','Cover_Type'])
df.to_csv('submission4.csv',index=False)

