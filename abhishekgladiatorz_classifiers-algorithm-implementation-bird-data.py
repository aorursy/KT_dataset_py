

import numpy as np 
import pandas as pd 
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

import seaborn as sns
import matplotlib.pyplot as plt




data = pd.read_csv('../input/bird.csv')




data.head(3)


data['type'].unique()


data.describe()




data.isnull().any()


data = data.dropna()
data.describe()
data.isnull().any()


data['type'].unique()


#we can assign the  SW,W,T,R,P,SO as 0,1,2,3,4,5 respectively for the types .


data.type[data.type == 'SW'] = 0
data.type[data.type == 'W'] = 1
data.type[data.type == 'T'] = 2
data.type[data.type == 'R'] = 3
data.type[data.type == 'P'] = 4
data.type[data.type == 'SO'] = 5



data['type'].unique()




data = data.drop('id',axis=1)




data.head(3)


y = data['type']


x = data.drop('type',axis=1)


plt.figure(figsize=(2,1))
sns.pairplot(x,height = 2.5)
plt.show()
plt.figure(figsize=(40,20))
cm = np.corrcoef(x.values.T)
sns.set(font_scale = 1.5)
hm = sns.heatmap(cm,cbar = True , annot=True,square=True,fmt='.1f',annot_kws={'size':15})
plt.show()

x


y = y.astype('int')




y.head(10)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x = scaler.fit_transform(x)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
print (x_train.shape, y_train.shape)
print (x_test.shape, y_test.shape)
from sklearn.decomposition import PCA
pca = PCA(n_components=9)
pca.fit(data)
print(pca.explained_variance_ratio_)  
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 1).fit(x_train, y_train) 
accuracy = knn.score(x_test, y_test) 
print (accuracy) 
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB().fit(x_train, y_train) 
gnb_predictions = gnb.predict(x_test) 
accuracy = gnb.score(x_test, y_test) 
print (accuracy)
from sklearn.cluster import KMeans
kmeansclust = KMeans(n_clusters=6,algorithm = 'elkan',max_iter = 40 , n_jobs = 10)
kmeansclust.fit(x_train,y_train)
ykmeans = kmeansclust.predict(x_test)


from sklearn.metrics import precision_score

print (precision_score(y_test, ykmeans,average='macro'))


from sklearn import metrics
confusion_matrix=metrics.confusion_matrix(y_test,ykmeans)
confusion_matrix


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
ldaalg = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None)
ldaalg.fit(x_train ,y_train)


ldaalg.predict(x_test)
ldaalg.score(x_train,y_train)
ldaalg.score(x_test,y_test)
from sklearn.tree import DecisionTreeClassifier
dectree = DecisionTreeClassifier(random_state=0,criterion='entropy')
dectree.fit(x_train,y_train)
dectree.predict(x_test)
dectree.score(x_train,y_train)
dectree.score(x_test,y_test)
from sklearn.svm import SVC 
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(x_train, y_train) 
svm_predictions = svm_model_linear.predict(x_test) 
svm_model_linear.score(x_train, y_train)
svm_model_linear.score(x_test, y_test)
svm_svc = SVC(kernel = 'rbf',random_state = 0 , gamma = 0.10 ,C=10.0)
svm_svc.fit(x_train, y_train) 

svm_svc.score(x_train, y_train) 
svm_svc.score(x_test, y_test) 
svm_svc1 = SVC(kernel = 'rbf',random_state = 0 , gamma = 0.2 ,C=10.0)
svm_svc1.fit(x_train, y_train) 

svm_svc1.score(x_train, y_train) 
svm_svc1.score(x_test, y_test) 
svm_svc2 = SVC(kernel = 'rbf',random_state = 0 , gamma = 2 ,C=10.0)
svm_svc2.fit(x_train, y_train) 
svm_svc2.score(x_train, y_train) 
svm_svc2.score(x_test, y_test) 
svm_svc3 = SVC(kernel = 'rbf',random_state = 0 , gamma = 0.40 ,C=10.0)
svm_svc3.fit(x_train, y_train) 
svm_svc3.score(x_train, y_train) 
svm_svc3.score(x_test, y_test) 
svm_svc3 = SVC(kernel = 'linear',random_state = 0 , gamma = 0.35 ,C=70.0)
svm_svc3.fit(x_train, y_train) 
svm_svc3.score(x_train, y_train) 
svm_svc3.score(x_test, y_test) 
from sklearn.ensemble import RandomForestClassifier

random_clf = RandomForestClassifier(n_estimators=300,random_state=1,criterion='entropy',n_jobs=4)
random_clf.fit(x_train,y_train)

random_clf.score(x_train,y_train)
random_clf.score(x_test,y_test)
#We can use grid search in order to find the parameters efficiently , 
#which can be used for this implementation in order to determine the best results for training set

from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
pipe_svc = Pipeline([('clf', RandomForestClassifier(random_state=1))])
estim_range = [1,10,50,100,200,300,400,700,1000]
jobs_range = [1,2,3,5,8,10,16]
crit_range = ['gini','entropy']
param_grid = [{'clf__n_estimators': estim_range,'clf__criterion': crit_range ,'clf__n_jobs': jobs_range}]
gs = GridSearchCV(estimator=pipe_svc,param_grid=param_grid,scoring='accuracy')
gs = gs.fit(x_train, y_train)
print(gs.best_score_)
print(gs.best_params_)


random_clf1 = RandomForestClassifier(n_estimators=100,criterion='gini',n_jobs=1)
random_clf1.fit(x_train,y_train)


random_clf1.score(x_train,y_train)
random_clf1.score(x_test,y_test)
from sklearn.linear_model import SGDClassifier
clfsgd = SGDClassifier(loss="log", penalty="l2", max_iter=400)
clfsgd.fit(x_train,y_train)
clfsgd.score(x_train,y_train)
clfsgd.predict(x_test)
clfsgd.score(x_test,y_test)
from sklearn.linear_model import LogisticRegression
logclf = LogisticRegression(random_state=1,multi_class='multinomial',solver='newton-cg',max_iter=300)
logclf.fit(x_train,y_train)
logclf.score(x_train,y_train)
logclf.score(x_test,y_test)
from sklearn.cluster import AgglomerativeClustering
aggclustering = AgglomerativeClustering(n_clusters=6)
aggclustering.fit(x_train,y_train)
aggclusterval = aggclustering.fit_predict(x_test)
from sklearn.metrics import precision_score

print (precision_score(y_test, aggclusterval,average='macro'))
import xgboost as xgb

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)
#defining parameters 
param = {'max_depth': 6, 'eta': 0.3,'silent': 1,'objective': 'multi:softprob','num_class': 6}  
num_round = 20
xgbstimp = xgb.train(param, dtrain, num_round)
preds = xgbstimp.predict(dtest)
best_preds = np.asarray([np.argmax(line) for line in preds])
from sklearn.metrics import precision_score

print (precision_score(y_test, best_preds, average='macro'))
from keras.models import Sequential 
from keras.layers import Dense 
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle
y1 = y
x1 = x
encoder = LabelEncoder()
encoder.fit(y1)
y1 = encoder.transform(y1)
y1 = np_utils.to_categorical(y1)
from sklearn.model_selection import train_test_split

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.20)
print (x1_train.shape, y1_train.shape)
print (x1_test.shape, y1_test.shape)
#sample y1 here 
y1
#selecting the dense with numbe rof neurons 
input_dim = len(data.columns) - 1

model = Sequential()
model.add(Dense(11, input_dim = input_dim , activation = 'relu'))
model.add(Dense(11, activation = 'relu'))
model.add(Dense(11, activation = 'relu'))
model.add(Dense(11, activation = 'relu'))
model.add(Dense(6, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )

model.fit(x1_train, y1_train, epochs = 90, batch_size = 2)

scores = model.evaluate(x1_test, y1_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))





