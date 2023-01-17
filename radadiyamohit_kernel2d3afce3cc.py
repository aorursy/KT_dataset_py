import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
# reading dataset
df = pd.read_csv("/kaggle/input/hackerearth-ml-challenge-pet-adoption/train.csv")
df = df.fillna(df.mean())

# This is for Features
x = df[['condition', 'color_type', 'length(m)', 'height(cm)','X1', 'X2']]

# We have Two Targer Value that is y1 and y2
y1 = df['breed_category']
y2 = df['pet_category']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x['color_type'] = le.fit_transform(x['color_type'])
x.head()

df_t = pd.read_csv("/kaggle/input/hackerearth-ml-challenge-pet-adoption/test.csv")
df_t = df_t.fillna(df.mean())
xt = df_t[['condition', 'color_type', 'length(m)', 'height(cm)','X1', 'X2']]
xt['color_type'] = le.transform(xt['color_type'])
yt1 = []
yt2 = []
xt.head()
from catboost import CatBoostRegressor
cbc = CatBoostRegressor()
cbc.fit(x,y1)
y_pred1=cbc.predict(xt)

cbc.fit(x,y2)
y_pred2=cbc.predict(xt)

o = list(zip(df_t["pet_id"], y_pred1, y_pred2))
o = np.array(o)
df = pd.DataFrame(o, columns = ["pet_id", 'breed_category', 'pet_category'])
df.to_csv("CatBoost.csv", index = False)
from sklearn.ensemble import AdaBoostRegressor
rf_boost=AdaBoostRegressor()

model = rf_boost.fit(x, y1)
y_pred1 = model.predict(xt)

model1 = rf_boost.fit(x, y2)
y_pred2 = model1.predict(xt)

o = list(zip(df_t["pet_id"], y_pred1, y_pred2))
o = np.array(o)
df = pd.DataFrame(o, columns = ["pet_id", 'breed_category', 'pet_category'])
df.to_csv("AdaBoost.csv", index = False)
from sklearn.ensemble import GradientBoostingRegressor
gboost=GradientBoostingRegressor()

model = gboost.fit(x, y1)
y_pred1 = model.predict(xt)

model2 = gboost.fit(x, y2)
y_pred2 = model2.predict(xt)

o = list(zip(df_t["pet_id"], y_pred1, y_pred2))
o = np.array(o)
df = pd.DataFrame(o, columns = ["pet_id", 'breed_category', 'pet_category'])
df.to_csv("GradientBoost.csv", index = False)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()

model = lr.fit(x, y1)
y_pred = model.predict(xt)

model1 = lr.fit(x, y2)
y_pred1 = model1.predict(xt)

o = list(zip(df_t["pet_id"], y_pred, y_pred1))
o = np.array(o)

df = pd.DataFrame(o, columns = ["pet_id", 'breed_category', 'pet_category'])
df.to_csv("LinearRegressor.csv", index = False)
from xgboost import XGBClassifier
model = XGBClassifier()

model = model.fit(x, y1)
y_pred1 = model.predict(xt)

model1 = model.fit(x, y2)
y_pred2 = model1.predict(xt)

o = list(zip(df_t["pet_id"], y_pred1, y_pred2))
o = np.array(o)

df = pd.DataFrame(o, columns = ["pet_id", 'breed_category', 'pet_category'])
df.to_csv("XGBoost.csv", index = False)
from sklearn.svm import SVC
svm = SVC(probability=True)

svm1 = svm.fit(x, y1)
y_pred1 = svm1.predict(xt)

svm2 = svm.fit(x, y2)
y_pred2 = svm2.predict(xt)

o = list(zip(df_t["pet_id"], y_pred1, y_pred2))
o = np.array(o)

df = pd.DataFrame(o, columns = ["pet_id", 'breed_category', 'pet_category'])
df.to_csv("SVM.csv", index = False)
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

nb1 = nb.fit(x, y1)
y_pred1 = nb1.predict(xt)

nb2 = nb.fit(x, y2)
y_pred2 = nb2.predict(xt)

o = list(zip(df_t["pet_id"], y_pred1, y_pred2))
o = np.array(o)

df = pd.DataFrame(o, columns = ["pet_id", 'breed_category', 'pet_category'])
df.to_csv("NaiveBayes.csv", index = False)
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()

rf1 = rf.fit(x, y1)
y_pred1 = rf1.predict(xt)

rf2 = rf.fit(x, y2)
y_pred2 = rf2.predict(xt)

o = list(zip(df_t["pet_id"], y_pred1, y_pred2))
o = np.array(o)

df = pd.DataFrame(o, columns = ["pet_id", 'breed_category', 'pet_category'])
df.to_csv("RandomForest.csv", index = False)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()

knn1 = knn.fit(x, y1)
y_pred1 = knn1.predict(xt)

knn2 = knn.fit(x, y2)
y_pred2 = knn2.predict(xt)

o = list(zip(df_t["pet_id"], y_pred1, y_pred2))
o = np.array(o)

df = pd.DataFrame(o, columns = ["pet_id", 'breed_category', 'pet_category'])
df.to_csv("KNN.csv", index = False)
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(random_state= 0, max_iter = 10000, learning_rate = 'optimal', penalty = 'l2')
sgd1 = sgd_reg.fit(x, y1)
y_pred1 = sgd1.predict(xt) 

sgd_reg = SGDRegressor(random_state= 0, max_iter = 25000, learning_rate = 'optimal', penalty = 'l2')
sgd2 = sgd_reg.fit(x, y2)
y_pred2 = sgd2.predict(xt) 

o = list(zip(df_t["pet_id"], y_pred1, y_pred2))
o = np.array(o)

df = pd.DataFrame(o, columns = ["pet_id", 'breed_category', 'pet_category'])
df.to_csv("SGDRegressor.csv", index = False)
from sklearn.svm import LinearSVR
from sklearn.datasets import make_regression
lin_SVR = LinearSVR(random_state=0, tol=0.05, C=100, epsilon=0.1)
SVR1 = lin_SVR.fit(x, y1)
y_pred1 = SVR1.predict(xt)

SVR2 = lin_SVR.fit(x, y2)
y_pred2 = SVR2.predict(xt)

o = list(zip(df_t["pet_id"], y_pred1, y_pred2))
o = np.array(o)

df = pd.DataFrame(o, columns = ["pet_id", 'breed_category', 'pet_category'])
df.to_csv("linear_SVR.csv", index = False)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("/kaggle/input/hackerearth-ml-challenge-pet-adoption/train.csv")
df = df.fillna(df.mean())

# This is for Features
x = df[['condition', 'color_type', 'length(m)', 'height(cm)','X1', 'X2']]

# We have Two Targer Value that is y1 and y2
y1 = df['breed_category']
y2 = df['pet_category']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x['color_type'] = le.fit_transform(x['color_type'])
x.head()
df_t = pd.read_csv("/kaggle/input/hackerearth-ml-challenge-pet-adoption/test.csv")
df_t = df_t.fillna(df.mean())
xt = df_t[['condition', 'color_type', 'length(m)', 'height(cm)','X1', 'X2']]
xt['color_type'] = le.transform(xt['color_type'])
xt.head()
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier


results=[]
names=[]
models=[]
lr=LogisticRegression()
knn=KNeighborsClassifier()
svm=SVC(probability=True)
rf=RandomForestClassifier()
nb=GaussianNB()
rf_boost=AdaBoostClassifier()
gboost=GradientBoostingClassifier()
cbc=CatBoostClassifier()

models.append(('cbc',cbc))
models.append(('lr',lr))
models.append(('knn',knn))
models.append(('svm',svm))
models.append(('rf',rf))
models.append(('nb',nb))
models.append(('rf_boost',rf_boost))
models.append(('gboost',gboost))

for name,model in models:
    kfold=KFold(shuffle=True,n_splits=10,random_state=1)
    cv_results=cross_val_score(model,x,y1,cv=kfold) 
    results.append(cv_results)
    names.append(name)
    print("%s:%f (%f)" % (name,np.mean(cv_results),np.var(cv_results,ddof=1)))  ##we have 10 recall scores. we are taking its average.
    
#boxplot algorithm comparison
fig=plt.figure()
fig.suptitle('algorithm comparison')
ax=fig.add_subplot(111)
plt.boxplot(results)
plt.show()
y_pre = []
y_pre1 = []
for i in range(len(y_pred)):
    y_pre.append(y_pred[i][0])
    y_pre1.append(y_pred1[i][0])
import numpy as np
import pandas as pd
import lightgbm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
df = pd.read_csv("/kaggle/input/hackerearth-ml-challenge-pet-adoption/train.csv")
df = df.fillna(df.mean())

# This is for Features
x = df[['condition', 'color_type', 'length(m)', 'height(cm)','X1', 'X2']]

# We have Two Targer Value that is y1 and y2
y1 = df['breed_category']
y2 = df['pet_category']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x['color_type'] = le.fit_transform(x['color_type'])
x.head()


df_t = pd.read_csv("/kaggle/input/hackerearth-ml-challenge-pet-adoption/test.csv")
df_t = df_t.fillna(df.mean())
xt = df_t[['condition', 'color_type', 'length(m)', 'height(cm)','X1', 'X2']]
xt['color_type'] = le.transform(xt['color_type'])
xt.head()
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from matplotlib import pyplot

model = LGBMClassifier(boosting_type='gbdt', 
                       num_leaves=31, 
                       max_depth=- 1, 
                       learning_rate=0.05, 
                       n_estimators=1000, 
                       subsample_for_bin=15, 
                       min_split_gain=0.0, 
                       min_child_weight=0.001, 
                       min_child_samples=20, 
                       subsample=1.0, 
                       random_state=5)
cv = RepeatedStratifiedKFold(n_splits=100, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, x, y1, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# fit the model on the whole dataset
model = LGBMClassifier()
model.fit(X, y)
