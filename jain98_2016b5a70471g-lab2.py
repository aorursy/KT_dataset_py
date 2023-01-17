import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.utils import resample

%matplotlib inline



pd.set_option('display.max_columns', 100)
#getting data

testdf = pd.read_csv('../input/eval-lab-2-f464/test.csv')

df = pd.read_csv('../input/eval-lab-2-f464/train.csv')

[df.shape, testdf.shape]
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn import tree

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import VotingClassifier
# importing required libraries

import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

from sklearn.ensemble import ExtraTreesClassifier



#testdf.head()

from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
#Constructing 4 sets of data, (x,y)-> (x_test,~); (x_train,y_train);(x_cv,y_cv)

columns = ['chem_2','chem_1','chem_4','chem_6','attribute']

x = df[columns].copy()

y = df["class"].copy()

#x = x.drop(['id'],axis=1);

#no need to scale in xgboost

#test = testdf.drop(['id'],axis=1);

test = testdf[columns].copy()
from sklearn.model_selection import train_test_split

x_train,x_cv,y_train,y_cv = train_test_split(x,y,test_size=0.33)

[x_train.shape, y_train.shape, x.shape, y.shape, x_cv.shape, y_cv.shape]

#x_train
#x_train = np.delete([x_train],0,axis=1);

#np.delete(x_cv,0,axis=1);

#np.delete(x,0,axis=1);

#[x_train]#.shape] #[y_train].shape, [x].shape, [y].shape, [x_cv].shape, [y_cv].shape]
#model3 = XGBClassifier(learning_rate=0.1,min_child_weight=4,max_depth=8,objective='multi:softmax',num_class =6,cv=20)

#model1 = ExtraTreesClassifier(n_estimators=5000,random_state=103231,max_depth=15,max_features = 6,min_samples_leaf=1,class_weight='balanced')

#model = ExtraTreesClassifier(random_state=103231,max_depth=24,n_estimators=5000)

#model2 = RandomForestClassifier(max_depth=3,max_features=6,n_estimators=5000)

#model = BaggingClassifier(ExtraTreesClassifier(n_estimators=5000,random_state=103231,max_depth=15,max_features = 6,min_samples_leaf=1))

#from sklearn.naive_bayes import GaussianNB

#model = GaussianNB()

#model = ExtraTreesRegressor()

# # fit the model with the training data

model = RandomForestClassifier(n_estimators=2000)

# m2 = ExtraTreesClassifier()

# m3 = XGBClassifier()

# model = VotingClassifier(estimators=[('rfc', m1), ('extra', m2), ('xgb', m3)], voting='hard')



# classifiers = [

#     ('sgd', SGDClassifier(max_iter=1000)),

#     ('logisticregression', LogisticRegression()),

#     ('svc', SVC(gamma='auto')),

# ]

# model = VotingClassifier(classifiers, n_jobs=-1)

model.fit(x_train,y_train)



# predict the target on the train dataset

predict_train = model.predict(x_train)



# Accuray Score on train dataset

accuracy_train = accuracy_score(y_train,predict_train)

print('\naccuracy_score on train dataset : ', accuracy_train)



# predict the target on the test dataset

y_predict_cv = model.predict(x_cv)



# Accuracy Score on test dataset

accuracy_test = accuracy_score(y_cv,y_predict_cv)

print('\naccuracy_score on test dataset : ', accuracy_test)
plt.scatter(y_predict_cv,y_cv,alpha = 0.1)
# fit the model with the whole data

model.fit(x,y)



# predict the target on the train dataset

y_test = model.predict(x)



# Accuray Score on train dataset

accuracy_train = accuracy_score(y_test,y)

print('\naccuracy_score on train dataset : ', accuracy_train)



# predict the target on the test dataset



y_pred = model.predict(test)

# Accuracy Score on test dataset
#storing y_test in reuired format

ID = testdf['id']

#y_test= y_test.reshape(len(y_test),1)

ans = pd.concat([ID,pd.DataFrame(y_pred)],axis=1)

ans[0].value_counts()

#check the things

ans.astype('int64')

ans.dtypes
ans.info

#store in csv

ans.to_csv("submit32.csv",index=None,header=["id","class"])
# Compute the correlation matrix

corr = df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(12, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, center=0, cmap=cmap,square=True,

             linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
df['class'].value_counts()