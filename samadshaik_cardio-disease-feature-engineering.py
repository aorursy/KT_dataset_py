import numpy as np

import pandas as pd

df=pd.read_csv('/kaggle/input/cardio-train2/cardio_train2.csv')
df.isnull().sum()

#we didnt find any missing values
#we dont have any temporial variables
feature_scale=[feature for feature in df.columns if feature not in ['id','cardio']]



from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaler.fit(df[feature_scale])

data = pd.concat([df[['id', 'cardio']].reset_index(drop=True),

                    pd.DataFrame(scaler.transform(df[feature_scale]), columns=feature_scale)], axis=1)

data
from sklearn.linear_model import Lasso

from sklearn.feature_selection import SelectFromModel



# to visualise al the columns in the dataframe

pd.pandas.set_option('display.max_columns', None)
y_train=data[['cardio']]

X_train=data.drop(['id','cardio'],axis=1)
feature_sel_model = SelectFromModel(Lasso(alpha=0.009, random_state=0)) # remember to set the seed, the random state in this function

feature_sel_model.fit(X_train, y_train)

feature_sel_model.get_support()
# let's print the number of total and selected features



# this is how we can make a list of the selected features

selected_feat = X_train.columns[(feature_sel_model.get_support())]



# let's print some stats

print('total features: {}'.format((X_train.shape[1])))

print('selected features: {}'.format(len(selected_feat)))

selected_feat
X=data[selected_feat]

Y=y_train
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=0)
from sklearn import model_selection

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier, export_graphviz
#decision tree

model = DecisionTreeClassifier(criterion='entropy',max_depth=5)

model.fit(X_train,y_train)

predictions = model.predict(X_test)

predictions

from sklearn.metrics import accuracy_score,f1_score,log_loss

accuracy_score(y_test,predictions),f1_score(y_test,predictions)
#logistic regression

model= LogisticRegression(C=7,solver='liblinear')

model.fit(X_train,y_train)

predictions2=model.predict(X_test)

predictions2

accuracy_score(y_test,predictions2),f1_score(y_test,predictions2)
#k-nearest-neighbors

k=2

model=KNeighborsClassifier(n_neighbors=k)

model.fit(X_train,y_train)

#jaccard_index

predictions3 = model.predict(X_test)

print(accuracy_score(y_test, predictions3))

f1_score(y_test,predictions3)