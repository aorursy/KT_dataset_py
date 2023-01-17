# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from sklearn import preprocessing, neighbors

from sklearn.cross_validation import train_test_split

from sklearn.linear_model import LogisticRegression

import seaborn as sns

import matplotlib.pyplot as plt
# Reading and uploading the file

df = pd.read_csv('../input/voice.csv')

df.head(5)
corrmat=df.corr()

sns.heatmap(corrmat,linewidths=0.25,vmax=1.0, square=True, cmap="YlGnBu", linecolor='black')
# Name of the columns

col_names = list(df.columns.values)

print(col_names)

print (type(df.columns.values))
df = df.rename(columns={'label': 'gender'})

df.columns.values
#Lets use logistic Regression:



#Producing X and y

X = np.array(df.drop(['gender'], 1))

y = np.array(df['gender'])



#Dividing the data randomly into training and test set

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)



model=LogisticRegression()

model.fit(X_train,y_train)



print('Accuracy1 :',model.score(X_train,y_train))

print('Accuracy2 :',model.score(X_test,y_test))


#KNN Classifier

#Producing X and y

X = np.array(df.drop(['gender'], 1))

y = np.array(df['gender'])



#Dividing the data randomly into training and test set

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)



model = neighbors.KNeighborsClassifier()

model.fit(X_train, y_train)



accuracy = model.score(X_test, y_test)

print('Accuracy='+str(accuracy))





#The above was without any tuning ,now we will drop some columns which does not make any sense

#We will drop col=median,mode,Q25,Q75,IQR.

#next edit use only few=meanfreq,sd,median,gender(for no error)
df1=df[['meanfreq','sd','median','meanfun','gender']]

X = np.array(df1.drop(['gender'], 1))

y = np.array(df1['gender'])



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)



model = neighbors.KNeighborsClassifier()

model.fit(X_train, y_train)



accuracy2 = model.score(X_test, y_test)

print('Accuracy2='+str(accuracy2))



#All the models should be above the base_line model:Base line model acc=50:50

#But this is not very helpful,have to find new ways for k-nearest neibhors
df2=df[['meanfreq','sd','meanfun','gender']]

X = np.array(df2.drop(['gender'], 1))

y = np.array(df2['gender'])



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)



model = neighbors.KNeighborsClassifier()

model.fit(X_train, y_train)



accuracy2 = model.score(X_test, y_test)

print('Accuracy='+str(accuracy2))
#print(X_train.shape,y_train.shape,)

print(X_test.shape,y_test.shape)


from sklearn.feature_selection import SelectKBest, f_classif



def select_kbest_clf(data_frame, target, k=5):

    """

    Selecting K-Best features for classification

    :param data_frame: A pandas dataFrame with the training data

    :param target: target variable name in DataFrame

    :param k: desired number of features from the data

    :returns feature_scores: scores for each feature in the data as 

    pandas DataFrame

    """

    feat_selector = SelectKBest(f_classif, k=k)

    _ = feat_selector.fit(data_frame.drop(target, axis=1), data_frame[target])

    

    feat_scores = pd.DataFrame()

    feat_scores["F Score"] = feat_selector.scores_

    feat_scores["P Value"] = feat_selector.pvalues_

    feat_scores["Support"] = feat_selector.get_support()

    feat_scores["Attribute"] = data_frame.drop(target, axis=1).columns

    

    return feat_scores

k=select_kbest_clf(df, 'gender', k=5).sort(['F Score'],ascending=False)



k
k1=sns.barplot(x=k['F Score'],y=k['Attribute'])

k1.set_title('Feature Importance')
df3=df[['meanfun','IQR','Q25','sp.ent','sd','sfm','meanfreq','gender']]

X = np.array(df3.drop(['gender'], 1))

y = np.array(df3['gender'])



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)



model = neighbors.KNeighborsClassifier()

model.fit(X_train, y_train)



accuracy3 = model.score(X_test, y_test)

print('Accuracy3='+str(accuracy3))
df.replace({'male':0,'female':1},inplace=True)

X = np.array(df.drop(['gender'], 1))

y = np.array(df['gender'])



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_auc_score

from sklearn.ensemble import AdaBoostClassifier

#DecisionTreeClassifier

dt=DecisionTreeClassifier(max_depth=3,min_samples_leaf=int(0.5*len(X_train)))

boosted_dt=AdaBoostClassifier(dt,algorithm='SAMME',n_estimators=800,learning_rate=0.5)

boosted_dt.fit(X_train,y_train)

y_predicted=boosted_dt.predict(X_test)



print ("Area under ROC curve: %.4f"%(roc_auc_score(y_test, y_predicted)))

from sklearn import svm

svc = svm.SVC(kernel='rbf', C=1,gamma='auto').fit(X_train, y_train)



#To be continued