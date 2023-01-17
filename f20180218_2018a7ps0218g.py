# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats



%matplotlib inline
df = pd.read_csv("/kaggle/input/minor-project-2020/train.csv")
df.info()
df.describe()
df = df.drop(["id"],axis=1)
df = df.drop_duplicates()

df
y = df["target"]

X = df.drop(["target"], axis=1)
y.value_counts()
from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(X, y, test_size=0.2)
from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()

scaled_X_train = scalar.fit_transform(X_train)

scaled_X_val = scalar.transform(X_val)
X_test = pd.read_csv("/kaggle/input/minor-project-2020/test.csv")

from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import SMOTE

from imblearn.pipeline import Pipeline

over = SMOTE(sampling_strategy=0.2)

rus = RandomUnderSampler(sampling_strategy=1,random_state=42)

steps = [('o',over),('u',rus)]

pipeline = Pipeline(steps = steps)

X_res, y_res = pipeline.fit_resample(scaled_X_train, y_train)

X_res.shape
from sklearn.feature_selection import RFE

from sklearn.linear_model import SGDClassifier



estimator = SGDClassifier(max_iter=1000, tol=1e-3)

selector = RFE(estimator, n_features_to_select=20, step=1,verbose = 1)

selector = selector.fit(X_res,y_res)
selector.support_
scaled_X_train = scaled_X_train[:,selector.ranking_== 1]
X_res = X_res[:,selector.ranking_== 1]
scaled_X_val = scaled_X_val[:,selector.ranking_== 1]
y_res.value_counts()
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

param_grid = {}

             

eclf = RandomForestClassifier(random_state=42,n_estimators=180,n_jobs=-1,verbose=True,max_depth=10)

clf = GridSearchCV(estimator=eclf, param_grid = param_grid, cv=5,verbose=1,n_jobs=-1)

clf.fit(X_res,y_res)
#from sklearn.model_selection import cross_val_score

#from sklearn.linear_model import LogisticRegression

#from sklearn.naive_bayes import GaussianNB

#from sklearn.ensemble import RandomForestClassifier

#from sklearn.ensemble import VotingClassifier

#from sklearn.model_selection import GridSearchCV



#params = {'lr__C': [1.0, 100.0], 'rf__n_estimators': [20, 200]}



#clf1 = LogisticRegression(random_state=1)

#clf2 = RandomForestClassifier(n_estimators=50, random_state=42,n_jobs=-1,verbose=True)

#clf3 = GaussianNB()



#eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],voting='hard')

#clf = GridSearchCV(estimator=eclf, param_grid=params, cv=5,verbose=1,n_jobs=-1)

#clf.fit(X_res,y_res)
y_pred = clf.predict(scaled_X_val)
y_pred.sum()
from sklearn import metrics

print(metrics.classification_report(y_val, y_pred[:114266],zero_division=0))

print(metrics.confusion_matrix(y_val, y_pred))
print("roc_auc_score: ", metrics.roc_auc_score(y_val, y_pred[:114266]))

print("f1 score: ", metrics.f1_score(y_val, y_pred[:114266]))
from sklearn.metrics import roc_curve, auc

plt.style.use('seaborn-pastel')



FPR, TPR, _ = roc_curve(y_val, y_pred)

ROC_AUC = auc(FPR, TPR)

print (ROC_AUC)



plt.figure(figsize =[11,9])

plt.plot(FPR, TPR, label= 'ROC curve(area = %0.2f)'%ROC_AUC, linewidth= 4)

plt.plot([0,1],[0,1], 'k--', linewidth = 4)

plt.xlim([0.0,1.0])

plt.ylim([0.0,1.05])

plt.xlabel('False Positive Rate', fontsize = 18)

plt.ylabel('True Positive Rate', fontsize = 18)

plt.title('ROC', fontsize= 18)

plt.show()
X_test2 = X_test.iloc[:,1:]

predictions = clf.predict(X_test2.iloc[:,selector.ranking_==1])
predictions.sum()
data = {'id' : X_test.iloc[:,0], 

        'target' : predictions

       }

submission = pd.DataFrame(data,columns=['id','target'])
submission
filename = 'submission.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)