import pandas as pd

df = pd.read_csv("../input/apndcts/apndcts.csv")

df.head()
from sklearn.model_selection import train_test_split

print(df.shape)

df_train, df_test=train_test_split(df,test_size=0.3,random_state=123)

print("Training Data Size:",df_train.shape)

print("Test Data Size:",df_test.shape)
#KFold cross-validation

from sklearn.model_selection import KFold

kf=KFold(n_splits=5)

for train_index,test_index in kf.split(df):

    df_train=df.iloc[train_index]

    df_test=df.iloc[test_index]

    print("Training Data Size:",df_train.shape)

    print("Test Data Size:",df_test.shape)

#Bootstrap sampling

from sklearn.utils import resample

X=df.iloc[:,0:9]

resample(X,n_samples=50,random_state=0)

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix



predictors=df.iloc[:,0:7]

target=df.iloc[:,7]

predictors_train,predictors_test,target_train,target_test=train_test_split(predictors,target,test_size=0.3)

dtree_entropy=DecisionTreeClassifier(criterion="entropy",random_state=100,max_depth=3,min_samples_leaf=5)



model=dtree_entropy.fit(predictors_train,target_train)

prediction=model.predict(predictors_test)

acc_score=accuracy_score(target_test,prediction,normalize=True)

print(acc_score)

con=confusion_matrix(target_test,prediction)

print(con)
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

con=confusion_matrix(target_test,prediction)

sns.heatmap(con,annot=True)

plt.ylabel("Predicted")

plt.xlabel("Truth")