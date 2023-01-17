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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import seaborn as sns

# Import things I need
data = pd.read_csv("../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")
display(data)
display(data.describe())
null_value = data.isnull().sum()
print(null_value)

# We check whether there are null_values and there are no null_values so we can pass this process
val_check = ["age","creatinine_phosphokinase","ejection_fraction","platelets","serum_creatinine","serum_sodium","time"]

plt.figure(figsize=(30,15))
n1 = 0
for i in range(1,8):
    plt.subplot(2,4,i)
    plt.hist(data[val_check[n1]])
    plt.title(val_check[n1])
    n1 +=1
    
# We should also check whether data follow normal distribuion. We can find that "creatinine_phosphokinase" and "serum_creatinine" columns skewed a lot.
def logarithm(x):
    return np.log(x+1)
df_2 = data.copy()

log1 = data.creatinine_phosphokinase.apply(lambda x : logarithm(x)).to_frame()
log2 = data.serum_creatinine.apply(lambda x : logarithm(x)).to_frame()

df_2.drop(["creatinine_phosphokinase","serum_creatinine"],axis=1,inplace=True)


display(df_2)

# Normalize data through logarithm function.
df_3 = pd.concat([df_2,log1,log2],axis=1)
display(df_3)

result_val = ["creatinine_phosphokinase","serum_creatinine"]

plt.figure(figsize=(10,10))
for j in range(0,2):
    plt.subplot(1,2,j+1)
    plt.hist(df_3[result_val[j]])
    plt.title(result_val[j])
from sklearn.preprocessing import MinMaxScaler

# As I mentioned earlier, we should stasndardilize each variables so that one variable can not affect to the result more than other variables. So I use MinMaxScaler 
scaler = MinMaxScaler()
df_4 = pd.DataFrame(scaler.fit_transform(df_3),columns=df_3.columns)
display(df_4)
corr = df_4.corr()
plt.figure(figsize=(16,16))
cmap = sns.cubehelix_palette(as_cmap=True)
sns.heatmap(corr,fmt=".2f",annot=True,cmap=cmap,vmin=0.2)

# When we see "Death_event", "age" and "serum_creatinine" affect most
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score

# For classification I choose RandomForestClassifier, AdaBoostClassifier and Support Vector machine.
X = df_4.loc[:,df_4.columns != "DEATH_EVENT"]
y = df_4.loc[:,"DEATH_EVENT"]

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=62)

# Split data and define X_train, X_test, y_train, y_test to train model. I define y variable as "Death event" which indicate whether patient deceased.
# I import Fbeta_score and I will weight more to recall rather than precision because data that I classified should reflect well about real data. 
model1 = RandomForestClassifier()
model1.fit(X_train,y_train)
model1_preds = model1.predict(X_test)
accuracy1 = accuracy_score(y_test,model1_preds)
fbeta_1 = fbeta_score(y_test,model1_preds,beta=1.5)
print("Accuracy of RandomForestClassifier : {}  fbeta score : {}".format(accuracy1,fbeta_1))

# First I train RandomForestClassifier and test it. Accuracy_score is 0.84 and f1_score is 0.74. Not bad
model2 = AdaBoostClassifier()
model2.fit(X_train,y_train)
model2_preds = model2.predict(X_test)
Accuracy2 = accuracy_score(y_test,model2_preds)
fbeta_2 = fbeta_score(y_test,model2_preds,beta=1.5)
print("Accuracy of AdaBoostClassifier : {} fbeta score : {}".format(Accuracy2,fbeta_2))
model3 = SVC()
model3.fit(X_train,y_train)
model3_preds = model3.predict(X_test)
Accuracy3 = accuracy_score(y_test,model3_preds)
fbeta_3 = fbeta_score(y_test,model3_preds,beta=1.5)
print("Accuracy of SVC : {} fbeta score : {}".format(Accuracy3,fbeta_3))
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import time

# Import things that I need. Also I import make_scorer for scoring. In this case I also use fbeta_score that has beta = 1.5. 
# Also I import time so that I can measure time for searching best estimator.
scorer = make_scorer(fbeta_score,beta=1.5)
parameters = {
    "n_estimators" : [100,150,200,250,300],
    "min_samples_split" : [2,4,6],
    "min_samples_leaf" : [4,6,8],
    "max_depth" : [80,100,150,200]
    
}

start = time.time()
grid = GridSearchCV(estimator=model1,param_grid=parameters,scoring=scorer,n_jobs=-1,cv=2)
grid.fit(X_train,y_train)
end = time.time()

print("Search Time : {} seconds".format(end-start))

grid.best_params_

# Check RandomForestClassifier parameters here "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"
model4 = grid.best_estimator_
model4_preds = model4.predict(X_test)
Accuracy4 = accuracy_score(y_test,model4_preds)
fbeta_4 = fbeta_score(y_test,model4_preds,beta=1.5)
print("GridSearch accuracy : {} fbeta score : {}".format(Accuracy4,fbeta_4))

# Through GridSearch I can improve my model. Accuracy : 0.84 → 0.86 Fbeta : 0.74 → 0.78
Importance = np.sort(np.round(model4.feature_importances_*100,3))
df_feature = pd.DataFrame({
    "importance" : Importance
},index=X_train.columns)
display(df_feature)

plt.figure(figsize=(16,16))
plt.barh(df_feature.index.to_list(),df_feature.importance)
plt.title("Feature importance")

# Through RandomForestClassifier, we can check feature importance. Through using numpy, sort values and make dataframe.