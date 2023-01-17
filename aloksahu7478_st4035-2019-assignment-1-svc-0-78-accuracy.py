

"""

@author: Alok Kumar Sahu

@email : alok.kr.sahu@outlook.com

@Linkedin : www.linkedin.com/in/alokkrsahu

"""

import os

import pandas as pd

from sklearn.svm import SVC

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from statsmodels.stats.outliers_influence import variance_inflation_factor



clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))



path = ("../input/st4035-2019-assignment-1/")

train = pd.read_csv(os.path.join(path,"vehicle_train.csv"))

test = pd.read_csv(os.path.join(path,"vehicle_test.csv"))

label = pd.read_csv(os.path.join(path,"vehicle_training_labels.csv"))

train = train.drop(['ID'],axis = 1)

test = test.drop(['ID'],axis = 1)
vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(train.values, i) for i in range(train.shape[1])]

vif["features"] = train.columns
corr= train.corr()

print(corr > 0.80)
vif.sort_values(by = 'VIF Factor')
clf.fit(train, label)
svms = pd.DataFrame(clf.predict(test))

svms.index.name = 'ID' 

svms.index += 1

svms.to_csv('./Submission.csv', index = True,header=['Class'])