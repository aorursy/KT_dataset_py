#import library for data read

import pandas as pd

df = pd.read_csv('../input/Dataset_spine.csv',

                 usecols=['Col1','Col2','Col3','Col4','Col5','Col6','Col7',

                          'Col8','Col9','Col10','Col11','Col12','Class_att'])

#renaming columns to appropriate names

df.columns = ['pelvic_incidence','pelvic_tilt','lumbar_lordosis_angle',

              'sacral_slope','pelvic_radius','degree_spondylolisthesis',

              'pelvic_slope','Direct_tilt','thoracic_slope','cervical_tilt',

              'sacrum_angle','scoliosis_slope','Class_att']
#analzing data

df.describe()
df[df['degree_spondylolisthesis']>400]
#dropping off 

clean_df = df.drop(115,0)

clean_df.describe()
#   splitting into features and outcome

features = clean_df.drop('Class_att', axis=1)

target = clean_df['Class_att']
#splitting into test/train datasets

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



logicModel = LogisticRegression()

logicModelFit = logicModel.fit(X_train, y_train)

logicModelPred = logicModelFit.predict(X_test)

logicModelPredScore = accuracy_score(y_test,logicModelPred)

print("Logistic Regression proves to be", logicModelPredScore*100, "% accurate here!")
from sklearn import svm



svmModel = svm.SVC(kernel='linear')

svmModelFit = svmModel.fit(X_train, y_train)

svmModelFitPred = svmModel.predict(X_test)

svmModelPredScore = accuracy_score(y_test, svmModelFitPred)

print("Support Vector Machines proves to be", svmModelPredScore*100, "% accurate here!")