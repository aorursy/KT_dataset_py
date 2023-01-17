import pandas  as pd

import numpy   as np
train  = pd.read_csv('../input/ingv-lgbm-baseline-the-train-test-csv-files/volcano_train.csv')

test   = pd.read_csv('../input/ingv-lgbm-baseline-the-train-test-csv-files/volcano_test.csv')

sample = pd.read_csv('../input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv')
X_train       = train.drop(["segment_id","time_to_eruption"],axis=1)

y_train       = train["time_to_eruption"]

X_test        = test.drop("segment_id",axis=1)
from rgf.sklearn import RGFRegressor



regressor = RGFRegressor(max_leaf=2000, 

                         algorithm="RGF_Sib", 

                         test_interval=100, 

                         loss="LS",

                         verbose=False)



regressor.fit(X_train, y_train)

predictions = regressor.predict(X_test)
sample.iloc[:,1:] = predictions

sample.to_csv('submission.csv',index=False)
train.to_csv('volcano_train.csv')

test.to_csv('volcano_test.csv')