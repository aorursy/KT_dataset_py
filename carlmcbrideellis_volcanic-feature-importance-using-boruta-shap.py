import numpy   as np

import pandas  as pd

pd.set_option('display.max_columns', None)

!pip install BorutaShap
train   = pd.read_csv('../input/the-volcano-and-the-regularized-greedy-forest/volcano_train.csv')

X_train = train.drop(["segment_id","time_to_eruption"],axis=1)

y_train = train["time_to_eruption"]



from xgboost import XGBRegressor

model = XGBRegressor()



from BorutaShap import BorutaShap

Feature_Selector = BorutaShap(model=model,importance_measure='shap', classification=False)

Feature_Selector.fit(X=X_train, y=y_train, n_trials=35, random_state=0);
Feature_Selector.plot(which_features='accepted', figsize=(20,12))
selected_features = Feature_Selector.Subset()

selected_features
selected_features.to_csv('selected_features.csv',index=False)
test   = pd.read_csv('../input/the-volcano-and-the-regularized-greedy-forest/volcano_test.csv')

X_train          = selected_features

selected_columns = selected_features.columns

X_test           = test[selected_columns]



from rgf.sklearn import RGFRegressor

regressor = RGFRegressor(max_leaf=10000, 

                         algorithm="RGF_Sib", 

                         test_interval=100, 

                         loss="LS",

                         verbose=False)

regressor.fit(X_train, y_train)

predictions = regressor.predict(X_test)



sample = pd.read_csv('../input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv')

sample.iloc[:,1:] = predictions

sample.to_csv('submission.csv',index=False)