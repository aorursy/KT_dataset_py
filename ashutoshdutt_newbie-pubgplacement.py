
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.preprocessing import Imputer
from sklearn.pipeline import make_pipeline

train_data= pd.read_csv('../input/train.csv')
test_data= pd.read_csv('../input/test.csv')
train_data=train_data.dropna(axis=0)
train_y=train_data.winPlacePerc
features=["DBNOs", 'damageDealt', 'matchId', 'killPoints', 'kills']
X=train_data[features]
test_X=test_data[features]
my_pipeline=make_pipeline(Imputer(),rfr())
my_pipeline.fit(X,train_y)
prediction=my_pipeline.predict(test_X)
submission = pd.read_csv('../input/sample_submission.csv')
submission['winPlacePerc'] = prediction

submission.to_csv('submission.csv', index=False)
