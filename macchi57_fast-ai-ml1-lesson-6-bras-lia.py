from fastai.imports import *
from fastai.torch_imports import *
from fastai.dataset import *
from fastai.learner import *
from fastai.structured import *
from fastai.column_data import *
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import sklearn.model_selection 
from sklearn.metrics import accuracy_score
import scipy
from scipy.cluster import hierarchy as hc
from IPython.display import Image
Image("../input/0312-1-drivetrain-approach-lg.png")
data =  pd.read_csv('../input/credit-card.csv')
data.head(10)
Image("../input/tab.PNG")
train, test = train_test_split(data, test_size=0.25, random_state=42)
remove = ['default payment next month','ID']
feats = [col for col in data.columns if col not in remove]
rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, max_features=.9, random_state=42)
rf.fit(train[feats],train['default payment next month'])
preds = rf.predict(test[feats])
preds_train = rf.predict(train[feats])
accuracy_score(train['default payment next month'], preds_train)
accuracy_score(test['default payment next month'], preds)
feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = train[feats].columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances.head(5)
feature_importances.plot.bar()
to_keep = feature_importances[feature_importances.importance>0.01].index; len(to_keep)
to_keep
rf.fit(train[to_keep],train['default payment next month'])
preds_train = rf.predict(train[to_keep])
accuracy_score(train['default payment next month'], preds_train)
preds = rf.predict(test[to_keep])
accuracy_score(test['default payment next month'], preds)