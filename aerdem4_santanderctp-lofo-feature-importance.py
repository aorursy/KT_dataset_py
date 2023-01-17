!pip install lofo-importance
import pandas as pd

import numpy as np



train_df = pd.read_csv("../input/train.csv")

train_df.shape
from sklearn.model_selection import StratifiedKFold

from lightgbm import LGBMClassifier



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

model = LGBMClassifier(n_estimators=50, num_leaves=7, n_jobs=-1)
from lofo import LOFOImportance, plot_importance



features = train_df.columns[2:]



lofo_imp = LOFOImportance(train_df, features, "target", model=model, cv=skf, scoring="roc_auc")



importance_df = lofo_imp.get_importance()

importance_df.head()
%matplotlib inline



plot_importance(importance_df, figsize=(12, 32))