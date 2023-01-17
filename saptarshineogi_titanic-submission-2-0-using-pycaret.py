!pip install pycaret
import pandas as pd

from pandas import Series,DataFrame



data_df = pd.read_csv('/kaggle/input/titanic/train.csv')

data_df.head()
from pycaret.classification import *

clf1 = setup(data_df, target = 'Survived', ignore_features = ['Name', 'Ticket', 'PassengerId'])
compare_models()
tuned_lightgbm = tune_model('lightgbm', optimize = 'AUC')
evaluate_model(tuned_lightgbm)
final_lightgbm = finalize_model(tuned_lightgbm)
print(final_lightgbm)
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

test_df.head()
predictions = predict_model(final_lightgbm, data = test_df)
predictions.head()