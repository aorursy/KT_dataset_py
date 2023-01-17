!pip install PyCaret
import pandas as pd

import numpy as np

from pycaret.classification import *

from pycaret.utils import version
data = pd.read_csv('../input/titanic/train.csv')

data.shape
data_test = pd.read_csv('../input/titanic/test.csv')

data_test.shape
data.describe
data.pop('Name')

#data.pop('Cabin')

#data.pop('Ticket')

#data.dropna(inplace=True)

data.head
data_setup = setup(data = data, target = 'Survived',train_size = 0.99, silent=True)
#compare_models()
models()
#xgboost = create_model('xgboost')

rf=create_model('rf')

dt=create_model('dt')

ridge = create_model('ridge')

catboost = create_model('catboost')

et=create_model('et')
#xgboost = tune_model(xgboost)

ridge = tune_model(ridge)

catboost = tune_model(catboost)

et=create_model(et)

rf=tune_model(rf)

dt=tune_model(dt)
#blender = blend_models(estimator_list = [ ridge, catboost, et])

blender = blend_models(estimator_list = [ rf,dt])

model = finalize_model(blender)
predictions = predict_model(model, data = data_test)

predictions.head
submission = pd.DataFrame({

        "PassengerId": predictions["PassengerId"],

        "Survived": predictions["Label"]

    })

submission.to_csv('submission.csv', index=False)

print("Your submission was successfully saved!")