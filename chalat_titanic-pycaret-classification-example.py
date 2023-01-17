!pip install pycaret -q
from pycaret.regression import *

models()
from pycaret.classification import *

models()
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from pycaret.classification import *
data = pd.read_csv('../input/titanic/train.csv')
test_data = pd.read_csv('../input/titanic/test.csv')
data.head()
data.info()
sns.countplot(data['Survived'])
test_data.head()
test_data.info()
clf = setup(data, target = 'Survived', 
            train_size = 0.8,
            numeric_imputation = 'median',
            categorical_imputation = 'mode',
            ignore_features = ['PassengerId','Name','Ticket'],
            feature_selection = True,
            remove_multicollinearity = True,
            folds_shuffle = True,
            session_id = 211, # Pseudo random number
            html = False, silent = True)
top5_model = compare_models(sort = 'Accuracy', fold = 5, n_select = 5)
for model in top5_model:
    print(model)
predict_model(top5_model[2]); # Predict on validation set
tuned_model = tune_model(top5_model[2], optimize = 'Accuracy', n_iter = 30, fold = 5)
print('After Tune:')
print(tuned_model)
predict_model(tuned_model); # Predict on validation set
plot_model(tuned_model, plot = 'confusion_matrix')
plot_model(tuned_model, plot='feature')
plot_model(tuned_model, plot = 'auc')
plot_model(tuned_model, plot = 'pr')
# All in one but slow

# evaluate_model(tuned_model)
interpret_model(tuned_model)
interpret_model(tuned_model, plot = 'reason', observation = 10)
final_model = finalize_model(tuned_model)
plot_model(final_model, plot = 'confusion_matrix')
predict_model(final_model); # Predict on validation set
predictions = predict_model(final_model, data=test_data)

predictions.head()
predictions.rename({'Label':'Survived'}, axis='columns', inplace = True)

predictions.head()
sns.countplot(predictions['Survived'])
predictions[['PassengerId','Survived']].to_csv('./result.csv', index = False)
# tuned_model_0 = tune_model(top5_model[0], optimize = 'Accuracy', n_iter = 30, fold = 5)
# tuned_model_1 = tune_model(top5_model[1], optimize = 'Accuracy', n_iter = 30, fold = 5)

# fin_0 = finalize_model(tuned_model_0)
# fin_1 = finalize_model(tuned_model_1)

# print(fin_0)
# print(fin_1)
# blender = blend_models(estimator_list = [final_model, fin_0, fin_1], method = 'hard')
# print(blender)
# predictions_blend = predict_model(blender, data=test_data)

# predictions_blend.rename({'Label':'Survived'}, axis='columns', inplace = True)
# sns.countplot(predictions_blend['Survived'])

# predictions_blend[['PassengerId','Survived']].to_csv('./result_blend.csv', index = False)
# save_model(final_rf,'<path_of_file>')
# load_model('<path_of_file>')