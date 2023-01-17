import pandas
import numpy as np

train_set = pandas.read_csv("../input/train.csv")
test_set = pandas.read_csv("../input/test.csv")
train_set = train_set.drop('id',axis=1)
print(train_set.describe())
train_set['type'], categories = train_set['type'].factorize()
print(train_set.describe())
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(20,10))
aux_plot = fig.add_subplot(111)
fig.colorbar(aux_plot.matshow(train_set.corr()))

aux_plot.set_xticklabels(train_set.columns)
aux_plot.set_yticklabels(train_set.columns)

plt.show()
from sklearn.base import BaseEstimator, TransformerMixin

class CreateExtraFeatures(BaseEstimator,TransformerMixin):
    def __init__(self):pass

    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        X['hair_soul'] = X['hair_length'] * X['has_soul']
        X['flesh_soul'] = X['rotting_flesh'] * X['has_soul']
        return np.c_[X]
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
pipeline_num = Pipeline([
    ("extra_feat",CreateExtraFeatures())
])

pipeline_cat = Pipeline([
    ("categorical_encoder", OneHotEncoder(sparse=False))
])

from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion([
    ("pip,num",pipeline_num),
    ("pip_cat",pipeline_cat)
])
X_train = train_set.drop('type',axis=1)
y_train = train_set.get('type')
X_train= X_train.append(test_set)

num_attributes = ["bone_length","rotting_flesh","hair_length","has_soul"]
cat_attributes = ["color"]
X_train= full_pipeline.fit_transform(X_train[num_attributes],X_train[cat_attributes].values)

X_test = X_train[371:]
X_train = X_train[:371]
from sklearn.neural_network import MLPClassifier

nn_clf = MLPClassifier(max_iter=3000)

from sklearn.model_selection import GridSearchCV

grid_params = [{"hidden_layer_sizes":range(3,20), "activation":['identity', 'logistic', 'tanh', 'relu'], "solver":["lbfgs","sgd","adam"],"learning_rate":["adaptive"]}]
grid_search = GridSearchCV(nn_clf,param_grid=grid_params,cv=3,verbose=0)

grid_search.fit(X_train,y_train)

print(grid_search.best_estimator_)
print(grid_search.best_score_)
y_pred = grid_search.predict(X_test)

submissions = pandas.DataFrame(y_pred, index=test_set.id,columns=["type"])
submissions["type"] = categories[submissions["type"]]
submissions.to_csv('./submission.csv', index=True)