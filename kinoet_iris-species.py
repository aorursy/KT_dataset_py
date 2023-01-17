# Load packages
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
# Load datasets
iris = pd.read_csv('../input/Iris.csv', index_col = 0)
iris.sample(10)
# observe data
sns.pairplot(iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']],
             hue='Species')
# take features
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
# split 2 folds, one for traing, one for validation
train_data, validate_data, train_answer, validate_answer = train_test_split(
    iris[features],
    iris['Species'].map({'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3}),
    random_state = 0
)

# create model
model = DecisionTreeRegressor()
# fit model
model.fit(train_data, train_answer)
# predict using model
predict_answer = model.predict(validate_data)
# model validation
result = pd.DataFrame({'validate': validate_answer, 'predict': predict_answer.astype('int64')})
result['Difference'] = result['validate'] == result ['predict']
result
# error rate
print("Error rate: " + str(len(result[result['Difference'] == False]) / len(result)))