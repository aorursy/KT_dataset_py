
# Import the required libraries
from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math   # yep! going to a bit of maths later!!
from scipy import stats as st # and some stats

# Load up the files 
dirty_training_set = pd.read_csv('../input/random-linear-regression/train.csv')
dirty_test_set = pd.read_csv('../input/random-linear-regression/test.csv')

# is NA Exists in Dataframe
dirty_training_set.isna().sum()

training_set = dirty_training_set.dropna()
dirty_test_set.head()
dirty_test_set.isnull().sum()

test_set = dirty_test_set.dropna() 
training_set.plot(x ='x', y='y', kind = 'scatter')
plt.show()
test_set.plot(x ='x', y='y', kind = 'scatter')
plt.show()
lm = linear_model.LinearRegression()
model=lm.fit(training_set[["x"]],training_set["y"])

print('R square: ',model.score(training_set[["x"]],training_set["y"]))
print('Coefficient for dataframe: ',model.coef_)

c=model.intercept_
c  # intercept at Y axis

y_predictions_train = model.predict(training_set[["x"]])
y_predictions_train
y_predictions_test = model.predict(test_set[["x"]])
y_predictions_test
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
mse = mean_squared_error(training_set["y"],y_predictions_train)
mse
rmse = math.sqrt(mse)
rmse
r2_score(training_set["y"],y_predictions_train)
test_mse = mean_squared_error(test_set["y"],y_predictions_test)
test_mse
rmse_test= math.sqrt(test_mse)
rmse_test
r2_score(test_set["y"],y_predictions_test)
!pip install pyspark
from pyspark.sql import SparkSession
spark= SparkSession.builder.appName("simple").getOrCreate()
from pyspark.ml.regression import LinearRegression
data = spark.read.csv('../input/simpledata/data.csv', header=True, inferSchema=True)
data.na.drop()
data.printSchema()
data.head(5)
data.columns
type(data['x'])
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=['x'], outputCol='features')
output = assembler.transform(data)
output.printSchema()
output.head(1)
final_data =output.select('features','y')
final_data.show()
train_data, test_data =final_data.randomSplit([0.7,0.3])
train_data=train_data.na.drop()
test_data=test_data.na.drop()
train_data.describe().show()
test_data.describe().show()
!pip3 install py4j
!pip install install-jdk
lr=LinearRegression(labelCol='y', featuresCol='features')
lr_model= lr.fit(train_data)
test_result = lr_model.evaluate(test_data)

test_result.residuals.show()
test_result.rootMeanSquaredError
test_result.r2