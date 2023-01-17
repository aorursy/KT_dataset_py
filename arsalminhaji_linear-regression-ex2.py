import  math
from IPython import display
from matplotlib import  cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from  tensorflow.python.data import Dataset



tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows=10
pd.options.display.float_format='{:.1f}'.format
calofornia_housing_data=pd.read_csv("../input/california_housing_train.csv", sep=",")
calofornia_housing_data
calofornia_housing_data=calofornia_housing_data.reindex(np.random.permutation(calofornia_housing_data.index))
calofornia_housing_data
calofornia_housing_data["median_house_value"]/=1000.0
calofornia_housing_data
calofornia_housing_data.describe()
my_feature=calofornia_housing_data[["total_rooms"]]
my_feature
feature_column=[tf.feature_column.numeric_column("total_rooms")]
feature_column
target_label=calofornia_housing_data['median_house_value']
target_label
my_optimizer =tf.train.GradientDescentOptimizer(learning_rate=0.00006)
my_optimizer=tf.contrib.estimator.clip_gradients_by_norm(my_optimizer,5.0)
linear_regerrsor=tf.estimator.LinearRegressor(
    feature_columns=feature_column,
    optimizer=my_optimizer)
linear_regerrsor
my_optimizer
def my_input_function(features,targets,batch_size=5,shuffle=True,num_epochs=None):
    features={key:np.array(value) for key ,value in dict(features).items()}
    #print(features)
    ds=Dataset.from_tensor_slices((features,targets))
    #print(ds)
    #print(targets)
    ds=ds.batch(batch_size).repeat(num_epochs)
    #print(ds)
    if shuffle:
        ds=ds.shuffle(buffer_size=10000)
    features,labels=ds.make_one_shot_iterator().get_next()
   # print(features)
   # print(labels)
    return features,labels
my_input_function(features=my_feature,targets=target_label)
_ =linear_regerrsor.train(
    input_fn=lambda:my_input_function(my_feature,target_label),
    steps=150 )
calofornia_housing_data.dtypes
prediction_input_fn =lambda: my_input_function(my_feature, target_label, num_epochs=1, shuffle=False)

prediction_input_fn
predictions=linear_regerrsor.predict(input_fn=prediction_input_fn,)
predictions
predictions = np.array([item['predictions'][0] for item in predictions])

predictions
mean_squared_error = metrics.mean_squared_error(predictions, target_label)
mean_squared_error
root_mean_squared_error = math.sqrt(mean_squared_error)

root_mean_squared_error
min_house_value = calofornia_housing_data["median_house_value"].min()
max_house_value = calofornia_housing_data["median_house_value"].max()
min_max_difference = max_house_value - min_house_value

print ("Min. Median House Value: %0.3f" % min_house_value)
print ("Max. Median House Value: %0.3f" % max_house_value)
print ("Difference between Min. and Max.: %0.3f" % min_max_difference)
print ("Root Mean Squared Error: %0.3f" % root_mean_squared_error)
calibration_data=pd.DataFrame()
calibration_data['predictions']=pd.Series(predictions)
calibration_data['target']=pd.Series(target_label)
calibration_data

sample = calofornia_housing_data.sample(n=300)
x_0 = sample["total_rooms"].min()
x_1 = sample["total_rooms"].max()

# Retrieve the final weight and bias generated during training.
weight = linear_regerrsor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
bias = linear_regerrsor.get_variable_value('linear/linear_model/bias_weights')

# Get the predicted median_house_values for the min and max total_rooms values.
y_0 = weight * x_0 + bias 
y_1 = weight * x_1 + bias

# Plot our regression line from (x_0, y_0) to (x_1, y_1).
plt.plot([x_0, x_1], [y_0, y_1], c='r')

# Label the graph axes.
plt.ylabel("median_house_value")
plt.xlabel("total_rooms")

# Plot a scatter plot from our data sample.
plt.scatter(sample["total_rooms"], sample["median_house_value"])

# Display graph.
plt.show()
#TWEAK PARAMETERS LIKE BATCH SIZE,LEARNING RATE , SIZE for better result
