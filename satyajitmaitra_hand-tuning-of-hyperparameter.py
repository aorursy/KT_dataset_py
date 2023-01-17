import math
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf



print(tf.__version__)
print(tf.version.VERSION)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.2f}'.format
df = pd.read_csv("https://storage.googleapis.com/ml_universities/california_housing_train.csv", sep = ",")
df.info()
df.describe()
df.head()
df.tail()
df[:30]
df.iloc[4:7]

df['num_rooms'] = df['total_rooms'] / df['households']
df.describe()
#split the data
np.random.seed(seed=1)
msk = np.random.rand(len(df)) <0.85
traindf = df[msk]
evaldf = df[~msk]
traindf.describe()
evaldf.describe()
OUTDIR ="./housing_trained"
def train_and_evaluate(output_dir, num_train_steps):
    estimator = tf.compat.v1.estimator.LinearRegressor(
                                                       model_dir=output_dir,
                                                       feature_columns=[tf.feature_column.numeric_column("num_rooms")])
    
    def rmse(labels,predictions):
        pred_values = tf.cast(predictions["predictions"],tf.float64)
        return {"rmse" : tf.compat.v1.metrics.root_mean_squared_error(labels,pred_values)}
    estimator = tf.compat.v1.estimator.add_metrics(estimator,rmse)
    
    
    train_spec = tf.estimator.TrainSpec(
                                       input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(x = traindf[["num_rooms"]],
                                              y = traindf["median_house_value"],  # note the scaling
                                              num_epochs = None,
                                              shuffle = True),
                       max_steps = num_train_steps)
    eval_spec=tf.estimator.EvalSpec(
                   input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(x = evaldf[["num_rooms"]],
                                          y = evaldf["median_house_value"],  # note the scaling
                                          num_epochs = 1,
                                          shuffle = False),
                   steps = None,
                   start_delay_secs = 1, # start evaluating after N seconds
                   throttle_secs = 10,  # evaluate every N seconds
                   )
    tf.estimator.train_and_evaluate(estimator,train_spec,eval_spec)
    
    
# Run training    
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time
train_and_evaluate(OUTDIR, num_train_steps = 100)
    
SCALE = 100000
OUTDIR = './housing_trained'
def train_and_evaluate(output_dir, num_train_steps):
  estimator = tf.compat.v1.estimator.LinearRegressor(
                       model_dir = output_dir, 
                       feature_columns = [tf.feature_column.numeric_column('num_rooms')])
  
  #Add rmse evaluation metric
  def rmse(labels, predictions):
    pred_values = tf.cast(predictions['predictions'],tf.float64)
    return {'rmse': tf.compat.v1.metrics.root_mean_squared_error(labels*SCALE, pred_values*SCALE)}
  estimator = tf.compat.v1.estimator.add_metrics(estimator,rmse)
  
  train_spec=tf.estimator.TrainSpec(
                       input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(x = traindf[["num_rooms"]],
                                              y = traindf["median_house_value"] / SCALE,  # note the scaling
                                              num_epochs = None,
                                              shuffle = True),
                       max_steps = num_train_steps)
  eval_spec=tf.estimator.EvalSpec(
                       input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(x = evaldf[["num_rooms"]],
                                              y = evaldf["median_house_value"] / SCALE,  # note the scaling
                                              num_epochs = 1,
                                              shuffle = False),
                       steps = None,
                       start_delay_secs = 1, # start evaluating after N seconds
                       throttle_secs = 10,  # evaluate every N seconds
                       )
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

# Run training    
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time
train_and_evaluate(OUTDIR, num_train_steps = 100)
!ls
!ls ./$OUTDIR *
SCALE = 100000
OUTDIR = './housing_trained'
def train_and_evaluate(output_dir, num_train_steps):
  myopt = tf.compat.v1.train.FtrlOptimizer(learning_rate = 0.2) # note the learning rate
  estimator = tf.compat.v1.estimator.LinearRegressor(
                       model_dir = output_dir, 
                       feature_columns = [tf.feature_column.numeric_column('num_rooms')],
                       optimizer = myopt)
  
  #Add rmse evaluation metric
  def rmse(labels, predictions):
    pred_values = tf.cast(predictions['predictions'],tf.float64)
    return {'rmse': tf.compat.v1.metrics.root_mean_squared_error(labels*SCALE, pred_values*SCALE)}
  estimator = tf.compat.v1.estimator.add_metrics(estimator,rmse)
  
  train_spec=tf.estimator.TrainSpec(
                       input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(x = traindf[["num_rooms"]],
                                              y = traindf["median_house_value"] / SCALE,  # note the scaling
                                              num_epochs = None,
                                              batch_size = 512, # note the batch size
                                              shuffle = True),
                       max_steps = num_train_steps)
  eval_spec=tf.estimator.EvalSpec(
                       input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(x = evaldf[["num_rooms"]],
                                              y = evaldf["median_house_value"] / SCALE,  # note the scaling
                                              num_epochs = 1,
                                              shuffle = False),
                       steps = None,
                       start_delay_secs = 1, # start evaluating after N seconds
                       throttle_secs = 10,  # evaluate every N seconds
                       )
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

# Run training    
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time
train_and_evaluate(OUTDIR, num_train_steps = 100)
