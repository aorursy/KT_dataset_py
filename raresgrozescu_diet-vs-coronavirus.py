#@title Import relevant modules



import pandas as pd

import tensorflow as tf

import numpy as np

from matplotlib import pyplot as plt

from tensorflow.keras import layers



# The following lines adjust the granularity of reporting. 

pd.options.display.max_rows = 30

pd.options.display.float_format = "{:.8f}".format
column_names = [

  'country',

  'alchoholic_beverages',

  'animal_products',

  'animal_fats',

  'aquatic_products',

  'cereals_excluding_beer',

  'eggs',

  'fish_and_seafood',

  'fruits',

  'meat',

  'miscellaneous',

  'milk_excluding_butter',

  'offals',

  'oilcrops',

  'pulses',

  'spices',

  'starchy_roots',

  'stimulants',

  'sugar_crops',

  'sugar_and_sweeteners',

  'treenuts',

  'vegetal_products',

  'vegetal_oils',

  'vegetables',

  'obesity',

  'undernourished',

  'confirmed',

  'deaths',

  'recovered',

  'active',

  'population',

  'unit',

]



diet_data = pd.read_csv(

  filepath_or_buffer='https://raw.githubusercontent.com/GrozescuRares/diet_vs_corona/master/diet_vs_corona.csv',

  skiprows=1,

  names=column_names,

)

diet_data = diet_data.reindex(np.random.permutation(diet_data.index))



diet_data.head()
#@title Get statistics on the dataset.



diet_data.describe()
#@title Get correlation matrix



diet_data.corr()
# Define features and labels.

feature_names = ['animal_products', 'cereals_excluding_beer', 'obesity', 'vegetal_products']

label_name = 'deaths'
# Inspect features data

diet_data[feature_names].head()
for feature_name in feature_names:

  diet_data.hist(column=feature_name)
# Get a data frame which only contains the features and the label

training_columns = feature_names + [label_name]

training_df = diet_data[training_columns]

training_df = training_df.astype(np.float32)



# Drop records with nan values

training_df = training_df.dropna()



print('Dropped records with missing values.')
def zscore(mean, std, val):

  epsilon = 0.000001

  

  return (val - mean) / (epsilon + std)



z_score_scaled_feature_names = ['animal_products', 'obesity', 'vegetal_products']

log_scaled_feature_names = ['cereals_excluding_beer']



training_df_copy = training_df.copy()

z_score_scaled_features = training_df_copy[z_score_scaled_feature_names].copy()



# Apply z-score on 'Animal Products', 'Obesity' and 'Vegetal Products'

for feature_name in z_score_scaled_feature_names:

  mean = z_score_scaled_features[feature_name].mean()

  std = z_score_scaled_features[feature_name].std()

  z_score_scaled_features[feature_name] = zscore(mean, std, z_score_scaled_features[feature_name])

  z_score_scaled_features.hist(column=feature_name)



log_scaled_features = training_df_copy[log_scaled_feature_names].copy()

for feature_name in log_scaled_feature_names:

  # Apply log scaling for 'Cereals - Excluding Beer'

  log_scaled_features[feature_name] = np.log(log_scaled_features[feature_name])

  log_scaled_features.hist(column=feature_name)
training_df[label_name] = training_df[label_name].astype(np.float32) * 100.0

training_df[label_name] = training_df[label_name].round(4)

training_df[label_name] = training_df[label_name].map(lambda val: np.log(val + 1))



training_df.describe()
training_df.hist(column=label_name)
from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(training_df[feature_names], training_df[label_name], test_size=0.10)



print('We have {} training records and {} records for evaluating the model.'.format(len(X_train), len(X_test)))
# Create the features normalized using z-score.

z_score_scaled_features = [

  tf.feature_column.numeric_column(

      feature_name,

      normalizer_fn=lambda val: zscore(X_train.mean()[feature_name], X_train.std()[feature_name], val),

  )

  for feature_name in z_score_scaled_feature_names

]



# Create the features normalized using log scaling

log_scaled_features = [

  tf.feature_column.numeric_column(

      feature_name,

      normalizer_fn=lambda val: tf.math.log(val),

  )

  for feature_name in log_scaled_feature_names

]



# Create the input layer

input_layer = layers.DenseFeatures(z_score_scaled_features + log_scaled_features)



print('Created input layer.')
def create_model(my_learning_rate, input_layer):

  """Create and compile a simple linear regression model."""



  model = tf.keras.models.Sequential()



  # Add the layer containing the feature columns to the model.

  model.add(input_layer)



  # Add one linear layer to the model to yield a simple linear regressor.

  model.add(tf.keras.layers.Dense(units=1, input_shape=(1, )))



  # Construct the layers into a model that TensorFlow can execute.

  model.compile(

    optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),

    loss='mean_squared_error',

    metrics=[tf.keras.metrics.RootMeanSquaredError()],

  )



  return model



print('Defined create_model function.')
def train_model(model, x, y, epochs, batch_size):

  """Feed a dataset into the model in order to train it."""



  features = {name:np.array(value) for name, value in x.items()}

  label = y.to_numpy()



  history = model.fit(

    x=features,

    y=label,

    batch_size=batch_size,

    epochs=epochs,

    shuffle=True,

  )



  # The list of epochs is stored separately from the rest of history.

  epochs = history.epoch

  

  # Isolate the mean absolute error for each epoch.

  hist = pd.DataFrame(history.history)

  rmse = hist['root_mean_squared_error']



  return epochs, rmse



print('Defined train_model function.')   
def plot_the_loss_curve(epochs, rmse):

  """Plot a curve of loss vs. epoch."""



  plt.figure()

  plt.xlabel('Epoch')

  plt.ylabel('Root Mean Squared Error')



  plt.plot(epochs, rmse, label="Loss")

  plt.legend()

  plt.ylim([rmse.min()*0.94, rmse.max()* 1.05])

  plt.show()



print('Defined plot function.')
# The following variables are the hyperparameters.

learning_rate = 0.003

epochs = 64

batch_size = 12



# Create and compile the model.

model = create_model(learning_rate, input_layer)



# Train the model on the training set.

epochs, rmse = train_model(model, X_train, Y_train, epochs, batch_size)



plot_the_loss_curve(epochs, rmse)
print("\n: Evaluate the new model against the test set:")



test_features = {name:np.array(value) for name, value in X_test.items()}



results = model.evaluate(x=test_features, y=Y_test.to_numpy(), batch_size=batch_size)
new_data = {

  'animal_products': [17.7],

  'cereals_excluding_beer': [7.9],

  'obesity': [10.5],

  'vegetal_products': [26.2],

}



new_data = {name:np.array(value) for name, value in new_data.items()}



results = model.predict(new_data)



print('The predicted deaths percentage is {}.'.format(results[0][0]))
#@title Import relevant modules



import pandas as pd

import numpy as np

from matplotlib import pyplot as plt



# The following lines adjust the granularity of reporting. 

pd.options.display.max_rows = 30

pd.options.display.float_format = "{:.8f}".format
column_names = [

  'country',

  'alchoholic_beverages',

  'animal_products',

  'animal_fats',

  'aquatic_products',

  'cereals_excluding_beer',

  'eggs',

  'fish_and_seafood',

  'fruits',

  'meat',

  'miscellaneous',

  'milk_excluding_butter',

  'offals',

  'oilcrops',

  'pulses',

  'spices',

  'starchy_roots',

  'stimulants',

  'sugar_crops',

  'sugar_and_sweeteners',

  'treenuts',

  'vegetal_products',

  'vegetal_oils',

  'vegetables',

  'obesity',

  'undernourished',

  'confirmed',

  'deaths',

  'recovered',

  'active',

  'population',

  'unit',

]

used_column_names = [

  'animal_products',

  'cereals_excluding_beer',

  'vegetal_products',

  'obesity',

  'deaths',

]



diet_data_simple = pd.read_csv(

  filepath_or_buffer='https://raw.githubusercontent.com/GrozescuRares/diet_vs_corona/master/diet_vs_corona.csv',

  skiprows=1,

  names=column_names,

  usecols=used_column_names,

)



diet_data_simple = diet_data_simple.dropna()

diet_data_simple.head()
#@title Get statistics on the dataset.



diet_data_simple.describe()
#@title Sort data by deaths

diet_data_sorted = diet_data_simple.sort_values(by=['deaths'])



diet_data_sorted
#@title Separate data in groups

diet_data_sorted = diet_data_sorted[diet_data_sorted.deaths != 0.0]



highest_deaths_rate_data = diet_data_sorted.tail(10)

lowest_deaths_rate_data = diet_data_sorted.head(10)



print('Data was separated in two groups by deaths rate.')
#@title Check records with the highest death rate



highest_deaths_rate_data
#@title Check records with the lowest death rate



lowest_deaths_rate_data
#@title Compute average for both groups



highest_deaths_rate_mean = {column_name:highest_deaths_rate_data[column_name].mean() for column_name in used_column_names[:-1]}

print('Average values for records with highest death rate: \n{}'.format(highest_deaths_rate_mean))



lowest_deaths_rate_mean = {column_name:lowest_deaths_rate_data[column_name].mean() for column_name in used_column_names[:-1]}

print('Average values for records with lowest death rate: \n{}'.format(lowest_deaths_rate_mean))
#@title Visualize charts



labels = used_column_names[:-1]



x = np.array([0, 2, 4, 6])  # the label locations

width = 0.7  # the width of the bars



fig, ax = plt.subplots(figsize=(20, 12))

rects1 = ax.bar(x - width/2, highest_deaths_rate_mean.values(), width, label='High deaths rate group')

rects2 = ax.bar(x + width/2, lowest_deaths_rate_mean.values(), label='Low deaths rate group')



# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_ylabel('Percentage')

ax.set_title('Percentage of fat income')

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.legend(loc='upper right')



def autolabel(rects):

    """Attach a text label above each bar in *rects*, displaying its height."""

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(0, 1),  # 3 points vertical offset

                    textcoords="offset points",

                    ha='center', va='bottom')

autolabel(rects1)

autolabel(rects2)



plt.show()