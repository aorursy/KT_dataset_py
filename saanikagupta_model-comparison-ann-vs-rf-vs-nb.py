from google.cloud import bigquery

import matplotlib.pyplot as plt

import matplotlib.ticker as mtick

import matplotlib.patheffects as PathEffects

import matplotlib.pylab as pylab

import numpy as np

import pandas as pd

import itertools

from sklearn.metrics import  confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.manifold import TSNE

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

import time

import seaborn as sns

from keras import utils, optimizers

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.losses import binary_crossentropy
miner_limit = 5000

non_miner_limit = 5000
# SQL query adapted from https://gist.github.com/allenday/16cf63fb6b3ed59b78903b2d414fe75b

sql = '''

WITH 

output_ages AS (

  SELECT

    ARRAY_TO_STRING(outputs.addresses,',') AS output_ages_address,

    MIN(block_timestamp_month) AS output_month_min,

    MAX(block_timestamp_month) AS output_month_max

  FROM `bigquery-public-data.crypto_bitcoin.transactions` AS transactions JOIN UNNEST(outputs) AS outputs

  GROUP BY output_ages_address

)

,input_ages AS (

  SELECT

    ARRAY_TO_STRING(inputs.addresses,',') AS input_ages_address,

    MIN(block_timestamp_month) AS input_month_min,

    MAX(block_timestamp_month) AS input_month_max

  FROM `bigquery-public-data.crypto_bitcoin.transactions` AS transactions JOIN UNNEST(inputs) AS inputs

  GROUP BY input_ages_address

)

,output_monthly_stats AS (

  SELECT

    ARRAY_TO_STRING(outputs.addresses,',') AS output_monthly_stats_address, 

    COUNT(DISTINCT block_timestamp_month) AS output_active_months,

    COUNT(outputs) AS total_tx_output_count,

    SUM(value) AS total_tx_output_value,

    AVG(value) AS mean_tx_output_value,

    STDDEV(value) AS stddev_tx_output_value,

    COUNT(DISTINCT(`hash`)) AS total_output_tx,

    SUM(value)/COUNT(block_timestamp_month) AS mean_monthly_output_value,

    COUNT(outputs.addresses)/COUNT(block_timestamp_month) AS mean_monthly_output_count

  FROM `bigquery-public-data.crypto_bitcoin.transactions` AS transactions JOIN UNNEST(outputs) AS outputs

  GROUP BY output_monthly_stats_address

)

,input_monthly_stats AS (

  SELECT

    ARRAY_TO_STRING(inputs.addresses,',') AS input_monthly_stats_address, 

    COUNT(DISTINCT block_timestamp_month) AS input_active_months,

    COUNT(inputs) AS total_tx_input_count,

    SUM(value) AS total_tx_input_value,

    AVG(value) AS mean_tx_input_value,

    STDDEV(value) AS stddev_tx_input_value,

    COUNT(DISTINCT(`hash`)) AS total_input_tx,

    SUM(value)/COUNT(block_timestamp_month) AS mean_monthly_input_value,

    COUNT(inputs.addresses)/COUNT(block_timestamp_month) AS mean_monthly_input_count

  FROM `bigquery-public-data.crypto_bitcoin.transactions` AS transactions JOIN UNNEST(inputs) AS inputs

  GROUP BY input_monthly_stats_address

)

,output_idle_times AS (

  SELECT

    address AS idle_time_address,

    AVG(idle_time) AS mean_output_idle_time,

    STDDEV(idle_time) AS stddev_output_idle_time

  FROM

  (

    SELECT 

      event.address,

      IF(prev_block_time IS NULL, NULL, UNIX_SECONDS(block_time) - UNIX_SECONDS(prev_block_time)) AS idle_time

    FROM (

      SELECT

        ARRAY_TO_STRING(outputs.addresses,',') AS address, 

        block_timestamp AS block_time,

        LAG(block_timestamp) OVER (PARTITION BY ARRAY_TO_STRING(outputs.addresses,',') ORDER BY block_timestamp) AS prev_block_time

      FROM `bigquery-public-data.crypto_bitcoin.transactions` AS transactions JOIN UNNEST(outputs) AS outputs

    ) AS event

    WHERE block_time != prev_block_time

  )

  GROUP BY address

)

,input_idle_times AS (

  SELECT

    address AS idle_time_address,

    AVG(idle_time) AS mean_input_idle_time,

    STDDEV(idle_time) AS stddev_input_idle_time

  FROM

  (

    SELECT 

      event.address,

      IF(prev_block_time IS NULL, NULL, UNIX_SECONDS(block_time) - UNIX_SECONDS(prev_block_time)) AS idle_time

    FROM (

      SELECT

        ARRAY_TO_STRING(inputs.addresses,',') AS address, 

        block_timestamp AS block_time,

        LAG(block_timestamp) OVER (PARTITION BY ARRAY_TO_STRING(inputs.addresses,',') ORDER BY block_timestamp) AS prev_block_time

      FROM `bigquery-public-data.crypto_bitcoin.transactions` AS transactions JOIN UNNEST(inputs) AS inputs

    ) AS event

    WHERE block_time != prev_block_time

  )

  GROUP BY address

)

--,miners AS (

--)



(SELECT

  TRUE AS is_miner,

  output_ages_address AS address,

  UNIX_SECONDS(CAST(output_ages.output_month_min AS TIMESTAMP)) AS output_month_min,

  UNIX_SECONDS(CAST(output_ages.output_month_max AS TIMESTAMP)) AS output_month_max,

  UNIX_SECONDS(CAST(input_ages.input_month_min AS TIMESTAMP)) AS input_month_min,

  UNIX_SECONDS(CAST(input_ages.input_month_max AS TIMESTAMP)) AS input_month_max,

  UNIX_SECONDS(CAST(output_ages.output_month_max AS TIMESTAMP)) - UNIX_SECONDS(CAST(output_ages.output_month_min AS TIMESTAMP)) AS output_active_time,

  UNIX_SECONDS(CAST(input_ages.input_month_max AS TIMESTAMP)) - UNIX_SECONDS(CAST(input_ages.input_month_min AS TIMESTAMP)) AS input_active_time,

  UNIX_SECONDS(CAST(output_ages.output_month_max AS TIMESTAMP)) - UNIX_SECONDS(CAST(input_ages.input_month_max AS TIMESTAMP)) AS io_max_lag,

  UNIX_SECONDS(CAST(output_ages.output_month_min AS TIMESTAMP)) - UNIX_SECONDS(CAST(input_ages.input_month_min AS TIMESTAMP)) AS io_min_lag,

  output_monthly_stats.output_active_months,

  output_monthly_stats.total_tx_output_count,

  output_monthly_stats.total_tx_output_value,

  output_monthly_stats.mean_tx_output_value,

  output_monthly_stats.stddev_tx_output_value,

  output_monthly_stats.total_output_tx,

  output_monthly_stats.mean_monthly_output_value,

  output_monthly_stats.mean_monthly_output_count,

  input_monthly_stats.input_active_months,

  input_monthly_stats.total_tx_input_count,

  input_monthly_stats.total_tx_input_value,

  input_monthly_stats.mean_tx_input_value,

  input_monthly_stats.stddev_tx_input_value,

  input_monthly_stats.total_input_tx,

  input_monthly_stats.mean_monthly_input_value,

  input_monthly_stats.mean_monthly_input_count,

  output_idle_times.mean_output_idle_time,

  output_idle_times.stddev_output_idle_time,

  input_idle_times.mean_input_idle_time,

  input_idle_times.stddev_input_idle_time

FROM

  output_ages, output_monthly_stats, output_idle_times,

  input_ages,  input_monthly_stats, input_idle_times

WHERE TRUE

  AND output_ages.output_ages_address = output_monthly_stats.output_monthly_stats_address

  AND output_ages.output_ages_address = output_idle_times.idle_time_address

  AND output_ages.output_ages_address = input_monthly_stats.input_monthly_stats_address

  AND output_ages.output_ages_address = input_ages.input_ages_address

  AND output_ages.output_ages_address = input_idle_times.idle_time_address

  AND output_ages.output_ages_address IN

(

  SELECT 

    ARRAY_TO_STRING(outputs.addresses,',') AS miner

  FROM 

  `bigquery-public-data.crypto_bitcoin.blocks` AS blocks,

  `bigquery-public-data.crypto_bitcoin.transactions` AS transactions JOIN UNNEST(outputs) AS outputs

  WHERE blocks.hash = transactions.block_hash 

    AND is_coinbase IS TRUE

    AND ( FALSE

      --

      -- miner signatures from https://en.bitcoin.it/wiki/Comparison_of_mining_pools

      --

      OR coinbase_param LIKE '%4d696e656420627920416e74506f6f6c%' --AntPool

      OR coinbase_param LIKE '%2f42434d6f6e737465722f%' --BCMonster

      --BitcoinAffiliateNetwork

      OR coinbase_param LIKE '%4269744d696e746572%' --BitMinter

      --BTC.com

      --BTCC Pool

      --BTCDig

      OR coinbase_param LIKE '%2f7374726174756d2f%' --Btcmp

      --btcZPool.com

      --BW Mining

      OR coinbase_param LIKE '%456c6967697573%' --Eligius

      --F2Pool

      --GHash.IO

      --Give Me COINS

      --Golden Nonce Pool

      OR coinbase_param LIKE '%2f627261766f2d6d696e696e672f%' --Bravo Mining

      OR coinbase_param LIKE '%4b616e6f%' --KanoPool

      --kmdPool.org

      OR coinbase_param LIKE '%2f6d6d706f6f6c%' --Merge Mining Pool

      --MergeMining

      --Multipool

      --P2Pool

      OR coinbase_param LIKE '%2f736c7573682f%' --Slush Pool

      --ZenPool.org

    )

  GROUP BY miner

  HAVING COUNT(1) >= 20 

)

LIMIT {})

UNION ALL

(SELECT

  FALSE AS is_miner,

  output_ages_address AS address,

  UNIX_SECONDS(CAST(output_ages.output_month_min AS TIMESTAMP)) AS output_month_min,

  UNIX_SECONDS(CAST(output_ages.output_month_max AS TIMESTAMP)) AS output_month_max,

  UNIX_SECONDS(CAST(input_ages.input_month_min AS TIMESTAMP)) AS input_month_min,

  UNIX_SECONDS(CAST(input_ages.input_month_max AS TIMESTAMP)) AS input_month_max,

  UNIX_SECONDS(CAST(output_ages.output_month_max AS TIMESTAMP)) - UNIX_SECONDS(CAST(output_ages.output_month_min AS TIMESTAMP)) AS output_active_time,

  UNIX_SECONDS(CAST(input_ages.input_month_max AS TIMESTAMP)) - UNIX_SECONDS(CAST(input_ages.input_month_min AS TIMESTAMP)) AS input_active_time,

  UNIX_SECONDS(CAST(output_ages.output_month_max AS TIMESTAMP)) - UNIX_SECONDS(CAST(input_ages.input_month_max AS TIMESTAMP)) AS io_max_lag,

  UNIX_SECONDS(CAST(output_ages.output_month_min AS TIMESTAMP)) - UNIX_SECONDS(CAST(input_ages.input_month_min AS TIMESTAMP)) AS io_min_lag,

  output_monthly_stats.output_active_months,

  output_monthly_stats.total_tx_output_count,

  output_monthly_stats.total_tx_output_value,

  output_monthly_stats.mean_tx_output_value,

  output_monthly_stats.stddev_tx_output_value,

  output_monthly_stats.total_output_tx,

  output_monthly_stats.mean_monthly_output_value,

  output_monthly_stats.mean_monthly_output_count,

  input_monthly_stats.input_active_months,

  input_monthly_stats.total_tx_input_count,

  input_monthly_stats.total_tx_input_value,

  input_monthly_stats.mean_tx_input_value,

  input_monthly_stats.stddev_tx_input_value,

  input_monthly_stats.total_input_tx,

  input_monthly_stats.mean_monthly_input_value,

  input_monthly_stats.mean_monthly_input_count,

  output_idle_times.mean_output_idle_time,

  output_idle_times.stddev_output_idle_time,

  input_idle_times.mean_input_idle_time,

  input_idle_times.stddev_input_idle_time

FROM

  output_ages, output_monthly_stats, output_idle_times,

  input_ages,  input_monthly_stats, input_idle_times

WHERE TRUE

  AND output_ages.output_ages_address = output_monthly_stats.output_monthly_stats_address

  AND output_ages.output_ages_address = output_idle_times.idle_time_address

  AND output_ages.output_ages_address = input_monthly_stats.input_monthly_stats_address

  AND output_ages.output_ages_address = input_ages.input_ages_address

  AND output_ages.output_ages_address = input_idle_times.idle_time_address

  AND output_ages.output_ages_address NOT IN

(

  SELECT 

    ARRAY_TO_STRING(outputs.addresses,',') AS miner

  FROM 

  `bigquery-public-data.crypto_bitcoin.blocks` AS blocks,

  `bigquery-public-data.crypto_bitcoin.transactions` AS transactions JOIN UNNEST(outputs) AS outputs

  WHERE blocks.hash = transactions.block_hash 

    AND is_coinbase IS TRUE

    AND ( FALSE

      --

      -- miner signatures from https://en.bitcoin.it/wiki/Comparison_of_mining_pools

      --

      OR coinbase_param LIKE '%4d696e656420627920416e74506f6f6c%' --AntPool

      OR coinbase_param LIKE '%2f42434d6f6e737465722f%' --BCMonster

      --BitcoinAffiliateNetwork

      OR coinbase_param LIKE '%4269744d696e746572%' --BitMinter

      --BTC.com

      --BTCC Pool

      --BTCDig

      OR coinbase_param LIKE '%2f7374726174756d2f%' --Btcmp

      --btcZPool.com

      --BW Mining

      OR coinbase_param LIKE '%456c6967697573%' --Eligius

      --F2Pool

      --GHash.IO

      --Give Me COINS

      --Golden Nonce Pool

      OR coinbase_param LIKE '%2f627261766f2d6d696e696e672f%' --Bravo Mining

      OR coinbase_param LIKE '%4b616e6f%' --KanoPool

      --kmdPool.org

      OR coinbase_param LIKE '%2f6d6d706f6f6c%' --Merge Mining Pool

      --MergeMining

      --Multipool

      --P2Pool

      OR coinbase_param LIKE '%2f736c7573682f%' --Slush Pool

      --ZenPool.org

    )

  GROUP BY miner

  HAVING COUNT(1) >= 20 

)

LIMIT {})

'''.format(miner_limit, non_miner_limit)
client = bigquery.Client()

df = client.query(sql).to_dataframe()
df.info()
# Dropping the columns with null values

df.drop(labels = ['stddev_output_idle_time','stddev_input_idle_time'], axis = 1, inplace = True)
df.tail(5)
df.head(5)
df.shape
# Dropping the non-numeric features

features = df.drop(labels = ['is_miner', 'address'], axis = 1)

target = df['is_miner'].values

indices = range(len(features))
sns.set_style('darkgrid')

sns.set_palette('muted')

sns.set_context("notebook", font_scale = 1.5, rc = {"lines.linewidth": 2.5})
# Utility function to visualize the outputs of t-SNE

def bitcoin_scatter(x, colors):

    # choose a color palette with seaborn.

    num_classes = len(np.unique(colors))

    palette = np.array(sns.color_palette("hls", num_classes))



    # create a scatter plot.

    f = plt.figure(figsize = (8, 8))

    ax = plt.subplot(aspect = 'equal')

    sc = ax.scatter(x[:,0], x[:,1], lw = 0, s = 40, c=palette[colors.astype(np.int)])

    plt.xlim(-25, 25)

    plt.ylim(-25, 25)

    ax.axis('off')

    ax.axis('tight')

    plt.title('t-SNE to visualize features')



    # add the labels for each digit corresponding to the label

    txts = []



    for i in range(num_classes):



        # Position of each label at median of data points.



        xtext, ytext = np.median(x[colors == i, :], axis = 0)

        txt = ax.text(xtext, ytext, str(i), fontsize = 24)

        txt.set_path_effects([

            PathEffects.Stroke(linewidth = 5, foreground = "w"),

            PathEffects.Normal()])

        txts.append(txt)



    return f, ax, sc, txts
time_start = time.time()

RS = 123

bitcoin_tsne = TSNE(random_state = RS).fit_transform(features)

print('Time elapsed: {} seconds' .format(time.time() - time_start))
bitcoin_scatter(bitcoin_tsne, target)
plt.savefig('tSNE.jpg')
# Splitting the training and testing dataset

x_train, x_test, y_train, y_test, indices_train, indices_test = train_test_split(features, target, indices,  test_size = 0.2)
x_train.head()
x_train.shape
y_train
# Feature Scaling

sc_x = StandardScaler()

x_train_ann = sc_x.fit_transform(x_train)

x_test_ann = sc_x.transform(x_test)
num_classes = 2



# Hyperparameters

learn_rate = 0.001

batch_size = 500

epochs = 270
# Encoding the Dependent Variable

labelencoder_y = LabelEncoder()

y_train_ann = labelencoder_y.fit_transform(y_train)



# Converting to binary class matrix

y_train_ann = utils.to_categorical(y_train_ann, num_classes)
y_train_ann.shape
# Encoding the Dependent Variable

labelencoder_y = LabelEncoder()

y_test_ann = labelencoder_y.fit_transform(y_test)



# Converting to binary class matrix

y_test_ann = utils.to_categorical(y_test_ann, num_classes)
y_test_ann.shape
seed = 1

np.random.seed(seed)



# Creating model

ann = Sequential()

ann.add(Dense(26, activation = 'tanh', kernel_initializer = 'glorot_uniform'))

ann.add(Dense(11, activation = 'tanh'))

ann.add(Dropout(0.5))

ann.add(Dense(6, activation = 'tanh'))

ann.add(Dense(num_classes, activation = 'softmax'))
rmsprop = optimizers.RMSprop(learn_rate)

ann.compile(loss = 'binary_crossentropy', optimizer = rmsprop, metrics = ['binary_accuracy']) # Compiling the model
# Model fitting

ann.fit(np.array(x_train_ann), y_train_ann, batch_size = batch_size, epochs = epochs, validation_data = (x_test_ann, y_test_ann))
ann.summary()
scores = ann.evaluate(x_test_ann, y_test_ann, verbose = 0)

print("Test Accuracy (Artificial Neural Network): {}%" .format(scores[1] * 100))
scores
y_pred = ann.predict(x_test_ann)



# Compute confusion matrix

matrix = confusion_matrix(y_test_ann.argmax(axis = 1), y_pred.argmax(axis = 1)) # Building the confusion matrix
matrix
# Training the model

rf = RandomForestClassifier(n_estimators = 100, class_weight = 'balanced')

rf.fit(x_train, y_train)
# Model predictions

y_pred = rf.predict(x_test)

probs = rf.predict_proba(x_test)[:, 1] # Positive class probabilities
params = {'legend.fontsize': 'small',

         'axes.labelsize': 'x-small',

         'axes.titlesize':'small',

         'xtick.labelsize':'x-small',

         'ytick.labelsize':'x-small'}

pylab.rcParams.update(params)
# Confusion matrix code adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion matrix', cmap = plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize = True.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)

    dummy = np.array([[0, 0], [0, 0]])

    plt.figure(figsize = (6, 6))

    plt.imshow(dummy, interpolation = 'nearest', cmap = cmap)

    plt.title(title)

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation = 45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment = "center",

                 color = "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()





# Compute confusion matrix

cnf_matrix = confusion_matrix(y_test, y_pred)

class_names = ['not mining pool', 'mining pool']

np.set_printoptions(precision = 2)



# Plot confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes = class_names, normalize = False, title = 'Bitcoin Mining Pool Detector using Random Forest - Confusion Matrix')



plt.show()
# Calculating Accuracy

acc = (cnf_matrix[0][0] + cnf_matrix[1][1]) / (cnf_matrix[0][0] + cnf_matrix[1][1] + cnf_matrix[0][1] + cnf_matrix[1][0])
print("Test Accuracy (Random Forest Classification): {}%" .format(acc * 100))
plt.savefig('RF_CM.jpg')
params = {'legend.fontsize': 'small',

         'axes.labelsize': 'small',

         'axes.titlesize':'small',

         'xtick.labelsize':'small',

         'ytick.labelsize':'small'

         }

pylab.rcParams.update(params)
x_pos = np.arange(len(features.columns))

btc_importances = rf.feature_importances_



inds = np.argsort(btc_importances)[::-1]

btc_importances = btc_importances[inds]

cols = features.columns[inds]

bar_width = .8



# How many features to plot?

n_features = 26

x_pos = x_pos[:n_features][::-1]

btc_importances = btc_importances[:n_features]



# Plot

plt.figure(figsize = (26, 10))

plt.barh(x_pos, btc_importances, bar_width, label = 'BTC model')

plt.yticks(x_pos, cols, rotation = 0, fontsize = 14)

plt.xlabel('feature importance', fontsize = 14)

plt.title('Mining Pool Detector', fontsize = 20)

plt.tight_layout()
plt.savefig('RFfeatureImportance.jpg')
# Data points where model predicts true, but are labelled as false

false_positives = (y_test == False) & (y_pred == True)
# Subset to test set data only

df_test = df.iloc[indices_test, :]



print('False Positive addresses')



# Subset test set to false positives only

df_test.iloc[false_positives].head(15)
index = inds[: -1]

index
data_top = x_train.columns

xann = pd.DataFrame(data = x_train_ann[0:, 0:], index = [i for i in range(x_train_ann.shape[0])], columns = [str(i) for i in data_top])

xtann = pd.DataFrame(data = x_test_ann[0:, 0:], index = [i for i in range(x_test_ann.shape[0])], columns = [str(i) for i in data_top])

xrf = x_train

xtrf = x_test

i = 1
ann_acc = [scores[1] * 100]

rf_acc = [acc * 100]

print("{}- All 26 features taken as input:" .format(i))

print("ANN test accuracy = {}%, RF test accuracy = {}%" .format(ann_acc[0], rf_acc[0]))

for x in reversed(index):

    xann = xann.drop(columns = [data_top[x]])	

    xtann = xtann.drop(columns = [data_top[x]])

    xrf = xrf.drop(columns = [data_top[x]])

    xtrf = xtrf.drop(columns = [data_top[x]])

    

    # Creating model

    ann = Sequential()

    ann.add(Dense(26, activation = 'tanh', kernel_initializer = 'glorot_uniform'))

    ann.add(Dense(11, activation = 'tanh'))

    ann.add(Dropout(0.5))

    ann.add(Dense(6, activation = 'tanh'))

    ann.add(Dense(num_classes, activation = 'softmax'))



    rmsprop = optimizers.RMSprop(learn_rate)

    ann.compile(loss = 'binary_crossentropy', optimizer = rmsprop, metrics = ['binary_accuracy']) # Compiling the model



    # Model fitting

    ann.fit(np.array(xann), y_train_ann, batch_size = batch_size, verbose = 0, epochs = epochs)

    scores_ann = ann.evaluate(xtann, y_test_ann, verbose = 0)



    # Training the model

    rf = RandomForestClassifier(n_estimators = 100, class_weight = 'balanced')

    rf.fit(xrf, y_train)



    # Model predictions

    y_pred = rf.predict(xtrf)

    cnf_matrix = confusion_matrix(y_test, y_pred)

    acc = (cnf_matrix[0][0] + cnf_matrix[1][1]) / (cnf_matrix[0][0] + cnf_matrix[1][1] + cnf_matrix[0][1] + cnf_matrix[1][0])

    i += 1

    print("{}- Dropping {}:" .format(i, data_top[x]))

    print("ANN test accuracy = {}%, RF test accuracy = {}%" .format(scores_ann[1] * 100, acc * 100))

    ann_acc.append(scores_ann[1] * 100)

    rf_acc.append(acc * 100)
# Create plots with pre-defined labels

f = plt.figure(figsize = (10, 10))

ax = f.add_subplot(121)

t = list(np.arange(1., 27., 1))

t.reverse()

ax.plot(t, rf_acc, 'r-', label = 'Random Forest')

ax.plot(t, ann_acc, 'b--', label = 'Artificial Neural Network')



legend = ax.legend(loc = 'lower right', shadow = True, fontsize = 'x-small')



# Put a nicer background color on the legend.

legend.get_frame().set_facecolor('C6')

plt.ylabel('Test Accuracy (%)')

plt.xlabel('Number of features')

plt.title('RF vs ANN (Accuracy Comparison)')

plt.show()
plt.savefig('RFvsANN.jpg')
# Fitting Naive Bayes to the Training set

nb = GaussianNB()

nb.fit(x_train, y_train)
# Predicting the Test set results

y_pred = nb.predict(x_test)
# Compute confusion matrix

cnf_matrix = confusion_matrix(y_test, y_pred)

class_names = ['not mining pool', 'mining pool']

np.set_printoptions(precision = 3)



# Plot confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes = class_names, normalize = False, title = 'Bitcoin Mining Pool Detector using Naive Bayes - Confusion Matrix')



plt.show()
# Calculating Accuracy

acc = (cnf_matrix[0][0] + cnf_matrix[1][1]) / (cnf_matrix[0][0] + cnf_matrix[1][1] + cnf_matrix[0][1] + cnf_matrix[1][0])

print("Test Accuracy (Naive Bayes Classification): {}%" .format(acc * 100))
plt.savefig('NaiveBayesCM.jpg')