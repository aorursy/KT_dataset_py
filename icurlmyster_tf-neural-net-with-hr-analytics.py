import tensorflow as tf

import pandas as pd

import numpy as np
np.random.seed(7)

tf.set_random_seed(7)



init_data = pd.read_csv("../input/HR_comma_sep.csv")
print(init_data.info())
print("Sales: {0}".format(init_data["sales"][:5]))

print("Salary: {0}".format(init_data["salary"][:5]))
sales_unique_n = init_data["sales"].nunique()

salary_unique_n = init_data["salary"].nunique()

print("Unique sale categories: {0}".format(sales_unique_n))

print("Unique salary categories: {0}".format(salary_unique_n))
sales_unique_feature_names = init_data["sales"].unique()

salary_unique_feature_names = init_data["salary"].unique()



# Function to breakdown a category into individual binary features

def break_down_features(feature_list, category, orig_data):

    for name in feature_list:

        orig_data[category+"_"+name] = [1 if x == name else 0 for _, x in enumerate(orig_data[category])]



break_down_features(sales_unique_feature_names, "sales", init_data)

break_down_features(salary_unique_feature_names, "salary", init_data)
init_data = init_data.drop(["sales", "salary"], axis=1)
print(init_data["left"].value_counts() / len(init_data["left"]))
def stratified_split_data(data, ratio):

    # Grab the data into its own category

    stayed_data = data.loc[data["left"] == 0]

    left_data = data.loc[data["left"] == 1]

    # mix up the data

    stayed_data = stayed_data.iloc[np.random.permutation(len(stayed_data))]

    left_data = left_data.iloc[np.random.permutation(len(left_data))]

    test_stayed_set_size = int(len(stayed_data) * ratio)

    test_left_set_size = int(len(left_data) * ratio)

    # Concatenate the partitioned data

    train_set = pd.concat([stayed_data[test_stayed_set_size:], left_data[test_left_set_size:]], ignore_index=True)

    test_set = pd.concat([stayed_data[:test_stayed_set_size], left_data[:test_left_set_size]], ignore_index=True)

    # Now mix up the concatenated data

    train_shuffled_indices = np.random.permutation(len(train_set))

    test_shuffled_indices = np.random.permutation(len(test_set))

    return train_set.iloc[train_shuffled_indices], test_set.iloc[test_shuffled_indices]



train_set, test_set = stratified_split_data(init_data, 0.2)
print(train_set["left"].value_counts() / len(train_set["left"]))
data = (train_set.drop("left", axis=1)).values

data_labels = train_set["left"].values

data_labels = data_labels.reshape([len(data_labels), 1])

num_features = data.shape[1]
X_init = tf.placeholder(tf.float32, [None, num_features])

Y_init = tf.placeholder(tf.float32, [None, 1])
w_1 = tf.Variable(tf.truncated_normal([num_features, 10], stddev=0.01))

b_1 = tf.Variable(tf.truncated_normal([10], stddev=0.01))



layer_1 = tf.nn.elu(tf.add(tf.matmul(X_init, w_1), b_1))
w_2 = tf.Variable(tf.truncated_normal([10, 8], stddev=0.01))

b_2 = tf.Variable(tf.truncated_normal([8], stddev=0.01))



layer_2 = tf.nn.elu(tf.add(tf.matmul(layer_1, w_2), b_2))
w_3 = tf.Variable(tf.truncated_normal([8, 1], stddev=0.01))

b_3 = tf.Variable(tf.truncated_normal([1], stddev=0.01))



output_layer = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, w_3), b_3))
cost = -tf.reduce_mean(tf.multiply(Y_init, tf.log(output_layer)) + (1 - Y_init)*tf.log(1 - output_layer) )
optimizer = tf.train.AdamOptimizer(1e-3).minimize(cost)
init = tf.global_variables_initializer()



sess = tf.Session()



sess.run(init)



loss_values = []
num_epochs = 600

batch_size = 50

count = len(data) # helper variable for our mini-batch training
for epoch in range(num_epochs):

    start_n = 0

    c = None

    while start_n < count:

        sess.run(optimizer, feed_dict={X_init:data[start_n:(start_n + batch_size)], Y_init:data_labels[start_n:(start_n + batch_size)]})

        start_n += batch_size

    c = sess.run(cost, feed_dict={X_init:data, Y_init:data_labels})

    loss_values.append(c)

print("Final cost = {0}".format(sess.run(cost, feed_dict={X_init:data, Y_init:data_labels})) )
import matplotlib.pyplot as plt

%matplotlib inline

plt.plot(loss_values);
predictions = sess.run(output_layer, feed_dict={X_init:data})
def confusion_matrix(pred_data, act_data, threshold=0.7):

    stayed_true = 0

    stayed_false = 0

    left_true = 0

    left_false = 0

    for i in range(len(pred_data)):

        if pred_data[i][0] >= threshold and act_data[i][0] == 1:

            left_true += 1

        elif pred_data[i][0] < threshold and act_data[i][0] == 1:

            left_false += 1

        elif pred_data[i][0] >= threshold and act_data[i][0] == 0:

            stayed_false += 1

        elif pred_data[i][0] < threshold and act_data[i][0] == 0:

            stayed_true += 1

    precision = left_true/np.max([1e-5, (left_true + left_false)])

    recall = left_true/np.max([1e-5, (left_true + stayed_false)])

    f1_score = 2*((precision*recall)/(precision+recall))

    print("Stayed True: {0}\nStayed False: {1}\nLeft True: {2}\nLeft False: {3}".format(stayed_true, stayed_false, left_true, left_false))

    print("Precision = {0}".format(precision))

    print("Recall = {0}".format(recall))

    print("F1 score = {0}".format(f1_score))

    print("Total Accuracy = {0}".format((stayed_true+left_true)/(len(pred_data))) )
confusion_matrix(predictions, data_labels)
confusion_matrix(predictions, data_labels, 0.33)
test_data = (test_set.drop("left", axis=1)).values

test_data_labels = test_set["left"].values

test_data_labels = test_data_labels.reshape([len(test_data_labels), 1])

test_predictions = sess.run(output_layer, feed_dict={X_init:test_data})



confusion_matrix(test_predictions, test_data_labels, 0.33)
confusion_matrix(test_predictions, test_data_labels)
print("Cost for train data: {0}".format(sess.run(cost, feed_dict={X_init:data, Y_init:data_labels})) )

print("Cost for test  data: {0}".format(sess.run(cost, feed_dict={X_init:test_data, Y_init:test_data_labels}) ) )

sess.close()