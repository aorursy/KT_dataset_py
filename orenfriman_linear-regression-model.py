# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# func that returns a dummified DataFrame of significant dummies in a given column
def dum_sign(dummy_col, threshold=0.1):
    dummy_col = dummy_col.copy()
    count = pd.value_counts(dummy_col) / len(dummy_col)
    mask = dummy_col.isin(count[count > threshold].index)
    dummy_col[~mask] = "others"
    return pd.get_dummies(dummy_col, prefix=dummy_col.name)

# Importing data
products = pd.read_csv("../input/olist_products_dataset.csv")
translation = pd.read_csv("../input/product_category_name_translation.csv")
items = pd.read_csv("../input/olist_order_items_dataset.csv")
# Data preparation 
dp0 = products.join(translation.set_index('product_category_name'), on='product_category_name')
dp1 = items.join(dp0.set_index('product_id'), on='product_id')

# Data exploration
dp1.groupby('product_category_name_english')['product_id'].nunique().sort_values(ascending=False).plot(kind='bar', figsize=(15, 10), fontsize=12)
dp1.plot(kind='scatter', x='product_name_lenght', y='price')
dp1.plot(kind='scatter', x='product_description_lenght', y='price')
dp1.plot(kind='scatter', x='product_photos_qty', y='price')                                                   
dp1.plot(kind='scatter', x='product_weight_g', y='price')
dp1.plot(kind='scatter', x='product_length_cm', y='price')
dp1.plot(kind='scatter', x='product_height_cm', y='price')
dp1.plot(kind='scatter', x='product_width_cm', y='price')

# Nulls manipulation
df = dp1.dropna()

# Feature selection
features_cols = ['product_name_lenght', 'product_description_lenght', 'product_photos_qty', 'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm', 'freight_value']
predict_cols = ['price']
cat_cols = ['seller_id']
extra_hot_df = dum_sign(df['product_category_name_english'], 0.01)
extra_cat_df = df[cat_cols].apply(lambda x: pd.factorize(x)[0])
dfx = pd.concat([df[features_cols], extra_hot_df, extra_cat_df], axis=1)
dfy = df[predict_cols]

#Data preparation
train_dfx = dfx.sample(frac=0.8,random_state=200)
train_dfy = dfy.sample(frac=0.8,random_state=200)
test_dfx = dfx.drop(train_dfx.index)
test_dfy = dfy.drop(train_dfy.index)

#Normalization
train_x_mean = train_dfx.mean()
train_x_std = train_dfx.std()
train_dfx = (train_dfx - train_x_mean)/train_x_std
test_dfx = (test_dfx - train_x_mean)/train_x_std

print("train data contains", train_dfx.shape[0], "rows")
print("test data contains", test_dfx.shape[0], "rows")
learning_rate = 1e-5
training_epochs = 10000000
batch_size = 500 # Mini-Batch Gradient Descent 
display = 50000
seed = 200
(hidden1_size, hidden2_size) = (100, 50)

X = tf.placeholder(tf.float32, shape=[None,train_dfx.shape[1]])
Y = tf.placeholder(tf.float32, shape=[None, train_dfy.shape[1]])
training_dataset = tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(500).repeat().batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((X, Y)).repeat().batch(test_dfx.shape[0])
iterator = tf.data.Iterator.from_structure(training_dataset.output_types,training_dataset.output_shapes)
dx, dy = iterator.get_next()
training_init_op = iterator.make_initializer(training_dataset)
test_init_op = iterator.make_initializer(test_dataset)
with tf.Session() as sess:
    W1 = tf.Variable(tf.random_normal([train_dfx.shape[1], hidden1_size], seed = seed))
    b1 = tf.Variable(tf.random_normal([hidden1_size], seed = seed))
    z1 = tf.nn.relu(tf.add(tf.matmul(dx,W1), b1))
    W2 = tf.Variable(tf.random_normal([hidden1_size, hidden2_size], stddev=0.1))
    b2 = tf.Variable(tf.random_normal([hidden2_size], seed = seed))
    z2 = tf.nn.relu(tf.add(tf.matmul(z1,W2), b2))
    W3 = tf.Variable(tf.random_normal([hidden2_size, train_dfy.shape[1]], stddev=0.1))
    b3 = tf.Variable(tf.random_normal([train_dfy.shape[1]], seed = seed))
    h = tf.add(tf.matmul(z2,W3), b3)                                    
    loss = tf.reduce_mean(tf.pow(h - dy, 2)) + 0.1 * tf.nn.l2_loss(W3)
    update = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(training_init_op, feed_dict={X: train_dfx.values, Y: train_dfy.values})
    for epoch in range(training_epochs):
        sess.run(update)
        if (epoch + 1) % display == 0:
            print("iter: {}, loss: {:.3f}".format(epoch + 1, sess.run(loss)))            
print("Training Finished!")
print("Train MSE:", sess.run(loss))
sess.run(test_init_op, feed_dict={X: test_dfx.values, Y: test_dfy.values})
print("Test MSE:", sess.run(loss))