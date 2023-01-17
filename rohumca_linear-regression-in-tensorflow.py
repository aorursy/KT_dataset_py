import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import time

batch_size = 32
learning_rate = 0.003
n_epoches = 6000

df1 = pd.read_csv('../input/correlate-artificial_intelligence.csv')
df1 = df1[['neural network', 'artificial intelligence']]

# splitting the data into train and test dataframes 
train_d = df1.sample(frac=0.7, random_state=101)
test_d = df1.drop(train_d.index)

# converting train and test dataframes to matices to creat datasets
train_d = train_d.astype('float32').as_matrix()
test_d = test_d.astype('float32').as_matrix()

# creating datasets
train_dset = tf.data.Dataset.from_tensor_slices((train_d[:,0], train_d[:,1]))
test_dset = tf.data.Dataset.from_tensor_slices((test_d[:,0], test_d[:,1]))

# combining consecutive elements of the train dataset into batches
train_dset = train_dset.batch(batch_size)

# creating an (uninitialized) iterator for enumerating the elements of the dataset with the given structure
iterator = tf.data.Iterator.from_structure(train_dset.output_types, train_dset.output_shapes)
train_init = iterator.make_initializer(train_dset)

# get_next() returns a nested structure of `tf.Tensor`s containing the next element
X, Y = iterator.get_next()

def R_squared(y, y_pred):
    '''
    R_squared computes the coefficient of determination.
    It is a measure of how well the observed outcomes are replicated by the model.
    '''
    residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
    r2 = tf.subtract(1.0, tf.div(residual, total))
    return r2

# Model
w = tf.Variable(tf.truncated_normal((1,), mean=0, stddev=0.1, seed=123), name='Weight')
b = tf.Variable(tf.constant(0.1), name='Bias')
y_pred = tf.multiply(w, X) + b
    
# Cost function
loss = tf.reduce_mean(tf.square(Y - y_pred), name='Loss')
        
# training
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

start = time.time()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(n_epoches):
    sess.run(train_init)
    try:
        # Loop until all elements have been consumed
        while True:
            sess.run(optimizer)
    except tf.errors.OutOfRangeError:
        pass

end = time.time()

w_curr, b_curr = sess.run([w,b])
y_pred_train = w_curr * train_d[:,0] + b_curr
y_pred_test = w_curr * test_d[:,0] + b_curr

r2_train = R_squared(train_d[:,1], y_pred_train)
r2_test = R_squared(test_d[:,1], y_pred_test)

print('R^2_train:', sess.run(r2_train))
print('R^2_test:', sess.run(r2_test))
print('elapsed time:', end - start)
sess.close()
plt.scatter(test_d[:,0], test_d[:,1], label='Real Data')
plt.plot(test_d[:,0], y_pred_test, 'r', label='Predicted Data')
plt.xlabel('Neural Network') 
plt.ylabel('Artificial Intelligence')
plt.legend();