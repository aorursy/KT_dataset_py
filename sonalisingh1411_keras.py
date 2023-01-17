# Run in vscode or as a script python C:\Spark\tf-with_anim.py

import tensorflow as tf

import numpy as np

import math

import matplotlib.pyplot as plt

import matplotlib.animation as animation



num_house = 60

np.random.seed(42)

house_size = np.random.randint(low=100,high=3500,size=num_house)

house_size



np.random.seed(42)

house_price = house_size*100.0 + np.random.randint(low=2000,high=70000,size=num_house)

house_price



plt.plot(house_size,house_price,"bx")

plt.ylabel("Price")

plt.xlabel("Size")

plt.show()







def normalize(array):

    return (array-array.mean())/array.std()



num_train_samples = math.floor(num_house*0.7)

num_train_samples



train_house_size = np.asarray(house_size[:num_train_samples])

train_house_size





train_price = np.asanyarray(house_price[:num_train_samples:])

train_price



train_house_size_norm = normalize(train_house_size)

train_house_size_norm



train_price_norm = normalize(train_price)

train_price_norm



test_house_size = np.array(house_size[num_train_samples:])

test_house_price = np.array(house_price[num_train_samples:])



test_house_size_norm = normalize(test_house_size)

test_house_price_norm = normalize(test_house_price)



tf_house_size = tf.placeholder("float",name="house_size")

tf_price = tf.placeholder("float",name="price")



tf_size_factor = tf.Variable(np.random.randn(),name="size_factor")

tf_size_factor



tf_price_offset = tf.Variable(np.random.randn(),name="price_offset")

tf_price_offset



tf_price_pred = tf.add(tf.multiply(tf_size_factor,tf_house_size),tf_price_offset)

tf_price_pred



tf_cost = tf.reduce_sum(tf.pow(tf_price_pred-tf_price, 2))/(2*num_train_samples)

tf_cost



learning_rate = 0.1

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)



init = tf.global_variables_initializer()



with tf.Session() as sess:

    sess.run(init)

    

    display_every=2

    num_training_iter = 50

    

    fit_num_plots = math.floor(num_training_iter/display_every)

    fit_size_factor = np.zeros(fit_num_plots)

    fit_price_offsets = np.zeros(fit_num_plots)

    fit_plot_idx = 0

    

    # keep iterating the training data

    for iteration in range(num_training_iter):



        # Fit all training data

        for (x, y) in zip(train_house_size_norm, train_price_norm):

            sess.run(optimizer, feed_dict={tf_house_size: x, tf_price: y})



        # Display current status

        if (iteration + 1) % display_every == 0:

            c = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price:train_price_norm})

            print("iteration #:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(c),                 "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset))

            # Save the fit size_factor and price_offset to allow animation of learning process

            fit_size_factor[fit_plot_idx] = sess.run(tf_size_factor)

            fit_price_offsets[fit_plot_idx] = sess.run(tf_price_offset)

            fit_plot_idx = fit_plot_idx + 1



    print("Optimization Finished!")

    training_cost = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price: train_price_norm})

    print("Trained cost=", training_cost, "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset), '\n')

    train_house_size_std = train_house_size.std()

    #print('train_house_size_std =' + train_house_size_std )

    train_house_size_mean = train_house_size.mean()

    train_price_std = train_price.std()

    train_price_mean = train_price.mean()

    

    





# 

# Plot another graph that animation of how Gradient Descent sequentually adjusted size_factor and price_offset to 

# find the values that returned the "best" fit line.

fig, ax = plt.subplots()

line, = ax.plot(house_size, house_price)



plt.rcParams["figure.figsize"] = (10,8)

plt.title("Gradient Descent Fitting Regression Line")

plt.ylabel("Price")

plt.xlabel("Size (sq.ft)")

plt.plot(train_house_size, train_price, 'go', label='Training data')

plt.plot(test_house_size, test_house_price, 'mo', label='Testing data')

plt.show()


