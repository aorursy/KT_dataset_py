# Import tensor library
import tensorflow as tf
with tf.name_scope('counter'):
    counter = tf.Variable(1, name="counter")
    tf.summary.scalar('counter', counter)
# Creation of a constant to allow multiplication
two_op = tf.constant(2, name="const")

# Operation to increase counter variable
new_value = tf.multiply(counter, two_op)
update = tf.assign(counter, new_value)

merged = tf.summary.merge_all()

# Initialize all variables
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)

    # Configure output
    summary_writer = tf.summary.FileWriter(".", sess.graph)

    for i in range(5):
        summary, _ = sess.run([merged, update])
        # write add data to the output
        summary_writer.add_summary(summary, i)
        print(sess.run(counter))