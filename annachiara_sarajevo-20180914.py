import numpy as np
import tensorflow as tf
import time
num_samples = 100
data_size = (1920, 1080)
data = np.random.random((num_samples, *data_size))
tic = time.time()

for i in range(num_samples-1):
    s = data[i] - data[i+1]

toc = time.time()
print (toc-tic)
curr_frame = data[0]
for i in range(1, num_samples):
    next_frame = data[i]
    diff = next_frame - curr_frame
    curr_frame = next_frame
    
tf.reset_default_graph()

# Switch between /gpu:0 and /cpu:0 to test execution on GPU and CPU
with tf.device("/gpu:0"):  
    tf_prev_frame = tf.placeholder(shape=data_size, dtype=tf.float32)
    tf_next_frame = tf.placeholder(shape=data_size, dtype=tf.float32)

    tf_diff = tf_next_frame - tf_prev_frame
tic = time.time()
with tf.Session() as session:
    for i in range(num_samples-1):
        p = {
            tf_prev_frame: data[i],
            tf_next_frame: data[i+1]
        }
        diff = session.run(tf_diff.op, p)
        
toc = time.time()
print (toc-tic)
frame_A = None
frame_B = None

def diff_A_B():
    return frame_A - frame_B

def diff_B_A():
    return frame_B - frame_A

tic = time.time()
frame_A = data[0]

for i in range(1, num_samples):
    if i % 2 == 0:
        frame_A = data[i]
        diff = diff_B_A()
    else:
        frame_B = data[i]
        diff = diff_A_B()
    
toc = time.time()
print(toc - tic)
tf.reset_default_graph()
with tf.device("/gpu:0"):

    tf_A = tf.get_variable(name="TF_A", shape=data_size)
    tf_B = tf.get_variable(name="TF_B", shape=data_size)
    tf_frame_reg = tf.placeholder(shape=data_size, dtype=tf.float32)

    tf_set_A = tf.assign(tf_A, tf_frame_reg)
    tf_set_B = tf.assign(tf_B, tf_frame_reg)

    tf_diff_A_B = tf_A - tf_B
    tf_diff_B_A = tf_B - tf_A
tic = time.time()
with tf.Session() as session:
    # Init
    session.run(tf_set_A.op, {tf_frame_reg:data[0]})
    
    for i in range(1, num_samples):
        if i % 2 == 0:
            session.run(tf_set_A.op, {tf_frame_reg: data[i]})
            session.run(tf_diff_B_A.op)
        else:
            session.run(tf_set_B.op, {tf_frame_reg: data[i]})
            session.run(tf_diff_A_B.op)
        
toc = time.time()
print (toc-tic)
tic = time.time()

for i in range(num_samples-1):
    s = data[i].dot(data[i+1].T)

toc = time.time()
print (toc-tic)
tf_dot_A_B = tf.matmul(tf_A, tf.transpose(tf_B))
tf_dot_B_A = tf.matmul(tf_B, tf.transpose(tf_A))

tic = time.time()
with tf.Session() as session:
    # Init
    session.run(tf_set_A.op, {tf_frame_reg:data[0]})
    
    for i in range(1, num_samples):
        if i % 2 == 0:
            session.run(tf_set_A.op, {tf_frame_reg: data[i]})
            session.run(tf_dot_B_A.op)

            
        else:
            session.run(tf_set_B.op, {tf_frame_reg: data[i]})
            session.run(tf_dot_A_B.op)
        
toc = time.time()
print (toc-tic)
# Tip: try to increase the computation on the GPU, maybe executing more dot products, 
# to see how the total execution time increases 