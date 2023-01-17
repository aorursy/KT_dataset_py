import tensorflow as tf
x=tf.placeholder(tf.float32)
#notice the difference among "placeholder(), Variable(), constant()"
w=tf.Variable([-.3],tf.float32)
b=tf.Variable([.3],tf.float32)

linear_model=w*x+b

y=tf.placeholder(tf.float32)

square_delta=tf.square(linear_model-y)

loss=tf.reduce_sum(square_delta)

#optimize with learning rate of 0.01
optimizer=tf.train.GradientDescentOptimizer( 0.01)
train=optimizer.minimize(loss)

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
for i in range(1000):
    sess.run(train, {x:[1,2,3,4],y:[0,-1,-2,-3]})
    
print(sess.run([w,b]))
