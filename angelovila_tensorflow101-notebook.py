import tensorflow as tf
node1 = tf.constant(3.0, tf.float32)  #can provide value and type
node2 = tf.constant(4.0)   #can provide only value
print(node1, node2)   
# will print out, but will not be evaluated until 
#it's part of the session

sess = tf.Session()
print(sess.run([node1,node2]))
node3 = tf.add(node1, node2)   #adder node, adds node1 and node2
print("node3: ", node3)  #will only print out a blah
print("sess.run(node3): ", sess.run(node3))   # will print out if it's in sessions
###tensorflow has operators are overloaded ###
###in basic wording, tensorflow can simply add the two nodes without needing to have the adder node

print("sess.run(node1 + node2): ", sess.run(node1+node2))
### placeholders ###
### this represents a structure of computation, it's 
### not doing anything until we pass data into it
### as part of a running session that it will execute

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
### running the adder node in a session
### passing values as a dictionary
print(sess.run(adder_node, {a:3, b:4.5}))
## can also pass a different shape tensor
print(sess.run(adder_node, {a:[1,3], b:[2,4]}))

add_and_triple = adder_node * 3
print(sess.run(add_and_triple, {a:3, b:4.5}))
#TensorBoard
import tensorflow as tf
a = tf.constant(5, name="input_a")
b = tf.constant(3, name="input_b")
c = tf.multiply(a,b, name="multiply_c")
d = tf.add(a,b, name="add_d")
e = tf.add(c,d, name="add_e")
sess = tf.Session()
output = sess.run(e)
#writer = tf.train.SummaryWriter('./my_graph', sess.graph)  #this has been depracated
writer = tf.summary.FileWriter('./my_graph', sess.graph)  #replaced with this instead
writer.close()
sess.close()
# image displaydoesn't work
# display(HTML('<img=src='tensorboard1.png> <img src='tensorboard2.png>'))



#more complicated examples

W = tf.Variable([.3], tf.float32)  #weight value
b = tf.Variable([-.3], tf.float32)  #bias value
x = tf.placeholder(tf.float32)
linear_model = W * x + b

#it's the calculation to get the slope
#y = Wx + b
#y = mx + b  #more popularly known as this equation

#we're using variables other than placeholders
#variables gets updated as we iterate over  learning phases
#placeholders get reset each time we pass through the graph


#why put values [.3] and [-.3]??
# we're seeding the model with this two values, these values are incorrect but
# as we learn these values would be adjusted and build a predictive model from the data
# we passed through it


# variables need to be initialized before running through a session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print(sess.run(linear_model, {x:[1,2,3,4]}))

#what we're doing here is we're running all the values in x
#and generate the results
    ###*remember y = Wx + b 
    # or
    ###linear_model = W * x + b
    
#without plotting the results for visualizations, it's hard 
#to know if the results are accurate based on our data


# to have a learning phase, we usually need to add a loss function or an error function

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y) #measures how far off every entry in the linear model
loss = tf.reduce_sum(squared_deltas) #sum how far off all entries to have a flat value
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]})) #result would say how far are we to the correct answer

## for sake of showing, if we hard code the correct weight and bias, see the loss function is 0
## this is not how it usually works for machine learning, this is just an example
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])

print(sess.run(loss, {x:[1,2,3,4],y:[0,-1,-2,-3]}))  #loss function is 0 with the correc

##if we don't know the correct weight and bias? how do we use machine learning to find
## out the correct weight and bias?

##we're going to use an optimizer. a common optimizer is Gradient Descent
##it basically tweaks the result and see if the loss function is less over and over

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init)  #initialize variables
# reset values to incorrect defaults
for i in range(1000):  #iterate 1000 times
    sess.run(train, {x:[1,2,3,4],
                     y:[0,-1,-2,-3]})
print(sess.run([W,b]))

#we can see the weight and bias is reassigned and is really
#close to the correct value which is
# W=-1, b=1
