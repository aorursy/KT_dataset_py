import mxnet as mx

a = mx.sym.Variable('a')

b = mx.sym.Variable('b')

c = a + b

assert a.name == "a", "Symbol name incorrect."

assert b.name == "b", "Symbol name incorrect."

(a, b, c)





# elemental wise times

d = a * b  

# matrix multiplication

e = mx.sym.dot(a, b)   

# reshape

f = mx.sym.Reshape(a, shape=(2,6))  

# broadcast

g = mx.sym.broadcast_to(f, shape=(3,2,6))  



# Output may vary

net = mx.sym.Variable('data')

net = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=128)

net = mx.sym.Activation(data=net, name='relu1', act_type="relu")

net = mx.sym.FullyConnected(data=net, name='fc2', num_hidden=10)

net = mx.sym.SoftmaxOutput(data=net, name='out')

#mx.viz.plot_network(net, shape={'data':(100,200)})

mx.viz.plot_network(net, save_format='png', shape={'data':(100,200)}).render()
from IPython.display import Image, display

display(Image(filename='./plot.gv.png'))