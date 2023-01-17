from IPython.display import YouTubeVideo

YouTubeVideo("yMgFHbjbAW8")
import numpy as np
x = np.array([7, 2, 3, 5])

x
w = np.array([0.9, 0.99, 0.75, 0.4])

w
y = np.dot(w, x)

y
y1 = np.sum( [w[j] * x[j] for j in range(len(x))] )

y1
a = 100

a
y = np.dot(w, x) + a

y
# 2 training samples - inputs

xs = np.array([ [5], [4] ])

print('xs = %s' % xs)

# and labels / outputs

ys = np.array([ 15, 12 ])

print('ys = %s <-- try to make "w • x + a" equal this!' % ys)
# Now how do we find out (a, w) so we can calculate y for any x ???

def predict(xs, a, w):

    return [np.dot(w, x_i) + a for x_i in xs]



# Let's guess...

predict(xs, a = 1, w = [2])



# Nope :(
# Let's try again

predict(xs, a = 3, w = [3])



# Still nope...
# Let's try again

predict(xs, a = 0, w = [3])