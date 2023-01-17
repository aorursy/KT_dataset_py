import pandas as pd

import numpy as np



data_path = '../input/iris/Iris.csv'



data = pd.read_csv(data_path)



print(data.shape)

print(data.sample)



print(data.info)

species = data['Species']



data_c1 = data[data['Species'] == 'Iris-setosa'].iloc[:,1:3]

data_c2 = data[data['Species'] == 'Iris-virginica'].iloc[:,1:3]

import matplotlib.pyplot as plt





plt.scatter(data_c1['SepalLengthCm'],data_c1['SepalWidthCm'],marker='*',c = 'y')

plt.scatter(data_c2['SepalLengthCm'],data_c2['SepalWidthCm'],marker='o',c = 'r')



plt.show()



def f(x,w,b):

    return np.dot(w,x) + b



def shuffle(x, y):

    indexes = np.arange(x.shape[0])

    np.random.shuffle(indexes)

    return x[indexes], y[indexes]



l1,l2 = data_c1.shape[0],data_c2.shape[0]

y1 = np.ones(l1)

y2 = -np.ones(l2)

d1,d2 = data_c1.values,data_c2.values

y = np.append(y1,y2)

x = np.append(d1,d2,axis = 0)

l = x.shape[0]

epochs = 1000

lr = 0.001

w = np.zeros(2,)

b = np.zeros(1,)

print(w,b)





history_w = [w]

history_b = [b]

for epoch in range(epochs):

    x, y = shuffle(x, y)

    error = 0

    for i in range(l):

        y_i = y[i]*f(x[i],w,b)

        if y_i <= 0:

            error = error + 1

            w = w + lr*y[i]*x[i]

            b = b + lr*y[i]

    history_w.append(w)

    history_b.append(b)

    print('{} epoch, error is {}'.format(epoch,error))

    if error == 0:

        break



print(w)        

w1 = w[0]

w2 = w[1]



x1 = np.arange(4,8)

x2 = -1/w2*(b + w1*x1)



plt.scatter(data_c1['SepalLengthCm'],data_c1['SepalWidthCm'],marker='*',c = 'y')

plt.scatter(data_c2['SepalLengthCm'],data_c2['SepalWidthCm'],marker='o',c = 'r')

plt.plot(x1,x2,c='g')

plt.show()



    

    

import matplotlib.pyplot as plt

import matplotlib.animation as animation



fig, ax = plt.subplots()

ax.scatter(data_c1['SepalLengthCm'],data_c1['SepalWidthCm'],marker='*',c = 'y')

ax.scatter(data_c2['SepalLengthCm'],data_c2['SepalWidthCm'],marker='o',c = 'r')



def f(w, x, b):

    w1 = w[0]

    w2 = w[1]

    if w2 == 0:

        return np.zeros(len(x),)

    return -1/(w2)*(b + w1*x1)



def init():  # only required for blitting to give a clean slate.

    line.set_ydata(f(history_w[0],x1,history_b[0]))

    return line,



def animate(i):

    line.set_ydata(f(history_w[i],x1,history_b[i]))  # update the data.

    return line,



x1 = np.arange(4,8)

line, = ax.plot(x1, f(history_w[0],x1,history_b[0]),c='b')



ani = animation.FuncAnimation(

    fig, animate, frames = len(history_w), init_func=init, interval=2, blit=True, save_count=50)



# To save the animation, use e.g.





# or



from matplotlib.animation import FFMpegWriter

writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)

ani.save("movie.mp4", writer=writer)

plt.show()
!ls -la