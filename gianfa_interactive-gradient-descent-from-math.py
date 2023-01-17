import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from ipywidgets import interact, interactive, fixed, interact_manual, Layout

import ipywidgets as widgets



from bokeh.plotting import ColumnDataSource

from bokeh.io import push_notebook, show, output_notebook

from bokeh.plotting import figure

from bokeh.layouts import widgetbox, row

output_notebook()
# f = lambda x: 0.5 * x**2 # <-- this is the func above..

f = lambda x: 0.2 + x+ (1/2)*x**2 -0.2*x**3 # <-- ...but this is funnier to see

x = np.linspace(-2,6,100) # x points

y = f(x) # y, function values in x
plt.plot(x, y)
dx = 1e-6 # variable increment, x differential

dy = f(x + dx) - f(x) # function values increment, function increrment

d_y = dy/dx # derivative of f in x



plt.plot(x, y) # blue

plt.plot(x, d_y) # orange
d = lambda f, x, dx: (( f(x+dx) - f(x) ) / dx)
plt.plot(x,f(x))

plt.plot(x, d(f, x, dx))
x0 = 1

dx = 4e-2

lr = 1e-2

df_x = d(f,x0,dx)/dx

x_hist = []



for it in range(600):

    x1 = x0 - lr * df_x

    df_x = d(f,x1,dx)# x1 -x0

    x0 = x1

    x_hist.append(x0)



print('x = ', x0)

plt.plot(x_hist);
def gd_1(iterations):

    x0 = 1

    lr = 1e-2

    df_x = 1e-6



    x_hist = []

    for i in range(iterations):

        x1 = x0 - lr * df_x

        df_x = d(f,x1,dx)# x1 -x0

        x0 = x1

        x_hist.append(x0)

    

    # Plot #

    r1.data_source.data= { 'x': range(len(x_hist)), 'y': x_hist }

    r2.data_source.data['y'] = f(x)

    pallino.data = {'x':[x0], 'y':[f(x0)]}

    push_notebook()
### Visualization ###

p1 = figure(title="x value", plot_height=300, plot_width=300, y_range=(-5,5),

           background_fill_color='#efefef', x_axis_label='iterations')

r1 = p1.line(range(len(x_hist)), x_hist, color="#8888cc", line_width=1.5, alpha=0.8)

pallino = ColumnDataSource(data={'x':[x0], 'y':[f(x0)]})



p2 = figure(title="Minimum reaching", plot_height=300, plot_width=600, y_range=(-5,5),

           background_fill_color='#efefef', x_axis_label='x')

r2 = p2.line(x, y, color="#8888cc", line_width=1.5, alpha=0.8)

pallino = ColumnDataSource(data={'x':[x0], 'y':[f(x0)]})

p2.circle('x', 'y', source=pallino, size=20, color='orange')

layout = row([p1,p2])



interact(gd_1, iterations=widgets.IntSlider(min=0, max=150, continous_update=False, layout=Layout(width='500px')))       

show(layout, notebook_handle=True);
def gd_1(iterations,lrp):

    x0 = 1

    df_x = 1e-6

    

    x_hist = []

    lr = 10**lrp

    for i in range(iterations):

        x1 = x0 - lr * df_x

        df_x = d(f,x1,dx)# x1 -x0

        x0 = x1

        x_hist.append(x0)



    # Plot #

    r1.data_source.data= { 'x': range(len(x_hist)), 'y': x_hist }

    r2.data_source.data['y'] = f(x)

    pallino.data = {'x':[x0], 'y':[f(x0)]}

    push_notebook()
### Visualization ###

p1 = figure(title="x value", plot_height=300, plot_width=300, y_range=(-5,5),

           background_fill_color='#efefef', x_axis_label='iterations')

r1 = p1.line(range(len(x_hist)), x_hist, color="#8888cc", line_width=1.5, alpha=0.8)

pallino = ColumnDataSource(data={'x':[x0], 'y':[f(x0)]})



p2 = figure(title="Minimum reaching", plot_height=300, plot_width=600, y_range=(-5,5),

           background_fill_color='#efefef', x_axis_label='x')

r2 = p2.line(x, y, color="#8888cc", line_width=1.5, alpha=0.8)

pallino = ColumnDataSource(data={'x':[x0], 'y':[f(x0)]})

p2.circle('x', 'y', source=pallino, size=20, color='orange')

layout = row([p1,p2])



interact(

    gd_1,

    iterations=widgets.IntSlider(min=0, max=150, layout=Layout(width='500px')),

    lrp=widgets.IntSlider(min=-5, max=3, value=-5, layout=Layout(width='500px')),

        )       

show(layout, notebook_handle=True);
x = np.linspace(-2,2,100)



# 1. define the true points

y_true = 0.5*x # I like this, you choose another one :P

# 2. define the model

g = lambda w: w*x # x is given, in this context our variable is w.

# 3. define the objective function

J = lambda y: np.sum( (y - y_true)**2 )/len(y) # J(w) = J( g(w) )

# 4. define the derivative of the objective function

dJ_w = lambda y_new, y, dw: (J(y_new) - J(y))/dw









w0 = 0.1

lr = 5e-3



# first step

y = g(w0)

dJ_wn = 1e-5

# loop

w_hist = []

J_hist = []

for it in range(400):

    w1 = w0 - lr * dJ_wn

    y_new = g(w1)

    dw = w1 - w0

    Jn = J(y_new)

    dJ_wn = dJ_w(y_new,y,dw)

    w0 = w1

    y = y_new

    # store

    w_hist.append(w0)

    J_hist.append(Jn)



print('w_true = 0.5\nw_pred = ', w0)

plt.plot(w_hist);

plt.title('w');
def gd_lr(its, lrp):

    # params

    w0 = 0.1

    lr = 10**lrp



    # 1. define the true points

    y_true = 0.5*x 

    g = lambda w: w*x 

    d = lambda f, x, dx: (( f(x+dx) - f(x) ) / dx)

    J = lambda y: np.sum( (y - y_true)**2 )/len(y) # J(w) = J( g(w) )

    dJ_w = lambda y_new, y, dw: (J(y_new) - J(y))/dw





    y = g(w0)

    dJ_wn = 1e-5

    # loop

    w_hist = []

    J_hist = []

    for it in range(its):

        w1 = w0 - lr * dJ_wn

        y_new = g(w1)

        dw = w1 - w0

        Jn = J(y)

        dJ_wn = dJ_w(y_new,y,dw)

        w0 = w1

        y = y_new

        # store

        w_hist.append(w0)

        J_hist.append(Jn)

  

    print('w= 0.5\nw_calc = ',w0,  '\ncost= ', J_hist[-1])

    fig, ax = plt.subplots(1,3, figsize=(12,5));

    ax[0].plot(range(its), J_hist);

    ax[0].set_xlabel('iterations');

    ax[0].set_title('cost');

    ax[1].plot(range(its), w_hist);

    ax[1].set_xlabel('iterations');

    ax[1].set_title('w calculated');

    ax[2].plot(y);

    ax[2].plot(y_true);

    ax[2].set_xlabel('x');

    ax[2].set_ylabel('y');

    ax[2].legend(['y_true', 'y']);

    plt.subplots_adjust(left=0);



interact(gd_lr, its=widgets.IntSlider(min=1, max=230, continous_update=False), lrp=(-5,2) );
x = np.linspace(-2,2,100)



# 1. define the true points

y_true = -10*x + 3*x**2

# 2. define the model

g = lambda w1, w2: +w1*x + w2*x**2 # x is given, remember.

# 3. define the objective function

J = lambda w1, w2: np.sum( (g(w1, w2) - y_true)**2 )/len(y_true) # <-- g(w1,w2), so to indipendently act on each w

# 4. define the derivative of the objective function

dJ_w1 = lambda w1, w2, dw1: ( J(w1+dw1, w2) - J(w1, w2) )/dw1

dJ_w2 = lambda w1, w2, dw2: ( J(w1, w2+dw2) - J(w1, w2) )/dw2











def gd_2p(its, lrp):

    w1 = 0.1

    w2 = 0.15

    lr = 10**lrp



    # first step

    dJ_w1n = 1e-2

    dJ_w2n = 1e-2



    # loop

    w_hist = []

    J_hist = []

    for it in range(its):

        w1_new = w1 - lr * dJ_w1n

        w2_new = w2 - lr * dJ_w2n # you have 2 params here, thus you must update both

      #   y_new = g(w1_new, w2_new)

        dw1 = w1_new - w1

        dw2 = w2_new - w2

        Jn = J(w1_new, w2_new) 

        dJ_w1n = dJ_w1(w1, w2, dw1); #print(dJ_w1n)

        dJ_w2n = dJ_w2(w1, w2, dw2)



        w1 = w1_new

        w2 = w2_new

        # store

        w_hist.append([w1, w2])

        J_hist.append(Jn)



    w_hist = np.vstack(w_hist)

    print(w_hist.shape)

    print('w1_true = -10\nw_pred = ', w_hist[-1,0])

    # plt.plot(J_hist)#w_hist[:,0]);



    print('w= 0.5\nw_calc = ',w0,  '\ncost= ', J_hist[-1])



    fig, ax = plt.subplots(2,2, figsize=(12,8));

    ax[0,0].plot(range(its), w_hist[:,0]);

    ax[0,0].set_xlabel('iterations');

    ax[0,0].set_title('w1 calculated');

    ax[0,1].plot(range(its), w_hist[:,1]);

    ax[0,1].set_xlabel('iterations');

    ax[0,1].set_title('w2 calculated');

    ax[1,0].plot(x, g(w1,w2));

    ax[1,0].plot(x, y_true);

    ax[1,0].set_xlabel('x');

    ax[1,0].legend();

    ax[1,0].set_title('functions');

    ax[1,1].plot(range(its), J_hist);

    ax[1,1].set_xlabel('iterations');

    ax[1,1].set_title('J(w1,w2)');

plt.subplots_adjust(hspace =0.9);



interact(gd_2p, its=widgets.IntSlider(min=1, max=230, continous_update=False), lrp=(-5,2) );
from mpl_toolkits.mplot3d import axes3d

import matplotlib.pyplot as plt

from matplotlib import cm







E = lambda w1, w2: (g(w1, w2) - y_true)**2

ncts = 100

w1 = w2 = np.linspace(-12, 5, ncts)







fig = plt.figure(figsize=(12,12))

ax = fig.gca(projection='3d')



# X, Y, Z = axes3d.get_test_data(0.05)

X, Y = np.meshgrid(w1,w2); print(X.shape, Y.shape)

Z = E(X,Y); print(Z.shape)

#ax.view_init(-5, 10)  #  <---- play with this

ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.45, color ='yellow')

cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)

cset = ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)

cset = ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)



Z = np.ones((ncts, ncts)) * 0

ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.9, color ='grey')



ax.set_xlabel('w1')

# ax.set_xlim(-40, 40)

ax.set_ylabel('w2')

# ax.set_ylim(-40, 40)

ax.set_zlabel('E(w1,w2)')

#ax.set_zlim(-3, 1)   #  <----  play with this



plt.show()

npts = 100

X = np.array([

    np.linspace(-2,2,npts),

    np.linspace(-2,2,npts),

])

X.shape
x1 = X[0]

x2 = X[1]
# define a function and its partial derivatives

f = lambda x1, x2: x1 + x2**2

dx1 = lambda f, x1, x2, ep: (( f(x1+ep,x2) - f(x1,x2) ) / ep)

dx2 = lambda f, x1, x2, ep: (( f(x1,x2+ep) - f(x1,x2) ) / ep)



eps = 1e-7
# https://matplotlib.org/examples/mplot3d/surface3d_demo.html

from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm





def plot_f_df(f, x1,x2, fs=15, figsize=(12,18)):

    fig = plt.figure(figsize=figsize)





    # Make data.

    x1grid, x2grid = np.meshgrid(x1, x2)

    zgrid = f(x1grid, x2grid)



    d_x1grid = dx1(f, x1grid, x2grid, 1e-1) # <--- eps lower makes the fx1 to be noisy

    d_x2grid = dx2(f, x1grid, x2grid, eps)





    # Data about a line, parallel to x2, on the f(x1,x2) surface

    rip = 35

    x1_pc = (x1.max()-x2.min())/rip

    x1b = x1_pc * rip/4

    x1_mini_rng = x1[ (x1 > -x1_pc + x1b) & (x1 < x1_pc +x1b) ]

    x1_mini_grid, x2_mini_grid =  np.meshgrid(x1_mini_rng, x2)

    z_mini_grid = f(x1_mini_grid, x2_mini_grid) + 0.1





    # Plot the surface

    ax1  = plt.subplot(311, projection='3d')

    ax1.plot_surface(x1grid, x2grid, zgrid, linewidth=0, antialiased=False, alpha=0.7)

    ax1.plot_surface(x1_mini_grid, x2_mini_grid, z_mini_grid, color='red', linewidth=5 )

    ax1.set_xlabel('x1', fontsize=fs)

    ax1.set_ylabel('x2', fontsize=fs)

    ax1.set_zlabel('f(x1,x2)', fontsize=fs)



    # Plot d_x2 surface

    ax2  = plt.subplot(312, projection='3d')

    ax2.plot_surface(x1grid, x2grid, d_x1grid)

    ax2.set_xlabel('x1', fontsize=fs)

    ax2.set_ylabel('x2', fontsize=fs)

    ax2.set_zlabel('f_x1', fontsize=fs)



    # Plot d_x2 surface

    ax3  = plt.subplot(313, projection='3d')

    ax3.plot_surface(x1grid, x2grid, d_x2grid)

    ax3.set_xlabel('x1', fontsize=fs)

    ax3.set_ylabel('x2', fontsize=fs)

    ax3.set_zlabel('f_x2', fontsize=fs)

plot_f_df(f, x1,x2)
f = lambda x1,x2: np.array([ -5*x1, 0.5*x2**2])  # f(x1,x2), here w1_true and w2_true are fixed



model = lambda w1, w2, x1, x2: np.array([  w1*x1, w2*x2**2  ])  # f(x1, x2; w1, w2) = w1x1 i + w2x2 j
J = lambda y, y_true: np.sum((y - y_true)**2 )/len(y) 

J2 = lambda y1, y2, y1_true, y2_true: np.array([ J(y1, y1_true),  J(y2, y2_true)]) # J2 = [J(y1), J(y2)]
# now we can define the differential of the cost

dJ2 = lambda J2, y1,y2, y1_old,y2_old, y1_true,y2_true: np.array([  # <- is just an array, please don't get lost

  J2(y1,y2, y1_true,y2_true)[0] - J2(y1_old,y2_old, y1_true,y2_true)[0], # J2(y1, y_true) - J2(y1_old, y_true) = dJ2

  J2(y1,y2, y1_true,y2_true)[1] - J2(y1_old,y2_old, y1_true,y2_true)[1]

])
dW = lambda w1_new, w2_new, w1, w2: np.array([w1_new - w1, w2_new - w2])   # [dw1, d2w]
dJ2_W = lambda dJ, dW: np.array([ dJ[0]/dW[0], dJ[1]/dW[1]  ])
f = lambda x1,x2: np.array([-5*x1, 0.5*x2**2])  # f(x1,x2), here w1_true and w2_true are fixed

model = lambda w1, w2, x1, x2: np.array([  w1*x1, w2*x2**2  ])  # f(x1, x2; w1, w2) = w1x1 i + w2x2 j







iterations = 10



def gd_plot2(iterations):

    npts = 100

    X = np.array([

      np.linspace(-2,2,npts),

      np.linspace(-2,2,npts),

    ])

    lr = 1e-2



    # Initialize

    W = [0.1, 0.15]

    W_new = [0.1+1e-5, 0.15+1e-5]

    y_true = f(X[0], X[1])

    y   = model( W[0],W[1], X[0],X[1] )          # I know there are a lot of cuttable things here, but i thought...

    y_new = model(W_new[0], W_new[1], X[0],X[1]) #    ...it was a simpler way to let follow the reasoning...

    dWn = dW( W_new[0],W_new[1], W[0],W[1] )     #    ...so, if you want, you can rewrite it and let me know :)

    dJ2n = dJ2(J2, y_new[0],y_new[1], y[0],y[1], y_true[0],y_true[1])

    dJ2_Wn = dJ2_W(dJ2n, dWn)



  

  # Loop

    W_hist = []

    J2_hist= []

    for i in range(iterations):

        W_new = [ W[0] - lr * dJ2_Wn[0], W[1] - lr * dJ2_Wn[1] ]

        y_new = model( W_new[0],W_new[1], X[0],X[1] )

        J2n = J2( y[0],y[1], y_true[0],y_true[1] )



        # update

        dWn = dW( W_new[0],W_new[1], W[0],W[1] )

        dJ2n = dJ2(J2, y_new[0],y_new[1], y[0],y[1], y_true[0],y_true[1])

        dJ2_Wn = dJ2_W(dJ2n, dWn)

        y = y_new

        W = W_new



        W_hist.append(W)

        J2_hist.append(J2n)



    J2_hist = np.vstack(J2_hist)

    W_hist  = np.vstack(W_hist)



    print('W_true:', [-5, 0.5], '\nW_calc:', W, '\nJ:', J2n)

    fig, ax = plt.subplots(1,2, figsize=(8,6))

    ax[0].plot(range(iterations), J2_hist[:,0])

    ax[0].set_xlabel('iterations')

    ax[0].set_ylabel('J2(w1)')

    # ax[1].plot(W_hist[:,1], J2_hist[:,1])

    ax[1].plot(range(iterations), J2_hist[:,1])

    ax[1].set_xlabel('iterations')

    ax[1].set_ylabel('J2(w2)')



interact(gd_plot2, iterations=(0,250,3));
## Interacrive visualization of surface shaping through gradient descent



def gd_2(its):

    npts = 100

    X = np.array([

        np.linspace(-2,2,npts),

        np.linspace(-2,2,npts),

    ])

    lr = 1e-2

    f = lambda x1,x2: np.array([-5*x1, 0.5*x2**2])  # f(x1,x2), here w1_true and w2_true are fixed

    model = lambda w1, w2, x1, x2: np.array([  w1*x1, w2*x2**2  ])  # f(x1, x2; w1, w2) = w1x1 i + w2x2 j



    # Initialize

    W = [0.1, 0.15]

    W_new = [0.1+1e-5, 0.15+1e-5]

    y_true = f(X[0], X[1])

    y   = model( W[0],W[1], X[0],X[1] )

    y_new = model(W_new[0], W_new[1], X[0],X[1])

    dWn = dW( W_new[0],W_new[1], W[0],W[1] )

    dJ2n = dJ2(J2, y_new[0],y_new[1], y[0],y[1], y_true[0],y_true[1])

    dJ2_Wn = dJ2_W(dJ2n, dWn)



  

  # Loop

    W_hist = []

    J2_hist= []

    for i in range(its):

        W_new = [ W[0] - lr * dJ2_Wn[0], W[1] - lr * dJ2_Wn[1] ]

        y_new = model( W_new[0],W_new[1], X[0],X[1] )

        J2n = J2( y[0],y[1], y_true[0],y_true[1] )



        # update

        dWn = dW( W_new[0],W_new[1], W[0],W[1] )

        dJ2n = dJ2(J2, y_new[0],y_new[1], y[0],y[1], y_true[0],y_true[1])

        dJ2_Wn = dJ2_W(dJ2n, dWn)

        y = y_new

        W = W_new



        W_hist.append(W)

        J2_hist.append(J2n)

    return W_hist, J2_hist





def plot_model_W(iterations):

    f     = lambda x1,x2: np.array([ -5*x1, -x1 + 0.5*x2**2 ])  # f(x1,x2), here w1_true and w2_true are fixed

    model = lambda w1, w2, x1, x2: np.array([  w1*x1, -x1 + w2*x2**2  ])  # f(x1, x2; w1, w2) = w1x1 i + w2x2 j



    W_hist, J2_hist = gd_2(iterations)



    W = np.vstack(W_hist)



    # Make the Grid

    X_g = np.meshgrid(X[0], X[1])





    # The function has to be calculated over the X grid, but per W single values.

    Z = model( W[-1][0],W[-1][1], X_g[0],X_g[1] ).sum(axis=0)

    Y_true = f(X_g[0],X_g[1]).sum(axis=0)

    #Y_true = model( 0.2,5, X_g[0],X_g[1] ).sum(axis=0); print('Z: ', Z.shape, 'Y_true: ', Y_true.shape)







    ## Plot

    fs=15

    fig = plt.figure(figsize=(9,6))



    # Plot the model

    ax = plt.subplot(221, projection='3d')

    ax.plot_surface(X_g[0], X_g[1], Z, color='red', linewidth=5 );

    ax.set_xlabel('x1', fontsize=fs);

    ax.set_ylabel('x2', fontsize=fs);

    ax.set_title('f(x1,x2; w1,w2)', fontsize=fs);

    ax.set_zlim(-1,130)

    # Plot the true surface

    ax = plt.subplot(222, projection='3d')

    ax.plot_surface(X_g[0], X_g[1], Y_true, color='red', linewidth=5 );

    ax.set_xlabel('x1', fontsize=fs);

    ax.set_ylabel('x2', fontsize=fs);

    ax.set_title('Y_true', fontsize=fs);

    ax.set_zlim(-1,130);



    J2_hist = np.vstack(J2_hist)

    ax = plt.subplot(223)

    xt = np.arange(iterations);# print('xt.shape: ', xt.shape, '. J2: ', J2_hist.shape)

    ax.plot(xt, J2_hist[:,0])

    ax.set_xlabel('iterations')

    ax.set_ylabel('J2(w1)')



    ax = plt.subplot(224)

    ax.plot(np.arange(iterations), J2_hist[:,1])

    ax.set_xlabel('iterations')

    ax.set_ylabel('J(w2)')



#sl =  widgets.IntSlider(min=0, max=150, step=3, value=0)

sl = widgets.IntSlider(min=0, max=40, step=1, value=1, continuous_update=False)

interact(plot_model_W, iterations=sl);



# J2_hist


npts = 300

###############



def plot_params_space(elev, azim):

    J = lambda W: (W[0]*X[0] + W[1]*X[1]**2) - (0.2*X[0] + 5*X[1]**2)



    # Make the Grid

    X = [

      np.linspace(-2, 2, npts),

      np.linspace(-2, 2, npts),

    ]

    W = [

      np.linspace(-6, 6, npts),

      np.linspace(-6, 6, npts),

    ]

    W_g = np.meshgrid(W[0], W[1])



    Jn = J(W_g); print('Jn: ', np.shape(Jn))









    ## Plot

    fs=15

    fig = plt.figure(figsize=(9,6))



    # Plot the model

    ax = plt.subplot(111, projection='3d')

    #angle = 190

    ax.view_init(elev, azim)

    ax.plot_surface(W_g[0], W_g[1], Jn, color='red', linewidth=5 );

    ax.set_xlabel('w1', fontsize=fs);

    ax.set_ylabel('w2', fontsize=fs);

    ax.set_title('f(x1,x2; w1,w2)', fontsize=fs);

    ax.set_zlim(-1,50)



sl1 =  widgets.IntSlider(min=0, max=360, step=1, value=0, continous_update=False)

sl2 =  widgets.IntSlider(min=0, max=360, step=1, value=0, continous_update=False)

interact(plot_params_space, elev=sl1, azim=sl2) 


