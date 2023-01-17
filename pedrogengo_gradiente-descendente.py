from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import pyplot as plt
from celluloid import Camera
from IPython.display import HTML
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
fig = plt.figure(figsize=(10,7))
ax = plt.axes(projection='3d')
u = np.linspace(-5.5, 5.5, 100)
x,y = np.meshgrid(u,u)
z = ((x)**2 + (y)**2)
ax.plot_surface(x,y,z,rstride=1,cstride=1,
                cmap='viridis', edgecolor = 'none')
ax.set_title('Visualização de f(w)');
ax.set_xlabel('$w_1$')
ax.set_ylabel('$w_2$')
ax.set_zlabel('f(w)');
fig = plt.figure(figsize=(10,7))
ax = plt.axes()
contours = plt.contour(x,y,z,15, colors = 'black')
plt.clabel(contours, inline = True, fontsize = 8)
plt.imshow(z, extent=[-5.5, 5.5, -5.5, 5.5], origin = 'lower',
           cmap = 'viridis', alpha = 0.5, )
ax.set_title('Curvas de nível')
ax.set_xlabel('$w_1$')
ax.set_ylabel('$w_2$')
plt.colorbar();
fig = plt.figure(figsize=(10,7))
ax = plt.axes(projection='3d')
ax.contour3D(x,y,z, 15, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Curvas de Nível de f(x,y)');
def objective_f(w):
    return w[0]**2 + w[1]**2
def gradient(w):
    first_term = w[0]*2
    second_term = w[1]*2
    return np.array([first_term, second_term])
def gradient_descent(w, ts, max_iter = 400, verbose = False):
    ##Lista de todos os valores da função e dos pontos estimados para todos os t
    f_values = list()
    ws = list()
    for t in ts:
        ##Lista de valores da função e dos pontos estimados para cada valor de t especifico
        w_t = list()
        f_value = list()
        k = 0
        w_atual = w
        w_ant = np.array([np.inf] * len(w))
        while k < max_iter:
            w_t.append(w_atual)
            k += 1
            f_value.append(objective_f(w_atual))            
            grad = gradient(w_atual)
            w_ant = w_atual.copy()
            ##Atualização da estimativa de w
            w_atual = w_atual - t * grad
            if verbose:
                print('Iteração {}: {}\nt: {}'.format(k, w_atual, t))
        f_values.append(f_value)
        ws.append(w_t)
        print('------------------------------------------------------------------------')
        print('\033[1m'+'Método do Gradiente Estocástico para max_iter = {} e t = {}'.format(max_iter, t) + '\033[0m')
        print('------------------------------------------------------------------------')
        print('W = {}\nf = {}\nIterações = {}'.format(w_atual, objective_f(w_atual), k))
        print('------------------------------------------------------------------------')
    return f_values, ts, ws
points, ts, ws = gradient_descent(np.array([1,4]),[1, 0.1, 0.001], 50)
fig = plt.figure(figsize=(10,7))      
for point, t in zip(points, ts):
    plt.plot(point, label = 't = ' + str(round(t,10)));

plt.legend(loc='upper right', frameon = False)
plt.title('Descida da função', size = 14)
plt.xlabel('Número de interações', size = 14)
plt.ylabel(r'$f(x)$', size = 14);
fig = plt.figure(figsize=(10,7))
camera = Camera(fig)

for i in range(50):
    ax = plt.axes()
    contours = plt.contour(x,y,z,15, colors = 'black')
    plt.clabel(contours, inline = True, fontsize = 8)
    plt.imshow(z, extent=[-5.5, 5.5, -5.5, 5.5], origin = 'lower',
               cmap = 'viridis', alpha = 0.5, )
    ax.set_title('Curvas de nível')
    ax.set_xlabel('$w_1$')
    ax.set_ylabel('$w_2$')
    plt.plot([ws[0][i][0]], [ws[0][i][1]], marker='o', markersize=5, color="red")
    camera.snap()
animation = camera.animate()
plt.close("all")
HTML(animation.to_html5_video())
fig = plt.figure(figsize=(10,7))
camera = Camera(fig)

for i in range(50):
    ax = plt.axes()
    contours = plt.contour(x,y,z,15, colors = 'black')
    plt.clabel(contours, inline = True, fontsize = 8)
    plt.imshow(z, extent=[-5.5, 5.5, -5.5, 5.5], origin = 'lower',
               cmap = 'viridis', alpha = 0.5, )
    ax.set_title('Curvas de nível')
    ax.set_xlabel('$w_1$')
    ax.set_ylabel('$w_2$')
    plt.plot([ws[1][i][0]], [ws[1][i][1]], marker='o', markersize=5, color="red")
    camera.snap()
animation = camera.animate()
plt.close("all")
HTML(animation.to_html5_video())
fig = plt.figure(figsize=(10,7))
camera = Camera(fig)

for i in range(50):
    ax = plt.axes()
    contours = plt.contour(x,y,z,15, colors = 'black')
    plt.clabel(contours, inline = True, fontsize = 8)
    plt.imshow(z, extent=[-5.5, 5.5, -5.5, 5.5], origin = 'lower',
               cmap = 'viridis', alpha = 0.5, )
    ax.set_title('Curvas de nível')
    ax.set_xlabel('$w_1$')
    ax.set_ylabel('$w_2$')
    plt.plot([ws[2][i][0]], [ws[2][i][1]], marker='o', markersize=5, color="red")
    camera.snap()
animation = camera.animate()
plt.close("all")
HTML(animation.to_html5_video())