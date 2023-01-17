from IPython.display import YouTubeVideo
YouTubeVideo("pthIRkUsNLc")
YouTubeVideo("9jAkRSkdtzs")
!pip install pyswarm
import numpy as np
from scipy.integrate import odeint
from scipy.integrate import simps
from pyswarm import pso
from IPython.display import HTML
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc, animation, cm
%matplotlib inline
# system of ODE's
def dSdT (S,T,k,B,n):
    
    # vector to be returned
    dS = np.zeros(n+1)
    
    # length of each interval
    deltaR = 1/n
    
    # ODE's
    for i in range(1,n):
        dS[i] = (k*S[i]+1)*((S[i+1]-2*S[i]+S[i-1])/(deltaR**2)+(2/(i*deltaR))*(S[i+1]-S[i])/(deltaR))

    # boundary conditions    
    dS[0] = dS[1]
    dS[n] = B*np.exp(k*(1-np.exp(-B*T))-B*T)
    
    return dS
# parameters

k = 1.0
B = 286.2
n = 20         # number of points in spatial discretization
# initial condition
S_inicial = np.zeros(n+1)

# range of T in which S will be obtained
T = np.arange(0.0,0.15,1e-4)
# integrating!
sol = odeint(dSdT, S_inicial, T, args=(k,B,n))
# converting S to concentration C*
C_star = np.log(k*sol+1)/k

# creating vector for grid in R
R = np.linspace(0.0,1.0,n+1)

# T points to be plotted (same as in the article)
T_param = np.array([0.005, 0.015, 0.03, 0.05, 0.07, 0.09, 0.12])

# plotting

for j in range(len(C_star)):
    if any(abs(T_param-T[j])<1e-8):
        plt.plot(R,C_star[j,:],'-k')
        
plt.axis([0,1,0,1])
plt.xlabel('Adimensional radius, R')
plt.ylabel('Concentration, $(C_A-C_{A0})/(C_{As}-C_{A0})$');
max_theta = 2.0 * np.pi
theta = np.linspace(0.0, max_theta, 100)

grid_R, grid_theta = np.meshgrid(R, theta)

grid_C_star = np.tile(C_star[400,:],(len(theta),1))

print(np.shape(grid_R))
print(np.shape(grid_theta))
print(np.shape(grid_C_star))
# creating the window where the polar projection will be plotted
fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))

# plotting!
p1 = ax.contourf(grid_theta, grid_R, grid_C_star, 100, vmin=0, vmax=1, cmap=cm.Blues)

# removing the markings on the R and theta axes
ax.set_xticklabels([])
ax.set_yticklabels([])

# removing grids
ax.grid(False)

# title
ax.set_title('C* polar projection: T = %.2f'%(T[400]))

# getting the limits of the color map
vmin,vmax = p1.get_clim()

# defining a normalized scale
cNorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

# creating a new axis in the right corner to plot the color bar
ax3 = fig.add_axes([0.9, 0.1, 0.03, 0.8])

# plotting the color map on the newly created axis
cb1 = mpl.colorbar.ColorbarBase(ax3, norm=cNorm,cmap=cm.Blues)
grid_C_star = np.tile(C_star[0,:],(len(theta),1))
p1 = ax.contourf(grid_theta, grid_R, grid_C_star, 100, vmin=0, vmax=1, cmap=cm.Blues)
def init():
    
    return p1,
    
def animate(i):
    
    ax.clear()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)
    ax.set_title('C* polar projection:: T = %.2f'%(T[i]))
    C_grid = np.tile(C_star[i,:],(len(theta),1))
    p1 = ax.contourf(grid_theta, grid_R, C_grid, 100, vmin=0, vmax=1,cmap=cm.Blues)  
    return p1,
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=1200, interval=50, blit=False)
HTML(anim.to_html5_video())
Mt_M = np.zeros_like(T)

for i in range(len(T)):
    Mt_M[i] = simps(C_star[i,:]*R**2,R)/simps(R**2,R)
    
plt.plot(T,Mt_M,'-')
plt.axis([0,0.15,0,1])
plt.xlabel('Adimensional time, T')
plt.ylabel('Fractional absorption, Mt/M$\infty$');
data_jatropha = np.genfromtxt ('../input/curve-of-water-absorption-in-seeds-jatropha/data_to_humiliate_that_ungrateful.csv', delimiter=",")
plt.plot(data_jatropha[:,0],data_jatropha[:,1],'*')
plt.xlabel('Time (hours)')
plt.ylabel('Humidity (%)')

print(data_jatropha)
Mt_M_exp = (data_jatropha[:-10,1]-data_jatropha[0,1])
Mt_M_exp = Mt_M_exp/Mt_M_exp[-1]
t_exp = data_jatropha[:-10,0]

plt.plot(t_exp,Mt_M_exp,'*')
plt.xlabel('Time (hours)')
plt.ylabel('Fractional absorption, Mt/M$\infty$');
def ObjF (params):
    
    # parameters
    kapa = params[0]
    D0 = params[1]
    beta = params[2]
    
    # saturation and initial concentrations
    Cs = 0.456701031
    C0 = 0.169072165

    # seed radius
    a = 1

    # dimensionless model parameters
    D0_linha = D0*np.exp(kapa*C0)
    k = kapa*(Cs-C0)
    B = (beta*a**2)/(D0_linha)
    
    # number of points in the mesh
    n = 20
    
    # initial condition
    S_inicial = np.zeros(n+1)

    # range of T in which S will be obtained
    T = t_exp*D0_linha/(a**2)

    # integrating!
    sol = odeint(dSdT, S_inicial, T, args=(k,B,n))

    # converting S to concentration
    C_star = np.log(k*sol+1)/k

    # creating vector for grid in R
    R = np.linspace(0,1,n+1)
    
    # calculating fractional absorption
    
    Mt_M = np.zeros_like(T)
    
    for i in range(len(T)):
        Mt_M[i] = simps(C_star[i,:]*R**2,R)/simps(R**2,R)
        
    # error (difference between calculated and experimental curves)
    error = Mt_M - Mt_M_exp
    
    # sum of squares of errors
    return np.sum(error*error)         
lb = [1e-1, 1e-5, 1e-1]
ub = [3,    1e-1, 3]

alpha_opt, fopt = pso(ObjF, lb, ub)
print(alpha_opt)
print(fopt)
# parameters
kapa = alpha_opt[0]
D0 = alpha_opt[1]
beta = alpha_opt[2]

# saturation and initial concentrations
Cs = 0.456701031
C0 = 0.169072165

# seed radius
a = 1

# dimensionless model parameters
D0_linha = D0*np.exp(kapa*C0)
k = kapa*(Cs-C0)
B = (beta*a**2)/(D0_linha)

# number of points in the mesh
n = 20
    
# initial condition
S_inicial = np.zeros(n+1)

# range of T in which S will be obtained
T = t_exp*D0_linha/(a**2)

# integrating!
sol = odeint(dSdT, S_inicial, T, args=(k,B,n))

# cconverting S to concentration
C_star = np.log(k*sol+1)/k

# creating vector for grid in R
R = np.linspace(0,1,n+1)

# calculating fractional absorption
    
Mt_M = np.zeros_like(T)
    
for i in range(len(T)):
    Mt_M[i] = simps(C_star[i,:]*R**2,R)/simps(R**2,R)

# plotting    
plt.plot(t_exp,Mt_M_exp,'*',t_exp,Mt_M,'-')
plt.xlabel('Time (hours)')
plt.ylabel('Fractional absorption, Mt/M$\infty$');
YouTubeVideo("ekzHIouo8Q4")