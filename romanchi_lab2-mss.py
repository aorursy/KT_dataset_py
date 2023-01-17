# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

from scipy.integrate import odeint

import matplotlib.pyplot as plt

from pylab import rcParams

rcParams['figure.figsize'] = 14, 8

%matplotlib inline
def create_df(RSS_next,numOfColX,yShape):

    df = pd.DataFrame(columns=['s', 'RSS', 'Cp', 'FPE'])

    Cp = RSS_next + 2 * numOfColX

    numOfRowY = yShape

    FPE = (numOfRowY + numOfColX) / (numOfRowY - numOfColX) * RSS_next

    df = df.append({'s': numOfColX, 'RSS': RSS_next, 'Cp': Cp, 'FPE': FPE},  ignore_index=True)

    return df

def RMNK(X, y, s=None, printt=False, cr_df=False):

    numOfColX = X.shape[1]

    if numOfColX > 1:

        if cr_df:

            w, H_inv, RSS, df = RMNK(X[:,:-1], y, s, printt, cr_df)

            if s is not None and numOfColX > s:

                return w, H_inv, RSS, df

        else:

            w, H_inv, RSS = RMNK(X[:,:-1], y, s, printt, cr_df)

            if s is not None and numOfColX > s:

                return w, H_inv, RSS

        

        h = (X[:,:-1].T @ X[:,-1]).reshape(-1,1) 

        η  = X[:,-1].T @ X[:,-1] 

        α = H_inv @ h 

        β  = η  - h.T @ α 

        β_inv = 1 /β  

        γ = X[:,-1].T @ y 

        ν =β_inv * (γ- h.T @ w) 

        w = np.vstack((w - ν * α, ν))  

        H_next_inv = np.vstack((np.hstack((H_inv + β_inv * α @ α.T, (- β_inv * α).reshape(-1, 1))),

                               np.hstack((-β_inv * α.T, β_inv))))

        RSS_next = (RSS - ν.flatten() ** 2 * β.flatten())[0]



    else: 

        H_inv = np.array([[0]])

        η  = β = X[:,-1].T @ X[:,-1]

        β_inv = 1 / β

        α = h = np.array([0])

        γ = X[:,-1].T @ y

        ν = np.array([β_inv * γ])

        w = np.array([ν])

        H_next_inv = np.array(β_inv).reshape(1, 1)

        RSS_next = (y.T @ y - y.T @ X[:,-1].reshape(-1, 1) @ w)[0]

        if cr_df:

            df = pd.DataFrame(columns=['s', 'RSS', 'Cp', 'FPE'])

        

    if printt:

        print("\t\t********\n")

        print('КРОК №{}'.format(numOfColX))

        '''print('h_{}:{}'.format(numOfColX, h.reshape(-1,1)[:,0]))

            print('η _{}:{}'.format(numOfColX, η ))

            print('α_{}:{}'.format(numOfColX, α.reshape(-1,1)[:,0]))

            print('β_{}:{}'.format(numOfColX, β))

            print('γ_{}:{}'.format(numOfColX, γ))

            print('ν_{}:{}'.format(numOfColX, ν))'''

        print('θ{}: {}'.format(numOfColX, w[:, 0]))

        print('RSS{}: {}'.format(numOfColX, RSS_next))

        print('H{}^-1:\n{}'.format(numOfColX, H_next_inv))

        print("\t\t********\n")

    

    if cr_df:

        Cp = RSS_next + 2 * numOfColX

        numOfRowY = y.shape[0]

        FPE = (numOfRowY + numOfColX) / (numOfRowY - numOfColX) * RSS_next

        df = df.append({'s': numOfColX, 'RSS': RSS_next, 'Cp': Cp, 'FPE': FPE},  ignore_index=True)

        return w, H_next_inv, RSS_next, df

    return w, H_next_inv, RSS_next
def print_plot(x, y):

    print("ГРАФІК (ПОТОЧКОВО):")

    for i in range(0, len(x), 1):

        print("(%.2f;%.2f)" % (x[i], y[i]))

    print("КІНЕЦЬ ГРАФІКУ")
def Fergulst(N, t, μ, k):

    return μ * N * (k - N)

def findtheta(μ,k):

    theta1 = μ * k + 1

    theta2 = -μ

    return np.array([theta1, theta2])

def findparams(theta1, theta2):

    μ = - theta2

    k = (1 - theta1) / theta2

    return μ, k



k_ideal = 10

μ_ideal = 0.0001



N0 = 100



t_start = 0

t_end = 1000



n = 100

n_list = [5, 20, 50]



C = 4

C_list = [1, 2, 5]



theta = findtheta(μ_ideal,k_ideal)

h = int((t_end - t_start) / (n - 1))

t = np.arange(t_start, t_end, h)

N = odeint(Fergulst, N0, t, (μ_ideal, k_ideal))

df = pd.DataFrame()

df['i'] = range(1, n+1)

df['t'] = list(map(int, t))

df['N(t)'] = N.flatten()

df['N^2(t)'] = np.square(N.flatten())

df['N(t+1)'] = np.array(df[['N(t)','N^2(t)']]) @ theta

df['N(t+1)'] = np.round(df['N(t+1)'], C)

X = np.array(df[['N(t)', 'N^2(t)']])

y = np.array(df['N(t+1)'])

print('МОДЕЛЬ ФЕРГЮЛЬСТА:\nm = {}\tk = {}\tN0 = {}'.format(μ_ideal, k_ideal, N0))

print('ШУМ: C = {}'.format(C))

print('n = {}'.format(n))

print('t =[{};{}] з h={}\n'.format(t_start,t_end,h))

plt.scatter(t, y)

print_plot(t, y)

plt.show()

df
import itertools
print('РМНКО:')

theta_est = RMNK(X, y, printt=True)[0][:,0]

μ_est, k_est = findparams(*theta_est)

print('Справжні значення/оцінені:θ_1 = {} θ_2 = {} θ_1* = {} θ_2* = {}'.format(*theta,*theta_est))

print('Справжні значення/оцінені: m = {} k = {} m* = {} k* = {}'.format(μ_ideal, k_ideal, μ_est, k_est))

plt.scatter(t, y)

t_plot = np.linspace(t_start, t_end, num=n * 10)

plt.plot(t_plot, odeint(Fergulst, N0, t_plot, (μ_est, k_est)), 'r')

plt.show()

param_df = pd.DataFrame(columns=['C', 'n', 'θ_1*', 'θ_2*', 'm*', 'k*'])

N_CONST = 100

H_CONST = int((t_end - t_start) / N_CONST)

T = np.arange(t_start, t_end, h)



for C,n in itertools.product(C_list, n_list):

        N = odeint(Fergulst, N0, T, (μ_ideal, k_ideal))

        df = pd.DataFrame({'N(t)': N.flatten(),

                          'N^2(t)': np.square(N.flatten())

                          })

        df['N(t+1)'] = np.array(df[['N(t)','N^2(t)']]) @ theta

        

#         df['i'] = range(1, n+1)

#         df['t'] = list(map(int, t))

#         df['N(t)'] = N.flatten()

#         df['N^2(t)'] = np.square(N.flatten())

#         df['N(t+1)'] = np.array(df[['N(t)','N^2(t)']]) @ theta

        df['N(t+1)'] = np.round(df['N(t+1)'], C)

        X = np.array(df[['N(t)', 'N^2(t)']])

        y = np.array(df['N(t+1)'])

        theta_est = RMNK(X, y, printt=False)[0][:,0]

        μ_est, k_est = findparams(*theta_est)

        param_df = param_df.append({'C': C, 'n': n,'θ_1*': theta_est[0], 

                                    'θ_2*': theta_est[1],'m*': μ_est, 'k*': k_est},

                                   ignore_index=True)
param_df
from sklearn.metrics import mean_squared_error
fig, axs = plt.subplots(nrows=param_df.shape[0], ncols=2, figsize=(10, 50))



for i in range(param_df.shape[0]):

    n = int(param_df.iloc[i]['n'])

    μ = param_df.iloc[i]['m*']

    k = param_df.iloc[i]['k*']

    C = param_df.iloc[i]['C']



    N = odeint(Fergulst, N0, T, (μ, k))

    N1 = odeint(Fergulst, N0, T, (μ_ideal, k_ideal))

    

    param_df.loc[i, 'RMSE'] = np.sqrt(mean_squared_error(N1, N))

    axs[i][0].plot(T, N, color='red')

    axs[i][0].plot(T, N1, color='blue')

    

    axs[i][1].plot(T[50:], N[50:], color='red')

    axs[i][1].plot(T[50:], N1[50:], color='blue')

    

    axs[i][0].legend(['Approx, C={}, n={}'.format(C, n), 'Real'])

    axs[i][1].legend(['Approx, C={}, n={}'.format(C, n), 'Real'])

    

    axs[i][0].set_title("Original")

    axs[i][1].set_title("Cropped")

    print('Iter: {}, RMSE: {}, m: {}, k: {}'.format(i, np.sqrt(mean_squared_error(N1, N)), μ, k))

    

param_df
from scipy.interpolate import interp1d
plt.style.use('ggplot')
def ZgasColiv(x, t, δ, ω0_sqr):

    return [x[1],  - 2 * δ * x[1] - ω0_sqr * x[0]]        

def findtheta(δ,ω0_sqr):

    d = 1 + 2 * δ

    theta1 = (2 + 2 * δ - ω0_sqr) / d

    theta2 = - 1 / d

    return np.array([theta1, theta2])

def findparams(theta1, theta2):

    δ = - (1 / theta2 + 1) / 2

    ω0_sqr = 1 - 1 / theta2 + theta1 / theta2

    return δ, ω0_sqr

 

δ = 0.005

ω0_sqr = 0.01

x0 = 5

x00 = 2

t_start = 0

t_end = 1000

n = 100

n_list = [10, 20, 40]

C = 2

C_list = [1, 3, 7]



theta = findtheta(δ,ω0_sqr)



h = int((t_end - t_start) / n)



t = np.arange(t_start, t_end, h)

x = odeint(ZgasColiv, np.array([x0, x00]), t, (δ, ω0_sqr))

x1 = x0 + x00

x11 = x00

x_1 = odeint(ZgasColiv, np.array([x1, x11]), t+1, (δ, ω0_sqr))

df = pd.DataFrame()

df['i'] = range(1, n+1)

df['t'] = list(map(int, t))

df['x(t)'] = x[:,0].flatten()

df['x(t+1)'] = x_1[:,0].flatten()

df['x(t+2)'] = np.array(df[['x(t)','x(t+1)']]) @ theta

df['x(t+2)'] = np.round(df['x(t+2)'], C)

X = np.array(df[['x(t)', 'x(t+1)']])

y = np.array(df['x(t+2)'])  

    

print('delta = {} w0^2 = {} x01 = {} x02 = {}'.format(δ, ω0_sqr, x0, x00))

print('C = {}'.format(C))

print('n = {}'.format(n))

print('t =[{};{}] з кроком h={}\n'.format(t_start,t_end,h))

#plt.scatter(t, y)

# plt.plot(t, y)

f = interp1d(t, y, kind='quadratic')

t_new = np.linspace(t[0], t[-1], 10000)

plt.plot(t_new, f(t_new))

print_plot(t, y)

plt.show()

df
print('РМНКО')

theta_pred = RMNK(X, y, printt=True)[0][:,0]

δ_pred, ω0_sqr_pred = findparams(*theta_pred)

print('Справжні значення/оцінені:θ_1 = {} θ_2 = {} θ_1* = {} θ_2* = {}'.format(*theta,*theta_pred))

print('Справжні значення/оцінені: delta = {} w0^2 = {} delta* = {} w0^2* = {}'.format(δ, 

                                                                                      ω0_sqr,

                                                                                      δ_pred,

                                                                                      ω0_sqr_pred))

# plt.scatter(t, y)

# print_plot(t, y)

# t_plot = np.arange(t_start, t_end, h / 10) # n*10

# plt.plot(t_new, f(t_new), c='b')

# plt.plot(t_plot, odeint(ZgasColiv, np.array([x0, x00]), t_plot, (δ_pred, ω0_sqr_pred))[:,0], 'r')

# print_plot(t_plot, odeint(ZgasColiv, np.array([x0, x00]),t_plot, (δ_pred, ω0_sqr_pred))[:,0])

# plt.show()

N_CONST = 100

H_CONST = int((t_end - t_start) / N_CONST)

T = np.arange(t_start, t_end, h)



param_df = pd.DataFrame(columns=['C', 'n', 'θ_1*', 'θ_2*','delta*', 'w0_sqr*'])

for C,n in itertools.product(C_list, n_list):

        x = odeint(ZgasColiv, np.array([x0, x00]), T, (δ, ω0_sqr))

        x1 = x0 + x00

        x11 = x00

        x_1 = odeint(ZgasColiv, np.array([x1, x11]), T+1, (δ, ω0_sqr))

        df = pd.DataFrame()

#         df['i'] = range(1, n+1)

#         df['t'] = list(map(int, T))

        df['x(t)'] = x[:,0].flatten()

        df['x(t+1)'] = x_1[:,0].flatten()

        df['x(t+2)'] = np.array(df[['x(t)','x(t+1)']]) @ theta

        df['x(t+2)'] = np.round(df['x(t+2)'], C)

        X = np.array(df[['x(t)', 'x(t+1)']])

        y = np.array(df['x(t+2)'])  

        theta_pred = RMNK(X, y, printt=False)[0][:,0]

        δ_pred, ω0_sqr_pred = findparams(*theta_pred)

        param_df =param_df.append({'C': C, 'n': n,'θ_1*': theta_pred[0],

                                   'θ_2*': theta_pred[1],'delta*': δ_pred, 

                                   'w0_sqr*': ω0_sqr_pred},

                                  ignore_index=True)
param_df
fig, axs = plt.subplots(nrows=param_df.shape[0], ncols=2, figsize=(20, 50))



for i in range(param_df.shape[0]):

    δ = param_df.iloc[i]['delta*']

    ω0_sqr = param_df.iloc[i]['w0_sqr*']

    C = param_df.iloc[i]['C']

    n = param_df.iloc[i]['n']

    

    x = odeint(ZgasColiv, np.array([x0, x00]), T, (δ, ω0_sqr))

    x1 = x0 + x00

    x11 = x00

    x_1 = odeint(ZgasColiv, np.array([x1, x11]), T+1, (δ, ω0_sqr))

        

    param_df.loc[i, 'RMSE'] = np.sqrt(mean_squared_error(x, x_1))

    f = interp1d(T, x[:, 0], kind='quadratic')

    t_new = np.linspace(T[0], T[-1], 10000)

    axs[i][0].plot(t_new, f(t_new), color='blue')

    axs[i][0].plot(T, x[:, 0], color='red')

    #axs[i][0].plot(T, x_1[:, 0], color='blue')

    

    axs[i][1].plot(T[:50], x[:50, 0], color='red')

    axs[i][1].plot(T[:50], x_1[:50, 0], color='blue')

    

    axs[i][0].legend(['Approx, δ={}, ω0_sqr={}'.format(round(δ, 4), round(ω0_sqr, 4)), 'Real'])

    axs[i][1].legend(['Approx, δ={}, ω0_sqr={}'.format(round(δ, 4), round(ω0_sqr, 4)), 'Real'])

    

    axs[i][0].set_title("Original, C={}, n={}".format(C, n))

    axs[i][1].set_title("Cropped, C={}, n={}".format(C, n))

    

    print('Iter: {}, RMSE: {}, δ: {}, ω0_sqr: {}'.format(i, np.sqrt(mean_squared_error(x_1, x)), 

                                                         δ, ω0_sqr))
param_df
import numpy as np

import pandas as pd

from scipy.integrate import odeint

import matplotlib.pyplot as plt

from pylab import rcParams

rcParams['figure.figsize'] = 12, 6





def create_df(RSS_next, numOfColX, yShape):

    df = pd.DataFrame(columns=['s', 'RSS', 'Cp', 'FPE'])

    Cp = RSS_next + 2 * numOfColX

    numOfRowY = yShape

    FPE = (numOfRowY + numOfColX) / (numOfRowY - numOfColX) * RSS_next

    df = df.append({'s': numOfColX, 'RSS': RSS_next, 'Cp': Cp, 'FPE': FPE}, ignore_index = True)

    return df





def RMNKO(X, y, s=None, printt=False, cr_df=False):

    numOfColX = X.shape[1]

    if numOfColX > 1:

        if cr_df:

            w, H_inv, RSS, df = RMNKO(X[:, :-1], y, s, printt, cr_df)

            if s is not None and numOfColX > s:

                return w, H_inv, RSS, df

        else:

            w, H_inv, RSS = RMNKO(X[:, :-1], y, s, printt, cr_df)

            if s is not None and numOfColX > s:

                return w, H_inv, RSS



        h = (X[:, :-1].T @ X[:, -1]).reshape(-1, 1)

        η = X[:, -1].T @ X[:, -1]

        α = H_inv @ h

        β = η - h.T @ α

        β_inv = 1 / β

        γ = X[:, -1].T @ y

        ν = β_inv * (γ - h.T @ w)

        w = np.vstack((w - ν * α, ν))

        H_next_inv = np.vstack((np.hstack((H_inv + β_inv * α @ α.T, (- β_inv * α).reshape(-1, 1))),

                                np.hstack((-β_inv * α.T, β_inv))))

        RSS_next = (RSS - ν.flatten() ** 2 * β.flatten())[0]



    else:

        H_inv = np.array([[0]])

        η = β = X[:, -1].T @ X[:, -1]

        β_inv = 1 / β

        α = h = np.array([0])

        γ = X[:, -1].T @ y

        ν = np.array([β_inv * γ])

        w = np.array([ν])

        H_next_inv = np.array(β_inv).reshape(1, 1)

        RSS_next = (y.T @ y - y.T @ X[:, -1].reshape(-1, 1) @ w)[0]

        if cr_df:

            df = pd.DataFrame(columns=['s', 'RSS', 'Cp', 'FPE'])



    if printt:

        print('Крок №{}'.format(numOfColX))

        print('---θ{}: {}'.format(numOfColX, w[:, 0]))

        print('---H{}inv:\n{}'.format(numOfColX, H_next_inv))

        print('---RSS{}: {}'.format(numOfColX, RSS_next))



    if cr_df:

        Cp = RSS_next + 2 * numOfColX

        numOfRowY = y.shape[0]

        FPE = (numOfRowY + numOfColX) / (numOfRowY - numOfColX) * RSS_next

        df = df.append({'s': numOfColX, 'RSS': RSS_next, 'Cp': Cp, 'FPE': FPE}, ignore_index = True)

        return w, H_next_inv, RSS_next, df

    return w, H_next_inv, RSS_next
import numpy as np

import pandas as pd

from scipy.integrate import odeint

import matplotlib.pyplot as plt

from pylab import rcParams

# from RMNKO import create_df, RMNKO

rcParams['figure.figsize'] = 12, 6

def ZgasColiv(x, t, δ, ω0_sqr):

    return [x[1], - 2 * δ * x[1] - ω0_sqr * x[0]]



def findtheta(δ, ω0_sqr):

    '''d = 1 + 2 * δ

    theta1 = (2 + 2 * δ - ω0_sqr) / d

    theta2 = - 1 / d

    '''

    theta1 = (2 - 2 * δ - ω0_sqr)

    theta2 = - 1 + 2 * δ



    return np.array([theta1, theta2])



def findparams(theta1, theta2):

    δ = - (1 / theta2 + 1) / 2

    ω0_sqr = 1 - 1 / theta2 + theta1 / theta2

    δ =  (theta2 + 1) / 2

    ω0_sqr = 1 - theta2 - theta1

    return δ, ω0_sqr



δ = 0.005

ω0_sqr = 0.01

x0 = 5

x00 = 2

t_start = 0

t_end = 1000

n = 100

n_list = [10, 20, 40]

C = 2

C_list = [1, 3, 7]



theta = findtheta(δ, ω0_sqr)

h = int((t_end - t_start) / (n - 1))

t = np.linspace(t_start, t_end, num=n)

x = odeint(ZgasColiv, np.array([x0, x00]), t, (δ, ω0_sqr))

x1 = x0 + x00

x11 = x00

x_1 = odeint(ZgasColiv, np.array([x1, x11]), t + 1, (δ, ω0_sqr))

df = pd.DataFrame()

df['i'] = range(1, n + 1)

df['t'] = list(map(int, t))

df['x(t)'] = x[:, 0].flatten()

df['x(t+1)'] = x_1[:, 0].flatten()

df['x(t+2)'] = np.array(df[['x(t)', 'x(t+1)']]) @ theta

df['x(t+2)'] = np.round(df['x(t+2)'], C)

X = np.array(df[['x(t)', 'x(t+1)']])

y = np.array(df['x(t+2)'])



print('δ = {} ω0^2 = {} x0 = {} x00 = {}'.format(δ, ω0_sqr, x0, x00))

print('C = {}'.format(C))

print('n = {}'.format(n))

print('t =[{};{}] з h={}\n'.format(t_start, t_end, h))



z = 'Коливання:' + 'delta:' + str(round(δ, 2)) + ';omega0:' + str(ω0_sqr) + ';x0:' + str(x0) + ';x00:' + str(x00)



plt.title(z)

plt.scatter(t, y,color = 'black')

plt.show()

print(df)

handle = open("text1.txt", "w")

ziz = df.values.tolist()

for item in ziz:

    handle.write("%s\n" % item)

handle.close()



print('РМНКО')

theta_pred = RMNKO(X, y, printt=True)[0][:, 0]

δ_pred, ω0_sqr_pred = findparams(*theta_pred)

print('Правдиві значення/Оцінка:θ_1 = {} θ_2 = {} θ_1* = {} θ_2* = {}'.format(*theta, *theta_pred))

print('Правдиві значення/Оцінка: δ = {} ω0^2 = {} δ* = {} ω0^2* = {}'.format(δ, ω0_sqr, δ_pred, ω0_sqr_pred))

plt.title(z)

plt.scatter(t, y,color = 'black')

t_plot = np.linspace(t_start, t_end, num=n * 10)

plt.plot(t_plot, odeint(ZgasColiv, np.array([x0, x00]), t_plot, (δ_pred, ω0_sqr_pred))[:, 0], 'r', color='yellow')

plt.show()



param_df = pd.DataFrame(columns=['C', 'n', 'θ_1*', 'θ_2*', 'δ*', 'ω0_sqr*'])

for C in C_list:

    for n in n_list:

        h = int((t_end - t_start) / (n - 1))

        t = np.linspace(t_start, t_end, num=n)

        x = odeint(ZgasColiv, np.array([x0, x00]), t, (δ, ω0_sqr))

        x1 = x0 + x00

        x11 = x00

        x_1 = odeint(ZgasColiv, np.array([x1, x11]), t + 1, (δ, ω0_sqr))

        df = pd.DataFrame()

        df['i'] = range(1, n + 1)

        df['t'] = list(map(int, t))

        df['x(t)'] = x[:, 0].flatten()

        df['x(t+1)'] = x_1[:, 0].flatten()

        df['x(t+2)'] = np.array(df[['x(t)', 'x(t+1)']]) @ theta

        df['x(t+2)'] = np.round(df['x(t+2)'], C)

        X = np.array(df[['x(t)', 'x(t+1)']])

        y = np.array(df['x(t+2)'])

        theta_pred = RMNKO(X, y, printt=False)[0][:, 0]

        δ_pred, ω0_sqr_pred = findparams(*theta_pred)

        param_df = param_df.append(

            {'C': C, 'n': n, 'θ_1*': theta_pred[0], 'θ_2*': theta_pred[1], 'δ*': δ_pred, 'ω0_sqr*': ω0_sqr_pred},

            ignore_index=True)
param_df
param_df = param_df.astype(float)
fig, axs = plt.subplots(nrows=param_df.shape[0], ncols=2, figsize=(20, 50))



for i in range(param_df.shape[0]):

    δ = param_df.iloc[i]['δ*']

    ω0_sqr = param_df.iloc[i]['ω0_sqr*']

    C = param_df.iloc[i]['C']

    n = param_df.iloc[i]['n']

    

    x = odeint(ZgasColiv, np.array([x0, x00]), T, (δ, ω0_sqr))

    x1 = x0 + x00

    x11 = x00

    x_1 = odeint(ZgasColiv, np.array([x1, x11]), T+1, (δ, ω0_sqr))

        

    param_df.loc[i, 'RMSE'] = np.sqrt(mean_squared_error(x, x_1))

    f = interp1d(T, x[:, 0], kind='quadratic')

    t_new = np.linspace(T[0], T[-1], 10000)

    axs[i][0].plot(t_new, f(t_new), color='blue')

    axs[i][0].plot(T, x[:, 0], color='red')

    #axs[i][0].plot(T, x_1[:, 0], color='blue')

    

    axs[i][1].plot(T[:50], x[:50, 0], color='red')

    axs[i][1].plot(T[:50], x_1[:50, 0], color='blue')

    

    axs[i][0].legend(['Approx, δ={}, ω0_sqr={}'.format(round(δ, 4), round(ω0_sqr, 4)), 'Real'])

    axs[i][1].legend(['Approx, δ={}, ω0_sqr={}'.format(round(δ, 4), round(ω0_sqr, 4)), 'Real'])

    

    axs[i][0].set_title("Original, C={}, n={}".format(C, n))

    axs[i][1].set_title("Cropped, C={}, n={}".format(C, n))

    

    print('Iter: {}, RMSE: {}, δ: {}, ω0_sqr: {}'.format(i, np.sqrt(mean_squared_error(x_1, x)), 

                                                         δ, ω0_sqr))
param_df
m = 5

n_list = [10, 30, 100]

theta = np.array([3, -2, 1, 0, 0])

sigma_list = [0.1, 0.5, 1]

s = 5

for n,sigma in itertools.product(n_list, sigma_list):

        X = np.random.uniform(0, 10, size=(n, m))

        ksi = np.random.normal(0, 0.01, size=n)

        y =X @ theta + ksi

        print('n = {}'.format(n))

        print('sigma = {}'.format(sigma))

        print('X:{}'.format(X))

        print('y:{}'.format(y))

        theta_pred, _, _, df = RMNK(X, y, s=s, printt=True, cr_df=True)

        #p = np.flip(np.arange(m), axis=0)

        #theta_pred, _, _, df = RMNK(X[:,p], y, s=s, printt=True, cr_df=True)

        print('Справжні значення: tetha={}'.format(theta))

        print('Оцінки: tetha*= {}'.format(theta_pred[:,0]))

        plt.plot(df['s'], df['RSS'], label='RSS')

        print_plot(df['s'], df['RSS'])

        plt.plot(df['s'], df['Cp'], label='Cp')

        print_plot(df['s'], df['Cp'])

        plt.plot(df['s'], df['FPE'], label='FPE')

        print_plot(df['s'], df['FPE'])

        plt.legend()

        plt.show()

        print(df)
