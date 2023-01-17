! conda install -y -c conda-forge hvplot==0.5.2 bokeh==1.4.0
import numba

import numpy as np

import pandas as pd

import hvplot.pandas

import holoviews as hv

from operator import add

from toolz.curried import *





hv.extension('bokeh')
@numba.jit(nopython=True, nogil=True)

def factorial(n):

    if n < 0:

        return 0

    elif n < 20:

        val = 1

        for k in range(1,n+1):

            val *= k

        return val

    else:

        return np.math.gamma(n)

    

@numba.jit(nopython=True, nogil=True)

def choose(n, r):

    return factorial(n)/(factorial(n-r)*factorial(r))



@numba.jit(nopython=True, nogil=True)

def legendre(x:np.ndarray, n:int=1):

    L = np.zeros(x.shape)

    for k in range(0, n+1):

        L = L + choose(n,k) * choose(n+k, k) * (((x-1)/2)**k)

    return L



@numba.jit(nopython=True, nogil=True)

def basis(x: np.ndarray, d:int = 2):

    L = legendre(x, n=0).reshape(-1,1)

    for n in range(1,d+1):

        L = np.hstack((L,legendre(x, n).reshape(-1,1)))

    return L
%%timeit

basis(np.random.uniform(-1, 1, 500), d=5)
(pd.DataFrame(basis(x=np.random.uniform(-1, 1, 500), d=5))

 .assign(x=lambda d: d.loc[:,1])

 .sort_values('x')

 .melt(id_vars='x', var_name=['Order'])

 .hvplot.line(x='x', y='value', groupby='Order', title='Different Order Legendre Polynomials')

 .overlay())


def plot(function=basis, degree=5, x=np.random.uniform(-1, 1, 500)):

    return ((pd.DataFrame(function(x, d=degree), index=['Legendre' for _ in range(x.shape[0])])

             .fillna(0)

             .dot(np.random.uniform(-1,1, size=(degree+1,3)))

             .assign(x=x.repeat(1))

             .reset_index()

             .rename(columns={'index':'Order'})

              .melt(id_vars=['Order','x'], value_name='f(x)')

              .sort_values('x')

             .hvplot.line(x='x', y='f(x)', color='red', label='Legendre', groupby=['variable'])

             .overlay()

             .opts(show_legend=False)) *

           (pd.DataFrame({d:x**d for d in range(degree+1)}, index=['Polynomial' for _ in range(x.shape[0])])

             .fillna(0)

             .dot(np.random.uniform(-1,1, size=(degree+1,3)))

             .assign(x=x.repeat(1))

             .reset_index()

             .rename(columns={'index':'Order'})

              .melt(id_vars=['Order','x'], value_name='f(x)')

              .sort_values('x')

             .hvplot.line(x='x', y='f(x)', color='green', label="Polynomial", groupby=['variable'])

             .overlay()

             .opts(show_legend=False))).opts(title='degree ' + str(degree))



x = np.random.uniform(-1, 1, 500)

reduce(add, [plot(degree = d, x=x) for d in range(4)]).cols(2)
@numba.jit()

def E(num_projections =1000,

      X_in = basis(np.random.uniform(-1, 1, 51),10),

      X_out = basis(np.random.uniform(-1, 1, 100),10),

      e_in = np.random.multivariate_normal(np.zeros(90), 

                                       np.diag(np.linspace(0.2,1.1, 90)), 

                                       111),

      e_out = np.random.multivariate_normal(np.zeros(90), 

                                        np.diag(np.linspace(0.2,1.1, 90)), 

                                        100),

      N = 10,

      order = 2):

    x_in_true = np.ascontiguousarray(X_in[:N,:order])

    x_out_true = np.ascontiguousarray(X_out[:,:order])

    X_in_2 = np.ascontiguousarray(X_in[:N,:2])

    X_out_2 = np.ascontiguousarray(X_out[:,:2])

    X_in_10 = np.ascontiguousarray(X_in[:N,:10])

    x_out_10 =np.ascontiguousarray(X_out[:,:10])

    e_in = np.ascontiguousarray(e_in[:N,:])

    

    E = np.zeros((num_projections, e_in.shape[1]))

    for run in numba.prange(num_projections):

        B_true = np.random.uniform(-1,1,size=(order,))

        y_true_in = (x_in_true@B_true).reshape(-1,1) + e_in

        y_true_out = (x_out_true@B_true).reshape(-1,1) + e_out



        B_2 = np.linalg.pinv(X_in_2.T @ X_in_2) @ X_in_2.T @ y_true_in

        mse_2 = np.ones((1,X_out_2.shape[0])) @ (y_true_out -  X_out_2 @ B_2)**2 / X_out_2.shape[0]



        B_10 = np.linalg.pinv(X_in_10.T @ X_in_10) @ X_in_10.T @ y_true_in

        mse_10 = np.ones((1,X_out_2.shape[0])) @ (y_true_out - x_out_10 @ B_10)**2 / X_out_2.shape[0]



        E[run,:] = mse_10 - mse_2



    return (np.ones((1,num_projections)) @ E /num_projections).reshape(-1,)



@numba.jit()

def experiment_2(order:int = 5, 

                  sigma:float = 0.2,

                  sample_sizes_lower = 20,

                  sample_sizes_upper = 50,

                  x_in_seed:np.ndarray = np.random.uniform(-1, 1, 111),

                  e_in:np.ndarray = np.random.multivariate_normal(np.zeros(90), 

                                                                   np.diag(np.linspace(0.2,1.1, 90)), 

                                                                   111),

                  x_out_seed:np.ndarray = np.random.uniform(-1, 1, 100),

                  e_out:np.ndarray = np.random.multivariate_normal(np.zeros(90), 

                                                                    np.diag(np.linspace(0.2,1.1, 90)), 

                                                                    100),

                  num_projections=1000):

    

    

    max_order = max(2,10, order)

    

    X_in = (basis(x=x_in_seed, d=max_order))

    X_out = (basis(x=x_out_seed, d=max_order))



    samples_sizes = np.arange(sample_sizes_lower, sample_sizes_upper)

    results = np.zeros(((samples_sizes.shape[0]), 

                        e_in.shape[1]))

    for index in numba.prange((samples_sizes.shape[0])):

        results[index, :] = E(num_projections=num_projections,

                              X_in=X_in,

                              X_out=X_out,

                              e_in = e_in,

                              e_out = e_out,

                              N=samples_sizes[index],

                              order=5)

        

    return results
results2 = experiment_2()



df2 = (pd.DataFrame(data=results2,

                    index=list(range(20, 50)),

                    columns=np.linspace(0.2,1.1, 90))

      .iloc[:,:]

     .reset_index()

     .rename(columns={'index':"N"})

     .melt(id_vars='N',var_name='Sigma'))



dataset2 = hv.Dataset(df2, vdims=[('value','E')])



heatmap2 = hv.HeatMap(dataset2.aggregate(['Sigma', 'N'], np.sum),

                     label='Out-of-Sample error')



heatmap2.opts(width=800, 

              height=600, 

              colorbar=True,

              xrotation=90,

              logz=True,

              colorbar_opts={'title':'E'},

              tools=['hover'])
@numba.jit()

def experiment_1(true_order:int = np.array(list(range(1,41))), 

      sigma:float = 0.2,

      sample_sizes = np.array(list(range(20,111))),

      x_in_seed:np.ndarray = np.random.uniform(-1, 1, 111),

      e_in:np.ndarray = np.random.normal(0, 0.2, 111),

      x_out_seed:np.ndarray = np.random.uniform(-1, 1, 100),

      e_out:np.ndarray = np.random.normal(0, 0.2, 100),

      num_projections=1000):

    

    max_order = np.max(true_order)

    

    X_in = (basis(x=x_in_seed, d=max_order))

    X_out = (basis(x=x_out_seed, d=max_order))

    

    H = []

    for N in sample_sizes:

        R = np.zeros(true_order.shape[0])

        for index, order in np.ndenumerate(true_order):

            x_in_true = np.ascontiguousarray(X_in[:N,:order])

            x_out_true = np.ascontiguousarray(X_out[:,:order])

            X_in_2 = np.ascontiguousarray(X_in[:N,:2])

            X_out_2 = np.ascontiguousarray(X_out[:,:2])

            X_in_10 = np.ascontiguousarray(X_in[:N,:10])

            x_out_10 =np.ascontiguousarray(X_out[:,:10])



            E = 0

            for run in numba.prange(num_projections):

                B_true = np.random.uniform(-1,1,size=(order,))

                y_true_in = (x_in_true@B_true + np.ascontiguousarray(e_in[:N]))

                y_true_out = (x_out_true@B_true + e_out)



                B_2 = np.linalg.pinv(X_in_2.T @ X_in_2) @ X_in_2.T @ y_true_in

                mse_2 = np.mean((y_true_out -  X_out_2 @ B_2)**2)



                B_10 = np.linalg.pinv(X_in_10.T @ X_in_10) @ X_in_10.T @ y_true_in

                mse_10 = np.mean((y_true_out - x_out_10 @ B_10)**2)



                E = E + mse_10 - mse_2



            R[index] = E/num_projections

        H.append(R)

        

    return H
results1 = experiment_1()



df1 = (pd.DataFrame(data=np.vstack(results1),

              index=np.array(list(range(20,111))),

              columns=np.array(list(range(1,41))))

     .reset_index()

     .rename(columns={'index':"N"})

     .melt(id_vars='N',var_name='Qf'))



df1.value.hvplot.hist(xlabel='E')



dataset1 = hv.Dataset(df1, vdims=[('value','E')])



heatmap1 = hv.HeatMap(dataset1.aggregate(['Qf', 'N'], np.sum),

                     label='Out-of-Sample error').opts(xlabel='Order of Polynomial')



heatmap1[:10, :].opts(width=800, 

                     height=600, 

                     colorbar=True,

                      logz=True,

                     colorbar_opts={'title':'E'},

                     tools=['hover'])