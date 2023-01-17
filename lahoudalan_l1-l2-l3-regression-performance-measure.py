import numpy as np

from matplotlib import pyplot as plt
def measu(v, k):

    return np.power((1/len(v))*(np.sum(np.power(np.abs(v),k))),1/k)



def compute_error_list_varying_outlier_value(n_iterations, v, k, n_outliers):

    

    v_new = v.copy()

    measure_values = []

    for outlier_value in range(0, 30):

        for n in range(0, n_outliers):

            v_new[n] = outlier_value



        measure_values.append(measu(v_new, k))

    return measure_values



def compute_l1_l2_l3_measure(percentage_outliers, v):

    l1 = compute_error_list_varying_outlier_value(n_iterations=30, v=v, k=1, n_outliers=int(m*percentage_outliers))

    l2 = compute_error_list_varying_outlier_value(n_iterations=30, v=v, k=2, n_outliers=int(m*percentage_outliers))

    l3 = compute_error_list_varying_outlier_value(n_iterations=30, v=v, k=3, n_outliers=int(m*percentage_outliers))

    

    return [l1, l2, l3]



def plot_measure_by_outliers_values(percentage_outliers, v):

    

    percentage_outliers = round(percentage_outliers,3)

    l = compute_l1_l2_l3_measure(percentage_outliers=percentage_outliers, v=v)

    str_percentage = str(round(100*percentage_outliers,3)) + '%'

    

    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)



    

    print('L2 is ', round(l[1][5]/l[0][5], 3), ' times L1 for outliers = 5')

    print('L3 is ', round(l[2][5]/l[0][5], 3), ' times L1 for outliers = 5')

    

    print('L2 is ', round(l[1][20]/l[0][20], 3), ' times L1 for outliers = 20')

    print('L3 is ', round(l[2][20]/l[0][20], 3), ' times L1 for outliers = 20')

    

    



    fig.suptitle('{st} of error outliers'.format(st = str_percentage))

    ax.plot(l[0], label='L1 norm')

    ax.plot(l[1], label='L2 norm')

    ax.plot(l[2], label='L3 norm')



    ax.axvline(3, linestyle='--', color='r', label='Not biiiig outliers')

    ax.axvline(20, linestyle='--', color='b', label='Very big outliers')



    ax.set_xlabel('Outlier value')

    ax.set_ylabel('Final measure value (L1, L2, L3)')



    ax.legend()
# number of examples to be evaluated

m = 1000



# The error vector (y - h(x))

v = np.random.normal(loc=0.0, scale=1.0, size=m)
p = 0

plot_measure_by_outliers_values(percentage_outliers=p, v=v)
p = 0.005

plot_measure_by_outliers_values(percentage_outliers=p, v=v)
p = 0.02

plot_measure_by_outliers_values(percentage_outliers=p, v=v)
p = 0.10

plot_measure_by_outliers_values(percentage_outliers=p, v=v)
p = 0.30

plot_measure_by_outliers_values(percentage_outliers=p, v=v)
p = 1

plot_measure_by_outliers_values(percentage_outliers=p, v=v)