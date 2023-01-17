import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
def get_moments(Xs, decay_rate=0.9):

    """

    REMEMBER AND FORGET

    ===================

    input: list, decay_rate

    output: list

    

    Recursive equation using loop

    """

    xs_w_moms = []



    gamma    = decay_rate

    prev_mom = 0 



    for x in Xs:

        # note: addition

        # ( (1-gamma)*x ) forgets

        # ( gamma * prev_mom ) remebers

        w_mom = ( (1-gamma)*x ) + ( gamma * prev_mom )

        xs_w_moms.append(w_mom)



        # update:

        prev_mom = w_mom

        

    return xs_w_moms







def get_moments_rem(Xs, decay_rate=0.9):

    """

    REMEMBER ONLY

    ===================

    input: list, decay_rate

    output: list

    

    Recursive equation using loop

    """

    xs_w_moms = []



    gamma    = decay_rate

    prev_mom = 0 



    for x in Xs:

        # note: addition

        w_mom = x + ( gamma * prev_mom )

        xs_w_moms.append(w_mom)



        # update:

        prev_mom = w_mom

        

    return xs_w_moms
# gen data

er = np.random.normal(0, 0.1, 500)

xs = np.arange(0, 5, 0.01)

ys = np.sin(xs) + er



fig, axarr = plt.subplots(2,2)

fig.set_size_inches(20,10)





# plot 1

axarr[0,0].scatter(xs, ys, color='blue', alpha=0.2)

axarr[0,0].set_title("Orignal Data")



# plot 2

axarr[0,1].scatter(xs, get_moments_rem(ys, 0.3), color='red', alpha=0.8)

axarr[0,1].scatter(xs, ys, color='blue', alpha=0.2)

axarr[0,1].set_title("Decay coeff: 0.3")



# plot 3

axarr[1,0].scatter(xs, get_moments_rem(ys, 0.6), color='red', alpha=0.8)

axarr[1,0].scatter(xs, ys, color='blue', alpha=0.2)

axarr[1,0].set_title("Decay coeff: 0.6")



# plot 4

axarr[1,1].scatter(xs, get_moments_rem(ys, 0.9), color='red', alpha=0.8)

axarr[1,1].scatter(xs, ys, color='blue', alpha=0.2)

axarr[1,1].set_title("Decay coeff: 0.9")



plt.show()
# gen data

er = np.random.normal(0, 0.1, 500)

xs = np.arange(0, 5, 0.01)

ys = np.sin(xs) + er



fig, axarr = plt.subplots(2,2)

fig.set_size_inches(20,10)





# plot 1

axarr[0,0].scatter(xs, ys, color='blue', alpha=0.2)

axarr[0,0].set_title("Orignal Data")



# plot 2

axarr[0,1].scatter(xs, get_moments(ys, 0.3), color='red', alpha=0.8)

axarr[0,1].scatter(xs, ys, color='blue', alpha=0.2)

axarr[0,1].set_title("Decay coeff: 0.3")



# plot 3

axarr[1,0].scatter(xs, get_moments(ys, 0.6), color='red', alpha=0.8)

axarr[1,0].scatter(xs, ys, color='blue', alpha=0.2)

axarr[1,0].set_title("Decay coeff: 0.6")



# plot 4

axarr[1,1].scatter(xs, get_moments(ys, 0.9), color='red', alpha=0.8)

axarr[1,1].scatter(xs, ys, color='blue', alpha=0.2)

axarr[1,1].set_title("Decay coeff: 0.9")



plt.show()