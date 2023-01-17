import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
xs_pt_1 = np.random.normal(0.5, 0.01, 100)

ys_pt_1 = np.random.normal(0.5, 0.01, 100)



xs_pt_2 = np.random.normal(0.2, 0.01, 100)

ys_pt_2 = np.random.normal(0.2, 0.01, 100)



xs_pt_3 = np.random.normal(0.2, 0.01, 100)

ys_pt_3 = np.random.normal(0.5, 0.01, 100)



xs_pt_4 = np.random.normal(0.5, 0.01, 100)

ys_pt_4 = np.random.normal(0.5, 0.01, 100)



x_q = [0.40]

y_q = [0.40]



x_q2 = [0.48]

y_q2 = [0.50]


fig, axarr = plt.subplots(1,2)

fig.set_size_inches(12, 4)



axarr[0].scatter(xs_pt_2, ys_pt_2)

axarr[0].scatter(xs_pt_1, ys_pt_1)

axarr[0].scatter(xs_pt_3, ys_pt_3)

axarr[0].scatter(x_q, y_q, label="query point")

axarr[0].legend()

axarr[0].grid()

axarr[0].set_title('Case 1')



axarr[1].scatter(xs_pt_1, ys_pt_1, color='green')

axarr[1].scatter(xs_pt_4, ys_pt_4, color='yellow')

axarr[1].scatter(x_q2, y_q2, label="query point", color='red')

axarr[1].legend()

axarr[1].grid()

axarr[1].set_title('Case 2')



plt.show()