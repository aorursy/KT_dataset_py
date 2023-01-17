import numpy as np # linear algebra

import matplotlib.pyplot as plt

import seaborn as sns

import math
sq2 = math.sqrt(2)



def one_metric(error, s):

    sigma = max(s, 70)

    d = min(error, 1000)

    return - sq2*d/sigma - math.log(sq2*sigma)



print(one_metric(100, 50))

print(one_metric(100, 60))

print(one_metric(200, 90))

print(one_metric(10, 50))

print(one_metric(70, 72))
print(one_metric(70/sq2, 70))

print(one_metric(70/sq2, 80))

print(one_metric(70/sq2, 60))

print(one_metric(70/sq2, 70+0.01))

print(one_metric(70/sq2, 70-0.01))
def plot_surface(d_min = 10, d_max = 1000, s_min = 50, s_max = 200):



    sigmas = np.arange(s_min, s_max + 1)



    deltas = np.arange(d_min, d_max + 1)



    surface = np.empty((deltas.size, sigmas.size))



    for i in range(deltas.size):

        for j in range(sigmas.size):

            surface[i,j] = one_metric(deltas[i], sigmas[j])



    print(f'max value is {surface.max()}')

    figure, axes = plt.subplots(figsize=(16, 16))



    c = axes.pcolormesh(sigmas, deltas, surface, cmap='magma')

    axes.contour(sigmas, deltas, surface, colors='red')

    axes.set_title('Heatmap')

    figure.colorbar(c)



    plt.show()



plot_surface(70, 1000)
plot_surface(70/sq2, 1000, 0, 400)
for delta in (60, 65, 70, 82.8, 300):

    print(f'{one_metric(delta, delta * sq2)} is better than {one_metric(delta, delta * sq2 + 1)} and {one_metric(delta, delta * sq2 - 1)}')
plot_surface(0, 50, 70 ,150)
def err(metric, sigma):

    return -(metric + math.log(sq2 * sigma)) * sigma / sq2





print(err(-7.02, 100))



print(err(-8.93, 71))