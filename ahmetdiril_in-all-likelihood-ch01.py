import numpy as np
import numpy.polynomial.polynomial as poly
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

k_list = np.array([-1e48, 0, 1, 2, 4, 5, 6, 7, 8])
k_list_cut = np.array([0, 1, 2, 4, 5, 6, 7, 8])
planets = ["Merkür", "Venüs", "Dünya", "Mars", "?", "Jüpiter", 
           "Satürn", "Uranüs (1781)", "Neptün (1846)", "Pluto (1930)"]
x_test = np.linspace(0, 8, 50)
#k_test = np.linspace(0, 2**8, 50)
#k_2 = [2**k for k in k_list]
Bode = np.array([ (4 + 3 *(2**k)) for k in x_test ])

Observed_dist = np.array([4.0, 7.2, 10, 15.3, 51.9, 95.5, 191.4, 300.0, 394.6])
Observed_dist_cut = np.array([7.2, 10, 15.3, 51.9, 95.5, 191.4, 300.0, 394.6])

# Create color maps for 3-class classification problem, as with iris
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

x = k_list_cut
y = Observed_dist_cut


coefs = poly.polyfit(x, y, 4)
ffit = poly.Polynomial(coefs)    # instead of np.poly1d
#print(ffit(10))

plt.figure(figsize=(6, 4))
plt.scatter(k_list, Observed_dist, s=30, label="Gözlenen")
plt.plot(x_test, ffit(x_test), label='4. derece', linestyle="dashed")
plt.plot(x_test, Bode, label="Bode kanunu")
plt.yscale("log")
plt.yticks((5, 50, 500), ('5', '50', '500'), color='k', size=15)

plt.legend(loc='best')
#plt.axis('tight')
plt.xlim(0,10)

plt.title('Dördüncü derece polinom fiti')
plt.ylabel('Uzaklık')
plt.xlabel('Sıra')
plt.show()