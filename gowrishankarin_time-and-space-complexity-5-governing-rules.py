%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np
n = np.linspace(1, 100)

plt.plot(n, 7*n*n + 6*n + 5)

plt.show()
plt.plot(n, 7*n*n + 6*n + 5, label="7n^2 + 6n +5")

plt.plot(n, 20 * n, label="20n")

plt.legend(loc="upper left")

plt.show()
n = np.linspace(1, 10)

plt.plot(n, 7*n*n + 6*n + 5, label="7n^2 + 6n +5")

plt.plot(n, 20 * n, label="20n")

plt.legend(loc="upper left")

plt.show()
n = np.linspace(1, 5)

plt.plot(n, 7*n*n + 6*n + 5, label="7n^2 + 6n +5")

plt.plot(n, 7*n*n , label="7n^2")

plt.legend(loc="upper left")

plt.show()
n = np.linspace(1, 100)

plt.plot(n, 7*n*n + 6*n + 5, label="7n^2 + 6n +5")

plt.plot(n, 7*n*n , label="7n^2")

plt.legend(loc="upper left")

plt.show()
plt.plot(n, (7 * n * n + 6 * n + 5)/(7 * n * n), label="7n^2 + 6n +5 / 7n^2")

plt.legend(loc="upper right")

plt.show()
plt.plot(n, (7 * n * n + 6 * n + 5)/(n * n), label="7n^2 + 6n +5 / n^2")

plt.legend(loc="upper right")

plt.show()
n = np.linspace(1, 10)

plt.plot(n, n, label="n")

plt.plot(n, n * n, label="n^2")

plt.plot(n, n * n * n, label="n^3")

plt.legend(loc='upper left')

plt.show()
n = np.linspace(1, 10)

plt.plot(n, n, label="n")

plt.plot(n, n * n, label="n^2")

plt.plot(n, n * n * n, label="n^3")

plt.plot(n, n * n * n * n, label="n^4")

plt.legend(loc='upper left')

plt.show()
n = np.linspace(1, 100)

plt.plot(n, n, label="n")

plt.plot(n, n * n, label="n^2")

plt.plot(n, n * n * n, label="n^3")

#plt.plot(n, n * n * n * n, label="n^4")

plt.legend(loc='upper left')

plt.show()
n = np.linspace(1, 10)

plt.plot(n, n ** 4, label="n^4")

plt.plot(n, 2 ** n, label="2^n")

plt.legend(loc='upper left')

plt.show()
n = np.linspace(1, 18)

plt.plot(n, n ** 4, label="n^4")

plt.plot(n, 2 ** n, label="2^n")

plt.legend(loc='upper left')

plt.show()
n = np.linspace(1, 20)

plt.plot(n, n, label="n")

plt.plot(n, np.log(n), label="log n")

plt.legend(loc='upper left')

plt.show()
n = np.linspace(1, 3)

plt.plot(n, n ** .5, label="n^.5")

plt.plot(n, np.log(n) ** 3, label="(log n)^3")

plt.legend(loc='upper left')

plt.show()
n = np.linspace(1, 5)

plt.plot(n, n ** .5, label="n^.5")

plt.plot(n, np.log(n) ** 3, label="(log n)^3")

plt.legend(loc='upper left')

plt.show()
n = np.linspace(1, 10**6)

plt.plot(n, n ** .5, label="n^.5")

plt.plot(n, np.log(n) ** 3, label="(log n)^3")

plt.legend(loc='upper left')

plt.show()
n = np.linspace(1, 10**7)

plt.plot(n, n ** .5, label="n^.5")

plt.plot(n, np.log(n) ** 3, label="(log n)^3")

plt.legend(loc='upper left')

plt.show()
n = np.linspace(1, 10**8)

plt.plot(n, n ** .5, label="n^.5")

plt.plot(n, np.log(n) ** 3, label="(log n)^3")

plt.legend(loc='upper left')

plt.show()
n = np.linspace(1, 100)

plt.plot(n, n ** .1, label="n^.1")

plt.plot(n, np.log(n) ** 5, label="(log n)^5")

plt.legend(loc='upper left')

plt.show()
n = np.linspace(1, 10**125)

plt.plot(n, n ** .1, label="n^.1")

plt.plot(n, np.log(n) ** 5, label="(log n)^5")

plt.legend(loc='upper left')

plt.show()