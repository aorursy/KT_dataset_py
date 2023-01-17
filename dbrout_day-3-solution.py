#Outlier exercise.

xs, ys = d3.T

p = np.polyfit(xs, ys,deg=5)

ps = np.polyval(p, xs)

plt.plot(xs, ys, ".", label="Data", ms=1)

plt.plot(xs, ps, label="Bad poly fit")

plt.legend();
x, y = xs.copy(), ys.copy()

for i in range(5):

    p = np.polyfit(x, y, deg=5)

    ps = np.polyval(p, x)

    good = y - ps < 3  # only remove positive outliers

    

    x_bad, y_bad = x[~good], y[~good]

    x, y = x[good], y[good]

    

    plt.plot(x, y, ".", label="Used Data", ms=1)

    plt.plot(x, np.polyval(p, x), label=f"Poly fit {i}")

    plt.plot(x_bad, y_bad, ".", label="Not used Data", ms=5, c="r")

    plt.legend()

    plt.show()

    

    if (~good).sum() == 0:

        break
from sklearn.neighbors import LocalOutlierFactor



lof = LocalOutlierFactor(n_neighbors=20, contamination=0.005)

good = lof.fit_predict(d2) == 1

plt.scatter(d2[good, 0], d2[good, 1], s=2, label="Good", color="#4CAF50")

plt.scatter(d2[~good, 0], d2[~good, 1], s=8, label="Bad", color="#F44336")

plt.legend();