import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
import sympy
import io

pd.options.display.float_format = '{:.6f}'.format
def read_cities(filename='../input/traveling-santa-2018-prime-paths/cities.csv'):
    return pd.read_csv(filename, index_col=['CityId'])

class Tour:
    cities = read_cities()
    coords = (cities.X + 1j * cities.Y).values
    penalized = ~cities.index.isin(sympy.primerange(0, len(cities)))

    def __init__(self, data):
        """
        Initializes from a list/iterable of indexes or
        a filename of tour in csv/tsplib/linkern format.
        """

        if type(data) is str:
            data = self._read(data)
        elif type(data) is not np.ndarray or data.dtype != np.int32:
            data = np.array(data, dtype=np.int32)
        self.data = data

        # Validate tour
        if len(data) != len(self.cities) + 1:
            raise Exception('Bad length')
        if data[0] != 0 or data[-1] != 0:
            raise Exception('Must start/end with 0');
        try:
            x = np.zeros(len(data) - 1, dtype=np.bool)
            x[data[1:]] = 1
            if np.min(x) == 0:
                raise Exception('Repeated/missing cities')
        except IndexError:
            raise Exception('Indexes out of bounds')

    @classmethod
    def _read(cls, filename):
        data = open(filename, 'r').read()
        if data.startswith('Path'):  # csv
            return pd.read_csv(io.StringIO(data)).Path.values
        offs = data.find('TOUR_SECTION\n')
        if offs != -1:  # TSPLIB/LKH
            data = np.fromstring(data[offs+13:], sep='\n', dtype=np.int32)
            data[-1] = 1
            return data - 1
        else:  # linkern
            data = data.replace('\n', ' ')
            data = np.fromstring(data, sep=' ', dtype=np.int32)
            if len(data) != data[0] + 1:
                raise Exception('Unrecognized format in %s' % filename)
            return np.concatenate((data[1:], [0]))

    def info(self):
        dist = np.abs(np.diff(self.coords[self.data]))
        penalty = 0.1 * np.sum(dist[9::10] * self.penalized[self.data[9:-1:10]])
        dist = np.sum(dist)
        return { 'score': dist + penalty, 'dist': dist, 'penalty': penalty }

    def dist(self):
        return self.info()['dist']

    def score(self):
        return self.info()['score']

    def __repr__(self):
        return 'Tour: %s' % str(self.info())

    def to_csv(self, filename):
        pd.DataFrame({'Path': self.data}).to_csv(filename, index=False)

    def reversed(self):
        return Tour(self.data[::-1])

    def roll(self, k):
        return Tour(np.concatenate(([0], np.roll(self.data[1:-1], k), [0])))

    def plot(self, cmap=mpl.cm.gist_rainbow, figsize=(25, 20)):
        fig, ax = plt.subplots(figsize=figsize)
        n = len(self.data)

        for i in range(201):
            ind = self.data[n//200*i:min(n, n//200*(i+1)+1)]
            ax.plot(self.cities.X[ind], self.cities.Y[ind], color=cmap(i/200.0), linewidth=1)

        ax.plot(self.cities.X[0], self.cities.Y[0], marker='*', markersize=15, markerfacecolor='k')
        ax.autoscale(tight=True)
        mpl.colorbar.ColorbarBase(ax=fig.add_axes([0.125, 0.075, 0.775, 0.01]),
                                  norm=mpl.colors.Normalize(vmin=0, vmax=n),
                                  cmap=cmap, orientation='horizontal')
filename = '../input/lkh-solver/submission.csv'

Tour(filename)
Tour(filename).reversed()
def multiscore(pattern):
    rows = []
    for filename in glob.glob(pattern):
        tour = Tour(filename)
        for rev in range(2):
            for k in range(-10, 11):
                info = tour.roll(k).info()
                info.update({'rev': rev, 'roll': k, 'filename': filename})
                rows.append(info)
            tour = tour.reversed()
    return pd.DataFrame(rows).sort_values('score').reset_index(drop=True)

multiscore(filename).head(10)
tour = Tour(filename)
print('Original:     %.2f' % tour.score())
print('reversed:     %.2f' % tour.reversed().score())
print('roll(6):      %.2f' % tour.roll(6).score())
print('------------')
print('Improvement:  %.2f' % (tour.score() - tour.roll(6).score()))
tour = tour.roll(6)
tour
tour.to_csv('submission.csv')
tour.plot()
%timeit Tour(filename)
tour = Tour(filename)
%timeit tour.score()