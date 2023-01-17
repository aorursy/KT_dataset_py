import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
cities = pd.read_csv('../input/cities.csv')
cities.head()
from matplotlib import collections  as mc
import matplotlib.animation as animation

def create_path_animation(cities, paths, interval=300):
    def update(i):
        coords = cities[['X', 'Y']].values
        ordered_coords = coords[paths[i]]
        xs, ys = zip(*ordered_coords)
        plt.cla()
        plt.plot(xs, ys,  lw=1., ms=10, c='blue')
        plt.xlim(0, cities.X.max())
        plt.ylim(0, cities.Y.max())
        plt.title('step=' + str(i))
        plt.axis('off')

    fig = plt.figure(figsize = (6, 4))
    ani = animation.FuncAnimation(fig, update, interval=interval, frames = len(paths), repeat=False)
    ani.save(f"sample.gif", writer='imagemagick')
# Create sample path
paths = []
for i in range(1, 10):
    paths.append(list(range(i)))
# Create and save animation gif.
create_path_animation(cities, paths)
