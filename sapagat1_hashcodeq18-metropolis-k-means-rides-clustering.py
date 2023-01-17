import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
input_df = pd.read_csv('../input/d_metropolis.in', sep=' ', header=None)
input_df.head()

R, C, F, N, B, T = input_df.iloc[0]
print('R:' + str(R), 'C:' + str(C), 'F: ' + str(F), 'N:' + str(N), 'B: ' + str(B), 'T:' + str(T))
rides = input_df.iloc[1:]
rides = rides.reset_index(drop=True)
rides.columns = ['a', 'b', 'x', 'y', 's', 'f']
rides = rides.assign(d = lambda ride: abs(ride.x- ride.a) + abs(ride.y - ride.b))
rides.head()
g = sns.scatterplot(rides.a, rides.b, color='red')
g.set_title('Origins')
axes = g.axes
axes.set_ylim(0,R);
axes.set_xlim(0,C);
g = sns.scatterplot(rides.x, rides.y, color='blue')
g.set_title('Destinations')
axes = g.axes
axes.set_ylim(0,R);
axes.set_xlim(0,C);
features = ['a', 'b', 'x', 'y']
K = 4
X = rides[features]
y_pred = KMeans(n_clusters=4, random_state=123).fit_predict(X)
P = X.copy()
P.loc[:, 'k'] = y_pred
P.describe().transpose()
P.groupby('k').count()
g = sns.scatterplot(x=P['a'], y=P['b'], hue=P['k'])
g.set_title('Origins')
axes = g.axes
axes.set_ylim(0,R);
axes.set_xlim(0,C);
g = sns.scatterplot(x=P['x'], y=P['y'], hue=P['k'])
g.set_title('Destinations')
axes = g.axes
axes.set_ylim(0,R);
axes.set_xlim(0,C);
rides_to_consider = P[P['k'] == 3]
rides_to_consider.describe().transpose()