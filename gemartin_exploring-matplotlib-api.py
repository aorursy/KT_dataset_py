import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('../input/train.csv')
# create a figure
fig = plt.figure(figsize=(6, 4))

# add an axes
ax = fig.add_subplot(111)

plt.show()
fig = plt.figure(figsize=(6, 4))

ax1 = fig.add_subplot(221)
ax1.set_title('first subplot')

ax2 = fig.add_subplot(222)
ax2.set_title('second subplot')

ax3 = fig.add_subplot(223)
ax3.set_title('third subplot')

ax4 = fig.add_subplot(224)
ax4.set_title('fourth subplot')

fig.tight_layout()
plt.show()
fig, axes = plt.subplots(nrows=2, ncols=2)

# we can now access any Axes the same way we would access
# an element of a 2D array
axes[0,0].set_title('first subplot')
axes[0,1].set_title('second subplot')
axes[1,0].set_title('third subplot')
axes[1,1].set_title('fourth subplot')

fig.tight_layout()
plt.show()
import matplotlib.gridspec as gridspec

fig = plt.figure()

# I use gridspec de set the grid
# I need a 2x2 grid
G = gridspec.GridSpec(2, 2)

# the first subplots is on the first row and span over all columns
ax1 = plt.subplot(G[0, :])

# the second subplot is on the first column of the second row
ax2 = plt.subplot(G[1, 0])

# the third subplot is on the second column of the second row
ax3 = plt.subplot(G[1, 1])

fig.tight_layout()
plt.show()
fig = plt.figure()

# the first subplot is two rows high
ax1 = plt.subplot(G[:, :1])

# the second subplot is on the second column of the first row
ax2 = plt.subplot(G[0, 1])

# the third subplot is on the second column of the second row
ax3 = plt.subplot(G[1, 1])

fig.tight_layout()
plt.show()
fig = plt.figure()

G = gridspec.GridSpec(2, 2,
                       width_ratios=[1, 2], # the second column is two times larger than the first one
                       height_ratios=[4, 1] # the first row is four times higher than the second one
                       )

# in this example, I use a different way to refer to a grid element
# note that it is not clear in which part of the grid the subplot is
ax1 = plt.subplot(G[0]) # same as plt.subplot(G[0, 0])
ax2 = plt.subplot(G[1]) # same as plt.subplot(G[0, 1])
ax3 = plt.subplot(G[2]) # same as plt.subplot(G[1, 0])
ax4 = plt.subplot(G[3]) # same as plt.subplot(G[1, 1])

fig.tight_layout()
plt.show()
fig, ax = plt.subplots(figsize=(6, 4), nrows=1, ncols=2)

# plot different charts on each axes
ax[0].bar(np.arange(0, 3), df["Embarked"].value_counts())
ax[1].bar(np.arange(0, 3), df["Pclass"].value_counts())

# customize a bit
ax[0].set_title('Embarked')
ax[1].set_title('Pclass')

fig.tight_layout()
plt.show()
x = np.arange(0, 10, 0.1)
y = x**2
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y)
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y)

# add labels
# we can also use LaTex notations in titles and labels
ax.set_xlabel('$x$')
ax.set_ylabel('$y = x^2$')

# reduce the axis limites to let the line touch the borders
ax.set_xlim(0, 10)
ax.set_ylim(0, 100)

# customize the ticks
ax.tick_params(labelleft=False,
               labelcolor='orange',
               labelsize=12, 
               bottom=False,
               color='green',
               width=4, 
               length=8,
               direction='inout'
              )

plt.show()
fig = plt.figure()
ax = fig.add_subplot(111)

# shorten the spines used for the x and y axis
ax.spines['bottom'].set_bounds(0.2, 0.8)
ax.spines['left'].set_bounds(0.2, 0.8)

# other customizations
ax.spines['bottom'].set_color('r')
ax.spines['bottom'].set_linewidth(2)

# remove the two other spinces
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()
# lets create some bars
x = np.arange(0, 5)
y = [2, 6, 7, 3, 4]
bars = plt.bar(x, y, color='b')

# we can get the height of the bars
for i, bar in enumerate(bars):
    print('bars[{}]\'s height = {}'.format(i, bar.get_height()))

# or we can set a different color for the third bar
bars[2].set_color('r')

# or set a different width for the first bar
bars[0].set_width(0.4)

plt.show()
fig = plt.figure()
ax = fig.add_subplot(111)

x = df['Age'].dropna()
bins = np.arange(0, 95, 5)
values, bins, bars = ax.hist(x, bins=bins)

# get the value of each bin
for bin, value in zip(bins, values):
    print('bin {}: {} passengers'.format(bin, int(value)))

# change the highest bin color    
max_idx = values.argmax()
bars[max_idx].set_color('r')

plt.show()
x = np.arange(0, 10, 0.1)
y = x**2
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y)

ax.annotate('bar', (5, 20))

ax.text(1, 
        50, 
        'foo',
        fontsize=14,
        color='red')

# add an arrow that starts at 'foo' and point at the line
ax.arrow(1.5, 48,
         0.5, -44,
         length_includes_head=True,
         width=0.3,
         head_length=4,
         facecolor='y',
         edgecolor='r',
         shape='left')

# we can also set an arrow in the annotate method
ax.annotate('quz', 
            xytext=(6, 60), # text coordinates 
            xy=(7.8, 60),   # arrow head coordinates
            arrowprops=dict(arrowstyle="->"))

plt.show()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(np.arange(0, 3), df["Embarked"].value_counts())
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111)

x = np.arange(0, 3)
y = df['Embarked'].value_counts()
bars = ax.bar(x, y, color='lightslategrey')

# remove the frame
ax.set_frame_on(False)

# we need only 3 ticks (one per bar)
ax.set_xticks(np.arange(0, 3))

# we don't want the ticks, only the labels
ax.tick_params(bottom='off')
ax.set_xticklabels(['Southampton', 'Cherbourg', 'Queenstown'],
                   {'fontsize': 12,
                    'verticalalignment': 'center',
                    'horizontalalignment': 'center',
                    })

# remove ticks on the y axis and show values in the bars
ax.tick_params(left='off',
               labelleft='off')

# add the values on each bar
for bar, value in zip(bars, y):
    ax.text(bar.get_x() + bar.get_width() / 2, # x coordinate
            bar.get_height() - 5,              # y coordinate
            value,                             # text
            ha='center',                       # horizontal alignment
            va='top',                          # vertical alignment
            color='w',                         # text color
            fontsize=14)

# use a different color for the first bar
bars[0].set_color("firebrick")

# add a title
ax.set_title('Most of passengers embarked at Southampton',
             {'fontsize': 18,
              'fontweight' : 'bold',
              'verticalalignment': 'baseline',
              'horizontalalignment': 'center'})

plt.show()
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

x = df['Age'].dropna()
age_bins = np.arange(0, 95, 5)
values, bins, bars = ax.hist(x, bins=age_bins)

ax.set_xticks(bins)

ax.set_ylim(values.min(), values.max() + 10)

ax.spines['bottom'].set_bounds(bins.min(), bins.max())
ax.spines['right'].set_bounds(values.min(), values.max())

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

cm = plt.cm.get_cmap('viridis')

for i in range(0, len(bins)):
    if i < len(bins) - 1:
        survival = df[(df.Age >= bins[i]) & (df.Age < bins[i + 1])]['Survived'].mean()
    else:
        survival = df[(df.Age >= bins[i])]['Survived'].mean()
    try:
        bars[i].set_color(cm(survival))
    except:
        pass

# add colorbar
# the survival rate is already normalized so we don't need norm=plt.Normalize(vmin=0, vmax=1)
# I left it as an example
sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=0, vmax=1)) 
sm._A = []
plt.colorbar(sm)
    
# add a title
ax.set_title('Survival rate per passenger age',
             {'fontsize': 18,
              'fontweight' : 'bold',
              'verticalalignment': 'baseline',
              'horizontalalignment': 'center'})

plt.show()
