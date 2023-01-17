import matplotlib.pyplot as plt

plt.rcdefaults()

import numpy as np

import matplotlib.pyplot as plt





plt.rcdefaults()

fig, ax = plt.subplots(dpi=244)



# Example data

people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')

y_pos = np.arange(len(people))

performance = 3 + 10 * np.random.rand(len(people))

error = np.random.rand(len(people))



ax.barh(y_pos, performance, xerr=error, align='center',

        color='green', ecolor='black')

ax.set_yticks(y_pos)

ax.set_yticklabels(people)

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Performance')

ax.set_title('How fast do you want to go today?')



plt.show()
import numpy as np

import matplotlib.pyplot as plt



x = np.linspace(0, 1, 500)

y = np.sin(4 * np.pi * x) * np.exp(-5 * x)



fig, ax = plt.subplots(dpi=244)



ax.fill(x, y, zorder=10)

ax.grid(True, zorder=5)

plt.show()
import numpy as np

import matplotlib.pyplot as plt



x = np.linspace(0, 2 * np.pi, 500)

y1 = np.sin(x)

y2 = np.sin(3 * x)



fig, ax = plt.subplots(dpi=244)

ax.fill(x, y1, 'b', x, y2, 'r', alpha=0.3)

plt.show()
import numpy as np

import matplotlib.pyplot as plt





x = np.linspace(0, 10, 500)

dashes = [10, 5, 100, 5]  # 10 points on, 5 off, 100 on, 5 off



fig, ax = plt.subplots(dpi=244)

line1, = ax.plot(x, np.sin(x), '--', linewidth=2,

                 label='Dashes set retroactively')

line1.set_dashes(dashes)



line2, = ax.plot(x, -1 * np.sin(x), dashes=[30, 5, 10, 5],

                 label='Dashes set proactively')



ax.legend(loc='lower right')

plt.show()
import numpy as np

import matplotlib.pyplot as plt



color = 'cornflowerblue'

points = np.ones(5)  # Draw 5 points for each line

text_style = dict(horizontalalignment='right', verticalalignment='center',

                  fontsize=12, fontdict={'family': 'monospace'})



def format_axes(ax):

    ax.margins(0.2)

    ax.set_axis_off()



def nice_repr(text):

    return repr(text).lstrip('u')



# Plot all line styles.

fig, ax = plt.subplots(dpi=300)



linestyles = ['-', '--', '-.', ':']

for y, linestyle in enumerate(linestyles):

    ax.text(-0.1, y, nice_repr(linestyle), **text_style)

    ax.plot(y * points, linestyle=linestyle, color=color, linewidth=3)

    format_axes(ax)

    ax.set_title('Line Styles')



plt.show()
import numpy as np

import matplotlib.pyplot as plt

from collections import OrderedDict

from matplotlib.transforms import blended_transform_factory



linestyles = OrderedDict(

    [('solid',               (0, ())),

     ('loosely dotted',      (0, (1, 10))),

     ('dotted',              (0, (1, 5))),

     ('densely dotted',      (0, (1, 1))),



     ('loosely dashed',      (0, (5, 10))),

     ('dashed',              (0, (5, 5))),

     ('densely dashed',      (0, (5, 1))),



     ('loosely dashdotted',  (0, (3, 10, 1, 10))),

     ('dashdotted',          (0, (3, 5, 1, 5))),

     ('densely dashdotted',  (0, (3, 1, 1, 1))),



     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),

     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])





plt.figure(figsize=(10, 6),dpi=300)

ax = plt.subplot(1, 1, 1)



X, Y = np.linspace(0, 100, 10), np.zeros(10)

for i, (name, linestyle) in enumerate(linestyles.items()):

    ax.plot(X, Y+i, linestyle=linestyle, linewidth=1.5, color='black')



ax.set_ylim(-0.5, len(linestyles)-0.5)

plt.yticks(np.arange(len(linestyles)), linestyles.keys())

plt.xticks([])



# For each line style, add a text annotation with a small offset from

# the reference point (0 in Axes coords, y tick value in Data coords).

reference_transform = blended_transform_factory(ax.transAxes, ax.transData)

for i, (name, linestyle) in enumerate(linestyles.items()):

    ax.annotate(str(linestyle), xy=(0.0, i), xycoords=reference_transform,

                xytext=(-6, -12), textcoords='offset points', color="blue",

                fontsize=8, ha="right", family="monospace")



plt.tight_layout()

plt.show()
import numpy as np

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D





points = np.ones(5)  # Draw 3 points for each line

text_style = dict(horizontalalignment='right', verticalalignment='center',

                  fontsize=12, fontdict={'family': 'monospace'})

marker_style = dict(color='cornflowerblue', linestyle=':', marker='o',

                    markersize=15, markerfacecoloralt='gray')





def format_axes(ax):

    ax.margins(0.2)

    ax.set_axis_off()





def nice_repr(text):

    return repr(text).lstrip('u')





fig, ax = plt.subplots(dpi=300)



# Plot all fill styles.

for y, fill_style in enumerate(Line2D.fillStyles):

    ax.text(-0.5, y, nice_repr(fill_style), **text_style)

    ax.plot(y * points, fillstyle=fill_style, **marker_style)

    format_axes(ax)

    ax.set_title('fill style')



plt.show()
from six import iteritems

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D





points = np.ones(3)  # Draw 3 points for each line

text_style = dict(horizontalalignment='right', verticalalignment='center',

                  fontsize=12, fontdict={'family': 'monospace'})

marker_style = dict(linestyle=':', color='cornflowerblue', markersize=10)





def format_axes(ax):

    ax.margins(0.2)

    ax.set_axis_off()





def nice_repr(text):

    return repr(text).lstrip('u')





def split_list(a_list):

    i_half = len(a_list) // 2

    return (a_list[:i_half], a_list[i_half:])





# Plot all un-filled markers

# --------------------------



fig, axes = plt.subplots(ncols=2,dpi=300)



# Filter out filled markers and marker settings that do nothing.

# We use iteritems from six to make sure that we get an iterator

# in both python 2 and 3

unfilled_markers = [m for m, func in iteritems(Line2D.markers)

                    if func != 'nothing' and m not in Line2D.filled_markers]

# Reverse-sort for pretty. We use our own sort key which is essentially

# a python3 compatible reimplementation of python2 sort.

unfilled_markers = sorted(unfilled_markers,

                          key=lambda x: (str(type(x)), str(x)))[::-1]

for ax, markers in zip(axes, split_list(unfilled_markers)):

    for y, marker in enumerate(markers):

        ax.text(-0.5, y, nice_repr(marker), **text_style)

        ax.plot(y * points, marker=marker, **marker_style)

        format_axes(ax)

fig.suptitle('un-filled markers', fontsize=14)





# Plot all filled markers.

# ------------------------



fig, axes = plt.subplots(ncols=2,dpi=300)

for ax, markers in zip(axes, split_list(Line2D.filled_markers)):

    for y, marker in enumerate(markers):

        ax.text(-0.5, y, nice_repr(marker), **text_style)

        ax.plot(y * points, marker=marker, **marker_style)

        format_axes(ax)

fig.suptitle('filled markers', fontsize=14)



plt.show()
import matplotlib.pyplot as plt

from numpy.random import rand



fig, ax = plt.subplots(dpi=300)

for color in ['red', 'green', 'blue']:

    n = 750

    x, y = rand(2, n)

    scale = 200.0 * rand(n)

    ax.scatter(x, y, c=color, s=scale, label=color,

               alpha=0.3, edgecolors='none')



ax.legend()

ax.grid(True)



plt.show()