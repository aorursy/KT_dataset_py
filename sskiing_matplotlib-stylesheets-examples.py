from numpy.random import beta

import matplotlib.pyplot as plt



## reset parameters

plt.rcParams.update(plt.rcParamsDefault)



plt.style.use('bmh')



def plot_beta_hist(ax, a, b):

    ax.hist(beta(a, b, size=10000), histtype="stepfilled",

            bins=25, alpha=0.8, normed=True)



fig, ax = plt.subplots(dpi=300)

plot_beta_hist(ax, 10, 10)

plot_beta_hist(ax, 4, 12)

plot_beta_hist(ax, 50, 12)

plot_beta_hist(ax, 6, 55)

ax.set_title("'bmh' style sheet")



plt.show()
import numpy as np

import matplotlib.pyplot as plt



## reset parameters

plt.rcParams.update(plt.rcParamsDefault)



plt.style.use('dark_background')



fig, ax = plt.subplots(dpi=300)



L = 6

x = np.linspace(0, L)

ncolors = len(plt.rcParams['axes.prop_cycle'])

shift = np.linspace(0, L, ncolors, endpoint=False)

for s in shift:

    ax.plot(x, np.sin(x + s), 'o-')

ax.set_xlabel('x-axis')

ax.set_ylabel('y-axis')

ax.set_title("'dark_background' style sheet")



plt.show()
from matplotlib import pyplot as plt

import numpy as np



## reset parameters

plt.rcParams.update(plt.rcParamsDefault)



plt.style.use('fivethirtyeight')



x = np.linspace(0, 10)



# Fixing random state for reproducibility

np.random.seed(19680801)



fig, ax = plt.subplots(dpi=300)



ax.plot(x, np.sin(x) + x + np.random.randn(50))

ax.plot(x, np.sin(x) + 0.5 * x + np.random.randn(50))

ax.plot(x, np.sin(x) + 2 * x + np.random.randn(50))

ax.plot(x, np.sin(x) - 0.5 * x + np.random.randn(50))

ax.plot(x, np.sin(x) - 2 * x + np.random.randn(50))

ax.plot(x, np.sin(x) + np.random.randn(50))

ax.set_title("'fivethirtyeight' style sheet")



plt.show()
import numpy as np

import matplotlib.pyplot as plt



## reset parameters

plt.rcParams.update(plt.rcParamsDefault)



plt.style.use('ggplot')



fig, axes = plt.subplots(ncols=2, nrows=2,dpi=300)

ax1, ax2, ax3, ax4 = axes.ravel()



# scatter plot (Note: `plt.scatter` doesn't use default colors)

x, y = np.random.normal(size=(2, 200))

ax1.plot(x, y, 'o')



# sinusoidal lines with colors from default color cycle

L = 2*np.pi

x = np.linspace(0, L)

ncolors = len(plt.rcParams['axes.prop_cycle'])

shift = np.linspace(0, L, ncolors, endpoint=False)

for s in shift:

    ax2.plot(x, np.sin(x + s), '-')

ax2.margins(0)



# bar graphs

x = np.arange(5)

y1, y2 = np.random.randint(1, 25, size=(2, 5))

width = 0.25

ax3.bar(x, y1, width)

ax3.bar(x + width, y2, width,

        color=list(plt.rcParams['axes.prop_cycle'])[2]['color'])

ax3.set_xticks(x + width)

ax3.set_xticklabels(['a', 'b', 'c', 'd', 'e'])



# circles with colors from default color cycle

for i, color in enumerate(plt.rcParams['axes.prop_cycle']):

    xy = np.random.normal(size=2)

    ax4.add_patch(plt.Circle(xy, radius=0.3, color=color['color']))

ax4.axis('equal')

ax4.margins(0)



plt.show()
import numpy as np

import matplotlib.pyplot as plt



## reset parameters

plt.rcParams.update(plt.rcParamsDefault)



def color_cycle_example(ax):

    L = 6

    x = np.linspace(0, L)

    ncolors = len(plt.rcParams['axes.prop_cycle'])

    shift = np.linspace(0, L, ncolors, endpoint=False)

    for s in shift:

        ax.plot(x, np.sin(x + s), 'o-')





def image_and_patch_example(ax):

    ax.imshow(np.random.random(size=(20, 20)), interpolation='none')

    c = plt.Circle((5, 5), radius=5, label='patch')

    ax.add_patch(c)



plt.style.use('grayscale')



fig, (ax1, ax2) = plt.subplots(ncols=2,dpi=300)

fig.suptitle("'grayscale' style sheet")



color_cycle_example(ax1)

image_and_patch_example(ax2)



plt.show()
import numpy as np

import matplotlib.pyplot as plt



## reset parameters

plt.rcParams.update(plt.rcParamsDefault)



def plot_scatter(ax, prng, nb_samples=100):

    """Scatter plot.

    """

    for mu, sigma, marker in [(-.5, 0.75, 'o'), (0.75, 1., 's')]:

        x, y = prng.normal(loc=mu, scale=sigma, size=(2, nb_samples))

        ax.plot(x, y, ls='none', marker=marker)

    ax.set_xlabel('X-label')

    return ax





def plot_colored_sinusoidal_lines(ax):

    """Plot sinusoidal lines with colors following the style color cycle.

    """

    L = 2 * np.pi

    x = np.linspace(0, L)

    nb_colors = len(plt.rcParams['axes.prop_cycle'])

    shift = np.linspace(0, L, nb_colors, endpoint=False)

    for s in shift:

        ax.plot(x, np.sin(x + s), '-')

    ax.set_xlim([x[0], x[-1]])

    return ax





def plot_bar_graphs(ax, prng, min_value=5, max_value=25, nb_samples=5):

    """Plot two bar graphs side by side, with letters as x-tick labels.

    """

    x = np.arange(nb_samples)

    ya, yb = prng.randint(min_value, max_value, size=(2, nb_samples))

    width = 0.25

    ax.bar(x, ya, width)

    ax.bar(x + width, yb, width, color='C2')

    ax.set_xticks(x + width)

    ax.set_xticklabels(['a', 'b', 'c', 'd', 'e'])

    return ax





def plot_colored_circles(ax, prng, nb_samples=15):

    """Plot circle patches.



    NB: draws a fixed amount of samples, rather than using the length of

    the color cycle, because different styles may have different numbers

    of colors.

    """

    for sty_dict, j in zip(plt.rcParams['axes.prop_cycle'], range(nb_samples)):

        ax.add_patch(plt.Circle(prng.normal(scale=3, size=2),

                                radius=1.0, color=sty_dict['color']))

    # Force the limits to be the same across the styles (because different

    # styles may have different numbers of available colors).

    ax.set_xlim([-4, 8])

    ax.set_ylim([-5, 6])

    ax.set_aspect('equal', adjustable='box')  # to plot circles as circles

    return ax





def plot_image_and_patch(ax, prng, size=(20, 20)):

    """Plot an image with random values and superimpose a circular patch.

    """

    values = prng.random_sample(size=size)

    ax.imshow(values, interpolation='none')

    c = plt.Circle((5, 5), radius=5, label='patch')

    ax.add_patch(c)

    # Remove ticks

    ax.set_xticks([])

    ax.set_yticks([])





def plot_histograms(ax, prng, nb_samples=10000):

    """Plot 4 histograms and a text annotation.

    """

    params = ((10, 10), (4, 12), (50, 12), (6, 55))

    for a, b in params:

        values = prng.beta(a, b, size=nb_samples)

        ax.hist(values, histtype="stepfilled", bins=30, alpha=0.8, normed=True)

    # Add a small annotation.

    ax.annotate('Annotation', xy=(0.25, 4.25), xycoords='data',

                xytext=(0.9, 0.9), textcoords='axes fraction',

                va="top", ha="right",

                bbox=dict(boxstyle="round", alpha=0.2),

                arrowprops=dict(

                          arrowstyle="->",

                          connectionstyle="angle,angleA=-95,angleB=35,rad=10"),

                )

    return ax





def plot_figure(style_label=""):

    """Setup and plot the demonstration figure with a given style.

    """

    # Use a dedicated RandomState instance to draw the same "random" values

    # across the different figures.

    prng = np.random.RandomState(96917002)



    # Tweak the figure size to be better suited for a row of numerous plots:

    # double the width and halve the height. NB: use relative changes because

    # some styles may have a figure size different from the default one.

    (fig_width, fig_height) = plt.rcParams['figure.figsize']

    fig_size = [fig_width * 2, fig_height / 2]



    fig, axes = plt.subplots(ncols=6, nrows=1, num=style_label,

                             figsize=fig_size, squeeze=True,dpi=200)

    axes[0].set_ylabel(style_label)



    plot_scatter(axes[0], prng)

    plot_image_and_patch(axes[1], prng)

    plot_bar_graphs(axes[2], prng)

    plot_colored_circles(axes[3], prng)

    plot_colored_sinusoidal_lines(axes[4])

    plot_histograms(axes[5], prng)



    fig.tight_layout()



    return fig



# Setup a list of all available styles, in alphabetical order but

# the `default` and `classic` ones, which will be forced resp. in

# first and second position.

style_list = list(plt.style.available)  # *new* list: avoids side effects.

style_list.remove('classic')  # `classic` is in the list: first remove it.

style_list.sort()

style_list.insert(0, u'default')

style_list.insert(1, u'classic')



# Plot a demonstration figure for every available style sheet.

for style_label in style_list:

    with plt.style.context(style_label):

        fig = plot_figure(style_label=style_label)



plt.show()