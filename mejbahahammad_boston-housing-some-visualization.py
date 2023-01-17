import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
housing_data = pd.read_csv('../input/boston-housing/housing.data', delim_whitespace=True, header=None)
housing_data.head()
columns_names = ['CRIM', 'ZN' , 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
housing_data.columns = columns_names
housing_data.info()
columns_names_new = ['CRIM', 'ZN' , 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'PTRATIO', 'LSTAT', 'MEDV']
means_data = []

for i in columns_names_new:

    means_data.append(np.ceil(np.mean(housing_data[str(i)])))
print(means_data)
std_data = []

for i in columns_names_new:

    std_data.append(np.ceil(np.std(housing_data[str(i)])))
print(std_data)
import matplotlib

import matplotlib.pyplot as plt

import numpy as np





labels = columns_names_new

means_data;

std_data;



x = np.arange(len(labels))  # the label locations

width = 0.35  # the width of the bars



fig, ax = plt.subplots()

rects1 = ax.bar(x - width/2, means_data, width, label='Mean')

rects2 = ax.bar(x + width/2, std_data, width, label='Std')



# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_ylabel('Amount')

ax.set_title('Grouped bar chart with labels')

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.legend()





def autolabel(rects):

    """Attach a text label above each bar in *rects*, displaying its height."""

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(0, 3),  # 3 points vertical offset

                    textcoords="offset points",

                    ha='center', va='bottom')





autolabel(rects1)

autolabel(rects2)



fig.tight_layout()



plt.show()
plt.figure(figsize=(10, 8))

plt.bar(columns_names_new, means_data, width = 0.3,yerr =std_data, bottom=means_data, label = columns_names_new)

plt.title("Stacked Bar Chart")

plt.xlabel("Data Amount")

plt.ylabel("Data Labels")

plt.legend()

plt.show()
fig, axs = plt.subplots(1, 3, figsize=(8, 6), sharey=True)

axs[0].bar(means_data, std_data)

axs[1].scatter(means_data, std_data)

axs[2].plot(means_data, std_data)

fig.suptitle('Categorical Plotting')
plt.figure()

plt.plot(means_data, std_data)

plt.show()
np.random.seed(19680801)





plt.rcdefaults()

fig, ax = plt.subplots()



# Example data

labels = columns_names_new

y_pos = np.arange(len(labels))

performance = 3 + 10 * np.random.rand(len(labels))

error = np.random.rand(len(labels))



ax.barh(y_pos, performance, xerr=error, align='center')

ax.set_yticks(y_pos)

ax.set_yticklabels(labels)

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Performance')

ax.set_title('Housing : Horizontal bar chart')



plt.show()




category_names = columns_names_new

results = {

    'Means': means_data,

    'Std': std_data

}





def survey(results, category_names):

    """

    Parameters

    ----------

    results : dict

        A mapping from question labels to a list of answers per category.

        It is assumed all lists contain the same number of entries and that

        it matches the length of *category_names*.

    category_names : list of str

        The category labels.

    """

    labels = list(results.keys())

    data = np.array(list(results.values()))

    data_cum = data.cumsum(axis=1)

    category_colors = plt.get_cmap('RdYlGn')(

        np.linspace(0.15, 0.85, data.shape[1]))



    fig, ax = plt.subplots(figsize=(9.2, 5))

    ax.invert_yaxis()

    ax.xaxis.set_visible(False)

    ax.set_xlim(0, np.sum(data, axis=1).max())



    for i, (colname, color) in enumerate(zip(category_names, category_colors)):

        widths = data[:, i]

        starts = data_cum[:, i] - widths

        ax.barh(labels, widths, left=starts, height=0.5,

                label=colname, color=color)

        xcenters = starts + widths / 2



        r, g, b, _ = color

        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'

        for y, (x, c) in enumerate(zip(xcenters, widths)):

            ax.text(x, y, str(int(c)), ha='center', va='center',

                    color=text_color)

    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),

              loc='lower left', fontsize='small')



    return fig, ax





survey(results, category_names)

plt.show()
# Fixing random state for reproducibility

np.random.seed(19680801)



# some random data

x = means_data

y = std_data





def scatter_hist(x, y, ax, ax_histx, ax_histy):

    # no labels

    ax_histx.tick_params(axis="x", labelbottom=False)

    ax_histy.tick_params(axis="y", labelleft=False)



    # the scatter plot:

    ax.scatter(x, y)



    # now determine nice limits by hand:

    binwidth = 0.25

    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))

    lim = (int(xymax/binwidth) + 1) * binwidth



    bins = np.arange(-lim, lim + binwidth, binwidth)

    ax_histx.hist(x, bins=bins)

    ax_histy.hist(y, bins=bins, orientation='horizontal')

    

    

    

# definitions for the axes

left, width = 0.1, 0.65

bottom, height = 0.1, 0.65

spacing = 0.005





rect_scatter = [left, bottom, width, height]

rect_histx = [left, bottom + height + spacing, width, 0.2]

rect_histy = [left + width + spacing, bottom, 0.2, height]



# start with a square Figure

fig = plt.figure(figsize=(8, 8))



ax = fig.add_axes(rect_scatter)

ax_histx = fig.add_axes(rect_histx, sharex=ax)

ax_histy = fig.add_axes(rect_histy, sharey=ax)



# use the previously defined function

scatter_hist(x, y, ax, ax_histx, ax_histy)



plt.show()
fig, ax = plt.subplots()

for color in ['tab:blue', 'tab:orange', 'tab:green']:

    n = len(columns_names_new)

    x = means_data

    y = std_data

    scale = 200.0 * np.random.rand(n)

    ax.scatter(x, y, c=color, s=scale, label=color,

               alpha=0.3, edgecolors='none')



ax.legend()

ax.grid(True)



plt.show()
x = means_data

y = std_data

plt.stem(x, y)

plt.show()
import matplotlib.pyplot as plt

import numpy as np



# Random test data

np.random.seed(19680801)

all_data = housing_data['AGE']

labels = "AGE"



fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))



# rectangular box plot

bplot1 = ax1.boxplot(all_data,

                     vert=True,  # vertical box alignment

                     patch_artist=True  # fill with color

                     )  # will be used to label x-ticks

ax1.set_title('Rectangular box plot')



# notch shape box plot

bplot2 = ax2.boxplot(all_data,

                     notch=True,  # notch shape

                     vert=True,  # vertical box alignment

                     patch_artist=True)

ax2.set_title('Notched box plot')



# fill with colors

colors = ['pink', 'lightblue', 'lightgreen']

for bplot in (bplot1, bplot2):

    for patch, color in zip(bplot['boxes'], colors):

        patch.set_facecolor(color)



# adding horizontal grid lines

for ax in [ax1, ax2]:

    ax.yaxis.grid(True)

    ax.set_xlabel('Three separate samples')

    ax.set_ylabel('Observed values')



plt.show()
mu = 100  # mean of distribution

sigma = 15  # standard deviation of distribution

value = mu + sigma * np.random.randn(437)

x = means_data

y =std_data

num_bins = 50



fig, ax = plt.subplots()



# the histogram of the data

n, bins, patches = ax.hist(x, num_bins, density=True)



# add a 'best fit' line

y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *

     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))

ax.plot(bins, y, '--')

ax.set_xlabel('Smarts')

ax.set_ylabel('Probability density')

ax.set_title(r'Histogram')



# Tweak spacing to prevent clipping of ylabel

fig.tight_layout()

plt.show()
labels = columns_names_new

sizes = means_data



fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
import numpy as np

import matplotlib.pyplot as plt

from matplotlib import cm

from mpl_toolkits.mplot3d import Axes3D



X = means_data

Y = std_data

X, Y = np.meshgrid(X, Y)

R = np.sqrt(X**2 + Y**2)

Z = np.sin(R)



fig = plt.figure()

ax = Axes3D(fig)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.viridis)



plt.show()