# Standard library

import random



# Specific imports from standard library

from cycler import cycler



# Basic imports

import numpy as np

import pandas as pd



# Graphs

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.dates as mdates

from matplotlib.lines import Line2D

from matplotlib.patches import Patch

import matplotlib.cm as cm

from matplotlib.colors import Normalize

from mpl_toolkits.axes_grid1 import make_axes_locatable
x = np.random.rand(100)

y = np.random.rand(100)
plt.figure()

plt.plot(x, y, "o", label="Points")

plt.xlabel("Label X")

plt.ylabel("Label Y")

plt.legend()

plt.show()
# Default graph settings



# Seaborn advanced                                                                                                                                                           

sns.set(style='ticks',          # 'ticks', 'darkgrid'                                                                                                                        

        palette='colorblind',   # 'colorblind', 'pastel', 'muted', 'bright'                                                                                                  

        #palette=sns.color_palette('Accent'),   # 'Set1', 'Set2', 'Dark2', 'Accent'                                                                                          

        rc = {                                                                                                                                                               

           'figure.autolayout': False,   # Automaticall set the figure size to fit in canvas                                                                       

           'figure.figsize': (16, 10),   # Figure size - width, height (in inches)    

           'figure.max_open_warning': False,

           'figure.titlesize': 32,      # Whole figure title size (plt.suptitle)

           'legend.frameon': True,      # Frame around the legend                                                                                                              

           'patch.linewidth': 2.0,      # Width of frame around the legend                                                                                                        

           'lines.markersize': 6,       # Size of marker points                                                                                                                      

           'lines.linewidth': 2.0,      # Width of lines                                                                                                                      

           'font.size': 14,             # Size of font on axes values                                                                                                           

           'legend.fontsize': 18,       # Font size in the legend                                                                                                           

           'axes.labelsize': 22,        # Font size of axes names                                                                                                                  

           'axes.titlesize': 26,        # Font size of subplot titles (plt.title)                                                                                                                 

           'axes.grid': True,           # Set grid on/off                                                                                                                             

           'grid.color': '0.9',         # Color of grid lines - 1 = white, 0 = black                                                                                          

           'grid.linestyle': '-',       # Style of grid lines                                                                                                              

           'grid.linewidth': 1.0,       # Width of grid lines                                                                                                                

           'xtick.labelsize': 22,       # Font size of values on X-axis                                                                                                  

           'ytick.labelsize': 22,       # Font size of values on Y-axis                                                                                                       

           'xtick.major.size': 8,       # Size of ticks on X-axis                                                                                                    

           'ytick.major.size': 8,       # Size of ticks on Y-axis                                                                                                 

           'xtick.major.pad': 10.0,     # Distance of axis values from X-axis                                                                                               

           'ytick.major.pad': 10.0,     # Distance of axis values from Y-axis   

           'image.cmap': 'viridis'      # Default colormap

           }                                                                                                                                                                 

       )     
plt.figure()

plt.plot(x, y, "o", label="Points")

plt.xlabel("Label X")

plt.ylabel("Label Y")

plt.legend()

plt.show()
x = np.random.rand(100)

y = np.random.rand(100)

t = np.arange(100)
plt.figure()

plt.plot(x, y, "o", label="Points - type 1")

plt.plot(y, x, "o", label="Points - type 2")

plt.xlabel("Label X")

plt.ylabel("Label Y")

plt.legend()

plt.show()
plt.figure()

plt.plot(t, y, "-o", label="Points - type 1")

plt.xlabel("Label X")

plt.ylabel("Label Y")

plt.legend()

plt.show()
N_points = 100

x = np.random.rand(N_points)

y = np.random.rand(N_points)

z = np.random.rand(N_points)

colors = [random.choice(["b", "r", "g", "y", "m", "c", "k", "w"]) for i in range(N_points)]
plt.figure()

plt.scatter(x, y, s=z*300, c=colors, alpha=0.5)

plt.show()
N_points = 100

x = np.random.rand(N_points)

y = np.random.rand(N_points)

z = np.random.rand(N_points)

colors = [random.choice(["blue", "red", "green", "yellow", "magenta", "cyan", "black", "white"]) for i in range(N_points)]
plt.figure()

plt.scatter(x, y, s=z*300, c=colors, alpha=0.5)

plt.show()
N_points = 100

x = np.random.rand(N_points)

y = np.random.rand(N_points)

z = np.random.rand(N_points)

colours = [(0.7812, 0.0859, 0.5234, opacity) for opacity in z]



plt.figure()

plt.scatter(x, y, s=15**2, c=colours)

plt.show()
def f(x, y):

    return x**2 + y**2



x, y = np.meshgrid(np.linspace(-50, 50, 100), np.linspace(-50, 50, 100))

Z = f(x, y)
plt.figure()

plt.pcolormesh(x, y, Z)

plt.colorbar()

plt.show()
y = np.random.rand(5)

x = range(len(y))

xlabels = ["A", "B", "C", "D", "E"]
plt.figure()

plt.bar(x, y, edgecolor="k")

plt.xticks(x, xlabels)

plt.show()
x = np.random.normal(0, 0.1, 100000)

y = np.random.normal(0, 0.1, 100000)
plt.figure()

plt.plot(x, y, "o", label="Points - type 1", alpha=0.01)

plt.xlabel("Label X")

plt.ylabel("Label Y")

plt.legend()

plt.show()
legend_elements = [Line2D([0], [0],

                          marker='o',              # Marker to show in a legend

                          color='w',               # Set this to white, so that line behind marker is not visible

                          label='Points - type 1', # Label next to marker

                          markerfacecolor='b',     # Color of the marker

                          markersize=15            # Size of the marker

                         )]



plt.figure()

plt.plot(x, y, "o", alpha=0.01)

plt.xlabel("Label X")

plt.ylabel("Label Y")

plt.legend(handles=legend_elements)

plt.show()
x = np.arange(1, 10)

y1 = 100 - x**2

y2 = 120 - 1.2 * x**2

z1 = np.sqrt(y1) - 3

z2 = np.sqrt(y2)



min_y = np.minimum(z1, z2).min()

max_y = np.maximum(z1, z2).max()
legend_elements = [Line2D([0], [0],                          

                          marker="o",                        

                          color="w",                         

                          label="Circles",                   

                          markerfacecolor="w",               

                          markeredgecolor="k",               

                          markersize=15                      

                         ),

                   Line2D([0], [0],                          

                          marker="d",

                          color="w",

                          label="Diamonds",

                          markerfacecolor="w",

                          markeredgecolor="k",

                          markersize=15)]



cmap = cm.viridis                                            

norm = Normalize(vmin=min_y, vmax=max_y)                     



plt.figure()

plt.scatter(x, y1, c=cmap(norm(z1)), marker="o", s=15**2)

plt.scatter(x, y2, c=cmap(norm(z2)), marker="d", s=15**2)

plt.xlabel("Label X")

plt.ylabel("Label Y")

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

plt.colorbar(sm)

plt.legend(handles=legend_elements)

plt.show()
N_points = 100



x = np.random.rand(N_points)

y = np.random.rand(N_points)

z = np.random.rand(N_points)
# fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)

# ax1 = axes[0][0]

# ax2 = axes[0][1]

# ax3 = axes[1][0]

# ax4 = axes[1][1]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)   # Identical to preceeding five lines



ax1.plot(x, y, "o", label="Points A")

ax1.set_ylabel("Label Y")

ax1.set_title("Title A")

ax1.legend(loc="upper right")



ax2.plot(y, z, "s", label="Points B")

ax2.set_title("Title B")

ax2.legend(loc="upper right")



ax3.plot(x, z, "d", label="Points C")

ax3.set_ylabel("Label Y")

ax3.set_xlabel("Label X")

ax3.set_title("Title C")

ax3.legend(loc="upper right")



ax4.plot(z, y, "v", label="Points D")

ax4.set_xlabel("Label X")

ax4.set_title("Title D")

ax4.legend(loc="upper right")



plt.suptitle("Four graphs")

plt.subplots_adjust(wspace=0.10, hspace=0.30)

plt.show()
def f1(x, y):

    return np.exp(-x)+np.exp(-y)



def f2(x, y):

    return 1/(1+x**2) + 1/(1+3*y**2)



x = np.linspace(0, 10, 100)

y = np.linspace(0, 10, 100)



X, Y = np.meshgrid(x, y)

Z1 = f1(X, Y)

Z2 = f2(X, Y)

Z_diff = Z2 - Z1

Z_diff_max_abs = np.maximum(Z_diff.max(), abs(Z_diff.min()))





fig = plt.figure(figsize=(15,27))

gs = fig.add_gridspec(ncols=2, nrows=6,

                      height_ratios=[1, 1, 0.2, 0.2, 0.2, 0.2])

ax1 = fig.add_subplot(gs[0, 0])

ax2 = fig.add_subplot(gs[0, 1])

ax3 = fig.add_subplot(gs[1, 0])

ax4 = fig.add_subplot(gs[1, 1])

ax5 = fig.add_subplot(gs[2, :])

ax6 = fig.add_subplot(gs[3, :])

ax7 = fig.add_subplot(gs[4, :])

ax8 = fig.add_subplot(gs[5, :])



im = ax1.pcolormesh(x, y, Z1)

divider = make_axes_locatable(ax1)

cax = divider.append_axes("right", size="5%", pad=0.05)

cbar = plt.colorbar(im, cax=cax)

ax1.set_title("Function 1")



im = ax2.pcolormesh(x, y, Z2)

divider = make_axes_locatable(ax2)

cax = divider.append_axes("right", size="5%", pad=0.05)

cbar = plt.colorbar(im, cax=cax)

ax2.set_title("Function 2")



im = ax3.pcolormesh(x, y, Z_diff, cmap="seismic")

divider = make_axes_locatable(ax3)

cax = divider.append_axes("right", size="5%", pad=0.05)

cbar = plt.colorbar(im, cax=cax)

im.set_clim(-Z_diff_max_abs, Z_diff_max_abs)

ax3.set_title("Difference 2-1")



im = ax4.pcolormesh(x, y, Z_diff, cmap="seismic")

divider = make_axes_locatable(ax4)

cax = divider.append_axes("right", size="5%", pad=0.05)

cbar = plt.colorbar(im, cax=cax)

im.set_clim(-Z_diff_max_abs, Z_diff_max_abs)

ax4.set_xlim((0,2.5))

ax4.set_ylim((0,2.5))

ax4.set_title("Zoomed difference 2-1")



im = ax5.pcolormesh(x, y, Z1)

divider = make_axes_locatable(ax5)

cax = divider.append_axes("right", size="5%", pad=0.05)

cbar = plt.colorbar(im, cax=cax)

ax5.set_xlim((0,10))

ax5.set_ylim((0,2.5))

ax5.set_title("Zoom to F1 along X-axis")



im = ax6.pcolormesh(x, y, Z2)

divider = make_axes_locatable(ax6)

cax = divider.append_axes("right", size="5%", pad=0.05)

cbar = plt.colorbar(im, cax=cax)

ax6.set_xlim((0,10))

ax6.set_ylim((0,2.5))

ax6.set_title("Zoom to F2 along X-axis")



im = ax7.pcolormesh(x, y, Z1)

divider = make_axes_locatable(ax7)

cax = divider.append_axes("right", size="5%", pad=0.05)

cbar = plt.colorbar(im, cax=cax)

ax7.set_xlim((0,2.5))

ax7.set_ylim((0,10))

ax7.set_title("Zoom to F1 along Y-axis")



im = ax8.pcolormesh(x, y, Z2)

divider = make_axes_locatable(ax8)

cax = divider.append_axes("right", size="5%", pad=0.05)

cbar = plt.colorbar(im, cax=cax)

ax8.set_xlim((0,2.5))

ax8.set_ylim((0,10))

ax8.set_title("Zoom to F2 along Y-axis")



plt.subplots_adjust(wspace=0.4, hspace=0.5)

plt.show()
x = np.arange(0, 100)
plt.figure()

for i in range(1, 16):

    plt.plot(x, x * i * 0.1, "-", label=f"{i}-th color")

plt.legend()

plt.show()
list_of_colors = ["#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#882255"]   # Paul Tol colorpalette



fig, ax = plt.subplots()

ax.set_prop_cycle(cycler('color', list_of_colors))

for i in range(1, 16):

    ax.plot(x, x * i * 0.1, "-", label=f"{i}-th color")

plt.legend()

plt.show()
fig, ax = plt.subplots()

ax.set_prop_cycle(color=plt.rcParams['axes.prop_cycle'].by_key()['color'][:5])



for i in range(1, 6):

    ax.plot(x, x * i * 0.1, "-", label=f"{i}-A")

for i in range(1, 6):

    ax.plot(x, x * (i+5) * 0.1, "--", label=f"{i}-B")

plt.legend()

plt.show()
plt.figure()

for i in range(1, 16):

    plt.plot(x, x * i * 0.1, "-", label=f"{i}-th color")

plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)

plt.show()
plt.figure()

for i in range(1, 16):

    plt.plot(x, x * i * 0.1, "-", label=f"{i}-th color")

plt.legend(bbox_to_anchor=(1.04,0.6), borderaxespad=0, ncol=3)

plt.show()
plt.figure()

for i in range(1, 16):

    plt.plot(x, x * i * 0.1, "-", label=f"{i}-th color")

plt.legend(bbox_to_anchor=(0.95,-0.1), borderaxespad=0, ncol=5)

plt.show()