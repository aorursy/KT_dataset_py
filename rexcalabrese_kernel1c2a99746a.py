# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install mplcursors
!pip install git+https://github.com/Davide-sd/pygasflow@master
import numpy as np

import matplotlib.pyplot as plt

import matplotlib.colors as colors

import matplotlib.cm as cmx

import mplcursors
%matplotlib notebook
import numpy as np



%matplotlib notebook



import matplotlib.pyplot as plt

import matplotlib.colors as colors

import matplotlib.cm as cmx

import mplcursors  #



from pygasflow.shockwave import (

    theta_from_mach_beta,

    beta_from_mach_max_theta,

    beta_theta_max_for_unit_mach_downstream

)
# upstream mach numbers

M = [1.1, 1.5, 2, 3, 5, 10, 1e9]

M = [1.1,1.2,1.3,1.4, 1.5,1.6,1.7,1.8,1.9, 2, 3, 5, 10, 1e9]

# specific heats ratio

gamma = 1.4

# number of points for each Mach curve

N = 100





# colors

jet = plt.get_cmap('hsv')

cNorm  = colors.Normalize(vmin=0, vmax=len(M))

scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

c = [scalarMap.to_rgba(i) for i in range(len(M))]



# labels

lbls = [r"$M_{1}$ = " + str(M[i]) for  i in range(len(M))]

lbls[-1] = r"$M_1$ = $\infty$"







%matplotlib notebook



############################### PART 1 ###############################



# plot the Mach curves

for i, m in enumerate(M):

    beta_min = np.rad2deg(np.arcsin(1 / m))

    betas = np.linspace(beta_min, 90, N)

    thetas = theta_from_mach_beta(m, betas, gamma)

    plt.plot(thetas, betas, color=c[i], linewidth=1, label=lbls[i])



############################### PART 2 ###############################



# compute the line M2 = 1

M1 = np.logspace(0, 3, 5 * N)

beta_M2_equal_1, theta_max = beta_theta_max_for_unit_mach_downstream(M1, gamma)



plt.plot(theta_max, beta_M2_equal_1, ':', color="0.3", linewidth=1)



# select an index where to put the annotation (chosen by trial and error)

i = 20

plt.annotate("$M_{2} > 1$", 

    (theta_max[i], beta_M2_equal_1[i]),

    (theta_max[i], beta_M2_equal_1[i] + 10),

    horizontalalignment='center',

    arrowprops=dict(arrowstyle = "<-", color="0.3"),

    color="0.3",

)

plt.annotate("$M_{2} < 1$", 

    (theta_max[i], beta_M2_equal_1[i]),

    (theta_max[i], beta_M2_equal_1[i] - 10),

    horizontalalignment='center',

    arrowprops=dict(arrowstyle = "<-", color="0.3"),

    color="0.3",

)



############################### PART 3 ###############################



# compute the line passing through (M,theta_max)

beta = beta_from_mach_max_theta(M1, gamma)



plt.plot(theta_max, beta, '--', color="0.2", linewidth=1)



# select an index where to put the annotation (chosen by trial and error)

i = 50

plt.annotate("strong", 

    (theta_max[i], beta[i]),

    (theta_max[i], beta[i] + 10),

    horizontalalignment='center',

    arrowprops=dict(arrowstyle = "<-")

)

plt.annotate("weak", 

    (theta_max[i], beta[i]),

    (theta_max[i], beta[i] - 10),

    horizontalalignment='center',

    arrowprops=dict(arrowstyle = "<-"),

)



plt.title(r"Mach - $\beta$ - $\theta$")

plt.xlabel(r"Flow Deflection Angle, $\theta$ [deg]")

plt.ylabel(r"Shock Wave Angle, $\beta$ [deg]")

plt.xlim((0, 50))

plt.ylim((0, 90))

plt.minorticks_on()

plt.grid(which='major', linestyle='-', alpha=0.7)

plt.grid(which='minor', linestyle=':', alpha=0.5)

plt.legend(loc="lower right")

plt.show()

mplcursors.cursor(hover=True)    
fn1()
from pygasflow.nozzles import CD_TOP_Nozzle, CD_Conical_Nozzle, CD_Min_Length_Nozzle

from pygasflow.utils import Ideal_Gas, Flow_State

from pygasflow.solvers import De_Laval_Solver

import numpy as np



import ipywidgets as widgets

from numba import jit,njit

import matplotlib.pyplot as plt

from ipywidgets import interact

from IPython.display import HTML, display

from IPython.display import display, Math,Markdown,Latex

import tabulate

import numpy as np



# Initialize air as the gas to use in the nozzle

gas = Ideal_Gas(287, 1.4)



# stagnation condition

upstream_state = Flow_State(

    p0 = 8 * 101325,

    t0 = 303.15

)
Ri = 0.4

Rt = 0.2

Re = 1.2



# half cone angle of the divergent

theta_c = 40

# half cone angle of the convergent

theta_N = 15



# Junction radius between the convergent and divergent

Rbt = 0.75 * Rt

# Junction radius between the "combustion chamber" and convergent

Rbc = 1.5 * Rt

# Fractional Length of the TOP nozzle with respect to a same exit

# area ratio conical nozzle with 15 deg half-cone angle.

K = 0.8

# geometry type

geom = "axisymmetric"



geom_con = CD_Conical_Nozzle(

    Ri,            # Inlet radius

    Re,            # Exit (outlet) radius

    Rt,            # Throat radius

    Rbt,           # Junction radius ratio at the throat (between the convergent and divergent)

    Rbc,           # Junction radius ratio between the "combustion chamber" and convergent

    theta_c,       # Half angle [degrees] of the convergent.

    theta_N,       # Half angle [degrees] of the conical divergent.

    geom,          # Geometry type

    1000           # Number of discretization points along the total length of the nozzle

)



geom_top = CD_TOP_Nozzle(

    Ri,            # Inlet radius

    Re,            # Exit (outlet) radius

    Rt,            # Throat radius

    Rbc,           # Junction radius ratio between the "combustion chamber" and convergent

    theta_c,       # Half angle [degrees] of the convergent.

    K,             # Fractional Length of the nozzle

    geom,          # Geometry type

    1000           # Number of discretization points along the total length of the nozzle

)



n = 15

gamma = gas.gamma



geom_moc = CD_Min_Length_Nozzle(

    Ri,            # Inlet radius

    Re,            # Exit (outlet) radius

    Rt,            # Throat radius

    Rbt,           # Junction radius ratio at the throat (between the convergent and divergent)

    Rbc,           # Junction radius ratio between the "combustion chamber" and convergent

    theta_c,       # Half angle [degrees] of the convergent.

    n,             # number of characteristics lines

    gamma          # Specific heat ratio

)
# Initialize the nozzle

nozzle_conical = De_Laval_Solver(gas, geom_con, upstream_state)

nozzle_top = De_Laval_Solver(gas, geom_top, upstream_state)

nozzle_moc = De_Laval_Solver(gas, geom_moc, upstream_state)

print(nozzle_conical)
%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.patches as patches



def Plot_Nozzle(geom, L, A, M, P, rho, T, flow_condition, Asw_At_ratio, title):

    fig, ax = plt.subplots(nrows=4, sharex=True)

    fig.set_size_inches(8, 12)

    radius_nozzle, radius_container = geom.get_points(False)

    ar_nozzle, ar_container = geom.get_points(True)



    # nozzle geometry

    ax[0].add_patch(patches.Polygon(radius_container, facecolor="0.85", hatch="///", edgecolor="0.4", linewidth=0.5))

    ax[0].add_patch(patches.Polygon(radius_nozzle, facecolor='#b7e1ff', edgecolor="0.4", linewidth=1))

    ax[0].set_ylim(0, max(radius_container[:, 1]))

    ax[0].set_ylabel("r [m]")

    ax[0].set_title(title + flow_condition)



    ax[1].add_patch(patches.Polygon(ar_container, facecolor="0.85", hatch="///", edgecolor="0.4", linewidth=0.5))

    ax[1].add_patch(patches.Polygon(ar_nozzle, facecolor='#b7e1ff', edgecolor="0.4", linewidth=1))

    ax[1].set_ylim(0, max(ar_container[:, 1]))

    ax[1].set_ylabel("$A/A^{*}$")

    

    # draw the shock wave if present in the nozzle

    if Asw_At_ratio:

        # get shock wave location in the divergent

        x = geom.location_divergent_from_area_ratio(Asw_At_ratio)

        rsw = np.sqrt((Asw_At_ratio * geom.critical_area) / np.pi)

        ax[0].plot([x, x], [0, rsw], 'r')

        ax[1].plot([x, x], [0, Asw_At_ratio], 'r')

        ax[0].text(x, rsw + 0.5 * (max(radius_container[:, 1]) - max(radius_nozzle[:, -1])),

            "SW", 

            color="r",

            ha='center',

            va="center",

            bbox=dict(boxstyle="round", fc="white", lw=0, alpha=0.85),

        )

        ax[1].text(x, Asw_At_ratio + 0.5 * (max(ar_container[:, 1]) - max(ar_nozzle[:, -1])),

            "SW", 

            color="r",

            ha='center',

            va="center",

            bbox=dict(boxstyle="round", fc="white", lw=0, alpha=0.85),

        )



    # mach number

    ax[2].plot(L, M)

    ax[2].set_ylim(0)

    ax[2].grid()

    ax[2].set_ylabel("M")



    # ratios

    ax[3].plot(L, P, label="$P/P_{0}$")

    ax[3].plot(L, rho, label=r"$\rho/\rho_{0}$")

    ax[3].plot(L, T, label="$T/T_{0}$")

    ax[3].set_xlim(min(ar_container[:, 0]), max(ar_container[:, 0]))

    ax[3].set_ylim(0, 1)

    ax[3].legend(loc="lower left")

    ax[3].grid()

    ax[3].set_xlabel("L [m]")

    ax[3].set_ylabel("ratios")



    plt.tight_layout()

    #mplcursors.cursor(hover=True)

    #fig.canvas.draw_idle()

    plt.show()
Pb_P0_ratio = 0.1



L1, A1, M1, P1, rho1, T1, flow_condition1, Asw_At_ratio1 = nozzle_conical.compute(Pb_P0_ratio)

L2, A2, M2, P2, rho2, T2, flow_condition2, Asw_At_ratio2 = nozzle_top.compute(Pb_P0_ratio)

L3, A3, M3, P3, rho3, T3, flow_condition3, Asw_At_ratio3 = nozzle_moc.compute(Pb_P0_ratio)



Plot_Nozzle(geom_con, L1, A1, M1, P1, rho1, T1, flow_condition1, Asw_At_ratio1, "Conical Nozzle: ")

print('Pb_P0_ratio=',Pb_P0_ratio)


