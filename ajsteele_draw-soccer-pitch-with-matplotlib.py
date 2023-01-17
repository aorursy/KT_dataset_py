import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import matplotlib.patches as patches
def draw_soccer_pitch(figsize=(12, 9)):

    """

    Function that plots a scaled soccer pitch of length 120*90 metres which 

    are the maximum dimensions allowed by FIFAs "Laws Of The Game"

    """

    rect = patches.Rectangle((-1, -1), 122, 92, linewidth=0.1,

                             edgecolor='r', facecolor='darkgreen', zorder=0)



    fig, ax = plt.subplots(1, figsize=figsize)

    ax.add_patch(rect)

    # Main pitch markings, ie sidelines, penalty area and halfway line

    plt.plot([0, 0,  0, 120, 120, 0,     0,  16.5,  16.5,     0,     0,   5.5,   5.5, 

                  0,  0, 60, 60, 120,   120, 103.5, 103.5,   120,   120, 114.5, 114.5,   120], 

             [0, 0, 90,  90,   0, 0, 25.85, 25.85, 66.15, 66.15, 55.15, 55.15, 36.85, 

              36.85, 90, 90,  0,   0, 25.85, 25.85, 66.15, 66.15, 55.15, 55.15, 36.85, 36.85], color='white')

    

    # Secondary pitch markings, ie penalty spots, centre circle etc

    plt.plot([11, 11.5],[45, 45], color='white')

    plt.plot([109, 108.5],[45, 45], color='white')

    

    centre_circle = patches.Circle([60, 45], 9.15, edgecolor='white', facecolor='darkgreen')

    ax.add_patch(centre_circle)

    

    left_arc = patches.Arc([16.5, 45], 9.15, 16, theta1=270.0, theta2=90.0, color='white')

    ax.add_patch(left_arc)

    right_arc = patches.Arc([103.5, 45], 9.15, 16, theta1=90.0, theta2=270.0, color='white')

    ax.add_patch(right_arc)

    

    bl_corner = patches.Arc([0, 0], 2.5, 2.5, theta1=0.0, theta2=90.0, color='white')

    tl_corner = patches.Arc([0, 90], 2.5, 2.5, theta1=270.0, color='white')

    br_corner = patches.Arc([120, 0], 2.5, 2.5, theta1=90.0, theta2=180.0, color='white')

    tr_corner = patches.Arc([120, 90], 2.5, 2.5, theta1=180.0, theta2=270.0,color='white')

    ax.add_patch(bl_corner)

    ax.add_patch(tl_corner)

    ax.add_patch(br_corner)

    ax.add_patch(tr_corner)

    

    plt.xlim(-1, 121)

    plt.ylim(-1, 91)

    plt.axis('off')    



    return fig, ax



draw_soccer_pitch()

plt.show()