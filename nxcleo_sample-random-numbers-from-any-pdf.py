# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import random

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from random import random
def sample_pdf(a, b, f, n=50):
    """
    :param a: lower domain
    :param b: upper domain
    :param f: Probability Density Function
    :param n: number of rectangles to simulate
    :return: a random number from the PDF
    """
    step = (b-a)/n
    r = random()
    for i in range(n):
        x = a + step * i
        y = f(x+0.5*step)
        if y*step > r:
            return x + (r/y)
        r -= y*step
    raise ValueError()
def pdf(x):
    s = 2
    return (s-abs(x))/s**2 if -s < x < s else 0


plt.figure(figsize = (10,6)) 
line_x = np.linspace(-2.5,2.5,100)
line_y = np.array([pdf(x) for x in line_x])
plt.plot(line_x, line_y)
plt.grid(True, which='both')
def make_boxes(ax, a, b, f, n, facecolor='#ffa6aeaa',
                     edgecolor='#ab5058', alpha=1):

    boxes = []
    step = (b - a) / n
    
    for i in range(n):
        x = a + i * step
        rect = Rectangle((x, 0), step, f(x+0.5*step))
        boxes.append(rect)

    pc = PatchCollection(boxes, facecolor=facecolor, alpha=alpha,
                         edgecolor=edgecolor)

    ax.add_collection(pc)

fig, ax = plt.subplots(1, figsize = (10,6))

ax.plot(line_x, line_y)
plt.grid(True, which='both')
make_boxes(ax, -2, 2, pdf, 10)
plt.show()
def make_boxes(ax, a, b, f, n, facecolor='#ffa6aeaa',
                     edgecolor='#ab5058', alpha=1):

    fill_boxes = []
    empty_boxes = []
    step = (b - a) / n
    
    r = 0.75
    
    for i in range(n):
        x = a + i * step
        y = f(x+0.5*step)
        if r > y*step:
            rect = Rectangle((x, 0), step, y)
            fill_boxes.append(rect)
        elif r > 0:
            rect = Rectangle((x, 0), r/y, y)
            fill_boxes.append(rect)
            sample = x+r/y
            rect = Rectangle((x+r/y, 0), step-r/y, y)
            empty_boxes.append(rect)
        else:
            rect = Rectangle((x, 0), step, y)
            empty_boxes.append(rect)
   
        r -= y*step

    ax.add_collection(PatchCollection(fill_boxes, facecolor=facecolor, alpha=alpha,
                         edgecolor=edgecolor))
    ax.add_collection(PatchCollection(empty_boxes, facecolor="#ffe1e4", alpha=alpha,
                         edgecolor=edgecolor))
    return sample
    
fig, ax = plt.subplots(1, figsize = (10,6))

ax.plot(line_x, line_y)
plt.grid(True, which='both')
sample = make_boxes(ax, -2, 2, pdf, 10)

var_y = np.linspace(-0.1,0.6,10)
var_x = np.array([sample for y in var_y])
ax.plot(var_x, var_y, color='g')

plt.show()
samples = [sample_pdf(-2, 2, pdf, 100) for _ in range(10000)]

plt.figure(figsize = (10,6)) 
plt.hist(samples, bins=50)
plt.show()