import numpy as np

import matplotlib.pyplot as plt



def sigmoid(x):

    return 1.0 / ( 1.0 + np.exp( -x ) )



xData = np.arange( -10.0, 40.0, 0.1 )

ySigm = sigmoid( xData )

yPositioned = 0.5 + ySigm/2

# Draw a banner; more detail, designed for meetup.com photo aspect ratio 600x338



banner = plt.figure(figsize=(6, 3.38))

ax = plt.axes(xlim=(-5.0, 40.0), ylim=(0.3, 1.1), frame_on=True, xticks=[], yticks=[])



logo_text = plt.text(2.5, 0.72, 'heffieldML',

                     fontsize=56,

                     verticalalignment='center',

                     color='b',

                     fontstretch='ultra-condensed')



subtext = plt.text(18, 0.40, 'Machine Learning & AI Meetup',

                   fontsize=22,

                   verticalalignment='center',

                   horizontalalignment='center',

                   color='b',

                   fontstretch='ultra-condensed')



plot = plt.plot( xData, yPositioned, 'b', linewidth=4)

plt.savefig('banner.png')

plt.savefig('banner.svg')
# Draw a square profile pic style logo for Twitter etc. - just the curve, thicker, cropping at the bounding box



logo = plt.figure(figsize=(3, 3))

ax = plt.axes(xlim=(-5.0, 5.0), ylim=(0.0, 1.0), frame_on=True, xticks=[], yticks=[])

plot = plt.plot( xData, ySigm, 'b', linewidth=8)

plt.savefig('logo.png')

plt.savefig('logo.svg')
import os

os.listdir('.')