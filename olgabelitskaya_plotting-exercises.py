%run ../input/python-recipes/dhtml.py

%run ../input/python-recipes/embedding_html_string.py

dhtml('Code Modules & Data Files')
import os,cv2,urllib

import numpy as np,pylab as pl

from skimage import color,measure

from skimage.transform import resize

import matplotlib.patches as pt

from IPython.core.magic import register_line_magic

from IPython.display import HTML

pi=np.pi

print(os.listdir(

    "../input/image-examples-for-mixed-styles"))
dhtml('Contour Detection')
@register_line_magic

def vector(params):

    [fn,cmn]=params.split()

    cm=['ocean','cool','gnuplot2','terrain',

        'winter','spring','summer','autumn']

    path1='../input/image-examples-for-mixed-styles/'

    path2='pattern0'+fn+'.png'

    img=cv2.imread(path1+path2)

    gray_img=color.colorconv.rgb2grey(img) 

    contours=measure.find_contours(gray_img,.7)

    n=len(contours); pl.figure(figsize=(10,10))

    pl.gca().invert_yaxis()

    [pl.plot(contours[i][:,1],contours[i][:,0],lw=.5,

             color=pl.get_cmap(cm[int(cmn)])(i/n)) 

     for i in range(n)]

    pl.xticks([]); pl.yticks([]); pl.show()
%vector 7 1
dhtml('Key Points of Sketch Images')
%run ../input/python-recipes/points2image.py

kp="""[[[139,141,128,109,84,50,37,21,14,6,1,1,14,39,59,80, 

         124,160,174,186,190,194,191,177,141,119,108,106], 

        [115,44,29,17,11,10,14,24,33,50,74,132,169,203,215,

         219,218,205,192,171,155,116,99,81,49,40,40,45]], 

        [[82,74,62,37,22,7,7,22,39,86,118,124,142,

          147,146,130,116,101,85,56,50,49], 

         [31,30,36,51,65,97,130,162,177,196,199,195,

          174,138,113,69,50,39,33,33,42,51]], 

        [[73,67,63,62,66,82,86,82,71,63],

         [93,94,103,114,120,119,109,96,90,91]], 

        [[109,89,79],[64,84,105]],[[89,111],[108,157]],

        [[63,47,44,45,57,71,83,83,79,70,52,46,48],

         [4,6,11,24,36,37,27,20,14,6,0,2,10]], 

        [[80,63,48],[203,218,247]],[[127,160],[197,255]]]"""

img1=get_image(kp,128)

img2=cv_get_image(kp,128,2)

fig=pl.figure(figsize=(10,5))

ax=fig.add_subplot(121)

pl.imshow(img1,cmap=pl.cm.Pastel1_r)

ax=fig.add_subplot(122)

pl.imshow(img2,cmap=pl.cm.Pastel1_r)

pl.show();
dhtml('Random Coefficients & Colors')
%run ../input/python-recipes/rotated_leaf.py

rotated_leaf(17,3,10)
dhtml('Recurrence Tables')
%run ../input/python-recipes/recursive_plot.py

recursive_plot(recursive_f1,5,7,30000,10)

recursive_plot(recursive_f1,5,7,100000,10)
dhtml('Random Patterns')
%run ../input/python-recipes/random_pattern01.py

random_pattern_plot('r',10)
dhtml('Coordinate Rotation')
%run ../input/python-recipes/circle_mandala.py

circle_mandala('b',8)
dhtml('"Universal" Functions for Sketch Images')
%run ../input/python-recipes/smiley_points.py

fig,ax=pl.subplots(figsize=(10,10))

ax.set_facecolor('ivory')

col=[[np.append([1],np.random.random(2))] 

     for i in range(16)]

[pl.scatter(XT[i],YT[i],s=3,c=col[i]) 

 for i in range(16)]

pl.grid(c='silver',alpha=.4);
dhtml('Bokeh')
import numpy as np; from bokeh.layouts import gridplot

from bokeh.plotting import figure,show,output_file

from IPython.core import display

t=np.linspace(0,2*np.pi,720); 

x=(np.cos(12*t)+np.cos(6*t))*np.cos(t)

y=(np.cos(12*t)+np.cos(6*t))*np.sin(t)

TOOLS="pan,wheel_zoom,box_zoom,reset,save,box_select"

p1=figure(title='PLotting Example 1',tools=TOOLS)

p1.circle(x,y,legend_label=u'ϱ = cos 12 θ + cos 6 θ')

p1.circle(2*x,2*y,legend_label=u'2 ϱ',color="red")

p1.legend.title='Polar Functions'

p2=figure(title='Plotting Example 2',tools=TOOLS)

p2.circle(x,y,legend_label=u'ϱ = cos 12 θ + cos 6 θ')

p2.line(x,y,legend_label=u'ϱ = cos 12 θ + cos 6 θ')

p2.square(2*x,2*y,legend_label=u'2 ϱ',

          fill_color=None,line_color="red")

p2.line(2*x,2*y,legend_label=u'2 ϱ',line_color="red")

output_file("bokeh.html",title="plotting examples")

show(gridplot([p1,p2],ncols=2,plot_width=265,plot_height=265))

display.HTML('''<div id='data'><iframe src="bokeh.html" 

height="350" width="550"></iframe></div>''')