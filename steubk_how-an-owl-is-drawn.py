import pandas as pd
import numpy as np
import ast

import matplotlib.pyplot as plt

from matplotlib import animation, rc
from IPython.display import HTML, Image

import cv2

rc('animation', writer='imagemagick')

cat = pd.read_csv('../input/train_simplified/owl.csv', nrows=500)
cat['drawing'] = cat['drawing'].apply(ast.literal_eval)



BASE_SIZE = 256


def draw_cv2(raw_strokes, step, size=256, lw=6):
   
    max_x = max( [max(x)  for x, y in raw_strokes ] )
    max_y = max( [max(y)  for x, y in raw_strokes ] )
   
    max_x, max_y =  max_x + 1, max_y + 1
    img = np.zeros((max_y, max_x), np.uint8)

   
    for i in range(step):
        stroke = raw_strokes[i]
        
        x,y=stroke[0], stroke[1]
        for i in range(len(x) - 1):
            _ = cv2.line(img, (x[i], y[i]),
                         (x[i + 1], y[i + 1]), 255, lw)
   
   
    top = max ( (BASE_SIZE - max_y) // 2 , 0 )
    bottom = max ( BASE_SIZE - top - max_y , 0)
   
    left = max (  (BASE_SIZE - max_x) // 2, 0 )
    right = max ( BASE_SIZE - max_x - left, 0 )
   
   
    img =  cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT)

    if size != BASE_SIZE:
        return cv2.resize(img, (size, size))
    else:
        return img



class Draw:
    def __init__(self, raw_strokes, fig):
        self.raw_strokes = raw_strokes
        self.count_frame = len(raw_strokes)
        self.fig = fig
    
    def animation(self, size=256, lw=6, interval=250, blit=True):
        ims = []
        for frame in range(self.count_frame):
            image = draw_cv2(self.raw_strokes, frame, size, lw)
            im=plt.imshow(image, animated=True)
            #im.set_cmap('hot')
            #plt.axis('off')
            ims.append([im])
        
        
        return animation.ArtistAnimation(self.fig, ims, interval, blit)
        
        
fig=plt.figure()    
draw = Draw (cat['drawing'].values[182], fig )
anim = draw.animation(size=256, lw=5)  
#HTML(anim.to_html5_video())    
anim.save("owl.gif")

plt.close(fig)

Image(url='owl.gif',width=512, height=512)




