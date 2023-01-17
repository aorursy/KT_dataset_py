import matplotlib.pyplot as plt
from skimage.morphology import convex_hull_image
import skimage.io
from skimage import data, img_as_float
#from skimage import util
from skimage.util import invert
from PIL import Image

#im1 = Image.open(r'C:\Users\Anik Chatterjee\AppData\Local\Programs\Python\Python37-32\Lib\site-packages\skimage\data\horse.png')
#im1.save(r'C:\Users\Anik Chatterjee\AppData\Local\Programs\Python\Python37-32\Lib\site-packages\skimage\data\horse.jpg')

#horse_image = data.horse()
#horse_image = invert(data.horse())
horse_image = skimage.io.imread("../input/image-horse/horse.png")
horse_inverted = invert(horse_image)
#plt.imshow(horse_inverted)
plt.imshow(horse_image)

fig, axes = plt.subplots(1,2, figsize= (12, 8), sharex=True , sharey=True)
ax = axes.ravel()

ax[0].set_title('Orginal')
ax[0].imshow(horse_image,cmap='gray')

ax[1].set_title('Inverted')
ax[1].imshow(horse_inverted,cmap='gray')
fig.tight_layout()
plt.show()
