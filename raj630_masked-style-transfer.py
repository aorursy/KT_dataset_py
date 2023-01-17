import numpy as np
from scipy.misc import imresize
import matplotlib.pyplot as plt
import PIL
def create_mask(mask_path, shape):
    mask = PIL.Image.open(mask_path).convert("L") 
    width, height, _ = shape
    mask = imresize(mask, (width, height), ).astype('float32')

    mask[mask <= 127] = 0
    mask[mask > 128] = 255

 

    return mask
mask_path = "../input/turtle.jpg"
content = plt.imread(mask_path)
art = plt.imread("../input/art.jpg")
content[36, :, 2:]

mask = create_mask(mask_path,content.shape)
mask.shape
plt.imshow(mask,cmap='gray')
p = art
plt.imshow(content)
plt.show()
plt.imshow(p)
plt.show()
art.setflags(write=1)

width,height,_ = content.shape
art = imresize(art, (width, height), interp='bicubic').astype('float32')
for i in range(width):
        for j in range(height):
            if mask[i, j] == 0.:
                art[i, j, :] = content[i, j, :]
final = art/255.0
plt.imshow(final)
final

def save(image, filename):
  
    image = np.clip(image, 0.0, 255.0)

    image = image.astype(np.uint8)
    
    with open(filename, 'wb') as file:
        Image.fromarray(image).save(file, 'jpeg')
save(mask*255.0,"mask.jpg")


