import matplotlib.pyplot as plt
import skimage
from skimage import data , io
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.transform import warp , AffineTransform
from skimage.draw import ellipse, circle

checkerboard = skimage.io.imread("../input/image-chessboard/chessboard_GRAY.png")
plt.figure(figsize = (6,6))
plt.imshow(checkerboard, cmap='gray')

# transform it to sclae rotate share or transform shear= lateral force 
#warp allows i\you to apply transformation 
transform = AffineTransform(scale =(0.9 , 0.8), rotation= 1 ,shear=0.6,translation=(150, -80))
warped_checkerboard =warp(checkerboard, transform, output_shape=(320,320))
plt.figure(figsize= (6,6))
plt.imshow(warped_checkerboard , cmap= 'gray')

corners = corner_harris(checkerboard)
plt.figure(figsize = (6,6))
plt.imshow(corners ,cmap= 'gray')

#gives the x,y coordinates
coords_peaks = corner_peaks(corners, min_distance=1)
coords_peaks.shape

#checks the corners 
coords_subpix = corner_subpix(checkerboard, coords_peaks, window_size=10)
coords_subpix[0:11]

fig, ax = plt.subplots(figsize=(8,8))

ax.imshow(checkerboard, interpolation = 'nearest',cmap='gray')
ax.plot(coords_peaks[:,1],coords_peaks[:,0],'.b',markersize=30)
ax.plot(coords_subpix[:,1],coords_subpix[:,0],'*r',markersize=10)
plt.tight_layout()
plt.show()
