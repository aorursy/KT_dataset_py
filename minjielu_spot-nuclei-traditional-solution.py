# load python libraries
%matplotlib inline

import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import csv
import sys
import math
from sklearn.decomposition import PCA
from os import listdir
picture = img.imread('../input/stage1_test/df40099c6306ca1f47fcc8a62e2fa39486d4e223177afdc51b2ad189691802d8/images/df40099c6306ca1f47fcc8a62e2fa39486d4e223177afdc51b2ad189691802d8.png')
nx = len(picture) # number of pixels for width
ny = len(picture[0]) # number of pixels for height
'''
# For colorful images, PCA is tried to reduce RGB images to white&black images
# Colorful images actually need more careful and complicated treatment
picture = picture[:,:,0:3]
pca = PCA(n_components = 3)
picture = picture.reshape(nx*ny,3)
picture = pca.fit_transform(picture)
picture = picture.reshape(nx,ny,3)
'''
picture = picture[:,:,0] # In this case the image is just white&black
# Normalize images
picture_stat = np.copy(picture)
picture_stat = picture_stat.reshape(1,nx*ny)
picture_stat.sort()
# Pixels are normalized according to the 100th largest and smallest numbers instead of the largest and smallest to avoid outlier pixels
picture = (picture-picture_stat[0,100])/(picture_stat[0,nx*ny-100]-picture_stat[0,100])
picture_stat = picture.reshape(1,nx*ny)
# Images can have bright background from fluorescent light or dark background
# Pixels are reversed if background is bright
average = np.mean(picture_stat)
if average > 0.5:
    picture = 1-picture
picture_copy = np.copy(picture)
for x in np.arange(0,nx):
    for y in np.arange(0,ny):
        if picture_copy[x,y] > 0.11:
            picture_copy[x,y] = 1
        else:
            picture_copy[x,y] = 0
plt.imshow(picture_copy)
_ = plt.title('Constant Threshold')
# This function smoothens the picture by assigning a pixel average brightness within a square of size s around it.
# Alternatively, a gaussian kernel can be used
def blurPicture(picture_copy,s):
    dp = np.zeros((nx+1,ny+1)) # numpy array dp[x,y] stores the summation of brightness of all pixels in the image to the left and top of point (x,y)
    for x in np.arange(0,nx):
        dp[x,0] = 0
    for y in np.arange(1,ny):
        dp[0,y] = 0
    for x in np.arange(1,nx+1):
        for y in np.arange(1,ny+1):
            dp[x,y] = dp[x-1,y]+dp[x,y-1]-dp[x-1,y-1]+picture_copy[x-1,y-1]
    for x in np.arange(0,nx):
        for y in np.arange(0,ny):
            left = max(x-s,0)
            right = min(x+s+1,nx)
            upper = max(y-s,0)
            lower = min(y+s+1,ny)
            picture_copy[x,y] = (dp[right,lower]-dp[left,lower]-dp[right,upper]+dp[left,upper])/(right-left)/(lower-upper)
            

# Images can have uneven illumination.
# Therefore, when certain threshold is applied to decide whether a pixel should be kept or not, the threshold should be adapted 
# according to local brightness.
# Function blurPicture can be used to calculate local brightness.
local_brightness = np.copy(picture)
blurPicture(local_brightness,math.floor(min(nx,ny)/2)) # Local brightness is defined as the average brightness of pixels in a square of size min((nx,ny)/2) around the point
picture_copy = np.copy(picture)
average = np.mean(picture_copy.reshape(1,nx*ny))
for x in np.arange(0,nx):
    for y in np.arange(0,ny):
        if picture_copy[x,y] > 0.11*local_brightness[x,y]/average:
            picture_copy[x,y] = 1
        else:
            picture_copy[x,y] = 0
plt.imshow(picture_copy)
_ = plt.title('Adaptive Threshold')
picture_stat = np.copy(picture)
picture_stat = picture_stat.reshape(1,nx*ny)
picture_stat.sort()
edges = []
for x in np.arange(0,100):
    edges.append(x/100)
distribution = np.histogram(picture_stat,bins = edges)
sns.regplot(distribution[1][:-1],distribution[0],fit_reg=False)
_ = plt.title('brightness histogram')
origin_picture = np.copy(picture)
largest_value = 0
second_value = 0
largest_pos = 0
second_pos = 0
for x in np.arange(0,len(distribution[0])):
    if distribution[0][x] > largest_value:
        second_value = largest_value
        second_pos = largest_pos
        largest_value = distribution[0][x]
        largest_pos = distribution[1][x]
    elif distribution[0][x] > second_value:
        second_value = distribution[0][x]
        second_pos = distribution[1][x]
half_height = 0
for x in np.arange(len(distribution[0])-1,-1,-1):
    if distribution[0][x] > 0.5*largest_value:
        half_height = distribution[1][x]
        break
# Threshold is set to the larger one of 0.1 or half height position.
valve = max(half_height,0.25)
average = np.mean(picture_stat)
for x in np.arange(0,nx):
    for y in np.arange(0,ny):
        if picture[x,y] > max(valve*local_brightness[x,y]/average,valve):
            picture[x,y] = 1
        else:
            picture[x,y] = 0
            
picture_helper = np.copy(picture) # picture_helper is used to deal with the problem caused by nuclei with large contrast described in part 4.        
_ = plt.imshow(picture)
blurPicture(origin_picture,1)
# Learnt from some experience, extracting gradient from the original image works better than from the smoothened one 
# px stores gradient in horizontal direction. py stores gradient in vertical direction.
# pz is the magnitude of gradient
px = np.zeros((nx,ny))
for x in np.arange(0,nx-1):
    for y in np.arange(0,ny):
        px[x,y] = origin_picture[x+1,y]-origin_picture[x,y]

py = np.zeros((nx,ny))
for x in np.arange(0,nx):
    for y in np.arange(0,ny-1):
        py[x,y] = origin_picture[x,y+1]-origin_picture[x,y]

pz = np.zeros((nx,ny))
for x in np.arange(0,nx):
    for y in np.arange(0,ny):
        pz[x,y] = np.sqrt(np.power(px[x,y],2)+np.power(py[x,y],2))
        
plt.imshow(pz)
_ = plt.title('Gradient Image')
# Normalize the gradient image and reverse it if necessary.
pz_stat = np.copy(pz)
pz_stat = pz_stat.reshape(1,nx*ny)
pz_stat.sort()
pz = (pz-pz_stat[0,100])/(pz_stat[0,nx*ny-100]-pz_stat[0,100])
pz_stat = pz.reshape(1,nx*ny)
pz_average = np.mean(pz_stat)
if pz_average > 0.5:
    pz = 1-pz
pz_stat = np.copy(pz)
pz_stat = pz_stat.reshape(1,nx*ny)
pz_stat.sort()

# We define threshold for the gradient image to identify boundaries between nuclei.
pz_valve = pz_stat[0,0]+0.07*(pz_stat[0,nx*ny-1]-pz_stat[0,0])
pz_local_brightness = np.copy(pz)
# Suppose thicknesses of boundaries are close, brighter nuclei should have boundaries with larger gradient.
# Therefore, an adaptive threshold changing with brightness of the original image should be used.
# In order to handle boundaries between a bright and a dark nuclei, a smoothened original image is used.
blurPicture(pz_local_brightness,5)
average = np.mean(pz_local_brightness.reshape(1,nx*ny))
for x in np.arange(0,nx):
    for y in np.arange(0,ny):
        if picture[x,y] == 1 and pz[x,y] > pz_valve*pz_local_brightness[x,y]/average:
            pz[x,y] = 1
        else:
            pz[x,y] = 0
plt.imshow(pz)
_ = plt.title('Boundaries between Nuclei')
for x in np.arange(0,nx):
    for y in np.arange(0,ny):
        picture[x,y] = max(picture[x,y]-pz[x,y],0)

_ = plt.imshow(picture)
# This function find connected areas in an image
def labelNuclei(picture_copy,x,y):
    if x < 0 or x > nx-1 or y < 0 or y > ny-1:
        return
    if picture_copy[x,y] != 1:
        return
    picture_copy[x,y] = num
    # Depth first search is not allowed to be performed along diagonal directions since sometimes after boundaries are substracted, some adjacent nuclei are still
    # connected through one pixel along diagonal directions.
    labelNuclei(picture_copy,x+1,y)
    labelNuclei(picture_copy,x-1,y)
    labelNuclei(picture_copy,x,y+1)
    labelNuclei(picture_copy,x,y-1)

# Label connected areas in an image with numbers
num = 1
for x in np.arange(0,nx):
    for y in np.arange(0,ny):
        if picture_helper[x,y] == 1:
            num = num+1
            labelNuclei(picture_helper,x,y)
directions = [[1,1],[1,0],[1,-1],[0,1],[0,0],[0,-1],[-1,1],[-1,0],[-1,-1]]

# This function returns numbers of pixels with the other value(0 vs 1) around point (x,y)
def getScatter(picture_helper_copy,picture_copy,x,y):
    result = 0
    for z in np.arange(0,9):
        x_new = x+directions[z][0]
        y_new = y+ directions[z][1]
        if x_new >= 0 and x_new < nx and y_new >= 0 and y_new < ny:
            # If (x,y) is on the boundary of a connected area, 0 is returned. This is to reduce the effect of the size of a nucleus.
            if picture_helper_copy[x_new,y_new] == 0:
                return 0
            if picture_copy[x,y] != picture_copy[x_new,y_new]:
                result = result+1
    return result

# Check whether some connected areas in picture_helper(before boundaries are subtracted) become very scattered in picture(after boundaries are subtracted).
# If yes, recombine the area to a single piece in picture.
scatter_valve = 1.7
need_union = set()
scatter = [0]*(num+1) # stores numbers of alternative 0,1 in picture within each connected area in picture_helper.
nuclei_sizes = [0]*(num+1) # stores sizes of connected areas in picture_helper.
# Sometimes a nucleus with large contrast will just become very sparse instead of scattered after boundaries are subtracted.
zero_cnt = [0]*(num+1) # countes numbers of 0 in picture within each connected area in picture_helper.
for x in np.arange(0,nx):
    for y in np.arange(0,ny):
        label = int(picture_helper[x,y])
        if label != 0:
            nuclei_sizes[label] = nuclei_sizes[label]+1
            scatter[label] = scatter[label]+getScatter(picture_helper,picture,x,y)
            if picture[x,y] == 0:
                zero_cnt[label] = zero_cnt[label]+1
max_size = np.max(nuclei_sizes)
for x in np.arange(2,num+1):
    if scatter[x]/nuclei_sizes[x] > scatter_valve and nuclei_sizes[x] > 0.1*max_size: # If the size of a nucleus is too small, it can be background noise.
        need_union.add(x)
    elif zero_cnt[x]/nuclei_sizes[x] > 0.7:
        need_union.add(x)
for x in np.arange(0,nx):
    for y in np.arange(0,ny):
        if picture_helper[x,y] in need_union:
            picture[x,y] = 1
# It's now time to label individual nucleus.
num = 1
for x in np.arange(0,nx):
    for y in np.arange(0,ny):
        if picture[x,y] == 1:
            num = num+1
            labelNuclei(picture,x,y)
            
plt.figure(figsize=(20,7),dpi=80)                       
plt.imshow(picture)
_ = plt.title('Nuclei Tagging')
# This function finds the majority vote of a pixels' neighbors.
def major(final,x,y):
    cnt = dict()
    for z in np.arange(0,9):
        x_1 = x+directions[z][0]
        y_1 = y+directions[z][1]
        if x_1 > 0 and x_1 < nx and y_1 > 0 and y_1 < ny:
            if final[x_1,y_1] not in cnt:
                cnt.update({final[x_1,y_1]:1})
            else:
                cnt[final[x_1,y_1]] = cnt[final[x_1,y_1]]+1
    majority = 0
    major_cnt = 0
    for x in cnt.items():
        if x[0] != 0:
            if x[1] > major_cnt:
                majority = x[0]
                major_cnt = x[1]
    if major_cnt <= 2:
        return 0
    return majority

# Nuclei are allowed to grow in the mentioned manner for two iterations.
picture_helper = np.zeros((nx,ny))
for x in np.arange(0,nx):
    for y in np.arange(0,ny):
        picture_helper[x,y] = major(picture,x,y)
for x in np.arange(0,nx):
    for y in np.arange(0,ny):
        picture[x,y] = major(picture_helper,x,y)

plt.figure(figsize=(20,7),dpi=80)                       
plt.imshow(picture)
_ = plt.title('After Growth')
# Calculate average brightness of each nucleus
nuclei_brightness = dict()
nuclei_sizes = dict()
for x in range(0,nx):
    for y in range(0,ny):
        if picture[x,y] not in nuclei_sizes:
            nuclei_sizes.update({picture[x,y]:1})
            nuclei_brightness.update({picture[x,y]:origin_picture[x,y]})
        else:
            nuclei_sizes[picture[x,y]] = nuclei_sizes[picture[x,y]]+1
            nuclei_brightness[picture[x,y]] = nuclei_brightness[picture[x,y]]+origin_picture[x,y]
for z in nuclei_brightness.items():
    if z[0] != 0:
        nuclei_brightness[z[0]] = nuclei_brightness[z[0]]/nuclei_sizes[z[0]]

# According to how the score is calculated by Kaggle, adjustment is needed to make sure masks covers exactly nuclei.
# After the above step, pixels are adjusted according to following criteria.
# 1. If a pixel is darker than the brightness of all surrounding nuclei times a factor, the pixel is removed.
# 2. Otherwise a pixel is assigned to the nucleus with the closest brightness around it.
def adjustNuclei(picture_copy,origin_pic,x,y,local_brightness,average,nuclei_brightness):
    factor = 0.7
    cnt = dict()
    for x_1 in range(x-1,x+2):
        for y_1 in range(y-1,y+2):
            if x_1 > 0 and x_1 < nx and y_1 > 0 and y_1 < ny:
                if picture_copy[x_1,y_1] not in cnt:
                    cnt.update({picture_copy[x_1,y_1]:1})
                else:
                    cnt[picture_copy[x_1,y_1]] = cnt[picture_copy[x_1,y_1]]+1
    darkest_nucleus = 5
    majority = 0
    major_cnt = 0
    for z in cnt.items():
        if z[0] != 0:
            if nuclei_brightness[z[0]] < darkest_nucleus:
                darkest_nucleus = nuclei_brightness[z[0]]
            if z[1] > major_cnt:
                major_cnt = z[1]
                majority = z[0]
    if darkest_nucleus == 5 or origin_pic[x,y] <= min(factor*darkest_nucleus*local_brightness[x,y]/average,factor*darkest_nucleus):
        return 0
    for z in cnt.items():
        if z[0] != 0 and z[0]!=majority:
            if abs(nuclei_brightness[z[0]]-nuclei_brightness[majority]) > 0.2:
                possibility_1 = abs(origin_pic[x,y]-nuclei_brightness[majority])
                possibility_2 = abs(origin_pic[x,y]-nuclei_brightness[z[0]])
                if possibility_1 > possibility_2:
                    return z[0]
    return majority

picture_helper = np.zeros((nx,ny))
for n in range(0,2):
    for x in range(0,nx):
        for y in range(0,ny):
            picture_helper[x,y] = adjustNuclei(picture,origin_picture,x,y,local_brightness,average,nuclei_brightness)
    for x in range(0,nx):
        for y in range(0,ny):
            picture[x,y] = picture_helper[x,y]
            
plt.figure(figsize=(20,7),dpi=80)                       
plt.imshow(picture)
_ = plt.title('After Adjustment')
nuclei_brightness = dict() # stores average brightness of each nucleus.
nuclei_sizes = dict() # stores sizes of each nucleus.
nuclei_pixels_x = dict() # stores an array of x coordinates of each nucleus
nuclei_pixels_y = dict() # stores an array of y coordinates of each nucleus
for x in np.arange(0,nx):
    for y in np.arange(0,ny):
        if picture[x,y] not in nuclei_sizes:
            nuclei_sizes.update({picture[x,y]:1})
            nuclei_brightness.update({picture[x,y]:origin_picture[x,y]})
            nuclei_pixels_x.update({picture[x,y]:[x]})
            nuclei_pixels_y.update({picture[x,y]:[y]})
        else:
            nuclei_sizes[picture[x,y]] = nuclei_sizes[picture[x,y]]+1
            nuclei_brightness[picture[x,y]] = nuclei_brightness[picture[x,y]]+origin_picture[x,y]
            nuclei_pixels_x[picture[x,y]].append(x)
            nuclei_pixels_y[picture[x,y]].append(y)

for z in nuclei_brightness.items():
    if z[0] != 0:
        nuclei_brightness[z[0]] = nuclei_brightness[z[0]]/nuclei_sizes[z[0]]
need_remove = set()
# Remove all nuclei smaller than 5 pixels.
for m in nuclei_sizes.items():
    if m[0] != 0 and m[1] < 5:
        need_remove.add(m[0])
        
# Remove all nuclei darker than 0.2 times the brightness of the brightest nucleus and smaller than 0.2 times the size of the largest nucleus.
largest = 0
brightest = 0
for m in nuclei_sizes.items():
    if m[0] != 0:
        if m[1] > largest:
            largest = m[1]
        if nuclei_brightness[m[0]] > brightest:
            brightest = nuclei_brightness[m[0]]

for m in nuclei_sizes.items():
    if m[0] != 0 and m[1] < 0.2*largest and nuclei_brightness[m[0]] < 0.2*brightest:
        need_remove.add(m[0])
        
# Remove all nuclei with a shape not close to a circle.
for m in nuclei_pixels_x.items():
    if m[0] != 0:
        covariance = np.cov(m[1],nuclei_pixels_y[m[0]])
        if abs(covariance[0,1]/covariance[0,0]/covariance[1,1]) > 0.5:
            need_remove.add(m[0])
            
for x in np.arange(0,nx):
    for y in np.arange(0,ny):
        if picture[x,y] in need_remove:
            picture[x,y] = 0
plt.figure(figsize=(20,7),dpi=80)                       
plt.imshow(picture)
_ = plt.title('Final Result')
