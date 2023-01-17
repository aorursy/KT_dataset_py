import glob
import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt
import seaborn as sns
#for train cat image pixel 
cy = []
cx = []
for i in (glob.glob("../input/dog vs cat/dataset/training_set/cats/*.jpg")):
    img = plt.imread(i)
    a = np.shape(img)
    c = np.reshape(img,(a[0]*a[1],a[2]))
    cy.append(np.shape(c)[0])
    cx.append(i)
columns = ['cat','c_pix']
dt = np.array([cx,cy])
cat_train_df = pd.DataFrame(dt.T, columns = columns)
cat_train_df['c_pix'] = cat_train_df['c_pix'].astype('int')
cat_train_df = cat_train_df.sort_values('c_pix')
cat_train_df.head()

sns.set(style="darkgrid")
mortality_age_plot = sns.barplot(x=cat_train_df['cat'],
                                 y=cat_train_df['c_pix'],
                                 palette = 'muted',
                                 order=cat_train_df['cat'].tolist())

plt.xticks(rotation=90)
plt.show()
new_style = {'grid': False}
plt.rc('axes', **new_style)
_, ax = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(12, 12))

for i in range(9):
    img = cv2.imread(cat_train_df['cat'][i])
    ax[i // 3, i % 3].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #ax[i // 3, i % 3].set_title('{} - {}'.format(f, l))
    #ax[i // 4, i % 4].show()
    print(cat_train_df['cat'][i])
  
new_style = {'grid': False}
plt.rc('axes', **new_style)
_, ax = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(12, 12))

for i in range(9):
    img = cv2.imread(cat_train_df['cat'][i])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,200,100)
    ax[i // 3, i % 3].imshow(edges)
    #ax[i // 3, i % 3].set_title('{} - {}'.format(f, l))
    #ax[i // 4, i % 4].show()
    print(cat_train_df['cat'][i])
new_style = {'grid': False}
plt.rc('axes', **new_style)
_, ax = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(12, 12))
j = 0
for i in range((len(cat_train_df['cat'])-1),(len(cat_train_df['cat'])-10),-1):
    img = cv2.imread(cat_train_df['cat'][i])
    ax[j // 3, j % 3].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #ax[i // 3, i % 3].set_title('{} - {}'.format(f, l))
    #ax[i // 4, i % 4].show()
    print(cat_train_df['cat'][i])
    j += 1
new_style = {'grid': False}
plt.rc('axes', **new_style)
_, ax = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(12, 12))
j = 0
for i in range((len(cat_train_df['cat'])-1),(len(cat_train_df['cat'])-10),-1):
    img = cv2.imread(cat_train_df['cat'][i])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,200,100)
    ax[j // 3, j % 3].imshow(edges)
    #ax[i // 3, i % 3].set_title('{} - {}'.format(f, l))
    #ax[i // 4, i % 4].show()
    print(cat_train_df['cat'][i])
    j += 1
def pixel_matrix(path):
    image = plt.imread(path)
    dims = np.shape(image)
    return np.reshape(image, (dims[0] * dims[1], dims[2]))# changing shape
#red = []
#green = []
#blue = []
#for i in range(4000):
    #img = pixel_matrix(cat_train_df['cat'][i])
    #red.append(img[:,0])
    #green.append(img[:,1])
    #blue.append(img[:,2])
#red = [j for i in red for j in i]
#green = [j for i in green for j in i]
#blue = [j for i in blue for j in i]
#sns.distplot(red, bins=12)
#sns.distplot(green, bins=12)
#sns.distplot(blue, bins=12)
def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()

count = 0
for imagePath in cat_train_df['cat']:
    # load the image, convert it to grayscale, and compute the
    # focus measure of the image using the Variance of Laplacian
    # method
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    #text = "Not Blurry"

    # if the focus measure is less than the supplied threshold,
    # then the image should be considered "blurry"
    
    if fm < 110.0:
        count += 1
        
print("Total blur image is ",count)
    # show the image
    #cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    #cv2.imshow("Image", image)
    #key = cv2.waitKey(0)
#Histogram equalization

