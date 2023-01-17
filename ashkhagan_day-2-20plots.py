import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import os

# Location of the image dir
img_dir = '../input/chest-xray-pneumonia/chest_xray/train/NORMAL/'
# Adjust the size of your images
plt.figure(figsize=(8,8))
norimg = plt.imread(os.path.join(img_dir,'IM-0125-0001.jpeg'))
plt.imshow(norimg, cmap='gray')
plt.colorbar()
plt.title('Normal Chest Xray Image')
plt.axis('on')
print(f"The dimensions of the image are {norimg.shape[0]} pixels width and {norimg.shape[1]} pixels height, one single color channel")
print(f"The maximum pixel value is {norimg.max():.4f} and the minimum is {norimg.min():.4f}")
print(f"The mean value of the pixels is {norimg.mean():.4f} and the standard deviation is {norimg.std():.4f}")    
plt.show()

# Location of the image dir
img_dir = '../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/'
# Adjust the size of your images
plt.figure(figsize=(6,6))
virimg= plt.imread(os.path.join(img_dir,'person1000_virus_1681.jpeg'))
plt.imshow(virimg, cmap='gray')
plt.colorbar()
plt.title('Virus Infected Pneumonia')
plt.axis('on')
print(f"The dimensions of the image are {virimg.shape[0]} pixels width and {virimg.shape[1]} pixels height, one single color channel")
print(f"The maximum pixel value is {virimg.max():.4f} and the minimum is {virimg.min():.4f}")
print(f"The mean value of the pixels is {virimg.mean():.4f} and the standard deviation is {virimg.std():.4f}")    
# Adjust subplot parameters to give specified padding
plt.show()    
# Adjust the size of your images
plt.figure(figsize=(6,6))
bacimg = plt.imread(os.path.join(img_dir,'person1000_bacteria_2931.jpeg'))
plt.imshow(bacimg, cmap='gray')
plt.colorbar()
plt.title('Bacteria Infected Pneumonia Chest Xray')
plt.axis('on')
print(f"The dimensions of the image are {bacimg.shape[0]} pixels width and {bacimg.shape[1]} pixels height, one single color channel")
print(f"The maximum pixel value is {bacimg.max():.4f} and the minimum is {bacimg.min():.4f}")
print(f"The mean value of the pixels is {bacimg.mean():.4f} and the standard deviation is {bacimg.std():.4f}")       
# Adjust subplot parameters to give specified padding
plt.show() 
# Plot a histogram of the distribution of the pixels
sns.distplot(norimg.ravel(), 
             label=f'Pixel Mean {np.mean(norimg):.4f} & Standard Deviation {np.std(norimg):.4f}', kde=False)
plt.legend(loc='upper center')
plt.title('Distribution of Pixel Intensities in the Normal Chest Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('# Pixels in Image')
# Plot a histogram of the distribution of the pixels
sns.distplot(virimg.ravel(), 
             label=f'Pixel Mean {np.mean(virimg):.4f} & Standard Deviation {np.std(virimg):.4f}', kde=False)
plt.legend(loc='upper center')
plt.title('Distribution of Pixel Intensities in the  Virus Infected Chest Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('# Pixels in Image')
# Plot a histogram of the distribution of the pixels
sns.distplot(bacimg.ravel(), 
             label=f'Pixel Mean {np.mean(bacimg):.4f} & Standard Deviation {np.std(bacimg):.4f}', kde=False)
plt.legend(loc='upper center')
plt.title('Distribution of Pixel Intensities in the  Bacteria Infected Chest Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('# Pixels in Image')
data_wine=pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
data_wine.head()
data_wine.info()
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'fixed acidity', data = data_wine)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'volatile acidity', data = data_wine)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y='citric acid', data = data_wine)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y='chlorides', data = data_wine)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y='residual sugar', data = data_wine)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y='free sulfur dioxide', data = data_wine)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y='total sulfur dioxide', data = data_wine)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y='density', data = data_wine)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y='pH', data = data_wine)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y='sulphates', data = data_wine)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y='alcohol', data = data_wine)
data_leag=pd.read_csv("../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv")
data_leag.head(10)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'blueWins', y='blueDragons', data = data_leag)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'blueWins', y='blueFirstBlood', data = data_leag)
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'blueWins', y='blueAssists', data = data_leag)