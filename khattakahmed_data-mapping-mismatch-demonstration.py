

import numpy as np

import matplotlib.pyplot as plt 



print(f"numpy version is {np.version.version}")



# Load Images

X_original = np.load("../input/sign-language-digits-dataset/Sign-language-digits-dataset/X.npy")

X_fixed = np.load("../input/signdigitdatasetfixed/X.npy")



# Reshape all images into 64x64x1 image 1 for the gray channel

X_original = X_original.reshape(-1, 64,64,1 )

X_fixed = X_fixed.reshape(-1, 64,64,1 )



#Data is 2062 x 64 x 64 x 1 array

print(f"X_original shape is {X_original.shape}")

print(f"X_fixed shape is {X_fixed.shape}")



# Load labels - One hot encoded: 2062 x 10

Y_original = np.load("../input/sign-language-digits-dataset/Sign-language-digits-dataset/Y.npy")

Y_fixed = np.load("../input/signdigitdatasetfixed/Y.npy")

                  

print(f"Y_original shape is {Y_original.shape}")

print(f"Y_fixed shape is {Y_fixed.shape}")

                  



# plot a sample images



# Change sample index and check associated labels

image_index = np.random.randint(2062, size=1)[0] # generate random index for image array



fig, ax = plt.subplots(1,2,figsize=(8, 18))

fig.tight_layout()

sample_image = X_original[image_index,:,:].reshape(64,64)

ax[0].imshow(sample_image, cmap='gray')

ax[0].set_title(f"Image from original data with index {image_index} \n Data label is {np.argmax(Y_original[image_index])}  ")



sample_image = X_fixed[image_index,:,:].reshape(64,64)

ax[1].imshow(sample_image,cmap='gray')

ax[1].set_title(f"Image from fixed data with index {image_index} \n Data label is {np.argmax(Y_fixed[image_index])}  ")


