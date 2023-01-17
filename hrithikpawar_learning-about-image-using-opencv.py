# Importing the libraries

import numpy as np

import cv2

import matplotlib.pyplot as plt
# Path of the image

image_path = '../input/picture/scenery.jpeg'



# Read the image

image = cv2.imread(image_path)
# Printing the image

print(image)
# Getting the size on the image

image_size = image.shape

print(image_size)


### Use this code if you want to only work with openCv I don't know why but this code is not working on kagggle notebook





### Use this code if you want to only work with openCv I don't know why but this code is not working on kagggle notebook

#while True:

#    cv2.imshow("Image", image)   ### This will show the pop up window having title as "Image" but it is only for 1 mili sec hence we applied the while loop to see it comtinuously

#    if cv2.waitKey(1) == ord('q'): ### When we press 'q' we exit the loop and image stop displaying but window is not closed

#        break

#cv2.destroyAllWindows() ## This Code at last distroys all the windows



## On kaggle notebook we should use

plt.imshow(image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # We are converting the color format to RGB

plt.imshow(image)
total_pixel_in_column = image_size[0]

total_pixel_in_row = image_size[1]



# Calculating the number of pixels in top half part of the column

column_top_half = total_pixel_in_column // 2

# Calculating the number of pixels in bottom half part of the column

column_bottom_half = total_pixel_in_column - column_top_half



# Calculating the number of pixels in left half of the row

row_left_half = total_pixel_in_row // 2

# Calculating the number of pixels in right half of the row

row_right_half = total_pixel_in_row - row_left_half



print(f'''

Top Half: {column_top_half}

Bottom Half: {column_bottom_half}

Left Half: {row_left_half}

Right Half: {row_right_half}

''')

# Cropping the image in four pieses

image_1 = image[0:column_top_half, 0:row_left_half]

image_2 = image[0:column_top_half, row_left_half:]

image_3 = image[column_top_half:, 0:row_left_half]

image_4 = image[column_top_half:, row_left_half:]



# Displaying the images

plt.figure()

f, axarr = plt.subplots(2,2) 





axarr[0][0].imshow(image_1)

axarr[0][1].imshow(image_2)

axarr[1][0].imshow(image_3)

axarr[1][1].imshow(image_4)
top_image = np.concatenate((image_4, image_3), axis=1)

bottom_image = np.concatenate((image_2, image_1), axis=1)



rearranged_image = np.concatenate((top_image, bottom_image))



# Display the re-arranged image

plt.imshow(rearranged_image)