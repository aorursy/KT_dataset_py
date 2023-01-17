from PIL import Image

import imagehash

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

def check_if_similar(im1_name, im2_name, cutoff):

    hash0 = imagehash.average_hash(Image.open('../input/images/images/' + im1_name)) 

    hash1 = imagehash.average_hash(Image.open('../input/images/images/' + im2_name)) 



    if hash0 - hash1 < cutoff:

      print('images are similar')

    else:

      print('images are not similar')

    

def show_image(image):

    image = Image.open('../input/images/images/' + image)

    imgplot = plt.imshow(image)

    plt.show()
show_image('master_im1.jpg')

show_image('wrong_im2.png')

check_if_similar('master_im1.jpg', 'wrong_im2.png', 20)

show_image('master_im1.jpg')

show_image('scraped_im1.jpeg')

check_if_similar('master_im1.jpg', 'scraped_im1.jpeg', 20)
# However, if you change the threshold, it might say that the similar images are not that similar. Let's change it from 20 to 18 in this example



show_image('master_im1.jpg')

show_image('scraped_im1.jpeg')

check_if_similar('master_im1.jpg', 'scraped_im1.jpeg', 18)