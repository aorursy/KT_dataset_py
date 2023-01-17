from PIL import Image, ImageChops

import imagehash

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

def check_if_similar(im1_name, im2_name, cutoff):    

    im1 = Image.open('../input/imagesnew/' + im1_name)

    

    show_image(im1)

    

    im1 = trim(im1)

    show_image(im1)

    

    im2 = Image.open('../input/imagesnew/' + im2_name)

    show_image(im2)



    im2 = trim(im2)

    show_image(im2)

    

    hash0 = imagehash.average_hash(im1) 

    hash1 = imagehash.average_hash(im2) 



    if hash0 - hash1 <= cutoff:

      print('images are similar ' + str(hash0 - hash1))

    else:

      print('images are not similar ' + str(hash0 - hash1))

    

def show_image(image):

    imgplot = plt.imshow(image)

    plt.show()

    

def trim(im):

    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))

    diff = ImageChops.difference(im, bg)

    diff = ImageChops.add(diff, diff, 2.0, -100)

    bbox = diff.getbbox()

    if bbox:

        return im.crop(bbox)
check_if_similar('bad2.jpeg', 'bad3.jpg', 20)

check_if_similar('good1.jpeg', 'good2.jpg', 20)

check_if_similar('good3.jpeg', 'good4.jpg', 20)

check_if_similar('badish1.jpg', 'badish2.jpeg', 20)

check_if_similar('jl.jpeg', 'master.jpg', 20)

check_if_similar('jl.jpeg', 'wrong.jpg', 20)
show_image('master_im1.jpg')

show_image('scraped_im1.jpeg')

check_if_similar('master_im1.jpg', 'scraped_im1.jpeg', 20)
# However, if you change the threshold, it might say that the similar images are not that similar. Let's change it from 20 to 18 in this example



show_image('master_im1.jpg')

show_image('scraped_im1.jpeg')

check_if_similar('master_im1.jpg', 'scraped_im1.jpeg', 18)