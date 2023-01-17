tile_h = 10

tile_w = 25



total = tile_h * tile_w
from glob import glob

from random import shuffle

from math import ceil, floor



from PIL import Image
def center_crop(img):

    h = img.height

    w = img.width

    d = min(w, h)



    if h == w:

        return img



    if h > w:

        left = 0

        right = w

        top = floor((h - d) / 2)

        bottom = h - top - 1

    else:

        left = floor((w - d) / 2)

        right = w - left - 1

        top = 0

        bottom = h



    return img.crop((left, top, right, bottom))





def resize(img, d=256):

    h = img.height

    w = img.width



    if h > w:

        new_h = d

        new_w = floor(h * d / w)

    else:

        new_h = floor(w * d / h)

        new_w = d



    return img.resize((new_h, new_w),resample=Image.BICUBIC)





def crop_resize(img):

    return center_crop(resize(img))
def concat_horizontal(images):

    w = sum(im.width for im in images)

    h = max(im.height for im in images)



    result = Image.new('RGB', (w, h))



    start_w = 0

    for im in images:

        result.paste(im, (start_w, 0))

        start_w = start_w + im.width



    return result



def concat_vertical(images):

    w = max(im.width for im in images)

    h = sum(im.height for im in images)



    result = Image.new('RGB', (w, h))



    start_h = 0

    for im in images:

        result.paste(im, (0, start_h))

        start_h = start_h + im.height

    

    return result
images = glob('../input/clothing-dataset-full/images_compressed/*')

shuffle(images)
images_pil = [Image.open(im) for im in images[:total]]
cropped = []



for img in images_pil:

    c = crop_resize(img)

    cropped.append(c)
lines = []



for i in range(0, total, tile_w):

    sublist = cropped[i:i + tile_w]

    line = concat_horizontal(sublist)

    lines.append(line)





result = concat_vertical(lines)
result.save('collage.jpg')
Image.open('../working/collage.jpg')