! pip install google_images_download
import os

import shutil



from pathlib import Path

from google_images_download import google_images_download



from fastai.vision import *
print(os.listdir('../input/food101classes'))

classes_filepath = '../input/food101classes/classes.txt'
categories = []

with open(classes_filepath) as classes_file:

    categories = classes_file.read().splitlines()

print(categories)

print(len(categories))
downloader = google_images_download.googleimagesdownload()
def download_and_get_failed(categs, img_dir='test-images'):

    failed_categs = []

    for categ in categs:

        kwargs = {

            "keywords": 'homemade ' + categ.replace('_', ' ') ,

            "limit": 20,

            "output_directory": img_dir,

            "image_directory": categ,

        }



        paths = downloader.download(kwargs)

        imgs = os.listdir(os.path.join(img_dir, categ))

        if not imgs:

            failed_categs.append(categ)

    

    return failed_categs
image_dir = 'test-images'

image_root_path = Path(image_dir)
selected_categories = categories

failed_categories = download_and_get_failed(selected_categories, image_dir)
while failed_categories:

    failed_categories = download_and_get_failed(failed_categories, image_dir)
downloaded_categories = os.listdir(image_dir)

print(len(downloaded_categories))
for category in downloaded_categories:

    print(os.path.join(image_dir, category))

    verify_images(os.path.join(image_dir, category), delete=True, max_workers=8)

    i = 0

    walks = os.walk(image_root_path/category)

    dirpaths, dirnames, filenames = next(walks)

    for filename in filenames:

        newname = ''.join([category, '-', str(i), filename[filename.rfind('.'):]])        

        os.rename(os.path.join(dirpaths, filename), os.path.join(dirpaths, newname))

        i += 1
img_root_path = Path(image_dir)

for category in downloaded_categories[:5]:

    print('\n', category)

    files = os.listdir(image_root_path/category)

    print(files)
zip_filename = 'food101-custom-test-homemade-and-keyword'



shutil.make_archive(zip_filename, 'zip', image_dir)
shutil.rmtree(image_dir, ignore_errors=True)

shutil.rmtree('unpacked', ignore_errors=True)