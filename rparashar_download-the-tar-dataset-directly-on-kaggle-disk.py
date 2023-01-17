TAR_FILE_COUNT_TO_PROCESS = 0  # <----------------------- Change this parameter to the number of tar files you want to process

ROOT_DIR = '/kaggle/working/'

TEMP_FOLDER_NAME = 'processing/'



IMAGE_W = 128                  # size of image, square images of this size will be generated

JPG_QUALITY = 70               # quality of the jpg file

VERBOSE = 0                    # if you want detailed log of what's going on

TESTING = 0                    # to test the whole process with a few files
urls = []

for i in range(TAR_FILE_COUNT_TO_PROCESS):

    url = 'https://s3.amazonaws.com/google-landmark/train/images_' + "{:03d}".format(i) + '.tar'

    urls.append(url)
import os

import shutil

from urllib.request import urlretrieve

from PIL import Image

import tarfile



class ImageDownloadWorker:

    

    def process_image_files(self):

        for i, new_url in enumerate(urls):

            print('URL: ' + new_url)

            self.download_zipfile_from_url(new_url)           #e.g. downloaded the tar file of images

            filename = new_url[new_url.rfind("/")+1:]         #e.g. image_000.tar

            tar_file_path = os.path.join(ROOT_DIR, filename)

            temp_dir_path = os.path.join(ROOT_DIR, TEMP_FOLDER_NAME)

            self.untar(tar_file_path, temp_dir_path)          #e.g. extracts image_000.tar to target dir

            self.remove_tar(tar_file_path)

            self.process_images(temp_dir_path)

            self.copytree(temp_dir_path, ROOT_DIR)

            self.remove_temp_dir(temp_dir_path)

            print('-------------- Done: ' + str(i+1))



    def copytree(self, src, dst, symlinks=False, ignore=None):

        if(VERBOSE == 1):

            print('Processing directory: ' + src)

        

        for item in os.listdir(src):

            s = os.path.join(src, item)

            d = os.path.join(dst, item)

            if os.path.isdir(s):

                self.copytree(s, d, symlinks, ignore)

            else:

                if(VERBOSE == 1):

                    print('source: ' + s + '\n' + 'dest: ' + d)

                head, tail = os.path.split(d)

                if(not os.path.exists(head)):

                    os.makedirs(head)

                shutil.copy2(s, d)

                if(VERBOSE == 1):

                    print('Copied')

            

    def download_zipfile_from_url(self, _url):

        filename = _url[_url.rfind("/")+1:]

        if(not os.path.exists(os.path.join(ROOT_DIR, filename)) ):

            print('Downloading url: ' + _url + ' to file: ' + filename)

            urlretrieve(_url, _url[_url.rfind("/")+1:])

            print('Done.')

        else:

            print('File already downloaded: ' + filename)

    

    def remove_tar(self, _file_path):

        print('Removing tar file: ' + _file_path)

        os.remove(_file_path)

        print('Done.')



    def remove_temp_dir(self, _dir_path):

        print('Removing temp dir: ' + _dir_path)

        shutil.rmtree(_dir_path)

        print('Done.')        



    def untar(self, _source_file_path, _target_dir_path):

        print('Extracting file data: ' + _source_file_path + ' to: ' + _target_dir_path)

        tf = tarfile.open(_source_file_path)

        if(os.path.exists(_target_dir_path)):

            print('Removing temp folder: ' + _target_dir_path)

            shutil.rmtree(_target_dir_path)

        

        print('Creating temp folder at: ' + _target_dir_path)

        os.mkdir(_target_dir_path)

        

        print('Extracting tar...')

        tf.extractall(_target_dir_path)

        print('Done.')

        

    def resize_image(self, _img_path):

        if (VERBOSE == 1):

            print('Resizing image: ' + _img_path)

            

        img = Image.open(_img_path)

        old_size = img.size

        ratio = float(IMAGE_W)/max(old_size)

        new_size = tuple([int(x*ratio) for x in old_size])

        img = img.resize(new_size, Image.ANTIALIAS)

        new_img = Image.new("RGB", (IMAGE_W, IMAGE_W))

        new_img.paste(img, ((IMAGE_W-new_size[0])//2,

                        (IMAGE_W-new_size[1])//2))

        new_img.save(_img_path, "JPEG", optimize=True, quality=JPG_QUALITY)



    

    def process_images(self, _folder_path):

        print('Processing images from folder: ' + _folder_path)

        for root, subFolders, files in os.walk(_folder_path):

            for file in files:

                full_file_path = os.path.join(root, file)

                if(VERBOSE == 1):

                    print('Full file path: ' + full_file_path)

                if(full_file_path[full_file_path.rfind('.jpg')+1:] == 'jpg'):

                    self.resize_image(full_file_path)

                    if(TESTING == 1):

                        break

    

    

    

# Run the program



worker = ImageDownloadWorker()

worker.process_image_files()