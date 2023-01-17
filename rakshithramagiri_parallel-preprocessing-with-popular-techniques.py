import multiprocessing

from tqdm import tqdm

import numpy as np

import glob

import cv2

import sys

import os





class ImagePreprocessor(object):

    def __init__(self, root_dir: str, save_dir: str, img_size: int, tolerance: int = 10, remove_outer_pixels: float = 0.0):

        """

        Preprocess images for kaggle competitions and general training purposes.



        args:

            root_dir  = absolute path to images folder

            save_dir  = folder in which to store processed images

            img_size  = final image dimensions, common values : 224, 512

            tolerance = tolerance value for pitch_black_remover func

            remove_outer_pixels = remove boundary pixels of image

        """

        if remove_outer_pixels > 0.50:

            print("ERROR: eroding more than 50% of image")

            raise InterruptedError



        self.root_dir = root_dir

        self.img_size = img_size

        self.tolerance = tolerance

        self.remove_outer_pixels = remove_outer_pixels



        self.images = glob.glob(f"{self.root_dir}/*.png") + glob.glob(

            f"{self.root_dir}/*.jpeg") + glob.glob(f"{self.root_dir}/*.jpg")

        self.save_dir = os.path.join(root_dir, save_dir)

        os.makedirs(self.save_dir, exist_ok=True)

        self.total_count = len(self.images)



    # counter decorator

    @staticmethod

    def _counter(func):

        def wrapper(self, *args, **kwargs):

            func(self, *args, **kwargs)

        return wrapper



    # different preprocessing methods



    @staticmethod

    def light_sensitivity_reducer(img: np.ndarray, alpha: int = 4, beta: int = -4, gamma: int = 128):

        """smooth image and apply ben's preprocessing"""

        return cv2.addWeighted(img, alpha, cv2.GaussianBlur(img, (0, 0), 10), beta, gamma)



    @staticmethod

    def outer_pixels_remover(img: np.ndarray, scale: float):

        """remove outer/boundary pixels of image"""

        scale_2 = scale / 2.0

        miny = int(img.shape[0]*scale_2)

        maxy = int(img.shape[0]-miny)

        minx = int(img.shape[1]*scale_2)

        maxx = int(img.shape[1]-minx)

        return img[miny:maxy, minx:maxx]



    @staticmethod

    def scale_image(img: np.ndarray, img_size: int):

        """resize image based on given scale"""

        return cv2.resize(img, (img_size, img_size))



    @staticmethod

    def pitch_black_remover(img: np.ndarray, tolerance: int = 10):

        """remove black pixels in image edges"""

        if img.ndim == 2:

            img_mask = img > tolerance

            return img[np.ix_(img_mask.any(1), img_mask.any(0))]

        elif img.ndim == 3:

            greyed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img_mask = greyed > tolerance

            img_1 = img[:, :, 0][np.ix_(img_mask.any(1), img_mask.any(0))]

            if img_1.shape[0] == 0:

                return img

            img_2 = img[:, :, 1][np.ix_(img_mask.any(1), img_mask.any(0))]

            img_3 = img[:, :, 2][np.ix_(img_mask.any(1), img_mask.any(0))]

            return np.stack([img_1, img_2, img_3], axis=-1)

        else:

            print("Image has more than 3 dimensions")

            raise InterruptedError



    # collaging different methods together and preprocessing images



    def replace_existing(self):

        import shutil

        shutil.rmtree(self.root_dir)

        os.makedirs(self.root_dir)

        processed_images = [i.path for i in os.scandir(

            self.save_dir) if i.is_file()]

        processor_pool = multiprocessing.Pool(64)

        for counter, _ in enumerate(processor_pool.imap_unordered(lambda x: shutil.move(x, self.root_dir), processed_images), 1):

            sys.stdout.write(

                f"\rMoving : {(counter/self.total_count)*100:3.2f}% \t[ {counter}/{self.total_count} ]")

        os.rmdir(self.save_dir)

        sys.stdout.write("\n\n")



    def forward(self, image: str):

        """take a single image path, preprocesse image and store preprocessed image"""

        img = cv2.imread(image)

        img = self.pitch_black_remover(img, tolerance=self.tolerance)

        img = self.scale_image(img, img_size=self.img_size)

        img = self.light_sensitivity_reducer(img)

        if self.remove_outer_pixels > 0.0:

            img = self.outer_pixels_remover(img, self.remove_outer_pixels)

        cv2.imwrite(os.path.join(self.save_dir, image.split('/')[-1]), img)



    def run(self, replace: bool = False):

        """process all images in root_dir in an iterative way"""

        for image in tqdm(self.images):

            # add logging if required

            self.forward(image)

        if replace:

            self.replace_existing()



    def parallel_run(self, workers: int = multiprocessing.cpu_count(), replace: bool = False):

        """process all images in root_dir parallely using python's multiprocessing library"""

        # haven't figured out a stable way for logging in case of multiprocessing

        processor_pool = multiprocessing.Pool(workers)



        for counter, _ in enumerate(processor_pool.imap_unordered(self.forward, self.images), 1):

            sys.stdout.write(

                f"\rProgress : {(counter/self.total_count)*100:3.2f}% \t[ {counter}/{self.total_count} ]")

        if replace:

            self.replace_existing()

        sys.stdout.write("\n\n")





if __name__ == "__main__":

    if True:

        sys.exit(1)

    # remove the above two lines

        

    # arguments

    import argparse

    import multiprocessing



    parser = argparse.ArgumentParser()

    parser.add_argument('-root_dir', required=True, type=str)

    parser.add_argument('-save_dir', required=True, type=str)

    parser.add_argument('-img_size', required=True, type=int)

    parser.add_argument('-tolerance', type=int, default=10)

    parser.add_argument('-remove_outer_pixels', type=float, default=0.0)

    parser.add_argument('-parallel', default=False, action='store_true')

    parser.add_argument('-workers', type=int,

                        default=multiprocessing.cpu_count())

    parser.add_argument('-replace', default=False, action='store_true')

    args = parser.parse_args()

    #



    preprocessor = ImagePreprocessor(root_dir=args.root_dir, save_dir=args.save_dir, img_size=args.img_size,

                                     tolerance=args.tolerance, remove_outer_pixels=args.remove_outer_pixels)

    if args.parallel:

        preprocessor.parallel_run(args.workers, args.replace)

    else:

        preprocessor.run(args.replace)
