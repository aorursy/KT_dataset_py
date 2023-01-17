import os

import cv2

import skimage.io

from tqdm.notebook import tqdm

import numpy as np

from albumentations.core.transforms_interface import ImageOnlyTransform

import matplotlib.pyplot as plt

TRAIN = '../input/prostate-cancer-grade-assessment/train_images/'

MASKS = '../input/prostate-cancer-grade-assessment/train_label_masks/'

OUT_TRAIN = 'train.zip'

OUT_MASKS = 'masks.zip'
names = [name[:-10] for name in os.listdir(MASKS)]

imgs = []

for name in tqdm(names[:10]):

    img = skimage.io.MultiImage(os.path.join(TRAIN,name+'.tiff'))[-1]

    imgs.append(img)


class SplitAndConcatTilesMine(ImageOnlyTransform):

    def __init__(

        self, always_apply=False, p=1.0, tile_size=128, pad_value=255, tiles_in_final_image_size=(5, 5)

    ):

        super().__init__(always_apply, p)

        self.tile_size = tile_size

        self.pad_value = pad_value

        self.tiles_in_final_image_size = tiles_in_final_image_size

        self.resulted_tiles = int(self.tiles_in_final_image_size[0] * self.tiles_in_final_image_size[1])



        self.need_reverse = True if self.pad_value == 0 else False



    def pad_image(self, image):

        height, width, channels = image.shape

        pad_h, pad_w = self.tile_size - height % self.tile_size, self.tile_size - width % self.tile_size

        res_size = (

            self.tiles_in_final_image_size[0] * self.tile_size,

            self.tiles_in_final_image_size[1] * self.tile_size,

        )

        if res_size[0] > height + pad_h:

            pad_h = res_size[0] - height

        if res_size[1] > width + pad_w:

            pad_w = res_size[1] - width



        padded_img = np.pad(image, [(pad_h, 0), (pad_w, 0), (0, 0)], "constant", constant_values=self.pad_value)

        height_padded, width_padded, channels_padded = padded_img.shape



        assert height_padded >= height

        assert width_padded >= width

        assert channels_padded >= channels

        assert height_padded % self.tile_size == 0

        assert width_padded % self.tile_size == 0

        return padded_img, height_padded, width_padded



    def cut_tiles(self, padded_img, height_padded, width_padded):

        w_len, h_len = width_padded // self.tile_size, height_padded // self.tile_size

        h_w_tile_storage = [[None for w in range(w_len)] for h in range(h_len)]

        tiles = []

        for h in range(h_len):

            for w in range(w_len):

                tile = padded_img[

                    self.tile_size * h : self.tile_size * (h + 1), self.tile_size * w : self.tile_size * (w + 1)

                ]

                tile_intensivity = tile.sum()

                h_w_tile_storage[h][w] = tile

                tiles.append([tile, h, w, tile_intensivity])

        sorted_tiles = sorted(tiles, key=lambda x: x[3], reverse=self.need_reverse)

        return h_w_tile_storage, sorted_tiles



    def constract_bin_matrix(self, sorted_tiles, height_padded, width_padded):

        # fill bin_mask [intence_block_bool, height, width]

        bin_mask = np.zeros((height_padded // self.tile_size, width_padded // self.tile_size, 3), dtype=int)

        for i in range(self.resulted_tiles):

            _, h, w, _ = sorted_tiles[i]

            bin_mask[h][w][0] = 1

            bin_mask[h][w][1] = h

            bin_mask[h][w][2] = w

        return bin_mask



    def apply(self, image, **params):



        padded_img, height_padded, width_padded = self.pad_image(image)



        h_w_tile_storage, sorted_tiles = self.cut_tiles(padded_img, height_padded, width_padded)



        bin_mask = self.constract_bin_matrix(sorted_tiles, height_padded, width_padded)



        resulted_img = [

            [None for _ in range(self.tiles_in_final_image_size[1])] for _ in range(self.tiles_in_final_image_size[0])

        ]

        region_of_interest = np.ones(self.tiles_in_final_image_size, dtype=bool)



        most_intencivity = 1 # crunch for while loop

        while most_intencivity > 0:

            bin_mask, region_of_interest, resulted_img, most_intencivity = self.process_region(

                bin_mask, region_of_interest, resulted_img

            )



        # deal with leftovers

        bin_mask, resulted_img

        bin_h, bin_w, _ = bin_mask.shape

        for h in range(bin_h):

            for w in range(bin_w):

                if bin_mask[h][w][0] == 1:

                    resulted_img = self.insert_value_in_res_im_array(resulted_img, bin_mask[h][w][1:].tolist())

                    bin_mask[h][w][0] = 0



        tiles_arr = [

            [None for _ in range(self.tiles_in_final_image_size[1])] for _ in range(self.tiles_in_final_image_size[0])

        ]

        for h in range(self.tiles_in_final_image_size[0]):

            for w in range(self.tiles_in_final_image_size[1]):

                target_h, target_w = resulted_img[h][w]

                tiles_arr[h][w] = h_w_tile_storage[target_h][target_w]



        return np.hstack(np.hstack(np.array(tiles_arr)))



    def get_transform_init_args_names(self):

        return ("tile_size", "pad_value", "tiles_in_final_image_size",)



    def insert_value_in_res_im_array(self, resulted_img, value):

        for h in range(len(resulted_img)):

            for w in range(len(resulted_img[0])):

                if resulted_img[h][w] is None:

                    resulted_img[h][w] = value

                    return resulted_img



    def process_region(self, bin_mask, region_of_interest, resulted_img):

        # select_region

        most_intensivity = 0

        most_intensive_region = None



        bin_mask_h, bin_mask_w, _ = bin_mask.shape

        for h in range(bin_mask_h - self.tiles_in_final_image_size[0]):

            for w in range(bin_mask_w - self.tiles_in_final_image_size[1]):

                h_slice = slice(h, h + self.tiles_in_final_image_size[0])

                w_slice = slice(w, w + self.tiles_in_final_image_size[1])

                bin_tile = bin_mask[h_slice, w_slice]

                intense = bin_tile[region_of_interest, 0].sum()

                if intense > most_intensivity:

                    most_intensivity = intense

                    most_intensive_region = bin_tile

                    most_intensive_region_slices = (h_slice, w_slice)

        if most_intensivity > 0:

            # fill resulted arr

            new_region_of_interest = np.zeros(self.tiles_in_final_image_size, dtype=bool)

            for h in range(self.tiles_in_final_image_size[0]):

                for w in range(self.tiles_in_final_image_size[1]):

                    interest = region_of_interest[h, w]

                    if interest:

                        is_filled_tile = most_intensive_region[h][w][0]

                        if is_filled_tile:

                            resulted_img[h][w] = most_intensive_region[h][w][1:].tolist()  # tolist important

                        else:

                            new_region_of_interest[h][w] = 1



            # clean selected

            bin_mask[most_intensive_region_slices][region_of_interest] = 0



            return bin_mask, new_region_of_interest, resulted_img, most_intensivity

        else:

            return bin_mask, region_of_interest, resulted_img, most_intensivity

class SplitAndConcatTilesIafoss(ImageOnlyTransform):

    def __init__(self, fake=None, always_apply=False, p=1.0, N=16):

        super().__init__(always_apply, p)

        self.fake = fake

        self.N = N

        self.buffer_size = np.round(self.N ** 0.5).astype(int)



    def tile(self, img, N=16, sz=128):

        shape = img.shape

        pad0, pad1 = (sz - shape[0] % sz) % sz, (sz - shape[1] % sz) % sz

        img = np.pad(img, [[pad0 // 2, pad0 - pad0 // 2], [pad1 // 2, pad1 - pad1 // 2], [0, 0]], constant_values=0)



        img = img.reshape(img.shape[0] // sz, sz, img.shape[1] // sz, sz, 3)

        img = img.transpose(0, 2, 1, 3, 4).reshape(-1, sz, sz, 3)

        if len(img) < N:

            img = np.pad(img, [[0, N - len(img)], [0, 0], [0, 0], [0, 0]], constant_values=0)

        idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[-N:]

        img = img[idxs]



        return img



    def apply(self, image: np.array, **params) -> np.array:

        tiles = self.tile(image, N=self.N)

        horizons = []

        buffer = []

        for t in tiles:

            buffer.append(t)

            if len(buffer) == self.buffer_size:

                horizons.append(np.vstack(buffer))

                buffer = []



        res = np.hstack(horizons)

        return res



    def get_transform_init_args_names(self):

        return ("fake",)



def display_images(images):

    '''

    This function takes in input a list of images. It then iterates through the image making openslide objects , on which different functions

    for getting out information can be called later

    source: https://www.kaggle.com/tanulsingh077/prostate-cancer-in-depth-understanding-eda-model

    '''

    mine_tiler = SplitAndConcatTilesMine()

    iafoss_tiles = SplitAndConcatTilesIafoss()

    f, ax = plt.subplots(len(images), 3, figsize=(18, 22))

    for i, image in enumerate(images):

        ax[i, 0].imshow(image) #Displaying Image

        

        tiled_image_iafoss = iafoss_tiles(image=255-image)["image"] # notice, my implementation works with invertd images

        ax[i, 1].imshow(255-tiled_image_iafoss)

        

        tiled_image_mine = mine_tiler(image=image)["image"]

        ax[i, 2].imshow(tiled_image_mine) #Displaying Image

    

    ax[0, 0].set_title(f"Default")

    ax[0, 1].set_title(f"Iafoss")

    ax[0, 2].set_title(f"Mine")



    plt.show() 
display_images(imgs)