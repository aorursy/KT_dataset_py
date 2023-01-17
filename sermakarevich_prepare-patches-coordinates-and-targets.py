import os

from typing import Optional, List, Tuple

from multiprocessing import Pool



import numpy as np

import openslide

import pandas as pd

from tqdm.auto import tqdm
class PatchSlicer:

    height: int

    width: int



    def __init__(self, path_to_wsi: str, step_size: int = 256):

        self.path_to_wsi = path_to_wsi

        self.step_size = step_size

        self.patch_size = (self.step_size, self.step_size)

        self.x = 0

        self.y = 0



    def patch_generator(self):

        with openslide.OpenSlide(self.path_to_wsi) as wsi:

            self.width = wsi.level_dimensions[0][0]

            self.height = wsi.level_dimensions[0][1]

            while self.y + self.step_size < self.height:

                while self.x + self.step_size < self.width:

                    coords = (self.x, self.y)

                    yield wsi.read_region(coords, 0, self.patch_size), coords

                    self.x += self.step_size

                self.x = 0

                self.y += self.step_size





class GridSearcher:

    def __init__(

            self,

            paths_to_wsis: List[str],

            step_size: Optional[int] = 512,

            white_area_score: Optional[int] = 240,

            max_white_area_mean: Optional[int] = 0.9,

            path_to_masks: Optional[str] = None,

    ):

        self.paths_to_wsis = paths_to_wsis

        self.path_to_masks = path_to_masks

        self.step_size = step_size

        self.white_area_score = white_area_score

        self.max_white_area_mean = max_white_area_mean



    def make_grid_mp(self):

        cpu_count = os.cpu_count()

        with Pool(processes=cpu_count) as p:

            dfs = list(tqdm(p.imap(self.make_grid, self.paths_to_wsis), total=len(self.paths_to_wsis)))

        df = pd.concat(dfs)

        return df



    def make_grid(self, path_to_wsi):

        path_to_mask = None

        wsi_filename = path_to_wsi.split('/')[-1]

        if self.path_to_masks is not None:

            mask_filename = wsi_filename.replace(".", "_mask.")

            path_to_mask = os.path.join(self.path_to_masks, mask_filename)

            if not os.path.exists(path_to_mask):

                path_to_mask = None

        if path_to_mask:

            df = self.make_grid_from_wsi_and_mask(

                path_to_wsi=path_to_wsi,

                path_to_mask=path_to_mask,

                step_size=self.step_size,

                white_area_score=self.white_area_score,

                max_white_area_mean=self.max_white_area_mean,

                wsi_filename=wsi_filename,

            )

        else:

            df = self.make_grid_from_wsi_only(

                path_to_wsi=path_to_wsi,

                step_size=self.step_size,

                white_area_score=self.white_area_score,

                max_white_area_mean=self.max_white_area_mean,

                wsi_filename=wsi_filename

            )

        return df



    @staticmethod

    def make_grid_from_wsi_only(

            path_to_wsi: str,

            step_size: int,

            white_area_score: int,

            max_white_area_mean: int,

            wsi_filename: str,

    ):

        patch_slicer = PatchSlicer(path_to_wsi=path_to_wsi, step_size=step_size)

        patches_generator = patch_slicer.patch_generator()

        good_patches_coords = []

        for patch, coords in patches_generator:

            patch_array = np.array(patch)[:, :, :3]

            patch_mean_pixel_value = patch_array.mean(axis=2)

            if (patch_mean_pixel_value > white_area_score).mean() > max_white_area_mean:

                continue

            else:

                good_patches_coords.extend([coords])

        df = GridSearcher.patches_coords_and_targets_to_df(

            good_patches_coords=good_patches_coords,

            wsi_filename=wsi_filename,

        )

        return df



    @staticmethod

    def make_grid_from_wsi_and_mask(

            path_to_wsi: str,

            path_to_mask: str,

            step_size: int,

            white_area_score: int,

            max_white_area_mean: int,

            wsi_filename: str

    ):

        wsi_patch_slicer = PatchSlicer(path_to_wsi=path_to_wsi, step_size=step_size)

        wsi_patches_generator = wsi_patch_slicer.patch_generator()

        mask_patch_slicer = PatchSlicer(path_to_wsi=path_to_mask, step_size=step_size)

        mask_patches_generator = mask_patch_slicer.patch_generator()

        good_patches_coords = []

        targets = []

        for (wsi_patch, wsi_coords), (mask_patch, _) in zip(wsi_patches_generator, mask_patches_generator):

            patch_array = np.array(wsi_patch)[:, :, :3]

            mask_array = np.array(mask_patch)[:, :, 0]

            non_zero_mask_array = (mask_array > 0).astype(int)

            if non_zero_mask_array.sum():

                patch_mean_pixel_value = patch_array.mean(axis=2)

                if (patch_mean_pixel_value > white_area_score).mean() > max_white_area_mean:

                    continue

                good_patches_coords.extend([wsi_coords])

                unique_mask_values = np.unique(mask_array)

                targets.extend([unique_mask_values])

        if not targets:

            targets = None

        df = GridSearcher.patches_coords_and_targets_to_df(

            good_patches_coords=good_patches_coords,

            targets=targets,

            wsi_filename=wsi_filename,

        )

        return df



    @staticmethod

    def patches_coords_and_targets_to_df(

            good_patches_coords: List[Tuple[int, int]],

            wsi_filename: str,

            targets: Optional[List[List[int]]] = None,

    ):

        df = pd.DataFrame({"patches_coords": good_patches_coords})

        df["targets"] = targets

        df['wsi_filename'] = wsi_filename

        return df
path_to_wsis = "../input/prostate-cancer-grade-assessment/train_images/"

path_to_masks = "../input/prostate-cancer-grade-assessment/train_label_masks/"



wsi_filenames = os.listdir(path_to_wsis)





gs = GridSearcher(

    paths_to_wsis = [os.path.join(path_to_wsis, wsi_filename) for wsi_filename in wsi_filenames[:8]],

    path_to_masks = path_to_masks

)

df = gs.make_grid_mp()

df.head()
df['targets'].astype(str).value_counts()