# All imports

import cv2

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import skimage.io

from skimage import morphology

import time

import os

from multiprocessing import Pool

from tqdm.notebook import tqdm

from operator import itemgetter 

from itertools import chain

import gc



class PandaProcess():

    def __init__(self, pipeline=1, image_location="", sensitivity=3500, lw_lvl=-1, hi_lvl=-2, size=None):

        self.img_loc = image_location

        self.sensitivity = sensitivity

        self.lw_lvl = lw_lvl

        self.lw_slide = None

        self.lw_w_crop = None

        self.lw_tiss_cnts = None

        self.lw_tiss_slide = None

        self.lw_tiss_only = None

        self.prod = None



        self.hi_lvl = hi_lvl

        self.hi_slide = None



        self.size = size



        options = {

            1:self.pipe1

            }

        

        if pipeline in options.keys(): options[pipeline]()

        

    def read_slide(self, location, level):

        return skimage.io.MultiImage(location)[level]



    def bkgrnd_cut(self, in_slide, bkgrnd=255):

        # Remove all rows of color

        row_not_blank = [row.all() for row in ~np.all(in_slide == [bkgrnd]*3, axis=1)]

        slide = in_slide[row_not_blank, :]

        # Remove all columns of color

        col_not_blank = [col.all() for col in ~np.all(slide == [bkgrnd]*3, axis=0)]

        out_slide = slide[:, col_not_blank]

        return out_slide



    def otsu_filter(self, channel, gaussian_blur=True):

        """Otsu filter."""

        if gaussian_blur:

            channel = cv2.GaussianBlur(channel, (5, 5), 0)

        channel = channel.reshape((channel.shape[0], channel.shape[1]))



        return cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]



    def get_tissue_contours(self, in_slide, sensitivity):

        # For timing

        self.times_tiss_detc = {}

        self.times_tiss_detc["start"] = time.time()



        # Convert from RGB to HSV color space

        slide_hsv = cv2.cvtColor(in_slide, cv2.COLOR_BGR2HSV)

        self.times_tiss_detc["Color Convert"] = time.time()



        # Compute optimal threshold values in each channel using Otsu algorithm

        _, saturation, _ = np.split(slide_hsv, 3, axis=2)

        mask = self.otsu_filter(saturation, gaussian_blur=True)

        self.times_tiss_detc["Otsu"] = time.time()



        # Make mask boolean

        mask = mask != 0



        # Perform Morphology

        mask = morphology.remove_small_holes(mask, area_threshold=sensitivity)

        self.times_tiss_detc["Morph 1"] = time.time()

        mask = morphology.remove_small_objects(mask, min_size=sensitivity)

        self.times_tiss_detc["Morph 2"] = time.time()



        # Get Contours

        mask = mask.astype(np.uint8)

        tissue_contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        self.times_tiss_detc["Contours"] = time.time()

        return tissue_contours



    def draw_tissue(self, in_slide, tissue_contours, polygon_type, line_thickness=None):

        for poly in tissue_contours:

            if polygon_type == "line":

                src_img = cv2.polylines(in_slide, [poly], True, [0, 255, 0], line_thickness)

            elif polygon_type == "area":

                if line_thickness is not None:

                    warnings.warn(

                        '"line_thickness" is only used if ' + '"polygon_type" is "line".'

                    )



                src_img = cv2.fillPoly(mask, [poly], tissue_color)

            else:

                raise ValueError('Accepted "polygon_type" values are "line" or "area".')



        return src_img



    def tissue_cutout(self, in_slide, tissue_contours):

        # https://stackoverflow.com/a/28759496

        # Get intermediate slide

        base_slide_mask = np.zeros(in_slide.shape[:2])

        crop_mask = np.zeros_like(

            base_slide_mask

        )  # Create mask where white is what we want, black otherwise

        cv2.drawContours(

            crop_mask, tissue_contours, -1, 255, -1

        )  # Draw filled contour in mask

        tissue_only = np.zeros_like(

            in_slide

        )  # Extract out the object and place into output image

        tissue_only[crop_mask == 255] = in_slide[crop_mask == 255]

        return tissue_only



    def getSubImage(self, src_img, rect, scale):

        rect = (

            (rect[0][0] * scale, rect[0][1] * scale),

            (rect[1][0] * scale, rect[1][1] * scale),

            rect[2],

        )

        width = int(rect[1][0])

        height = int(rect[1][1])

        box = cv2.boxPoints(rect)



        src_pts = box.astype("float32")

        dst_pts = np.array(

            [[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]],

            dtype="float32",

        )

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        warped = cv2.warpPerspective(src_img, M, (width, height))

        return warped



    def min_rect_crop(self, in_slide, tissue_cnts, scale):

        all_bounding_rect = cv2.minAreaRect(np.concatenate(tissue_cnts))

        return self.getSubImage(in_slide, all_bounding_rect, scale)



    def pipe1(self):

        # For timing

        self.times_pipe = {}

        self.times_pipe["Start"] = time.time()

        self.lw_slide = self.read_slide(self.img_loc, self.lw_lvl)

        self.times_pipe["Read"] = time.time()

        self.lw_tiss_cnts = self.get_tissue_contours(self.lw_slide, self.sensitivity)

        if len(self.lw_tiss_cnts) == 0: return

        self.times_pipe["Get Countours"] = time.time()

        self.lw_tiss_cut = self.tissue_cutout(self.lw_slide, self.lw_tiss_cnts)

        self.times_pipe["Cut Tissue"] = time.time()

        self.lw_rect_crop = self.min_rect_crop(self.lw_tiss_cut, self.lw_tiss_cnts, 1)

        self.times_pipe["Contour Rect"] = time.time()

        self.lw_w_crop = self.bkgrnd_cut(self.lw_rect_crop, bkgrnd=0)

        if self.lw_w_crop.size==0: return

        self.times_pipe["Final: Cut Background"] = time.time()

        self.prod = self.lw_w_crop

        return





class PandaDataStore(PandaProcess):

    def __init__(self, pct_store=.001, size=[512, 512], run=False):

        self.save_dir = "/kaggle/train_images/"

        os.makedirs(self.save_dir, exist_ok=True)

        self.slide_dir = "../input/prostate-cancer-grade-assessment/train_images/"

        self.mask_dir = "../input/prostate-cancer-grade-assessment/train_label_masks/"

        self.size = size

        self.train_data_df = pd.read_csv("../input/prostate-cancer-grade-assessment/train.csv")

        N_to_process = int(len(self.train_data_df) * pct_store)

        sample_df = self.categorical_sample(self.train_data_df, "isup_grade", N_to_process)

        self.image_ids = list(sample_df.image_id)

        if run: self.run_store()



    def run_store(self):

        with Pool(processes=4) as pool:

            prod_bool = list(

                pool.imap(self.img_prod_save, self.image_ids), total=len(self.image_ids)

            )

        return prod_bool



    def categorical_sample(self, df, cat_col, N):

        group = df.groupby(cat_col, group_keys=False)

        sample_df = group.apply(lambda g: g.sample(int(N * len(g) / len(df))))

        return sample_df



    def img_prod_save(self, image_id):

        load_path = self.slide_dir + image_id + '.tiff'

        save_path = self.save_dir + image_id + '.png'

        pprocess = PandaProcess(start_type="pipe1", image_location=load_path, size=self.size)

        if pprocess.prod is None: return 0

        cv2.imwrite(save_path, pprocess.prod)

        return 1

    

class PandaMetadata(PandaDataStore):

    def __init__(self, pct_review=.001, mode="load", meta_slide_df="", meta_cnts_df=""):

        PandaDataStore.__init__(self, pct_store=pct_review,size=None, run=False)

        if mode=="run" or mode=="save":

            results = self.run_meta()

            meta_slide, meta_cnts = zip(*results)

            self.meta_slide_df = pd.DataFrame(meta_slide)

            self.meta_cnts_df = pd.DataFrame(list(chain.from_iterable(meta_cnts)))

            self.meta_slide_df = pd.merge(self.train_data_df,self.meta_slide_df, on="image_id", how="right")

            self.meta_cnts_df = pd.merge(self.train_data_df,self.meta_cnts_df, on="image_id", how="right")

            

            if mode == "save":

                self.meta_slide_df.to_csv("PANDA_Tissue_Metadata_Slides.csv", index=False)

                self.meta_cnts_df.to_csv("PANDA_Tissue_Metadata_Contours.csv", index=False)

                

        elif mode=="load": 

            self.df = pd.read_csv(df_loc)

            self.meta_slide_df = pd.read_csv(meta_slide_df)

            self.meta_cnts_df = pd.read_csv(meta_cnts_df)

            

        else:

            print("Invalid Option")

            self.df = None

        

    def get_meta(self, image_id):

        

        load_path = self.slide_dir + image_id + '.tiff'

        pprocess = PandaProcess(pipeline=1, image_location=load_path, size=self.size, sensitivity=3000)

        

        if pprocess.prod is None: return {}, []

        

        #Base/Production image metadata

        meta_image = {

                "image_id": image_id,

                "base_shape": pprocess.lw_slide.shape,

                "base_area": np.prod(pprocess.lw_slide.shape[:-1]),

                "base_horz": pprocess.lw_slide.shape[1] > pprocess.lw_slide.shape[0],

                "base_ratio": pprocess.lw_slide.shape[1] / pprocess.lw_slide.shape[0],

                "prod_shape": pprocess.prod.shape,

                "prod_area": np.prod(pprocess.prod.shape[:-1]),

                "prod_horz": pprocess.prod.shape[1] > pprocess.prod.shape[0],

                "prod_ratio": pprocess.prod.shape[1] / pprocess.prod.shape[0],

                "prod_rect" : cv2.minAreaRect(np.concatenate(pprocess.lw_tiss_cnts)),

                "cnt_count":len(pprocess.lw_tiss_cnts)

               }

        

        #Timing Metadata

        meta_timimg = {

                "pipeline_times": pprocess.times_pipe,

                "tissue_detect_times": pprocess.times_tiss_detc

                }

        

        #Contours Metadata (one row for each contour)

        meta_cnts_list = []

        for i,cnt in enumerate(pprocess.lw_tiss_cnts):

            rect = cv2.minAreaRect(cnt)

            cnt_area = cv2.contourArea(cnt)

            cnt_rect_area = np.prod(rect[1])

            meta_cnts = {

                    "image_id":image_id,

                    "cnt_id": i,

                    "cnt_area":cnt_area,

                    "cnt_rects":rect,

                    "cnt_react_shape":rect[1],

                    "cnt_rect_area":np.prod(rect[1]),

                    "cnt_horz":rect[1][1] > rect[1][0],

                    "cnt_ratio":rect[1][1] / rect[1][0],

                    "cnt_tissue_pct": cnt_area / cnt_rect_area

                    }

            meta_cnts_list.append(meta_cnts)

            

        cnt_areas = list(map(itemgetter('cnt_area'), meta_cnts_list))

        meta_image["cnt_area_total"] = sum(cnt_areas)

        

        cnt_rect_areas = list(map(itemgetter('cnt_rect_area'), meta_cnts_list))

        meta_image["cnt_rect_area_total"] = sum(cnt_rect_areas)

        

        meta_image["pct_prod_base"] = meta_image["prod_area"] / meta_image["base_area"]

        meta_image["pct_tissue_base"] = meta_image["cnt_area_total"] / meta_image["base_area"]

        meta_image["pct_tissue_prod"] = meta_image["cnt_area_total"] / meta_image["prod_area"]

        meta_image["pct_tissue_rect"] = meta_image["cnt_area_total"] / meta_image["cnt_rect_area_total"]

        meta_image["pct_tissue_base_prod"] = meta_image["pct_tissue_base"] / meta_image["pct_tissue_prod"]

        meta_image["pct_tissue_prod_rect"] = meta_image["pct_tissue_prod"] / meta_image["pct_tissue_rect"]

        

        #Clean up

        del pprocess

        gc.collect()

        

        return [meta_image, meta_cnts_list]

    

    def run_meta(self):

        with Pool(processes=4) as pool:

            results = list(pool.imap(self.get_meta, self.image_ids))

        return results 

    

data_store = PandaMetadata(.001, mode="save")
import pandas as pd

import plotly.express as px

import ast



def categorical_sample(df, cat_col, pct):

    N = int(len(df) * pct)

    group = df.groupby(cat_col, group_keys=False)

    sample_df = group.apply(lambda g: g.sample(int(N * len(g) / len(df))))

    return sample_df



#Read in dataset from saved data

dataset_loc = "/kaggle/input/panda-tissue-metadata/PANDA_Tissue_Metadata_Slides.csv"

td_meta_df = pd.read_csv(dataset_loc)



#Can also get directly from the saved run above

#test_loc = "/kaggle/working/PANDA_Tissue_Metadata_Slides.csv"

#td_meta_df = pd.read_csv(test_loc)



#Drop any columns without base shapes

td_meta_df = td_meta_df.dropna(subset=["base_shape"])



#Set ISUP to int

td_meta_df["isup_grade"] = td_meta_df["isup_grade"].astype(int)



#Sample based on isup for memory saving reasons

td_meta_df = categorical_sample(td_meta_df, "isup_grade", .50)



#Take all columns that are suppose to be lists and transform

touple_cols = ["base_shape", "prod_shape"]



def str_to_list(col):

    return col.str.strip("[]").str.split(",")



def tpl_to_list(col):

    return col.str.strip("()").str.split(",")



#Convert touple cols to touple types

td_meta_df[touple_cols] = td_meta_df[touple_cols].apply(tpl_to_list, axis=1)

td_meta_df["prod_rect"] = td_meta_df["prod_rect"].apply(ast.literal_eval)



pd.set_option('display.max_columns', len(td_meta_df.columns))

td_meta_df.head(1)
#Read in dataset

dataset_loc_cnts = "/kaggle/input/panda-tissue-metadata/PANDA_Tissue_Metadata_Contours.csv"

cnts_meta_df = pd.read_csv(dataset_loc_cnts)



#Can also get directly from the saved run above

#test_loc_cnts = "/kaggle/working/PANDA_Tissue_Metadata_Contours.csv"

#cnts_meta_df = pd.read_csv(test_loc_cnts)



#Sample for data sake

cnts_meta_df = categorical_sample(cnts_meta_df, "isup_grade", .50)



#Take all columns that are suppose to be lists and transform

touple_cols_cnts = ["cnt_react_shape"]



#Convert touple cols to touple types

cnts_meta_df[touple_cols_cnts] = cnts_meta_df[touple_cols_cnts].apply(tpl_to_list, axis=1)

cnts_meta_df["cnt_rects"] = cnts_meta_df["cnt_rects"].apply(ast.literal_eval)

cnts_meta_df.head()
slide_dir = "../input/prostate-cancer-grade-assessment/train_images/"

def show_cnts(ids):

    for image_id in ids:

        #Get CMBRs

        tissue_CMBRs = cnts_meta_df[cnts_meta_df["image_id"] == image_id].cnt_rects

        slide_CMBR = list(td_meta_df[td_meta_df["image_id"] == image_id].prod_rect)

        #Get base slide

        slide_loc = f"{slide_dir}{image_id}.tiff"

        base_slide = skimage.io.MultiImage(slide_loc)[-1]



        if len(slide_CMBR) > 0:

            box = cv2.boxPoints(slide_CMBR[0])

            box = np.int0(box)

            cv2.drawContours(base_slide,[box],0,(255,255,0),10)



        for rect in tissue_CMBRs:

            box = cv2.boxPoints(rect)

            box = np.int0(box)

            cv2.drawContours(base_slide,[box],0,(0,255,255),5)



        print(f"Slide: {image_id} | Tissue Contours: {len(tissue_CMBRs)}")

        plt.imshow(base_slide)

        plt.show()

    return 



#train_data_df = pd.read_csv("../input/prostate-cancer-grade-assessment/train.csv")

sample_ids = list(td_meta_df.sample(5).image_id)

show_cnts(sample_ids)
td_meta_df["base_height"] = td_meta_df.base_shape.map(lambda x: int(x[0]))

td_meta_df["base_width"] = td_meta_df.base_shape.map(lambda x: int(x[1]))





fig = px.scatter(td_meta_df, x="base_width", y="base_height", color="isup_grade",

                 hover_data=["image_id", 'gleason_score'], trendline="ols")



fig.add_shape(

        # Line Diagonal

            type="line",

            x0=0,

            y0=0,

            x1=td_meta_df["base_width"].max(),

            y1=td_meta_df["base_height"].max(),

            line=dict(

                color="MediumPurple",

                width=4,

                dash="dot",

            )

)



fig.show()
td_meta_df["prod_height"] = td_meta_df.prod_shape.map(lambda x: int(x[0]))

td_meta_df["prod_width"] = td_meta_df.prod_shape.map(lambda x: int(x[1]))



fig = px.scatter(td_meta_df, x="prod_width", y="prod_height", color="isup_grade",

                 hover_data=["image_id", 'gleason_score'], trendline="ols")



fig.add_shape(

        # Line Diagonal

            type="line",

            x0=0,

            y0=0,

            x1=td_meta_df["prod_width"].max(),

            y1=td_meta_df["prod_height"].max(),

            line=dict(

                color="MediumPurple",

                width=4,

                dash="dot",

            )

)



fig.show()
cnts_meta_df["cnt_height"] = cnts_meta_df.cnt_react_shape.map(lambda x: float(x[0]))

cnts_meta_df["cnt_width"] = cnts_meta_df.cnt_react_shape.map(lambda x: float(x[1]))



fig = px.scatter(cnts_meta_df, x="cnt_width", y="cnt_height", color="isup_grade",

                 hover_data=["image_id", 'gleason_score'], trendline="ols")



fig.add_shape(

        # Line Diagonal

            type="line",

            x0=0,

            y0=0,

            x1=cnts_meta_df["cnt_width"].max(),

            y1=cnts_meta_df["cnt_height"].max(),

            line=dict(

                color="MediumPurple",

                width=4,

                dash="dot",

            )

)



fig.show()
import plotly.express as px

import plotly.graph_objects as go



def get_px_to_sq(shape):

    #Ensure that values are in float

    shape = [float(x) for x in shape]

    #Only keep the top two values (third would be channel values)

    shape = shape if len(shape) < 2 else shape[:2]

    #Sort values for arithmatic

    shape.sort()

    #Get base area

    base_area = np.prod(shape)

    #Find required area to amke sqaure

    px_req = (shape[1] - shape[0]) * shape[1]

    #Get percentage or original

    px_to_sq = px_req / base_area

    return px_to_sq



cnts_meta_df["cnts_px_to_sq"] = cnts_meta_df.cnt_react_shape.apply(get_px_to_sq)

td_meta_df["base_px_to_sq"] = td_meta_df.base_shape.apply(get_px_to_sq)

td_meta_df["prod_px_to_sq"] = td_meta_df.prod_shape.apply(get_px_to_sq)

cnt_px_to_sq = cnts_meta_df[["image_id", "cnts_px_to_sq"]]

slide_px_to_sq = td_meta_df[["image_id", "base_px_to_sq", "prod_px_to_sq"]]

to_graph = slide_px_to_sq.merge(cnt_px_to_sq, on="image_id", how="outer").melt("image_id")



# create the bins

graph_top = 2000



fig = px.histogram(to_graph, x="value", color="variable", marginal="rug", # can be `box`, `violin`

                         hover_data=["image_id"])







fig.add_trace(go.Scatter(

    x=[td_meta_df["base_px_to_sq"].mean()]*20,

    y=np.linspace(0,graph_top,20),

    mode="lines",

    name="base_px_to_sq mean",

    hoverinfo = "x",

    textposition="bottom center",

    line=dict(

                color=px.colors.qualitative.Plotly[0],

                width=4,

                dash="dot",

            )

))



fig.add_trace(go.Scatter(

    x=[td_meta_df["prod_px_to_sq"].mean()]*20,

    y=np.linspace(0,graph_top,20),

    mode="lines",

    name="prod_px_to_sq mean",

    hoverinfo = "x",

    textposition="bottom center",

    line=dict(

                color=px.colors.qualitative.Plotly[1],

                width=4,

                dash="dot",

            )

))



   

fig.add_trace(go.Scatter(

    x=[cnts_meta_df["cnts_px_to_sq"].mean()]*20,

    y=np.linspace(0,graph_top,20),

    mode="lines",

    name="cnts_px_to_sq mean",

    hoverinfo = "x",

    textposition="bottom center",

    line=dict(

                color=px.colors.qualitative.Plotly[2],

                width=4,

                dash="dot",

            )

))

    



fig.show()
cnt_tiss_pct = cnts_meta_df[["image_id", "cnt_tissue_pct"]]

slide_tiss_pct = td_meta_df[["image_id", "pct_tissue_base", "pct_tissue_prod"]]

to_graph = slide_tiss_pct.merge(cnt_tiss_pct, on="image_id", how="outer").melt("image_id")



graph_top = 1500



fig = px.histogram(to_graph, x="value", color="variable", marginal="rug", # can be `box`, `violin`

                         hover_data=["image_id"])







fig.add_trace(go.Scatter(

    x=[td_meta_df["pct_tissue_base"].mean()]*20,

    y=np.linspace(0,graph_top,20),

    mode="lines",

    name="pct_tissue_base mean",

    hoverinfo = "x",

    textposition="bottom center",

    line=dict(

                color=px.colors.qualitative.Plotly[0],

                width=4,

                dash="dot",

            )

))



fig.add_trace(go.Scatter(

    x=[td_meta_df["pct_tissue_prod"].mean()]*20,

    y=np.linspace(0,graph_top,20),

    mode="lines",

    name="pct_tissue_prod mean",

    hoverinfo = "x",

    textposition="bottom center",

    line=dict(

                color=px.colors.qualitative.Plotly[1],

                width=4,

                dash="dot",

            )

))



   

fig.add_trace(go.Scatter(

    x=[cnts_meta_df["cnt_tissue_pct"].mean()]*20,

    y=np.linspace(0,graph_top,20),

    mode="lines",

    name="cnt_tissue_pct mean",

    hoverinfo = "x",

    textposition="bottom center",

    line=dict(

                color=px.colors.qualitative.Plotly[2],

                width=4,

                dash="dot",

            )

))

    



fig.show()
to_graph = td_meta_df.cnt_count.value_counts().reset_index()

to_graph.columns = ["Number of Countors", "Count"]

fig = px.pie(to_graph, values='Count', names="Number of Countors")

fig.show()