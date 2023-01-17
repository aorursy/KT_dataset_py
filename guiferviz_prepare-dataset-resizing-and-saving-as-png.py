# Desired output size.

RESIZED_WIDTH, RESIZED_HEIGHT = 128, 128



OUTPUT_FORMAT = "png"



OUTPUT_DIR = "output"
import glob



import joblib



import numpy as np



import PIL



import pydicom



import tqdm
data_dir = "../input/rsna-intracranial-hemorrhage-detection"

!ls {data_dir}
train_dir = "stage_1_train_images"

train_paths = glob.glob(f"{data_dir}/{train_dir}/*.dcm")

test_dir = "stage_1_test_images"

test_paths = glob.glob(f"{data_dir}/{test_dir}/*.dcm")

len(train_paths), len(test_paths)
def get_first_of_dicom_field_as_int(x):

    if type(x) == pydicom.multival.MultiValue:

        return int(x[0])

    return int(x)



def get_id(img_dicom):

    return str(img_dicom.SOPInstanceUID)



def get_metadata_from_dicom(img_dicom):

    metadata = {

        "window_center": img_dicom.WindowCenter,

        "window_width": img_dicom.WindowWidth,

        "intercept": img_dicom.RescaleIntercept,

        "slope": img_dicom.RescaleSlope,

    }

    return {k: get_first_of_dicom_field_as_int(v) for k, v in metadata.items()}



def window_image(img, window_center, window_width, intercept, slope):

    img = img * slope + intercept

    img_min = window_center - window_width // 2

    img_max = window_center + window_width // 2

    img[img < img_min] = img_min

    img[img > img_max] = img_max

    return img 



def resize(img, new_w, new_h):

    img = PIL.Image.fromarray(img.astype(np.int8), mode="L")

    return img.resize((new_w, new_h), resample=PIL.Image.BICUBIC)



def save_img(img_pil, subfolder, name):

    img_pil.save(f"{OUTPUT_DIR}/{subfolder}/{name}.{OUTPUT_FORMAT}")



def normalize_minmax(img):

    mi, ma = img.min(), img.max()

    return (img - mi) / (ma - mi)



def prepare_image(img_path):

    img_dicom = pydicom.read_file(img_path)

    img_id = get_id(img_dicom)

    metadata = get_metadata_from_dicom(img_dicom)

    img = window_image(img_dicom.pixel_array, **metadata)

    img = normalize_minmax(img) * 255

    img_pil = resize(img, RESIZED_WIDTH, RESIZED_HEIGHT)

    return img_id, img_pil



def prepare_and_save(img_path, subfolder):

    try:

        l.error("loading eso")

        img_id, img_pil = prepare_image(img_path)

        save_img(img_pil, subfolder, img_id)

    except KeyboardInterrupt:

        # Rais interrupt exception so we can stop the cell execution

        # without shutting down the kernel.

        raise

    except:

        l.error(f"Error processing the image: {img_path}")



def prepare_images(imgs_path, subfolder):

    for i in tqdm.tqdm(imgs_path):

        prepare_and_save(i, subfolder)

import logging as l

def prepare_images_njobs(img_paths, subfolder, n_jobs=-1):

    joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(prepare_and_save)(i, subfolder) for i in tqdm.tqdm(img_paths))
!mkdir -p {OUTPUT_DIR}/{train_dir}

!mkdir -p {OUTPUT_DIR}/{test_dir}
# Running on the first 100 files of train and set!!!

prepare_images_njobs(train_paths[:100], train_dir)

prepare_images_njobs(test_paths[:100], test_dir)
train_output_path = glob.glob(f"{OUTPUT_DIR}/{train_dir}/*")
img_path = train_output_path[0]

PIL.Image.open(img_path)