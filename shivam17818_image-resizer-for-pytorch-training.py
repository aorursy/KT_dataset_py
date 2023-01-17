import os

import glob

from tqdm import tqdm

from PIL import Image,ImageFile

from joblib import Parallel,delayed

ImageFile.LOAD_TRUNCATED_IMAGES=True
def resize_images(image_path,output_folder,size):

    base_name=os.path.basename(image_path)

    outpath=os.path.join(output_folder,base_name)

    img=Image.open(image_path)

    img=img.resize(

        (size[1],size[0]),resample=Image.BILINEAR

    )

    img.save(outpath)
!mkdir /kaggle/working/train600

!mkdir /kaggle/working/test600
input_folder="../input/siim-isic-melanoma-classification/jpeg/train/"

output_folder="/kaggle/working/train600"

images=glob.glob(os.path.join(input_folder,"*.jpg"))

Parallel(n_jobs=12)(

    delayed(resize_images)(

    i,

    output_folder,

    (600,450)

) for i in tqdm(images)

)
input_folder="../input/siim-isic-melanoma-classification/jpeg/test/"

output_folder="/kaggle/working/test600"

images=glob.glob(os.path.join(input_folder,"*.jpg"))

Parallel(n_jobs=12)(

    delayed(resize_images)(

    i,

    output_folder,

    (600,450)

) for i in tqdm(images)

)