!pip install piexif
import json

import pydicom

import piexif

import csv

import numpy as np

from PIL import Image

from pathlib import Path

import matplotlib.pyplot as plt

%matplotlib inline
data_root = Path('..') / "input" / "sample images"
annotations = {}

annotation_path = data_root / 'train-rle-sample.csv'

with annotation_path.open(encoding="utf_8_sig") as fp:

    for image_id, encoded_pixels in csv.reader(fp):

        annotations.setdefault(image_id, []).append(encoded_pixels)
def get_attr(dcm, masks):

    attr = {

        "StudyInstanceUID": dcm.StudyInstanceUID,

        "SeriesInstanceUID": dcm.SeriesInstanceUID,

        "SOPInstanceUID": dcm.SOPInstanceUID,

        "PatientSex": dcm.PatientSex,

        "PatientAge": int(dcm.PatientAge),

        "ViewPosition": dcm.ViewPosition,

        "PixelSpacing": [float(s) for s in dcm.PixelSpacing],

    }

    if masks is not None:

        attr['Masks'] = masks

        

    return attr
def save_as_jpg(dst_path, pixel_array, attr):

    img = Image.fromarray(pixel_array)

    exif_ifd = {piexif.ExifIFD.MakerNote: json.dumps(attr).encode("ascii")}

    exif = {"Exif": exif_ifd}

    img.save(dst_path, format='jpeg', exif=piexif.dump(exif))    
for p in data_root.glob('*.dcm'):

    dcm = pydicom.dcmread(str(p))

    masks = annotations.get(p.stem)

    attr = get_attr(dcm, masks)

    dst_path = '{}.jpg'.format(p.stem)

    save_as_jpg(dst_path, dcm.pixel_array, attr)
def rle2mask(rle, width, height):

    mask= np.zeros(width* height)

    if rle != '-1':

        array = np.asarray([int(x) for x in rle.split()])

        starts = array[0::2]

        lengths = array[1::2]



        current_position = 0

        for index, start in enumerate(starts):

            current_position += start

            mask[current_position:current_position+lengths[index]] = 255

            current_position += lengths[index]



    return mask.reshape(height, width).T
def read_jpg(path):

    img = Image.open(path)

    makernote_bytes = piexif.load(img.info["exif"])["Exif"][piexif.ExifIFD.MakerNote]

    attr = json.loads(makernote_bytes.decode("ascii"))



    masks = None

    if 'Masks' in attr:

        masks = [rle2mask(encoded_pixels, img.width, img.height) for encoded_pixels in attr['Masks']]

        del attr['Masks']



    return np.asarray(img), attr, masks
for i, p in enumerate(Path('.').glob('*.jpg')):

    pixel_array, attr, masks = read_jpg(p)

    plt.figure(i)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(pixel_array)

    ax2.imshow(masks[0])

    