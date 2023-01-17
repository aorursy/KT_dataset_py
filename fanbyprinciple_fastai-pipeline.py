import pydicom as dcm
import matplotlib.animation as animation
import random, os
import numpy as np
import torch
from fastai.vision.all import *
from fastai.medical.imaging import *
path = Path('../input/rsna-str-pulmonary-embolism-detection/')
fname = path/ "train" /"0003b3d648eb"/ "d2b2960c2bbf" / "00ac73cfc372.dcm"
fname_dcm = fname.dcmread();fname_dcm
dcm = fname.dcmread()
dcm.show(scale=False)
dcm.show()
dcm.show(cmap="spring_r", figsize=(6,6))
