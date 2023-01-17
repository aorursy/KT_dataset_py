import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.graph_objects as go
import cv2
from tqdm import tqdm_notebook as tqdm
%matplotlib inline
melanoma_classification_dcm_img_info = pd.read_csv('/kaggle/input/image-properties/dicom_image_properties.csv')
melanoma_classification_dcm_img_info.info()
melanoma_classification_dcm_img_info['age'] = melanoma_classification_dcm_img_info['age'].apply(lambda x : x[:-1])
melanoma_classification_train_set = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
melanoma_classification_dcm_img_info.head()
melanoma_classification_dcm_img_info[melanoma_classification_dcm_img_info['image_name']=='ISIC_2637011']
melanoma_classification_train_set[melanoma_classification_train_set['image_name']=='ISIC_2637011']
