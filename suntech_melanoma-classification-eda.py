import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm_notebook as tqdm
import pydicom as dcm
def extract_DICOM_attributes(folder):
    images = list(os.listdir(os.path.join(PATH, folder)))
    df = pd.DataFrame()
    for image in images:
        image_name = image.split(".")[0]
        dicom_file_path = os.path.join(PATH,folder,image)
        dicom_file_dataset = dcm.read_file(dicom_file_path)
        Patient_ID = dicom_file_dataset.PatientID
        study_date = dicom_file_dataset.StudyDate
        modality = dicom_file_dataset.Modality
        age = dicom_file_dataset.PatientAge
        sex = dicom_file_dataset.PatientSex
        body_part_examined = dicom_file_dataset.BodyPartExamined
        Samples_Per_Pixel = dicom_file_dataset.SamplesPerPixel
        Bits_Allocated = dicom_file_dataset.BitsAllocated
        Bits_Stored = dicom_file_dataset.BitsStored
        High_Bit = dicom_file_dataset.HighBit
        patient_orientation = dicom_file_dataset.PatientOrientation
        photometric_interpretation = dicom_file_dataset.PhotometricInterpretation
        rows = dicom_file_dataset.Rows
        columns = dicom_file_dataset.Columns
        Pixel_Representation = dicom_file_dataset.PixelRepresentation
        Burned_In_Annotation = dicom_file_dataset.BurnedInAnnotation
        Lossy_Image_Compression = dicom_file_dataset.LossyImageCompression
        df = df.append(pd.DataFrame({'image_name': image_name, 'image_path': dicom_file_path,'Patient_ID' : Patient_ID,
                        'dcm_modality': modality,'dcm_study_date': study_date, 'age': age, 'sex': sex, 'dcm_body_part_examined': body_part_examined,
                        'Samples_Per_Pixel' : Samples_Per_Pixel, 'Bits_Allocated' : Bits_Allocated, 'Bits_Stored' : Bits_Stored, 'High_Bit' : High_Bit,'dcm_patient_orientation': patient_orientation,
                        'dcm_photometric_interpretation': photometric_interpretation,
                        'dcm_rows': rows, 'dcm_columns': columns,'Pixel_Representation' : Pixel_Representation, 'Burned_In_Annotation' : Burned_In_Annotation, 'Lossy_Image_Compression' : Lossy_Image_Compression}, index=[0]))
    return df
PATH = '/kaggle/input/siim-isic-melanoma-classification'
folder = 'train'
dicom_image_properties=extract_DICOM_attributes(folder)
dicom_image_properties.head()
dicom_image_properties.to_csv('dicom_image_properties.csv',index=False)
import pydicom as dcm
import pylab
ds=dcm.read_file('/kaggle/input/siim-isic-melanoma-classification/train/ISIC_0015719.dcm')
pylab.imshow(ds.pixel_array, cmap=pylab.cm.bone)
pylab.show()