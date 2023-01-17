import numpy as np

import pandas as pd

import time

import json

import glob

from tqdm import tqdm

import multiprocessing as mp

import cv2

import os

import mxnet as mx

import pydicom

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

from gluoncv.utils import viz
class GenerateMxNetRecordIOFile(object):

    """

    This class generates a binary file following RecordIO format from a list of images.

    Some code of this class is copied from https://raw.githubusercontent.com/apache/incubator-mxnet/master/tools/im2rec.py.

    """



    def __init__(self, n_cores = -1):

        """

        This is the class' constructor.



        Parameters

        ----------

        n_cores : integer (default = -1)

                Number of CPU cores to use to extract the data. 

                If n_cores == 1, then multiprocessing module is not used.

                If n_cores == -1, then all CPU cores are used.



        Returns

        -------

        None

        """



        self.n_cores = n_cores



        if n_cores == -1:

            self.n_cores = mp.cpu_count()



    def _extract_dcm_info(self, dcm_file_path_str):

        """

        This method extracts the content of a DCM file (image and metadata).



        Parameters

        ----------

        dcm_file_path_str: string

                Path where the DCM file is stored.

                

        Returns

        -------

        img: numpy array

                Extracted image from the DCM file.



        metadata_df: Pandas DataFrame

                DataFrame containing the extracted metadata.

        """



        # Read the DCM file

        dcm_data = pydicom.dcmread(dcm_file_path_str)



        # Extract the metadata

        metadata_dict = {"file_path": dcm_file_path_str}

        metadata_dict["storage_type"] = dcm_data.SOPClassUID

        metadata_dict["patient_name"] = dcm_data.PatientName.family_name + " " + dcm_data.PatientName.given_name

        metadata_dict["patient_id"] = dcm_data.PatientID

        metadata_dict["patient_age"] = dcm_data.PatientAge

        metadata_dict["patient_sex"] = dcm_data.PatientSex

        metadata_dict["modality"] = dcm_data.Modality

        metadata_dict["body_part_examined"] = dcm_data.BodyPartExamined

        metadata_dict["view_position"] = dcm_data.ViewPosition



        if "PixelData" in dcm_data:

            rows = int(dcm_data.Rows)

            cols = int(dcm_data.Columns)

            metadata_dict["image_height"] = rows

            metadata_dict["image_width"] = cols

            metadata_dict["image_size"] = len(dcm_data.PixelData)

        else:

            metadata_dict["image_height"] = np.nan

            metadata_dict["image_width"] = np.nan

            metadata_dict["image_size"] = np.nan



        if "PixelSpacing" in dcm_data:

            metadata_dict["pixel_spacing_x"] = dcm_data.PixelSpacing[0]

            metadata_dict["pixel_spacing_y"] = dcm_data.PixelSpacing[1]

        else:

            metadata_dict["pixel_spacing_x"] = np.nan

            metadata_dict["pixel_spacing_y"] = np.nan



        metadata_df = pd.DataFrame.from_records([metadata_dict])



        # Extract the image (in OpenCV BGR format)

        img = cv2.cvtColor(dcm_data.pixel_array, cv2.COLOR_GRAY2BGR)



        return img, metadata_df



    def _generate_lst_file(self, patients_id_lst, input_dir_path_str, output_file_path_str):

        """

        This method generates the RecordIO .lst file.

        File format is described here: https://mxnet.incubator.apache.org/versions/master/faq/recordio.html?highlight=rec%20file.



        Parameters

        ----------

        patients_id_lst : list

                List containing the IDs of the pets we want to extract data for.



        input_dir_path_str: string

                Path of the directory containing images to use.



        output_file_path_str: string

                Path where the .lst file will be written.



        Returns

        -------

        lst_file_content_lst: list of lists

                List of lists in the following format: [integer_image_index, path_to_image, label_index]

        """



        # Get images list

        images_lst = [os.path.basename(f) for f in glob.glob(input_dir_path_str + "*.*") if os.path.isfile(f)]



        # Create the file content

        lst_file_content_df = pd.DataFrame({"integer_image_index": list(range(len(images_lst))), "label_index": [0.000000 for _ in range(len(images_lst))], "path_to_image": images_lst})

        lst_file_content_df["ImageId"] = lst_file_content_df["path_to_image"].apply(lambda x: x.replace(".dcm", ""))

        lst_file_content_df = lst_file_content_df.loc[lst_file_content_df["ImageId"].isin(patients_id_lst)]

        lst_file_content_df.drop("ImageId", axis = 1, inplace = True)

        

        # Ensure the DataFrame has the correct column order

        lst_file_content_df = lst_file_content_df[["integer_image_index", "label_index", "path_to_image"]]



        # Save the .lst file

        lst_file_content_df.to_csv(output_file_path_str, sep = "\t", header = False, index = False)



        # Reshape the DataFrame

        lst_file_content_df = lst_file_content_df[["integer_image_index", "path_to_image", "label_index"]]



        # Convert the DataFrame to list

        lst_file_content_lst = lst_file_content_df.values.tolist()



        return lst_file_content_lst

    

    def _read_worker(self, input_dir_path_str, q_in, q_recordio_out, q_metadata_out):

        """

        This method gets an image, preprocess it and put in the output queue.



        Parameters

        ----------

        input_dir_path_str: string

                Path of the directory containing images to use.



        q_in: Multiprocessing Queue

                Input queue containing images names.



        q_recordio_out: Multiprocessing Queue

                Output queue containing extracted images in RecordIO format.



        q_metadata_out: Multiprocessing Queue

                Output queue containing extracted metadata.



        Returns

        -------

        None

        """



        while True:

            deq = q_in.get()

            

            if deq is None:

                break



            i, item = deq



            # Compute the DCM file full path

            dcm_path_str = os.path.join(input_dir_path_str, item[1])



            # Extract data from DCM file

            img, metadata_df = self._extract_dcm_info(dcm_path_str)

                        

            # Create one RecordIO item

            header = mx.recordio.IRHeader(0, item[2], item[0], 0)



            if img is None:

                print("Image was None for file: %s" % img_path_str)

                q_recordio_out.put((i, None, item))

            else:

                try:

                    s = mx.recordio.pack_img(header, img, quality = 95, img_fmt = ".jpg")

                    q_recordio_out.put((i, s, item))

                    q_metadata_out.put(metadata_df)

                except Exception as e:

                    print("pack_img error on file: %s" % img_path_str, e)

                    q_recordio_out.put((i, None, item))



    def _write_worker(self, q_out, output_file_prefix_str):

        """

        This method fetches a processed image from the output queue and write it to the .rec file.



        Parameters

        ----------

        q_out: Multiprocessing Queue

                Output queue containing result of processing.



        output_file_prefix_str: string

                Prefix indicating where both .rec and .idx files will be written.



        Returns

        -------

        None

        """



        pre_time = time.time()

        count = 0

        record = mx.recordio.MXIndexedRecordIO(output_file_prefix_str + ".idx", output_file_prefix_str + ".rec", "w")

        buf = {}

        more = True

        while more:

            deq = q_out.get()

            if deq is not None:

                i, s, item = deq

                buf[i] = (s, item)

            else:

                more = False

            while count in buf:

                s, item = buf[count]

                del buf[count]

                if s is not None:

                    record.write_idx(item[0], s)



                if count % 1000 == 0:

                    cur_time = time.time()

                    print("        ", count, "items saved in the RecordIO in", cur_time - pre_time, "secs")

                    pre_time = cur_time

                count += 1



    def _metadata_worker(self, q_out, output_file_prefix_str):

        """

        This method fetches a processed image from the output queue and write it to a csv file.



        Parameters

        ----------

        q_out: Multiprocessing Queue

                Output queue containing result of processing.



        output_file_prefix_str: string

                Prefix indicating where both .rec and .idx files will be written.



        Returns

        -------

        None

        """



        metadata_lst = []



        while True:

            deq = q_out.get()



            if deq is not None:

                metadata_lst.append(deq)

            else:

                # Merge all rows into one DataFrame

                metadata_df = pd.concat(metadata_lst, axis = 0)



                # Reorder columns

                metadata_df = metadata_df[["file_path", "storage_type", "patient_name", "patient_id", "patient_age", "patient_sex", "modality", "body_part_examined", "view_position", "image_height", "image_width", "image_size", "pixel_spacing_x", "pixel_spacing_y"]]



                # Generate index

                metadata_df["ImageId"] = metadata_df["file_path"].apply(lambda x: os.path.basename(x).replace(".dcm", ""))

                metadata_df.set_index("ImageId", inplace = True)



                # Save DataFrame to csv

                metadata_df.to_csv(output_file_prefix_str + ".csv")



                break



    def _generate_rec_file(self, input_dir_path_str, lst_file_content_lst, output_file_prefix_str):

        """

        This method generates the RecordIO .rec file.



        Parameters

        ----------

        input_dir_path_str: string

                Path of the directory containing images to use.



        lst_file_content_lst: list of lists

                List of lists in the following format: [integer_image_index, path_to_image, label_index]



        output_file_prefix_str: string

                Prefix indicating where both .rec and .idx files will be written.



        Returns

        -------

        None

        """



        # Create queues for Producer-Consumer paradigm

        q_in = [mp.Queue(1024) for i in range(self.n_cores - 1)]

        q_recordio_out = mp.Queue(1024)

        q_metadata_out = mp.Queue(1024)

        

        # Define the processes

        read_processes = [mp.Process(target = self._read_worker, args = (input_dir_path_str, q_in[i], q_recordio_out, q_metadata_out)) for i in range(self.n_cores - 1)]

        

        # Process images with n_cores - 1 process

        for p in read_processes:

            p.start()

            

        # Only use one process to write .rec to avoid race-condtion

        write_process = mp.Process(target = self._write_worker, args = (q_recordio_out, output_file_prefix_str))

        write_process.start()



        # Only use one process to write metadata to avoid race-condtion

        metadata_process = mp.Process(target = self._metadata_worker, args = (q_metadata_out, output_file_prefix_str))

        metadata_process.start()

        

        # Put the image list into input queue

        for i, item in enumerate(lst_file_content_lst):

            q_in[i % len(q_in)].put((i, item))

        

        for q in q_in:

            q.put(None)

            

        for p in read_processes:

            p.join()



        q_recordio_out.put(None)

        q_metadata_out.put(None)

        write_process.join()

        metadata_process.join()



    def generate_record_io_file(self, patients_id_lst, output_file_prefix, input_images_dir):

        """

        This method generates the .rec RecordIO file and its .lst associated file.

        It also apply a transformation on each image.



        Parameters

        ----------

        patients_id_lst : list

                List containing the IDs of the pets we want to extract data for.



        output_file_prefix: string

                Prefix indicating where both .rec and .idx files will be written.



        input_images_dir: string

                Path of the directory containing images to use.



        Returns

        -------

        None

        """



        st = time.time()

        print("Generating RecordIO file from images found in ", input_images_dir, "...") 



        # Generate the .lst file

        print("    Generating the .lst file as", output_file_prefix + ".lst", "...")

        lst_file_content_lst = self._generate_lst_file(patients_id_lst, input_images_dir, output_file_prefix + ".lst")



        # Generate the .rec file

        print("    Generating the .rec file as", output_file_prefix + ".rec", "...")

        self._generate_rec_file(input_images_dir, lst_file_content_lst, output_file_prefix)

        

        print("Generating RecordIO file from images... done in", round(time.time() - st, 3), "secs")
# Get number of available CPUs

cpu_count = mp.cpu_count()

print("Found", cpu_count, "CPUs")



# Get the list of patient IDs in sample

patients_ids_lst = [os.path.basename(f).replace(".dcm", "") for f in glob.glob("../input/sample images/*.dcm")]



# Start the timer

start_time = time.time()



# Load the data

gmrf = GenerateMxNetRecordIOFile(n_cores = cpu_count)

gmrf.generate_record_io_file(patients_ids_lst, "sample_recordio", "../input/sample images/")



# Stop the timer and print the exectution time

print("*** Test finished: Executed in:", time.time() - start_time, "seconds ***")
glob.glob("./*.*")
sample_data = mx.io.ImageRecordIter(

    path_imgrec = "sample_recordio.rec",

    path_imgidx = "sample_recordio.idx",

    data_shape  = (3, 1024, 1024),

    batch_size  = 2,

    shuffle     = True

)



i = 0

for batch in sample_data:

    for img in batch.data[0]:

        img = np.transpose(img.asnumpy(), (1, 2, 0))

        cv2.imwrite("sample_image_" + str(i) + ".png", img)

        i += 1
img = cv2.imread("sample_image_0.png")

new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(new_img)