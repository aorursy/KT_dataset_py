import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import glob

import time

import shutil

import os

import stat, sys 
def return_doc_index(subject, w2v_model):

    

    documents = np.where(corona_data['text_body'].str.contains(subject) ,corona_data['text_body'].index, 0)

    index = np.where(documents != 0)

    return index







def print_document_title(list_of_index):

    

    for index in list_of_index:

        for i in index:

            print('-----------\n')

            report = corona_data.iloc[i]

            print(report['title'])

            print(f"Paper Index {i}")

            print('-------------\n')

    

    







def print_document_body(list_of_index):

    

    for index in list_of_index:

        for i in index:

            print('-----------\n')

            report = corona_data.iloc[i]

            print(report['text_body'])

            print(f"Paper Index {i}")

            print('-------------\n')







def create_document_directory():

    

    current_directory = os.getcwd()

    current_time = time.strftime("%Y-%m-%d @ %H:%M:%S")

    final_directory = os.path.join(current_directory, r'Saved Documents: ' + current_time)

    if not os.path.exists(final_directory):

       os.makedirs(final_directory)

    folder_name = f"Saved Documents: {current_time}"

    

    return folder_name





def save_documents(list_of_index, corona_data, new_image_folder):

    

    dir_ = "../input/CORD-19-research-challenge/2020-03-13"

    output_dir = "../output/kaggle/working"

    

    for i in list_of_index:

        for ind in i:

            paper = corona_data.iloc[ind]

            filename = paper['doc_id']

            source = paper['source']

            ext = ".json"

            

            source_folder = f"/{source}/{source}/"

            output_dir = f"/{new_image_folder}"

            

            save_doc = shutil.copy2(os.path.join(dir_ + source_folder, filename + ext),

                                    os.path.join(output_dir, filename + ext))

            

corona_data = pd.read_csv("../input/kaggle-covid19/kaggle_covid-19_open_csv_format.csv")

corona_data = corona_data.drop(columns=['abstract'])

corona_data = corona_data.fillna("Unknown")

doc_folder = {"risk": return_doc_index("risk", corona_data),



              "preg": return_doc_index("pregnant", corona_data),



               "smoking": return_doc_index("smoking", corona_data),



               "co_infection": return_doc_index("co infection", corona_data),



                "neonates": return_doc_index("neonates", corona_data),



               "transmission": return_doc_index("transmission dynamics", corona_data),



                "high_risk": return_doc_index("high-risk patient", corona_data)

             }
print(f"Number of Documents that Mention Risk: {len(doc_folder['risk'][0])}")



print(f"Number of Documents that Mention Pregnancy: {len(doc_folder['preg'][0])}")



print(f"Number of Documents that Mention Smoking: {len(doc_folder['smoking'][0])}")



print(f"Number of Documents that Mention Neonates: {len(doc_folder['neonates'][0])}")



print(f"Number of Documents that Mention Transmission Dynamics: {len(doc_folder['transmission'][0])}")



print(f"Number of Documents that Mention High Risk Patients: {len(doc_folder['high_risk'][0])}")
new_image_folder = create_document_directory()



# save_docs = save_documents(doc_folder['high_risk'], corona_data, new_image_folder)
