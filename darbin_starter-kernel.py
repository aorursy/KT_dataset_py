import numpy as np

import pandas as pd

import os



bulletin_dir = "../input/cityofla/CityofLA/Job Bulletins"

data_list = []

for filename in os.listdir(bulletin_dir):

    with open(bulletin_dir + "/" + filename, 'r', errors='ignore') as f:

        for line in f.readlines():

            #Insert code to parse job bulletins

            if "Open Date:" in line:

                job_bulletin_date = line.split("Open Date:")[1].split("(")[0].strip()

        data_list.append([filename, job_bulletin_date])



df = pd.DataFrame(data_list)

df.columns = ["FILE_NAME", "OPEN_DATE"]

df.head()