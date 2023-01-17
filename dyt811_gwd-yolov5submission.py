# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        pass

        #print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import os, sys

import glob

from pathlib import Path

from shutil import copytree, rmtree

import pickle



# Default Path

path_input = Path("/kaggle/input/")

path_output = Path("/kaggle/working/")



# Source Data:

path_data = path_input / "global-wheat-detection"



# Source Data Packages

path_source_code = path_input / "gwdpipeline01"

path_target_code = path_output / "gwd"



# Source Dependency Packages

path_source_dep = path_input / "gwd-pip-dep"

path_target_dep = path_output / "gwd_dep"



# =================

# Data Preparation

# =================



#######################################

# Copy from INPUT DATSET to WORKING DIR

# Fom src.submissions.bengaliai import kaggle_project_submission

# This does NOT include the /depends folder.

if Path(path_target_code).exists():

    rmtree(path_target_code)

    print("Target Code cleaned.")

copytree(src=path_source_code, dst=path_target_code)



if Path(path_target_dep).exists():

    rmtree(path_target_dep)

    print("Target Dep cleaned.")

copytree(src=path_source_dep, dst=path_target_dep)

    

import os

for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:        

        #print(os.path.join(dirname, filename))

        pass



# =======================

# Dependency Preparation

# =======================



class SimpleObj:

    """

    This class data structure is required to decompress the entry and bin data.

    """



    def __init__(self, fname, bdata):

        self.name = fname

        self.bdata = bdata

        return







print(os.getcwd())



try:

    import yacs

except ImportError as e:

    os.chdir(path_target_dep)

    # The file that contain all the binary of all possible dependencies.

    picklefile = 'dill.pkl'



    # unpack pickle

    installers = []

    with open(picklefile, 'rb') as pf:

        installers = pickle.load(pf)

        #print(installers)



    for index, i in enumerate(installers):

        #print(f"Writing Dependencies {index} ")

        with open(i.name, 'wb') as p:

            p.write(i.bdata)



    # install

    os.system('pip install -r requirements.txt --no-index --find-links .')

print("Finished Installing all dependencies.")
import torch

import torchvision

print(torch.__version__)

print(torchvision.__version__)

assert torch.cuda.is_available()

assert torch.cuda.device_count() > 0
# =======================

# Running Prediction

# =======================

import os, sys

os.chdir(path_target_code)

print(os.getcwd())



sys.path.append("/kaggle/working/gwd/src/modeling/arch/yolov5")

sys.path.append("/kaggle/working/gwd/src/modeling/arch/yolov5")



from src.submissions.gwd import GWDSubmission

submission = GWDSubmission(input_path_model_weight="/kaggle/input/gwd-yolov5-weights/best.pt",

                           input_path_test="/kaggle/input/global-wheat-detection/test")

submission.configure()

path_submission = "/kaggle/working/gwd/submission_data"

Path(path_submission).mkdir(parents=True, exist_ok=True)

submission.eval(path_out=path_submission)

submission.parse(path_out=path_submission)

os.chdir(path_submission)
from shutil import copyfile

copyfile(Path(path_submission) / "Submission.csv", path_output / "submission.csv")