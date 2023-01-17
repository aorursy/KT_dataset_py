"""

Recording commands to train and build the neural_network_model package.



This script is saved in both of the following locations:

- Kaggle Kernel: To run the commands next to the data.

    - <https://www.kaggle.com/btw78jt/deploy-ml-commands>

- GitHub repo: Just for a back-up into version control of the above.

    - <https://github.com/A-Breeze/deploying-machine-learning-models>



To avoid having to download the accompanying dataset (which is 2GB),

these commands have been run on a Kaggle Kernel to build the 

neural_network_model package. The resulting package distribution can be

manually downloaded and added to the Git repo, to be deployed.



**IMPORTANT**

This script uses a dataset "deployingmachinelearningmodelsab" which contains

the project GitHub repo. The dataset does *not* automatically update when the

GitHub repo changes. You must update it *manually* by:

1.Ensure the branch you want is set as the *default branch* on GitHub.

    - Go to the repo home page - Settings - Branches - Default branch

2. Click "Update" on the Kaggle dataset page.

    - <https://www.kaggle.com/btw78jt/deployingmachinelearningmodelsab>

3. It will create a new version even if nothing has changed. Therefore,

check that a change from the most recent commit has correctly come 

through to the Kaggle dataset.

"""



print(

    '###############\n'

    '# Environment #\n'

    '###############'

)

# Output the current environment, so we know what the package should be run using

# Specific dependencies

import platform

import sys

# Setup

from pip import __version__ as pip_version

from setuptools import __version__ as setuptools_version

from wheel import __version__ as wheel_vers

# from kaggle import __version__ as kaggle_version  # Cannot do this without also inputting Kaggle credentials

# Data and modelling

from joblib import __version__ as joblib_version

from numpy import __version__ as np_version

from pandas import __version__ as pd_version

from matplotlib import __version__ as mpl_version

from skimage import __version__ as skimage_version

from sklearn import __version__ as sk_version

from tensorflow import __version__ as tf_version

from keras import __version__ as keras_version

from h5py import __version__ as h5py_version

# Development

from pytest import __version__ as pytest_version

from notebook import __version__ as notebook_version



# Confirm expected versions (i.e. the versions running in the Kaggle Kernel)

assert platform.python_version() == '3.6.6'

print(f"Python version:\t\t{sys.version}")

assert pip_version == '20.0.2'

print(f"pip version:\t\t{pip_version}")

assert setuptools_version == '46.1.3.post20200330'

print(f"setuptools version:\t{setuptools_version}")

assert wheel_vers == '0.34.2'

print(f"wheel version:\t{wheel_vers}")

# assert kaggle_version == '1.5.6'

# print(f"kaggle version:\t{kaggle_version}")

assert joblib_version == '0.14.1'

print(f"joblib version:\t\t{joblib_version}")

assert np_version == '1.18.2'

print(f"numpy version:\t\t{np_version}")

assert pd_version == '0.25.3'

print(f"pandas version:\t\t{pd_version}")

assert mpl_version == '3.2.1'

print(f"matplotlib version:\t{mpl_version}")

assert skimage_version == '0.16.2'

print(f"skimage version:\t{skimage_version}")

assert sk_version == '0.22.2.post1'

print(f"sklearn version:\t{sk_version}")

assert tf_version == '2.1.0'

print(f"tensorflow version:\t{tf_version}")

assert keras_version == '2.3.1'

print(f"keras version:\t\t{keras_version}")

assert h5py_version == '2.10.0'

print(f"h5py version:\t\t{h5py_version}")

assert pytest_version == '5.0.1'

print(f"pytest version:\t\t{pytest_version}")

assert notebook_version == '5.5.0'

print(f"notebook version:\t{notebook_version}")



print("\nShow versions of Jupyter packages")

!jupyter --version



print("\nCapturing full pip environment spec...")

print("(But note that not all these packages are required)")

!pip freeze > requirements_Kaggle.txt

print("...Done\n")



print(

    '#########\n'

    '# Setup #\n'

    '#########'

)

"""

There is some additional setup needed, 

given that we are running this on Kaggle,

not in the command line of the repo.

"""



# Import built-in modules

import os

from pathlib import Path

import shutil



# Set the DATA_FOLDER env var for use in Kaggle

os.environ['DATA_FOLDER'] = "/kaggle/input/deployingmachinelearningmodelsab/packages/neural_network_model/neural_network_model/datasets/test_data"

# Alternative:  "/kaggle/input/v2-plant-seedlings-dataset"

print("Data folder configured as follows (should print twice):")

!echo $DATA_FOLDER

!python -c "import os; print(os.environ['DATA_FOLDER'])"



# Copy scripts from (read-only) dataset to output area

folder_to_copy = Path('/kaggle/input') / 'deployingmachinelearningmodelsab' / 'packages' / 'neural_network_model'

target_location = Path('.') / 'packages' / 'neural_network_model'

target_location.parent.mkdir(parents=True, exist_ok=True)

shutil.copytree(str(folder_to_copy), str(target_location))

print("\nThe package source files have been copied to the sandbox area.")



# You should have already decided the model version that will be built

vers_str = (folder_to_copy / 'neural_network_model' / 'VERSION').read_text()

print(f"\nWe are going to be fitting version:\t{vers_str}\n")



print(

    '###############\n'

    '# Train model #\n'

    '###############'

)

# Override the default for EPOCHS

os.environ['EPOCHS'] = "2"  # 1 for testing, 8 for fitting the model

print("Number of epochs to use for training (should print twice):")

!echo $EPOCHS

!python -c "import os; print(os.environ['EPOCHS'])"



print("\nStarting model fitting...")

!PYTHONPATH=./packages/neural_network_model python packages/neural_network_model/neural_network_model/train_pipeline.py

print("\nModel fitting has completed.\n")



print(

    '#######################\n'

    '# Run automated tests #\n'

    '#######################'

)

!pytest packages/neural_network_model/tests



print(

    '#####################\n'

    '# Build the package #\n'

    '#####################'

)

os.chdir(Path('.') / 'packages' / 'neural_network_model')

!python setup.py sdist bdist_wheel

os.chdir(Path('.') / '..' / '..')



print(

    '####################\n'

    '# Clean the output #\n'

    '####################'

)

# Want to store the package distribution on Kaggle, 

# so it doesn't have to be saved within the repo.

print("Move the source distribution of the package to the top level")

src_dist_filename = f'neural_network_model-{vers_str}.tar.gz'

(target_location / 'dist' / src_dist_filename).rename(Path('.') / src_dist_filename)



print("Delete the remainder of the copied files")

shutil.rmtree(target_location.parent)



print("\n==== Script completed. =====")