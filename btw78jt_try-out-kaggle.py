# Import built-in modules

import sys

import os

from pathlib import Path



# Import external modules

import numpy as np

import pandas as pd



# Allow local modules to be imported

usr_lib_path_str = '/kaggle/usr/lib'  # For utility scripts 

repo_lib_path_str = '/kaggle/input'  # For GitHub repo modules

if sys.path[:2] != [usr_lib_path_str, repo_lib_path_str]:

    sys.path.insert(0, repo_lib_path_str)

    sys.path.insert(0, usr_lib_path_str)



# Import project modules (also see explanation below)

import try_out_kaggle_utils as kutils  # Example utility script

import kaggletestsrepo.mods_for_kaggle.utils as repo_utils  # Example repo module



# Check they imported OK

print(f"Python version:\t{sys.version}")

print(f"Python running here:\t{sys.executable}")

print("-" * 50)

print(f"numpy version:\t\t{np.__version__}")

print(f"pandas version:\t\t{pd.__version__}")

print("-" * 50)

print(f"kutils (utility script) version:\t{kutils.__version__}")

print(f"repo_utils (repo module) version:\t{repo_utils.__version__}")
%%time

# Takes approx 8 s

print("The conda environment is automatically created by Kaggle")

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

!conda env list
%%time

# Takes approx 20 s

if repo_utils.is_internet_accessible():

    print("Internet *is* available")

else:

    print("Internet is *not* available")
#  Example of the help() function on an imported object (with a docstring)

help(repo_utils.is_internet_accessible)
# Example of installing a package

# !conda install -c conda-forge pyprojroot==0.2.0
from xgboost import __version__ as xgb_version

print(f"xgboost available, version:\t{xgb_version}")

from lightgbm import __version__ as lgb_version

print(f"lightgbm available, version:\t{lgb_version}")

from hyperopt import __version__ as hypo_version

print(f"hyperopt available, version:\t{hypo_version}")

from sklearn_pandas import __version__ as sklpd_version

print(f"sklearn_pandas available, version:\t{sklpd_version}")

from pymc3 import __version__ as pymc3_version

print(f"pymc3 available, version:\t{pymc3_version}")

from bokeh import __version__ as bok_version

print(f"bokeh available, version:\t{bok_version}")

from altair import __version__ as alt_version

print(f"altair available, version:\t{alt_version}")

from pylint import __version__ as pyl_version

print(f"pylint available, version:\t{pyl_version}")

from pytest import __version__ as pyt_version

print(f"pytest available, version:\t{pyt_version}")

print("-" * 40)

try:

    import pyprojroot

    print("pyprojroot available")

except ModuleNotFoundError:

    print("pyprojroot *not* included")

try:

    import pygam

    print("pygam available")

except ModuleNotFoundError:

    print("pygam *not* included")
# Keras is special because it starts the backend on being loaded

from keras import __version__ as ker_version

print(f"keras available, version:\t{ker_version}")
print("The following shows that Git is available but this is *not* a Git repo")

!git status
print("In fact, no config has been set up, "

      "as seen by the fact the following produces no output")

!git config --list
# Get available directory structure

initial_folder_path = Path(os.getcwd())

print(f"Initial working directory:\t{initial_folder_path}")
# This is just one of many directories in the environment

print("\t".join([str(x) for x in Path('/').iterdir() if x.is_dir()]))
proj_dir_path = Path('/kaggle')

print(f"Files initially in the {proj_dir_path} directory:")

print(proj_dir_path)

for line in kutils.tree(proj_dir_path, prefix="    "):

    print(line)

print("")
# Create a new directory

op_dir_path = proj_dir_path / 'new_output'

try: 

    op_dir_path.mkdir()

    print(f"New directory created here:\t{op_dir_path}")

except FileExistsError:

    print(f"Specified directory already exists:\t{op_dir_path}")
# Remove it

try:

    op_dir_path.rmdir()

    print(f"Directory removed:\t{op_dir_path}")

except FileNotFoundError:

    print(f"Specified directory does not exist:\t{op_dir_path}")
%%time

# Write a file to the working directory

# This example writes a conda environment specification

# Takes approx 10 s

!conda env export --name base --from-history > base_env_from_hist.yml
conda_env_spec_path = Path('/kaggle/working') / 'base_env_from_hist.yml'

content = conda_env_spec_path.read_text().split('\n')

n_lines = 10

print(f"Top of file up to {n_lines} lines\n" + "-" * 25)

print('\n'.join(content[:n_lines]))
from IPython.display import FileLink

flink = FileLink(conda_env_spec_path.name)  # Note this is the *name* not the absolute path

flink