# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Imports

import os
all_files = os.listdir('../input')
all_files
def create_frame(files):

    for x in range(0, (len(files) - 1)):

        if x == 0:

            df = pd.read_csv(files[x])

        else:

            df_import = pd.read_csv(files[x])

            

            # Append the new frame to the original

            df.append(df_import)

            

    return df
# Make lists for all the files

training_files = []

test_files = []
# Make lists of each filepath in the directory

for x in all_files:

    if x.find('train') != -1:

        training_files.append(x)

    elif x.find('test') != -1:

        test_files.append(x)
training_files.sort()
training_files
# Unzip the training files

for x in training_files:

    filen = '/kaggle/working/' + x

    !unzip {filen}
train_frame = create_frame(training_files)
import os



def get_filepaths(directory):

    """

    This function will generate the file names in a directory 

    tree by walking the tree either top-down or bottom-up. For each 

    directory in the tree rooted at directory top (including top itself), 

    it yields a 3-tuple (dirpath, dirnames, filenames).

    """

    file_paths = []  # List which will store all of the full filepaths.



    # Walk the tree.

    for root, directories, files in os.walk(directory):

        for filename in files:

            # Join the two strings in order to form the full filepath.

            filepath = os.path.join(root, filename)

            file_paths.append(filepath)  # Add it to the list.



    return file_paths  # Self-explanatory.

# Run the above function and store its results in a variable.   

full_file_paths = get_filepaths("/kaggle/working/")
full_file_paths
from os import walk



f = []

for (dirpath, dirnames, filenames) in walk("/kaggle/working/"):

    f.extend(filenames)
f