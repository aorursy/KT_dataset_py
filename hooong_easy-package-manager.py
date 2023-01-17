import sys 

import platform

import os 

import zipfile 



def get_env_info(): 

    

    print(sys.platform)

    print(platform.python_implementation())

    print(sys.version)

    

get_env_info()
os.mkdir('packages')
#copy paste your requirements file contents to here 



open('packages/requirements.txt', 'w').write('''

numpy>=1.16.0

scipy>=1.4.1

opencv-python

efficientnet

''')
#download packages to directory 

!pip download -d packages -r packages/requirements.txt
# #zip packages so you can create a dataset from them easily

# #actually don't this is dumb. just create a new dataset from the output files

# import shutil

# shutil.make_archive('packages.zip', 'zip', 'packages')
# shutil.rmtree('packages')
#extract again using below 

# shutil.unpack_archive('packages.zip.zip', 'packages')