# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



count=0

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        count = count + 1

print (count)

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Clone the repository of XmlToTxt convertor

!git clone https://github.com/Isabek/XmlToTxt.git
# Enter to the directory XmlToTxt

%cd XmlToTxt/
# Install the requirements of XmlToTxt convertor

!pip install declxml==0.9.1
# Clean the Input & Output directories

!rm -rf xml/*.*

!rm -rf out/*.*
# Copy the XML files to be converted to text mode

!cp ../../input/mask-images-dataset/images/test/*.xml xml/
# All the XML files listed below

!ls xml/
# Modify the classes.txt files 

!cat classes.txt
!sed -i '4d' classes.txt

!sed -i '5d' classes.txt

!sed -i '6d' classes.txt



!sed -i 's/bus/with_mask/' classes.txt

!sed -i 's/minivan/mask_weared_incorrect/' classes.txt

!sed -i 's/auto/without_mask/' classes.txt



!sed -i '4d' classes.txt



!cat classes.txt
!python xmltotxt.py -xml xml -out out
# One of the sample input XMl file

!cat xml/maksssksksss640.xml
# Output text files of the same XML file above.

!cat out/maksssksksss640.txt