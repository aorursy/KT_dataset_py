# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import sys

import os

from PIL import Image
# sys.argv.append('../input/pokemon/')

# sys.argv.append('../output/pokemonconverted/')
path = '../input/pokemon/'

directory = '../output/pokemonconverted/'
# if not os.path.exists(directory):

#     os.mkdir(directory)
os.listdir(path)
for filename in os.listdir(path):

    clean_name = os.path.splitext(filename)[0]

    print(filename)

    print(clean_name)

    img = Image.open(f'{path}{filename}')

    img.save(f'{clean_name}.png','png')

    print('done')