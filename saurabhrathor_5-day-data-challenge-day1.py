# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import zipfile



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input/80-cereals"]).decode("utf8"))

#zip_ref = zipfile.ZipFile("../input/80-cereals", 'r')

#zip_ref.extractall("../input")

#zip_ref.close()



data = pd.read_csv("../input/80-cereals/cereal.csv")

data.describe()

# Any results you write to the current directory are saved as output.