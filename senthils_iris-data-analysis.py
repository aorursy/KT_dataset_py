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
ir = pd.read_csv("../input/Iris.csv")
df = ir.PetalLengthCm
import matplotlib.pyplot as plt
plt.scatter(ir.PetalLengthCm, ir.PetalWidthCm)
plt.show()
type(ir)
ir
def get_grayscale_intensity(species):
    print(species)
    return 100
    
ir.assign(SpeciesType = lambda x: ( get_grayscale_intensity(x['Species']) ))
ir


