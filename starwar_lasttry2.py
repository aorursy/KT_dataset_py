# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages
# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats
# For example, here's several helpful packages to load in 

#library(ggplot2) # Data visualization
#library(readr) # CSV file I/O, e.g. the read_csv function

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#system("ls ../input")

# Any results you write to the current directory are saved as output.
import numpy as np

size = 10.0;

x_step = 0.2
y_step = 0.2

x_ranges = zip(np.arange(0, size, x_step), np.arange(x_step, size + x_step, x_step))
y_ranges = zip(np.arange(0, size, y_step), np.arange(y_step, size + y_step, y_step))

for x,y in x_ranges:
    print ("hel")
    print (x)
    print (y)
    for a,b in y_ranges:
        print (a)
        print (b)

