# You can write R code here and then click "Run" to run it on our platform

#library(readr)

# The competition datafiles are in the directory ../input
# Read competition data files:
#train <- read_csv("../input/train.csv")
#test <- read_csv("../input/test.csv")

# Write to the log:
#cat(sprintf("Training set has %d rows and %d columns\n", nrow(train), ncol(train)))
#cat(sprintf("Test set has %d rows and %d columns\n", nrow(test), ncol(test)))

# Generate output files with write_csv(), plot() or ggplot()
# Any files you write to the current directory get shown as outputs


import numpy as np
import urllib
dataset = np.loadtxt("../input/train.csv", delimiter=",")