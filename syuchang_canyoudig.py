# You can write R code here and then click "Run" to run it on our platform

library(readr)

# The competition datafiles are in the directory ../input
# Read competition data files:
train <- read_csv("../input/train.csv")
test <- read_csv("../input/test.csv")

# Write to the log:
cat(sprintf("Training set has %d rows and %d columns\n", nrow(train), ncol(train)))
cat(sprintf("Test set has %d rows and %d columns\n", nrow(test), ncol(test)))

# Generate output files with write_csv(), plot() or ggplot()
# Any files you write to the current directory get shown as outputs

import pandas as pd
from pandas import DataFrame as df
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
alg = LogisticRegression(random_state=1)
scores = cross_validation.cross_val_score(alg,train.iloc[:,2:784], train["label"], cv=3)
print(scores.mean())