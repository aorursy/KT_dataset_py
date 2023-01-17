# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function

library(dplyr)

library(ggplot2)

library(plotly)

library(data.table)

library(formattable)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory





movie <- read.csv('../input/movie_metadata.csv',header=T,stringsAsFactors = F)

str(movie)







# Any results you write to the current directory are saved as output.