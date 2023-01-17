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
library(dplyr)

library(purrr)

library(tidyr)

library(ggplot2)

library(broom)

library(magrittr)

library(plotly)

library(RSQLite)

library(reshape2)

library(visNetwork)

library(networkD3)

library(jsonlite)

library(RColorBrewer)

library(gplots)

library(knitr)

library(DT)

library(data.table)

library(d3heatmap)

library(viridis)

library(maps)

library(ggmap)

library(circlize)





# Connect to data base ----------------------------------------------------

con <- dbConnect(SQLite(), dbname="../input/database.sqlite")



player       <- tbl_df(dbGetQuery(con,"SELECT * FROM player"))

# player_stats <- tbl_df(dbGetQuery(con,"SELECT * FROM player_stats"))

Match        <- tbl_df(dbGetQuery(con,"SELECT * FROM Match"))

Team        <- tbl_df(dbGetQuery(con,"SELECT * FROM Team"))

Country        <- tbl_df(dbGetQuery(con,"SELECT * FROM Country"))

League        <- tbl_df(dbGetQuery(con,"SELECT * FROM League"))


