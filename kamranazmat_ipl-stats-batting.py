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
deliveries_data = pd.read_csv("../input/deliveries.csv")

matches_data = pd.read_csv("../input/matches.csv")
def filter_data_oc():

    data = deliveries_data[["match_id", "batsman", "batsman_runs"]]

    return data
def most_runs_year():

    """

    Finds the player with most run for every year. ("Orange Cap")

    Calls two functions:

        i) Filters out data - match_id, batsman_runs and then finds out the year related to the given match_id

        ii) Reduces all the data provided by the previous function and displays Orange Cap holder year wise

    """

    

    filter_data = filter_data_oc()
d = pd.merge(left = deliveries_data, right = matches_data, left_on = "match_id", right_on = "id")[["match_id", "id", "season", "batsman", "batsman_runs"]]