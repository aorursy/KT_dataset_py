# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# https://www.census.gov/geo/reference/ansi_statetables.html

state_ids = pd.read_csv("../input/us-fips-codes-for-states/fips_state.csv", usecols=["fips_code", "post_code"], dtype= { "fips_code": np.object, "post_code": np.object })

state_ids.set_index("fips_code", inplace = True)

state_ids = state_ids["post_code"].to_dict()
# row 0 is label data, so skip it

raw_data = pd.read_csv("../input/us-congressional-district-populations/DEC_10_115_P1_with_ann.csv", skiprows = [0])

raw_data
# GEO.id starts with 500, which indicates state + cong district level specificity

# so GEO.id2 is FIPS state ID followed by district ID



data = pd.DataFrame({ 

    "state_id": raw_data["Id2"].str[0:2].map(state_ids).fillna(raw_data["Id2"].str[0:2]),

    "district_id": raw_data["Id2"].str[2:],

    "population": pd.to_numeric(raw_data["Total"])

})



data.set_index(["state_id", "district_id"], inplace = True)

data
data.nlargest(10, "population")