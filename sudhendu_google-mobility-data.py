# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# Any results you write to the current directory are saved as output.
out_file = "global_mobility_google.csv"

out_dir="/kaggle/output"



if not os.path.exists(out_dir):

    os.mkdir(out_dir)



fullname = os.path.join(out_dir, out_file)



global_mobility_google= pd.read_csv("https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv", low_memory = False)

global_mobility_google.to_csv("global_mobility_google.csv", index = False)