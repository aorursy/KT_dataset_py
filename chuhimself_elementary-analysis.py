import numpy as np 
import pandas as pd 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv("../input/train.csv")
# Any results you write to the current directory are saved as output.
#taking a look at the data
df.info()
df.columns
df['target'].value_counts
df['target']





