import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# First, look at everything.
#from subprocess import check_output
#print(check_output(["ls", "../input/activitylog"]).decode("utf8"))

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

data=pd.read_csv('../input/activitylog/finall_Activitylog.csv')

data.info
# Any results you write to the current directory are saved as output.

