import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
gender_age = pd.read_csv('../input/gender_age_train.csv')
gender_age.head()