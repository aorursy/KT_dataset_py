# Import Required Libraries

import numpy as np 

import pandas as pd

import seaborn as sns

%matplotlib inline
# Check for dataset availability

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df=pd.read_csv('../input/mushrooms.csv')

df.head()
sns.countplot(df['population']).set_title('Bar Chart for Population')
sns.countplot(y=df['population'],hue=df['class']).set_title('Bar Chart for Population By Class')