import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

df = pd.read_csv('../input/Loan payments data.csv')

df.shape
df.describe()
df.head(5).T
df.info()
df.isnull().sum()
df['loan_status'].value_counts()
df['Gender'].value_counts()
sns.countplot(x="past_due_days", data=df, palette="muted");
sns.countplot(x="education", data=df, palette="dark");