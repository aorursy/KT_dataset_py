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
raw_data = pd.read_csv('../input/xAPI-Edu-Data.csv')
raw_data.columns
# raw numeric summaries

raw_data.describe()
raw_data.StageID.value_counts()
# how about GradeId

raw_data.GradeID.value_counts()
# is it possible to find out which subset of the GradeID belong to a StageID?

gby = raw_data.groupby('StageID')
gby.GradeID.unique()
raw_data[(raw_data.GradeID == 'G-07') & (raw_data.StageID == 'lowerlevel')]
raw_data.loc[(raw_data.GradeID == 'G-07') & (raw_data.StageID == 'lowerlevel'), 'StageID'] = 'MiddleSchool'
raw_data[(raw_data.GradeID == 'G-07') & (raw_data.StageID == 'lowerlevel')]
pd.crosstab(raw_data.ParentschoolSatisfaction, raw_data.StudentAbsenceDays, margins=True)
raw_data.groupby('StudentAbsenceDays').raisedhands.plot(kind="hist", legend=True, title="raised hands")
raw_data.groupby('StudentAbsenceDays').boxplot(column='raisedhands')
pd.crosstab(raw_data.StageID, raw_data.StudentAbsenceDays, margins=True, normalize='index')