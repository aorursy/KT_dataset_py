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
salaries = pd.read_csv('../input/Salaries.csv',low_memory=False)
print(salaries.columns)
idx = np.array(np.argsort(salaries.TotalPay),dtype=int)[::-1]
job_titles = salaries.JobTitle.values
total_pays = salaries.TotalPay.values
for i in range(0,10):
    print(('%s\t%d' % (job_titles[idx[i]], total_pays[idx[i]])))