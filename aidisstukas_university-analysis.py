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
cwur = pd.read_csv('../input/cwurData.csv')

#education_expenditure_supplementary_data = pd.read_csv('../input/education_expenditure_supplementary_data.csv')

educational_attainment_supplementary_data = pd.read_csv('../input/educational_attainment_supplementary_data.csv')

school_and_country_table = pd.read_csv('../input/school_and_country_table.csv')

shanghaiData = pd.read_csv('../input/shanghaiData.csv')

timesData = pd.read_csv('../input/timesData.csv')
cwur.head(2)
educational_attainment_supplementary_data.head(2)
shanghaiData.head(2)
timesData.head(2)