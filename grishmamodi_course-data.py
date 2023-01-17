# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
data = pd.read_csv('../input/coursera-course-dataset/coursea_data.csv')

print(data.shape)

data.head()
data.info()

def find_org_greater_than_10(data):

    """Returns a dataframe with course_organization and number of courses > 10"""

    dict = {}

    course_org = data['course_organization'].to_list()

    for org in course_org:

        if org in dict:

            dict[org] += 1

        else:

            dict[org] = 1

    orgs = []

    counts = []

    for key, value in dict.items():

        if value > 10:

            orgs.append(key)

            counts.append(value)

        else:

            continue

    course_org_greater_than_1 = pd.DataFrame({'course_organization':orgs, 'count':counts})

    course_org_greater_than_1.sort_values(by='count', ascending=False, inplace=True)

    return course_org_greater_than_1
course_org_greater_than_1 = find_org_greater_than_10(data)



# plot a barh chart

course_org_greater_than_1.plot(kind='barh', x='course_organization', y='count')

plt.title('Organizations with more than 10 courses')

plt.xlabel('count')

plt.show()