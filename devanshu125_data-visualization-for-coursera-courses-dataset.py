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
# replacing Unnamed: 0 column with Id

data['Id'] = data['Unnamed: 0']

data.drop(['Unnamed: 0'], axis=1, inplace=True)
data.head()
data.info()
std_enroll = []

for i in range(0, len(data)):

    try:

        num = data['course_students_enrolled'].str.split('k')[i][0]

        num = float(num) * 1000 

        std_enroll.append(num)

    except:

        num = data['course_students_enrolled'].str.split('m')[i][0]

        num = float(num) * 1000000

        std_enroll.append(num)

data['course_students_enrolled'] = std_enroll

data['course_students_enrolled'] = data['course_students_enrolled'].astype(float)
# check using describe method

data.describe()
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
# dictionary containing organization as key and avg rating as value

dom_dict = round(data.groupby('course_organization')['course_rating'].mean(), 1).to_dict()
# Filter out organizations, as we only want those organizations with more than 10 courses

orgs = course_org_greater_than_1['course_organization'].to_list()

avg_rating = []

for org in orgs:

    for key, value in dom_dict.items():

        if key == org:

            avg_rating.append(value)

        else:

            continue

course_org_greater_than_1['avg_rating'] = avg_rating
course_org_greater_than_1
# plot a barh chart

course_org_greater_than_1.plot(kind='barh', x='course_organization', y='avg_rating')

plt.title('Average course rating of organizations with more than 10 courses')

plt.xlabel('Average course rating')

plt.show()
dom_dict = round(data.groupby('course_organization')['course_rating'].mean(), 1).to_dict()

dom_dict = {k: v for k, v in sorted(dom_dict.items(), key=lambda item: item[1], reverse=True)}

dom_dict
for key, value in dom_dict.items():

    if value == 4.9:

        print(key)
course_dict = data.groupby('course_title')['course_rating'].mean().to_dict()

course_dict = {k: v for k, v in sorted(course_dict.items(), key=lambda item: item[1], reverse=True)}

course_dict
cour_df = pd.DataFrame({'course_title':list(course_dict.keys()), 'course_rating':list(course_dict.values())})

cour_df[:10]
stud_dict = round(data.groupby('course_difficulty')['course_students_enrolled'].mean(), 0).to_dict()

stud_dict = {k: v for k, v in sorted(stud_dict.items(), key=lambda item: item[1], reverse=True)}

stud_dict
diff_stud = pd.DataFrame({'difficulty':list(stud_dict.keys()), 'avg_students':list(stud_dict.values())})

diff_stud
diff_stud.plot(kind='bar', x='difficulty', y='avg_students', title='Distribution by difficulty')

plt.ylabel('Number of students')
rate_dict = round(data.groupby('course_difficulty')['course_rating'].mean(), 1).to_dict()

rate_dict = {k: v for k, v in sorted(rate_dict.items(), key=lambda item: item[1], reverse=True)}

rate_dict
diff_stud['avg_rating'] = list(rate_dict.values())

diff_stud.plot(kind='bar', x='difficulty', y='avg_rating', title='Average rating by difficulty')
cert_dict = round(data.groupby('course_Certificate_type')['course_students_enrolled'].mean(), 0).to_dict()

cert_dict = {k: v for k, v in sorted(cert_dict.items(), key=lambda item: item[1], reverse=True)}

cert_dict
cert_df = pd.DataFrame({'course_Certificate_type':list(cert_dict.keys()), 'avg_students':list(cert_dict.values())})

cert_df.plot(kind='bar', x='course_Certificate_type', y='avg_students', title='Distribution by course_Certificate_type')

plt.ylabel('Number of students')

plt.show()
rate_cert_dict = round(data.groupby('course_Certificate_type')['course_rating'].mean(), 1).to_dict()

rate_cert_dict = {k: v for k, v in sorted(rate_cert_dict.items(), key=lambda item: item[1], reverse=True)}

rate_cert_dict
cert_df['avg_rating'] = list(rate_cert_dict.values())

cert_df.plot(kind='bar', x='course_Certificate_type', y='avg_rating', title='Average rating by certificate type')

plt.ylabel('Number of students')

plt.show()
data.plot(kind='scatter', x='course_students_enrolled', y='course_rating', title='Number of students enrolled vs. course rating')
data.corr()