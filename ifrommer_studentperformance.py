import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import matplotlib.pyplot as plt

%matplotlib inline



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Read in data; then preview it

sData = pd.read_csv('../input/xAPI-Edu-Data.csv')

sData.head()
# Have a look, make sure nothing is missing

sData.info()
import seaborn as sns # statistical data visualization

sns.set_style("whitegrid")



# Get an overview by school - elementary, middle, high

sns.countplot(x='StageID', data=sData)
# add a column with an indicator if they are in middle school

sData['midSchool'] = np.where(sData['StageID']=='MiddleSchool',1,0)

# extract only the middle schoolers

midSchoolData = sData.query('midSchool == 1')

# Have a look at courses taken distribution of middle schoolers

sns.countplot(x='Topic', data=midSchoolData)
sns.countplot(y='NationalITy', data=sData)
sns.countplot(y='Class', data=sData, hue='NationalITy',order=['L','M','H'])


sns.countplot(y='NationalITy', data=midSchoolData)
numKW = midSchoolData[midSchoolData["NationalITy"]=='KW'].count()["gender"]

numJord = midSchoolData[midSchoolData["NationalITy"]=='Jordan'].count()["gender"]

numOther = len(midSchoolData) - numKW - numJord

print("Kuwaitis = ",numKW,"; Jordanians = ",numJord,"; Other = ",numOther)
# Add a new column indicating which of the Kuwait, Jordan, or other the student comes from

def nat3(x):

    x = str(x)

    if x == 'KW' or x == 'Jordan':

        val = x

    else:

        val = "Other"

    return val



midSchoolData['Nat3'] = midSchoolData.NationalITy.apply(nat3)

sns.countplot(midSchoolData.Nat3, palette='Set3',order=['Jordan','KW','Other'])
sns.countplot(x='Class', data=midSchoolData, hue='Nat3',order=['L','M','H'],hue_order=['KW','Jordan','Other'])
sns.countplot(x='StudentAbsenceDays', data=midSchoolData, hue='Nat3', \

              hue_order=['KW','Jordan','Other'])
f, axes = plt.subplots(2, 2, figsize=(12,12), sharex=True)

sns.boxplot(x="Nat3", y="raisedhands", data=midSchoolData,order=['Jordan','KW','Other'],ax=axes[0, 0])

sns.boxplot(x="Nat3", y="VisITedResources", data=midSchoolData,order=['Jordan','KW','Other'],ax=axes[0, 1])

sns.boxplot(x="Nat3", y="AnnouncementsView", data=midSchoolData,order=['Jordan','KW','Other'],ax=axes[1, 0])

sns.boxplot(x="Nat3", y="Discussion", data=midSchoolData,order=['Jordan','KW','Other'],ax=axes[1, 1])
