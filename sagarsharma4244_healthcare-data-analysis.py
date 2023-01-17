import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
patient_profiles = pd.read_csv("/kaggle/input/healthcare-analytics/Train/Patient_Profile.csv")
First_Health_Camp_Attended = pd.read_csv("/kaggle/input/healthcare-analytics/Train/First_Health_Camp_Attended.csv")
Second_Health_Camp_Attended = pd.read_csv("/kaggle/input/healthcare-analytics/Train/Second_Health_Camp_Attended.csv")
Third_Health_Camp_Attended = pd.read_csv("/kaggle/input/healthcare-analytics/Train/Third_Health_Camp_Attended.csv")
Health_Camp_Detail = pd.read_csv("/kaggle/input/healthcare-analytics/Train/Health_Camp_Detail.csv")
Train = pd.read_csv("/kaggle/input/healthcare-analytics/Train/Train.csv")


patient_profiles.head()
patient_profiles.info()
patient_profiles.shape
patient_profiles
patient_profiles.dropna().info()
plt.figure(figsize=(16,6))
sns.barplot(x="Age", y="Facebook_Shared", data=patient_profiles.dropna().sort_values("Age"))
plt.figure(figsize=(16,6))
sns.barplot(x="Age", y="Online_Follower", data=patient_profiles.dropna().sort_values("Age"))
# Check for 
patient_profiles.isnull().sum()
sns.countplot(patient_profiles.sort_values("City_Type").City_Type)
plt.figure(figsize=(20,6))
sns.countplot(patient_profiles.dropna().sort_values("Age").Age)
sns.countplot(patient_profiles.dropna().Income)
plt.figure(figsize=(20,6))
sns.countplot(patient_profiles.dropna().Employer_Category)
Train.head()
sns.pairplot(Train)
First_Health_Camp_Attended.rename(columns={"Health_Camp_ID":"Health_Camp_ID_1" ,"Donation":"Donation_1","Health_Score":"Health_Score_1"})
Second_Health_Camp_Attended.rename(columns={"Health_Camp_ID":"Health_Camp_ID_2", "Health Score":"Health_Score_2"})
Third_Health_Camp_Attended.rename(columns={"Health_Camp_ID":"Health_Camp_ID_3", "Health Score":"Health_Score_2"})
First_Health_Camp_Attended.info()

Second_Health_Camp_Attended.info()

Third_Health_Camp_Attended.info()
sns.distplot(First_Health_Camp_Attended.Donation)

sns.scatterplot(x="Donation",y="Health_Score",data = First_Health_Camp_Attended)
