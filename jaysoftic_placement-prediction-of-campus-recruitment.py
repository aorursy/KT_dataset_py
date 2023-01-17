# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
main_data = pd.read_csv("../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")
main_data.sample(5)
# Let's Clean first

# we don't need sl_no column
main_data.drop(columns = "sl_no", inplace = True)
# find na value

main_data.isna().sum()
# here salary has 67 NaN values because of 67 Interviewee Not Placed

# fill that salary 0
main_data.salary.fillna(value = 0, inplace = True)
# creating function for define figure size
def figsize(h = 10, w = 6):
    return plt.figure(figsize = (h, w))
placed_salary = main_data.loc[main_data.status == "Placed"]
figsize()
sns.countplot(data = main_data, x = "gender", hue = "status")
plt.show()
figsize()
sns.boxplot("salary","gender", data = main_data)
plt.show()
figsize()
sns.kdeplot(main_data.loc[main_data.status == "Placed", "ssc_p"], label = "Placed")
sns.kdeplot(main_data.loc[main_data.status == "Not Placed", "ssc_p"], label = "Not Placed")
plt.xlabel("SSC Percentage")
plt.show()
figsize()
sns.lineplot( "ssc_p","salary", data = placed_salary)
plt.xlabel("SSC Percentage")
plt.show()
figsize()
sns.kdeplot(main_data.loc[main_data.status == "Placed", "hsc_p"], label  = "Placed")
sns.kdeplot(main_data.loc[main_data.status == "Not Placed", "hsc_p"], label = "Not Placed")
plt.xlabel("HSC Percentage")
plt.show()
figsize()
sns.lineplot("hsc_p", "salary", data = placed_salary)
plt.xlabel("HSC Percentage")
plt.show()
plt.figure(figsize = (12, 8))
plt.figure(1)
plt.subplot(221)
sns.countplot("ssc_b", hue = "status", data = main_data)
plt.xlabel("SSC Board")
plt.subplot(222)
sns.countplot("hsc_b", hue =  "status", data = main_data)
plt.xlabel("HSC Borad")
plt.show()
figsize(10, 10)

plt.subplot(211)
sns.boxplot(x = "salary", y = "ssc_b", data = placed_salary)
plt.xlabel("Salary")
plt.ylabel("SSC Board")
plt.title("Does SSC Board affect to salary?")

plt.subplot(212)
sns.boxplot(x = "salary", y = "hsc_b", data = placed_salary)
plt.xlabel("Salary")
plt.ylabel("HSC Board")
plt.title("Does HSC Board affect to salary?")
plt.show()
figsize()
sns.countplot("hsc_s", hue = "status", data = main_data)
plt.xlabel("HSC Specialisation")
plt.show()
figsize()
sns.boxenplot("salary", "hsc_s", data = placed_salary)
plt.ylabel("HSC Specialisation")
plt.show()
plt.figure(figsize = (10, 6))
sns.kdeplot(main_data.loc[main_data.status == "Placed", "degree_p"], label = "Placed")
sns.kdeplot(main_data.loc[main_data.status == "Not Placed", "degree_p"], label = "Not Placed")
plt.xlabel("Degree Percentage")
plt.show()
figsize()
sns.lineplot("degree_p", "salary", data = placed_salary)
plt.xlabel("Degree Percentage")
plt.show()
plt.figure(figsize = (10, 6))
sns.countplot("degree_t", hue = "status", data = main_data)
plt.xlabel("Degree Type")
plt.show()
figsize()
sns.boxplot("salary", "degree_t", data = main_data)
plt.xlabel("Degree Type")
plt.show()
plt.figure(figsize = (10, 6))
sns.countplot("workex", hue = "status", data = main_data)
plt.xlabel("Work Experience")
plt.show()
figsize()
sns.boxplot("salary", "workex", data = placed_salary)
plt.ylabel("Work Experience")
plt.show()
plt.figure(figsize = (10, 6))
sns.kdeplot(main_data.loc[main_data.status == "Placed", "etest_p"], label = "Placed")
sns.kdeplot(main_data.loc[main_data.status == "Not Placed", "etest_p"], label = "Not Placed")
plt.xlabel("Etest Percentage")
plt.show()
figsize()
sns.lineplot("etest_p", "salary", data = placed_salary)
plt.xlabel("Etest Percetage")
plt.show()
plt.figure(figsize = (10, 6))
sns.countplot("specialisation", hue = "status", data = main_data)
plt.xlabel("Specialisation")
plt.show()
figsize()
sns.boxenplot("salary", "specialisation", data = placed_salary)
plt.show()
plt.figure(figsize = (10, 6))
sns.kdeplot(main_data.loc[main_data.status == "Placed", "mba_p"], label = "Placed")
sns.kdeplot(main_data.loc[main_data.status == "Not Placed", "mba_p"], label = "Not Placed")
plt.xlabel("MBA Percentage")
plt.show()
figsize()
sns.lineplot("mba_p", "salary", data = placed_salary)
plt.xlabel("MBA Percentage")
plt.show()
# gathering for status prediction
status_data = main_data[["gender", "ssc_p", "hsc_p", "hsc_s", "degree_p", "degree_t", "workex", "specialisation", "status"]].copy()
status_data.sample(5)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
status_data["gender"] = le.fit_transform(status_data.gender) # 0 = female, 1 = male
status_data["hsc_s"] = le.fit_transform(status_data.hsc_s) # 0 = Arts, 1 = Commerce, 2 = Science
status_data["degree_t"] = le.fit_transform(status_data.degree_t) # 0 = Comm&Mgmt, 1 = Others, 2 = Sci&Tech
status_data["workex"] = le.fit_transform(status_data.workex) # 0 = No, 1 = Yes
status_data["specialisation"] = le.fit_transform(status_data.specialisation) # 0 = Mkt&Fin, 1 - Mkt&HR
status_data["status"] = le.fit_transform(status_data.status) # 0 = Not Placed, 1 = Placed 
status_data.head(2)
X = status_data.loc[:, status_data.columns != "status"]
y = status_data.status
# split data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 15)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
logreg.score(X_test, y_test)
y_pred = logreg.predict(X_test)
from sklearn.metrics import confusion_matrix
cf = confusion_matrix(y_test, y_pred)
sns.heatmap(cf, annot = True)
plt.xlabel("Predicted")
plt.ylabel("Actual")