import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



#constants

sns.set_style("dark")

sigLev = 3

pd.set_option("display.precision",sigLev)
#load in dataset

hrFrame = pd.read_csv("../input/HR_comma_sep.csv")
hrFrame.shape
hrFrame.info()
leftCountFrame = hrFrame.groupby("left",as_index = False)["satisfaction_level"].count()

leftCountFrame = leftCountFrame.rename(columns = {"satisfaction_level":"count"})

#then plot

sns.barplot(x = "left",y = "count",data = leftCountFrame)

plt.xlabel("Left")

plt.ylabel("Count")

plt.title("Distribution of Left")
plt.hist(hrFrame["satisfaction_level"])

plt.xlabel("Satisfaction Level")

plt.ylabel("Count")

plt.title("Distribution of Satisfaction Level")
plt.hist(hrFrame["last_evaluation"])

plt.xlabel("Last Evaluation")

plt.ylabel("Count")

plt.title("Distribution of Last Evaluation")
plt.hist(hrFrame["number_project"])

plt.xlabel("Number of Projects")

plt.ylabel("Count")

plt.title("Distribution of Number of Projects")
plt.hist(hrFrame["average_montly_hours"])

plt.xlabel("Average Monthly Hours")

plt.ylabel("Count")

plt.title("Distribution of Average Monthly Hours")

plt.hist(hrFrame["time_spend_company"])

plt.xlabel("Time Spent at the Company")

plt.ylabel("Count")

plt.title("Distribution of Time Spent at the Company")
waCountFrame = hrFrame.groupby("Work_accident",as_index = False)["left"].count()

waCountFrame = waCountFrame.rename(columns = {"left":"count"})

#then plot

sns.barplot(x = "Work_accident",y = "count",data = waCountFrame)

plt.xlabel("Work Accident")

plt.ylabel("Count")

plt.title("Distribution of Work Accident")
promoteCountFrame = hrFrame.groupby("promotion_last_5years",as_index = False)["left"].count()

promoteCountFrame = promoteCountFrame.rename(columns = {"left":"count"})

#then plot

sns.barplot(x = "promotion_last_5years",y = "count",data = promoteCountFrame)

plt.xlabel("Promotion in the Last 5 Years")

plt.ylabel("Count")

plt.title("Distribution of Promotion in the Last 5 Years")
deptCountFrame = hrFrame.groupby("sales",as_index = False)["left"].count()

deptCountFrame = deptCountFrame.rename(columns = {"left":"count"})

#then plot

sns.barplot(x = "sales",y = "count",data = deptCountFrame)

plt.xlabel("Department")

plt.ylabel("Count")

plt.title("Distribution of Department")
salaryCountFrame = hrFrame.groupby("salary",as_index = False)["left"].count()

salaryCountFrame = salaryCountFrame.rename(columns = {"left":"count"})

#then plot

sns.barplot(x = "salary",y = "count",data = salaryCountFrame)

plt.xlabel("Salary Level")

plt.ylabel("Count")

plt.title("Distribution of Salary Level")
sns.boxplot(x = "left",y = "satisfaction_level",data = hrFrame)

plt.xlabel("Left")

plt.ylabel("Satisfaction Level")

plt.title("Satisfaction Level on Left")
sns.boxplot(x = "left",y = "last_evaluation",data = hrFrame)

plt.xlabel("Left")

plt.ylabel("Last Evaluation")

plt.title("Last Evaluation on Left")
sns.boxplot(x = "left",y = "number_project",data = hrFrame)

plt.xlabel("Left")

plt.ylabel("Number of Projects")

plt.title("Number of Projects on Left")