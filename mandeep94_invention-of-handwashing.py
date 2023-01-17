
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML
pd.options.display.max_columns = None
pd.options.display.max_rows = None

CSS = """
          .output {
              flex-direction: row;
          } 
          """
HTML('<style>{}</style>'.format(CSS))
monthly_deaths=pd.read_csv("../input/monthly_deaths.csv")
yearly_deaths = pd.read_csv("../input/yearly_deaths_by_clinic.csv")
print("monthly deaths shape: ",monthly_deaths.shape)
print("yearly deaths shape: ", yearly_deaths.shape)
print("Monthly deaths column types: \n",monthly_deaths.dtypes)
display(monthly_deaths.head())
print("Yearly deaths column types: \n", yearly_deaths.dtypes)
display(yearly_deaths.head())
print(len(monthly_deaths.date.unique()) == monthly_deaths.date.count())
monthly_deaths['date'] = pd.to_datetime(monthly_deaths['date'])
display(monthly_deaths.dtypes)
display(monthly_deaths.describe())
display(yearly_deaths.describe())
display(yearly_deaths.clinic.unique())
display(monthly_deaths.isnull().any())
display(yearly_deaths.isnull().any())
plt.plot(monthly_deaths.date, monthly_deaths.births, color='g', label="births")
plt.plot(monthly_deaths.date, monthly_deaths.deaths, color='r', label="deaths")
plt.xlabel("Date")
plt.ylabel("people count")
plt.legend(loc="upper left")
plt.suptitle("Fig. 1: Total number of births and deaths monthly")
_ = plt.plot()
monthly_deaths['percent'] = monthly_deaths.deaths *100 / monthly_deaths.births
plt.plot(monthly_deaths.date, monthly_deaths.percent, color='black', label="deaths percentage on births")
plt.ylabel("percentage")
plt.xlabel("Date")
plt.ylim(0,100)
plt.yticks(range(0,100, 10))
plt.grid(linestyle='dotted', linewidth=1)
plt.legend(loc="upper right")
plt.suptitle("Fig. 2: Death %age on births monthly")
_ = plt.plot()
yearly_deaths['percent'] = yearly_deaths.deaths *100/ yearly_deaths.births
clinic1 = yearly_deaths[yearly_deaths['clinic']=='clinic 1']
clinic2 = yearly_deaths[yearly_deaths['clinic']=='clinic 2']
print("total records in clinic 1: ", len(clinic1))
print("total records in clinic 2: ", len(clinic2))
display(clinic1.describe())
display(clinic2.describe())
plt.plot(clinic1.year, clinic1.births, label="Clinic 1")
plt.plot(clinic2.year, clinic2.births, label="Clinic 2")
plt.legend(loc="best")
plt.ylabel("number of births")
plt.xlabel("Year")
plt.suptitle("Fig. 3: Total number of births yearly in both clinics")
_ = plt.plot()
plt.plot(clinic1.year, clinic1.deaths, label="Clinic 1")
plt.plot(clinic2.year, clinic2.deaths, label="Clinic 2")
plt.legend(loc="best")
plt.ylabel("number of deaths")
plt.xlabel("Year")
plt.suptitle("Fig. 4: Total number of deaths yearly in both clinics")
_ = plt.plot()
plt.plot(clinic1.year, clinic1.percent, label="Clinic 1")
plt.plot(clinic2.year, clinic2.percent, label="Clinic 2")
plt.legend(loc="best")
plt.ylabel("percent of deaths on births")
plt.xlabel("Year")
plt.suptitle("Fig. 5: Death %age on births in both clinics")
_ = plt.plot()
avg_c1 = clinic1.deaths.sum() *100/ clinic1.births.sum()
print("average rate of death in clinic1 :", avg_c1)
avg_c2 = clinic2.deaths.sum() *100/ clinic2.births.sum()
print("average rate of death in clinic1 :", avg_c2)
total_1847 = monthly_deaths[monthly_deaths.date <= pd.to_datetime('1847-12-01')]
total_1848 = monthly_deaths[monthly_deaths.date > pd.to_datetime('1847-12-01')]
print("average death rate till 1847 : ", (total_1847.deaths.sum()*100/ total_1847.births.sum()))
print("average death rate after 1847 : ", (total_1848.deaths.sum()*100/ total_1848.births.sum()))