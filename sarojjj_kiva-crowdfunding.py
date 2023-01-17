# Data processing
import numpy as np 
import pandas as pd

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Initial setup
%matplotlib inline
color = sns.color_palette()
sns.set_style('dark')
# kiva crowd funding data
kiva_loans = pd.read_csv("../input/kiva_loans.csv")
kiva_mpi_region_locations = pd.read_csv("../input/kiva_mpi_region_locations.csv")
loan_theme_ids = pd.read_csv("../input/loan_theme_ids.csv")
loan_themes_by_region = pd.read_csv("../input/loan_themes_by_region.csv")
sns.set(rc={"figure.figsize":(10, 5)})
sns.distplot(kiva_loans[kiva_loans['loan_amount'] < 5000].loan_amount, bins=[x for x in range(0, 5100, 200)], kde=False, color='c', label='loan_frequency')
plt.legend()
plt.show()
sns.set(rc={"figure.figsize":(15, 8)})
sns.countplot(y="country", data=kiva_loans, order=kiva_loans.country.value_counts().iloc[:20].index)
plt.title("Distribution of kiva loans by country")
plt.ylabel('')
plt.show()
sns.set(rc={"figure.figsize": (15, 8)})
sns.countplot(y="sector", data=kiva_loans, order=kiva_loans.sector.value_counts().iloc[:20].index)
plt.title("Distribution of loans by Sector")
plt.ylabel("")
plt.show()
sns.set(rc={"figure.figsize": (15, 10)})
sns.boxplot(x='loan_amount', y='country', data=kiva_loans, order=kiva_loans.country.value_counts().iloc[:10].index)
plt.title("Distribution of loan amount by country")
plt.ylabel("")
plt.show()
sns.set(rc={"figure.figsize": (15, 10)})
sns.boxplot(x='loan_amount', y='sector', data=kiva_loans, order=kiva_loans.sector.value_counts().iloc[:10].index)
plt.title("Distribution of loan amount by sector")
plt.ylabel("")
plt.show()
