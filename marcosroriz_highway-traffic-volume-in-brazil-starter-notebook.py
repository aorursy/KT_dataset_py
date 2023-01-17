# Basic libraries imports
import glob         # Library for handling file searching
import numpy as np  # Linear algebra library
import pandas as pd # Data processing, CSV file I/O (e.g. pd.read_csv)
# Library for handling graphics
import matplotlib.pyplot as plt    # For plotting
import matplotlib.ticker as ticker # For handling ticks specifics
# Standard Machine Learning Libraries
from sklearn.preprocessing import StandardScaler     # Normalization 
from sklearn.model_selection import train_test_split # For train/test split
from sklearn.metrics import r2_score                 # To compute R^2
from sklearn.linear_model import SGDRegressor        # Basic Linear Regressor
# Lets retrieve the datasets (* symbol means any csv file in the given folder)
search_files = "../input/traffic-volume-in-brazils-highway-toll-plazas/volume*.csv"
founded_files = glob.glob(search_files)

print("\n".join(founded_files))
# Read each file into a pandas dataframe, finally put them all in a single dataframe

database_list = []

for file in founded_files:
    database = pd.read_csv(file,                    # Given file
                           sep=";",                 # Database use ";" separator
                           decimal=",",             # Brazil number separator
                           encoding = "ISO-8859-1") # Encoding
    database_list.append(database)

dataset = pd.concat(database_list, ignore_index=True)
dataset.head()
dataset.dtypes
# Handling data (mon-year, e.g., jan-2019)
dataset["mes_ano"] = pd.to_datetime(dataset["mes_ano"], format='%b-%y')
# Number of operators
dataset["Concessionaria"].unique()
# Lets filter the dataset for only Nova Dutra entries
# Nova Dutra (BR-116) interconnects São Paulo and Rio de Janeiro
# it is one of the busiest highways.

dataset_dutra = dataset[dataset["Concessionaria"] == "01.Nova Dutra"]
dataset_dutra
# Lets filter the dataset for only Nova Dutra entries
# Nova Dutra (BR-116) interconnects São Paulo and Rio de Janeiro
# it is one of the busiest highways.

dataset_dutra = dataset[dataset["Concessionaria"].str.contains("Nova Dutra")]
dataset_dutra
# Lets see what are the categories of fees in this operator
dataset_dutra["Categoria"].unique()
# Lets filter category 1 ("cars")
dataset_dutra_cars = dataset_dutra[dataset_dutra["Categoria"].str.contains("Categoria 1")]
dataset_dutra_cars
# Let's get BR-116, one of the highway that have the highest volume in Brazil
dataset_dutra_car_br116 = dataset_dutra_cars[dataset_dutra_cars["Praca"].str.contains("BR-116/SP")]
dataset_dutra_car_br116
# Finally, lets group the data by month
dataset_br116_month = dataset_dutra_car_br116.groupby(["mes_ano"]).sum()
dataset_br116_month
# Let's plot the highway volume at every month
X = dataset_br116_month.index
y = dataset_br116_month[["Volume_total"]]

plt.xlabel("Year")
plt.ylabel("Number of Vehicles")
plt.scatter(X,y)

axis = plt.gca() 
yaxis = axis.get_yaxis()
yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.tight_layout()

# Save fig also
plt.savefig("/kaggle/working/fig1.png", dpi=300)
# Let's transform the year/month into month so that we can try to predict the traffic in the next months
dataset_br116_month["month"] = dataset_br116_month.index.to_period('M') - dataset_br116_month.index[0].to_period('M')
dataset_br116_month["month"] = dataset_br116_month["month"].apply(lambda x: x.n + 1)
# Let's plot the highway volume at every month (w.r.t. the number of days)
X = dataset_br116_month[["month"]]
y = dataset_br116_month[["Volume_total"]]

plt.xlabel("Months")
plt.ylabel("Number of Vehicles")
plt.scatter(X,y)

axis = plt.gca() 
yaxis = axis.get_yaxis()
yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.tight_layout()

# Save fig also
plt.savefig("/kaggle/working/fig2.png", dpi=300)
# Given X and Y, the goal is to predict how the traffic will behave in the next 24 months

X = dataset_br116_month[["month"]]
y = dataset_br116_month[["Volume_total"]]
# Try out many different regressors, from basic to more advanced ones
# Linear Regression, MLP, LTSM, etc
