import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")
my_filepath = "../input/data-analyst-jobs/DataAnalyst.csv"
my_data = pd.read_csv(my_filepath)
my_data.head()
sns.swarmplot(x= my_data['Job Title'] , y = my_data['Rating'])