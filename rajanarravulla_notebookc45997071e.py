import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")
my_filepath = "../input/statewise-ayush-registrations/statewise_ayush_registrations.csv"
my_data = pd.read_csv(my_filepath)
my_data.head()

plt.figure(figsize=(14,8))
sns.jointplot(x=my_data['Ayurveda'], y=my_data['Unani'], kind="kde")


sns.distplot(a=my_data['Total'], kde=False)