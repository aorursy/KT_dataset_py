# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

print(" ")
print(" ")
print("Setup Complete!")

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
breda_filepath = "../input/air-pollution-covid19-v3-gesorteerd/Breda.csv"
breda_data = pd.read_csv(breda_filepath)

dordrecht_filepath = "../input/air-pollution-covid19-v3-gesorteerd/Dordrecht.csv"
dordrecht_data = pd.read_csv(dordrecht_filepath)

eindhoven_filepath = "../input/air-pollution-covid19-v3-gesorteerd/Eindhoven.csv"
eindhoven_data = pd.read_csv(eindhoven_filepath)

groningen_filepath = "../input/air-pollution-covid19-v3-gesorteerd/Groningen.csv"
groningen_data = pd.read_csv(groningen_filepath)

heerlen_filepath = "../input/air-pollution-covid19-v3-gesorteerd/Heerlen.csv"
heerlen_data = pd.read_csv(heerlen_filepath)

nijmegen_filepath = "../input/air-pollution-covid19-v3-gesorteerd/Nijmegen.csv"
nijmegen_data = pd.read_csv(nijmegen_filepath)

rotterdam_filepath = "../input/air-pollution-covid19-v3-gesorteerd/Rotterdam.csv"
rotterdam_data = pd.read_csv(rotterdam_filepath)

utrecht_filepath = "../input/air-pollution-covid19-v3-gesorteerd/Utrecht.csv"
utrecht_data = pd.read_csv(utrecht_filepath)

sgravenhage_filepath = "../input/air-pollution-covid19-v3-gesorteerd/s-Gravenhage.csv"
sgravenhage_data = pd.read_csv(sgravenhage_filepath)
plt.figure(figsize=(12,6))
sns.lineplot(x = breda_data["Week of datebegin"], y = breda_data["Avg. AirQualityLevel"], sort=False)
plt.xlabel("Breda")
plt.xticks(rotation=90)
plt.tight_layout()
plt.figure(figsize=(12,6))
sns.lineplot(x = dordrecht_data["Week of datebegin"], y = dordrecht_data["Avg. AirQualityLevel"], sort=False)
plt.xlabel("Dordrecht")
plt.xticks(rotation=90)
plt.tight_layout()
plt.figure(figsize=(12,6))
sns.lineplot(x = eindhoven_data["Week of datebegin"], y = eindhoven_data["Avg. AirQualityLevel"], sort=False)
plt.xlabel("Eindhoven")
plt.xticks(rotation=90)
plt.tight_layout()
plt.figure(figsize=(12,6))
sns.lineplot(x = groningen_data["Week of datebegin"], y = groningen_data["Avg. AirQualityLevel"], sort=False)
plt.xlabel("Groningen")
plt.xticks(rotation=90)
plt.tight_layout()
plt.figure(figsize=(12,6))
sns.lineplot(x = heerlen_data["Week of datebegin"], y = heerlen_data["Avg. AirQualityLevel"], sort=False)
plt.xlabel("Heerlen")
plt.xticks(rotation=90)
plt.tight_layout()
plt.figure(figsize=(12,6))
sns.lineplot(x = nijmegen_data["Week of datebegin"], y = nijmegen_data["Avg. AirQualityLevel"], sort=False)
plt.xlabel("Nijmegen")
plt.xticks(rotation=90)
plt.tight_layout()
plt.figure(figsize=(12,6))
sns.lineplot(x = rotterdam_data["Week of datebegin"], y = rotterdam_data["Avg. AirQualityLevel"], sort=False)
plt.xlabel("Rotterdam")
plt.xticks(rotation=90)
plt.tight_layout()
plt.figure(figsize=(12,6))
sns.lineplot(x = utrecht_data["Week of datebegin"], y = utrecht_data["Avg. AirQualityLevel"], sort=False)
plt.xlabel("Utrecht")
plt.xticks(rotation=90)
plt.tight_layout()
plt.figure(figsize=(12,6))
sns.lineplot(x = sgravenhage_data["Week of datebegin"], y = sgravenhage_data["Avg. AirQualityLevel"], sort=False)
plt.xlabel("'s-Gravenhage")
plt.xticks(rotation=90)
plt.tight_layout()