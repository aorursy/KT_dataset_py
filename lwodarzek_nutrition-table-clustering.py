import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set()

from sklearn.mixture import GaussianMixture
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator

import os
print(os.listdir("../input"))
original = pd.read_csv("../input/en.openfoodfacts.org.products.tsv",delimiter='\t', encoding='utf-8', nrows = 50000)
original.head()
nan_values = original.isnull().sum()
nan_values = nan_values / original.shape[0] *100

plt.figure(figsize=(20,8))
sns.distplot(nan_values, kde=False, bins=np.int(original.shape[0]/100), color = "Red")
plt.xlabel("Percentage of nan_values", size = 15)
plt.ylabel("Frequency", size = 15)
plt.title("Missing values by feature", size = 20)
def split_data_by_nan(cutoff):
    cols_of_interest = nan_values[nan_values <= cutoff].index
    data = original[cols_of_interest]
    return data.copy()

low_nan_data = split_data_by_nan(10)

print("Original number of features: " + str(original.shape[1]))
print("Number of features with less than 10 % nans: " + str(low_nan_data.shape[1]))
low_nan_data.columns
nutrition_table_cols = ["fat_100g", "carbohydrates_100g", "sugars_100g", "proteins_100g", "salt_100g", "energy_100g"]
nutrition_table = low_nan_data[nutrition_table_cols].copy()
nutrition_table["isempty"] = np.where(nutrition_table.isnull().sum(axis=1) >= 1, 1, 0)
percentage = nutrition_table.isempty.value_counts()[1] / nutrition_table.shape[0] * 100
print("Percentage of incomplete tables: " + str(percentage))
nutrition_table = nutrition_table[nutrition_table.isempty==0].copy()
nutrition_table.isnull().sum()
nutrition_table.drop("isempty", inplace=True,axis=1)
nutrition_table["reconstructed_energy"] = nutrition_table["fat_100g"] * 39 + nutrition_table["carbohydrates_100g"] * 17 + nutrition_table["proteins_100g"] * 17

nutrition_table.head()
plt.figure(figsize = (10,10))
plt.scatter(nutrition_table["energy_100g"], nutrition_table["reconstructed_energy"], s = 0.6, c= "goldenrod")
plt.xlabel("given energy")
plt.ylabel("reconstructed energy")
nutrition_table["g_sum"] = nutrition_table.fat_100g + nutrition_table.carbohydrates_100g + nutrition_table.proteins_100g

nutrition_table["exceeded"] = np.where(nutrition_table.g_sum.values > 100, 1, 0)
nutrition_table[nutrition_table["exceeded"] == 1].head()
nutrition_table.exceeded.value_counts() 
nutrition_table["product"] = original.loc[nutrition_table.index.values]["product_name"] 
nutrition_table.to_csv("nutrition_table.csv", header=True, index=True) 
feature = "g_sum"
colors_dict = {"fat_100g": "lightskyblue", "carbohydrates_100g": "limegreen", "sugars_100g": "hotpink", "proteins_100g": "mediumorchid", "energy_100g": "gold", "salt_100g": "gray", "reconstructed_energy": "orange", "g_sum": "m"}
plt.figure(figsize=(20,5))
sns.distplot(nutrition_table[feature], kde = False, color = colors_dict[feature])
plt.xlabel(feature)
plt.ylabel("frequency")
features = ["fat_100g", "carbohydrates_100g", "sugars_100g", "proteins_100g", "salt_100g", "energy_100g", "reconstructed_energy", "g_sum"]
X_train = nutrition_table[features].values

model = GaussianMixture(n_components=14, covariance_type="full", n_init = 5, max_iter = 200)
model.fit(X_train)
log_prob = model.score_samples(X_train)
results = nutrition_table[features].copy()
results["cluster"] = model.predict(X_train)
results["product_name"] = original.loc[nutrition_table.index.values, "product_name"]

probas = np.round(model.predict_proba(X_train), 2)
cluster_values = results.cluster.values
certainty = np.zeros(cluster_values.shape[0])
for n in range(len(certainty)):
    certainty[n] = probas[n,cluster_values[n]]
    
results["certainty"] = certainty
results[results.cluster == 2].head(15)
def get_outliers(log_prob, treshold):
    epsilon = np.quantile(log_prob, treshold)
    outliners = np.where(log_prob <= epsilon, 1, 0)
    return outliners 
your_choice = 0.12
plt.figure(figsize=(20,5))

sns.distplot(log_prob, kde=False, bins=50, color="Red")
g1 = plt.axvline(np.quantile(log_prob, 0.25), color="Green", label="Q_25")
g2 = plt.axvline(np.quantile(log_prob, 0.5), color="Blue", label="Q_50 - Median")
g3 = plt.axvline(np.quantile(log_prob, 0.75), color="Green", label="Q_75")
g4 = plt.axvline(np.quantile(log_prob, your_choice), color="Purple", label="Q_ %i" % (int(your_choice*100)))
handles = [g1, g2, g3, g4]
plt.xlabel("log-probabilities of the data spots")
plt.xlim((-50,0))
plt.ylabel("frequency")
plt.legend(handles) 
results["anomaly"] = get_outliers(log_prob, your_choice)
results.head()
features = ["energy_100g", "reconstructed_energy"]
plt.figure(figsize = (10,10))
plt.scatter(results[features[0]], results[features[1]], c=results.anomaly.values, cmap = "coolwarm", s = 5.5)
plt.xlabel(features[0])
plt.ylabel(features[1])
results.to_csv("clustering_and_anomalies.csv", header=True, index=True) 
results[results.anomaly == 1].head(50)