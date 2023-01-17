import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
df = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict.csv', sep=r'\s*,\s*',
                           header=0, encoding='ascii', engine='python')
df
# Dropping the unnecessary iterative variable
df = df.drop("Serial No.", axis=1)
df.isnull().sum()
sns.countplot("University Rating", data=df)
# Dataframe containing all the enteries of students who have undertaken research
df_research = df[df["Research"] == 1]

# Dataframe containing all the enteries of students who havent undertaken research
df_no_research = df[df["Research"] == 0]

# Getting a distribution of the likelihood of admission for the students who have gotten an admit and who havent 
plt.figure(figsize=(9, 8))
sns.distplot((df_research["Chance of Admit"]), color='g', bins=100, hist_kws={'alpha': 0.4});
sns.distplot((df_no_research["Chance of Admit"]), color='b', bins=100, hist_kws={'alpha': 0.4});
corr = df.corr() 
plt.figure(figsize=(12, 10))

sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True)
for i in range(0, len(df.columns), 5):
    sns.pairplot(data=df,
                x_vars=df.columns[i:i+5],
                y_vars=['Chance of Admit'])
plt.figure(figsize=(15, 6))
ax = sns.boxplot(x="SOP", y="Chance of Admit", hue="SOP", data=df)
SOP_ratings_to_filter =  [4, 4.5, 5]

# We will consider the outliers to be the ones which lie in the lower 10 percentile of the enteries for that rating of the SOP
for i in SOP_ratings_to_filter:
    df_i = df[df["SOP"] == i]
    df_outliers = df_i[(df_i["Chance of Admit"] < df["Chance of Admit"].quantile(0.30))]
    display(df_outliers)
plt.figure(figsize=(15, 6))
ax = sns.boxplot(x="LOR", y="Chance of Admit", hue="LOR", data=df)
LOR_ratings_to_filter =  [4, 4.5, 5]

# We will consider the outliers to be the ones which lie in the lower 10 percentile of the enteries for that rating of the SOP
for i in LOR_ratings_to_filter:
    df_i = df[df["LOR"] == i]
    df_outliers = df_i[(df_i["Chance of Admit"] < df["Chance of Admit"].quantile(0.30))]
    display(df_outliers)
g =sns.scatterplot(x="GRE Score", y="TOEFL Score",
              hue="Chance of Admit",
              data=df);
g.set(xscale="log");
g =sns.scatterplot(x="SOP", y="LOR",
              hue="Chance of Admit",
              data=df);
g.set(xscale="log");
mean_liklehood = df["Chance of Admit"].mean()
mean_liklehood
df["Admit"] = [1 if df["Chance of Admit"][i] > mean_liklehood else 0 for i in range(len(df))]
sns.countplot("Admit", data=df)
df
df_admit = df[df["Admit"] == 1]
sns.countplot("University Rating", data=df_admit)
df_admit = df_admit[(df_admit["University Rating"] >= 2)]
plt.figure(figsize=(15, 7))
ax = sns.boxplot(x="Admit", y="GRE Score", hue="Admit", data=df)

df_stats = pd.DataFrame()
for i in range(3, max(df["University Rating"] + 1)):
    df_curr_rating = df_admit[(df_admit["University Rating"] == i)]
    stats = [df_curr_rating.mean()]
    df_stats = df_stats.append(stats)

df_stats
from scipy import stats

for col in df_stats:
    df_stats[col + "_percentile"] = [0 for i in range(len(df_stats))]
    arr = []
    for i, ent in enumerate(df_stats[col]):
        percentile = stats.percentileofscore(df[col], ent)
        arr.append(percentile)
    df_stats[col + "_percentile"] = arr

# df_stats = df_stats.drop(["Admit", "University Rating_percentile", "Research_percentile", "Chance of Admit_percentile", "Admit_percentile"], axis=1)
df_stats
# import matplotlib.pyplot as plt
# ax = df_stats[['GRE Score_percentile','TOEFL Score_percentile', 'CGPA_percentile', 'LOR_percentile', 'SOP_percentile']].plot(kind='bar', title ="Percentile Scores", x=df_stats["University Rating"], figsize=(15, 10), legend=True, fontsize=12)
# ax.set_xlabel("University Rating", fontsize=12)
# ax.set_ylabel("Percentile Scores", fontsize=12)
# plt.show()


df_stats.plot(x="University Rating", y=["GRE Score_percentile", "TOEFL Score_percentile", "SOP_percentile", "LOR_percentile", "CGPA_percentile"], kind="bar", figsize=(10,10))

y = df_admit["Chance of Admit"] 
X = df_admit.drop(["Chance of Admit", "Admit", "University Rating"], axis=1)
names = X.columns
rf = RandomForestRegressor()
rf.fit(X, y)
print("Features sorted by their score:")
arr = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), 
             reverse=True)
print(arr)

sns.barplot(x=[ent[0] for ent in arr], y=[ent[1] for ent in arr])
df_top = df_admit[df_admit["University Rating"] >= 4.0]
df_top
y_top = df_top["Chance of Admit"] 
X_top = df_top.drop(["Chance of Admit", "Admit", "University Rating"], axis=1)
names = X.columns
rf = RandomForestRegressor()
rf.fit(X_top, y_top)
print("Features sorted by their score:")
arr = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), 
             reverse=True)

print(arr)
sns.barplot(x=[ent[0] for ent in arr], y=[ent[1] for ent in arr])
model = sm.OLS(y,X)
results = model.fit()
results.summary()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(X_train, y_train);
predictions = rf.predict(X_test)
errors = abs(predictions - y_test)

mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
