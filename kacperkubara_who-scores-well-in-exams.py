import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
dataset = pd.read_csv("../input/StudentsPerformance.csv")
dataset.head()
# Create figures and axes
fig0, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize = [12.8, 9.6])
# Plot the histograms
sns.distplot(dataset["math score"], kde = False, label = "Maths", ax = ax0, color = 'b')
ax0.set_title("Math") 
ax0.set_xlabel("") # Remove xlabel
sns.distplot(dataset["reading score"], kde = False, label = "Reading", ax = ax1, color = 'g')
ax1.set_title("Reading")
ax1.set_xlabel("")
sns.distplot(dataset["writing score"], kde = False, label = "Writing", ax = ax2, color = 'y')
ax2.set_title("Writing")
ax2.set_xlabel("")
# Create figures and axes
fig1, ((ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(3, 2, figsize = [12.8, 9.6])

# Plot the histograms for exam scores distribiution based on gender
dataset_male = dataset[dataset["gender"] == "male"]
dataset_female = dataset[dataset["gender"] == "female"]

# Plot the exam distributions
sns.distplot(dataset_male["math score"], kde = False, label = "Maths", ax = ax3, color = 'b')
ax3.set_xlabel("Math_male")
sns.distplot(dataset_female["math score"], kde = False, label = "Maths", ax = ax4, color = 'b')
ax4.set_xlabel("Math_female")
sns.distplot(dataset_male["reading score"], kde = False, label = "Reading", ax = ax5, color = 'g')
ax5.set_xlabel("Reading_male")
sns.distplot(dataset["reading score"], kde = False, label = "Reading", ax = ax6, color = 'g')
ax6.set_xlabel("Reading_female")
sns.distplot(dataset_male["writing score"], kde = False, label = "Writing", ax = ax7, color = 'y')
ax7.set_xlabel("Writing_male")
sns.distplot(dataset_female["writing score"], kde = False, label = "writing", ax = ax8, color = 'y')
ax8.set_xlabel("Writing_female")
# Visualise the mean score based on gender
male_mean = dataset_male[["math score", "reading score", "writing score"]].mean()
female_mean = dataset_female[["math score", "reading score", "writing score"]].mean()
mean_scores_by_gender = pd.concat([male_mean, female_mean], axis = 1, names = ["test", "lol"])
mean_scores_by_gender.columns = ["Male Mean", "Female Mean"] 
display(mean_scores_by_gender)
# Display the labels for the education
display(dataset["parental level of education"].unique())
dataset["parental level of education"] = dataset["parental level of education"].map(lambda x: "high school" if x == "some high school" else x)
dataset["parental level of education"] = dataset["parental level of education"].map(lambda x: "college" if x == "some college" else x)
education_level_list = dataset["parental level of education"].unique()
display(education_level_list)
# Initialise the figure and df_mean to store mean values
df_mean = pd.Series()
fig2 , ax = plt.subplots(3, 1, figsize = [12.8, 15], sharex= True)

# Create neat table for mean values
for i, education_level in enumerate(education_level_list):
    mean = dataset[dataset["parental level of education"] == education_level].mean()
    mean = mean.rename(education_level)
    df_mean = pd.concat([df_mean, mean], axis = 1, sort = False)

df_mean = df_mean.drop(df_mean.columns[0], axis = 1)

# Plot the exam score based on parental education
ax[0] = sns.barplot(x = "parental level of education", y = "math score", 
                    data = dataset, estimator = np.mean, ax = ax[0])
ax[1] = sns.barplot(x = "parental level of education", y = "reading score", 
                    data = dataset, estimator = np.mean, ax = ax[1])
ax[2] = sns.barplot(x = "parental level of education", y = "writing score", 
                    data = dataset, estimator = np.mean, ax = ax[2])
for axes in ax:
    axes.set_xlabel("")
# Display the mean table
display(df_mean)

# Display a heatmap with the numeric values in each cell
fig4, ax9 = plt.subplots(figsize=(12.8, 6))
sns.heatmap(df_mean,linewidths=.1, ax=ax9)
# Results based on the lunch type
dataset_lunch = dataset[["lunch", "math score", "reading score", "writing score"]].copy()
dataset_lunch = dataset_lunch.groupby(by = ["lunch"]).mean()
# Display the table and the heatmap
display(dataset_lunch)
fig5, ax10 = plt.subplots(figsize=(12.8, 6))
sns.heatmap(dataset_lunch,linewidths=.1, ax=ax10)

dataset_preparation = dataset[["test preparation course", "math score", "reading score", "writing score"]].copy()
dataset_preparation = dataset_preparation.groupby(by = ["test preparation course"]).mean()
display(dataset_preparation)
def score_labels(x):
    if x<35:
        return "very low"
    if x>=35 and x<55:
        return "low"
    if x>=55 and x<65:
        return "average"
    if x>=65 and x<75:
        return "good"
    if x>=75 and x<85:
        return "high"
    if x>=85 and x<=100:
        return "very high"    
# Read the data
"""
Create classes for the exam scores
0-35%    - very low
35-55%   - low
55-65%   - average
65%-75%  - good
75-85%   - high
85%-100% - very high
"""
# Make an average score from 3 exams and label them as above
average_score = dataset.iloc[:,-3:]
x_num =  dataset.iloc[:,-3:]
average_score = average_score.applymap(score_labels)
x = average_score
x_copy = x.copy()

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
hot_enc_x   = OneHotEncoder()
label_enc_x = LabelEncoder()

x = x.apply(label_enc_x.fit_transform)
x = hot_enc_x.fit_transform(x).toarray()
display(x[:,:5])
# Fitting K-Means to the dataset
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 5, init = 'k-means++')
y_kmeans = kmeans.fit_predict(x)
x_num["cluster"] = y_kmeans
# Visualising the clusters
from mpl_toolkits.mplot3d import axes3d
fig6 = plt.figure(figsize = (12.8, 9))
ax11 = fig6.add_subplot(111, projection='3d')
ax11.scatter((x_num[x_num.cluster == 0])["math score"].values, (x_num[x_num.cluster == 0])["reading score"].values, (x_num[x_num.cluster == 0])["writing score"].values, s = 100, c = 'red', label = 'Cluster 1')
ax11.scatter((x_num[x_num.cluster == 1])["math score"].values, (x_num[x_num.cluster == 1])["reading score"].values, (x_num[x_num.cluster == 1])["writing score"].values, s = 100, c = 'blue', label = 'Cluster 2')
ax11.scatter((x_num[x_num.cluster == 2])["math score"].values, (x_num[x_num.cluster == 2])["reading score"].values, (x_num[x_num.cluster == 2])["writing score"].values, s = 100, c = 'green', label = 'Cluster 3')
ax11.scatter((x_num[x_num.cluster == 3])["math score"].values, (x_num[x_num.cluster == 3])["reading score"].values, (x_num[x_num.cluster == 3])["writing score"].values, s = 100, c = 'cyan', label = 'Cluster 4')
ax11.scatter((x_num[x_num.cluster == 4])["math score"].values, (x_num[x_num.cluster == 4])["reading score"].values, (x_num[x_num.cluster == 4])["writing score"].values, s = 100, c = 'magenta', label = 'Cluster 5')
ax11.set_title('Clusters of Students')
ax11.set_xlabel('Math')
ax11.set_ylabel('Reading')
ax11.set_zlabel('Writing')
ax11.legend()