# Import some packages
import numpy as np 
import pandas as pd 
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score as sl
import matplotlib.pyplot as plt
import seaborn as sbn

# Read the survey data and for convenience drop the question row
kd_init = pd.read_csv("../input/multipleChoiceResponses.csv").drop(index=0, axis=0)

# Lets impute some values for NaNs and remove some non-ascii chars for some troublesome Pandas Series
kd_init["Q8"] = kd_init["Q8"].fillna("0-1").astype(str)
kd_init["Q23"] = kd_init["Q23"].fillna("1% to 25% of my time").astype(str)
kd_init["Q24"] = kd_init["Q24"].fillna("< 1 year")
kd_init["Q25"] = kd_init["Q25"].fillna("I have never studied machine learning but plan to learn in the future")
kd_init["Q4"] = kd_init["Q4"].fillna("Masters degree")
kd_init["Q4"] = kd_init["Q4"].astype(str).apply(lambda x: re.sub(r'[^\x00-\x7F]', '', x).replace("/", " "))

# Define the clustering dataframe and modfiy question feature names to be more compact
kd = pd.DataFrame()
feature_names = {"Q4": "Degree Level",
                 "Q5": "Degree Major",
                 "Q8": "Years Experience",
                 "Q11": "Work Activities",
                 "Q12": "Data Analysis Tools Used",
                 "Q16": "Programming Langs Used",
                 "Q17": "Languages Used",
                 "Q19": "ML Frameworks Used",
                 "Q22": "Viz Used",
                 "Q23": "Coding Activity",
                 "Q24": "Coding Experience",
                 "Q25": "ML Experience",
                 "Q30": "Big Data Products",
                 "Q34": "Data Science Project Time",
                 "Q35": "ML Training"}

# Add coarse deep learning & machine learning categories for more granular Questions 3 and 20
US = ["United States of America"]
EU = ["United Kingdom of Great Britain and Northern Ireland", "France", 
      "Germany", "Italy", "Spain", "Netherlands", "Poland", "Sweden", 
      "Norway", "Greece", "Portugal", "Switzerland", "Denmark", 
      "Belgium", "Ireland", "Finland", "Hungary"]
BRICCJ = ["India", "China", "Russia", "Brazil", "Canada", "Japan"]
kd_init["Q3"] = kd_init["Q3"].apply(lambda x: 
                                    "US" if x in US else (
                                        "Europe" if x in EU else (
                                            x if x in BRICCJ else "Other")))

TF = ["TensorFlow"]
Other_DL = ["Keras", "PyTorch", "H2O", "Fastai", "Mxnet", "Caret", "CNTK", "Caffe"]
SKL = ["Scikit-Learn"]
Other_ML = ["Other", "catboost", "lightgbm", "randomForest", "Prophet", "mlr", "Xgboost", "Spark MLlib"]
kd_init["Q20"] = kd_init["Q20"].apply(lambda x: 
                                      "TensorFlow" if x in TF else (
                                          "Other Deep Learning" if x in Other_DL else (
                                              "SKLearn" if x in SKL else (
                                                  "Other Machine Learning" if x in Other_ML else "None"))))

# Custom label encoding for categorical questions with obvious ordinality
# N.B. Q4 has some noise due to no answer responses, so impute mode (master's degree)
ords = {
    "Q4": {
        "No formal education past high school": 0,
        "Some college university study without earning a bachelors degree": 1,
        "Bachelors degree": 2,
        "Professional degree": 3,
        "I prefer not to answer": 4,
        "Masters degree": 4,
        "Doctoral degree": 5
},
    "Q8": {
        "0-1": 0,
        "1-2": 1,
        "2-3": 1,
        "3-4": 2,
        "4-5": 2,
        "5-10": 3,
        "10-15": 4,
        "15-20": 5,
        "20-25": 5,
        "25-30": 5,
        "30 +": 5
    },
    "Q25": {
        "I have never studied machine learning and I do not plan to": 0,
        "I have never studied machine learning but plan to learn in the future": 1,
        "< 1 year": 2,
        "1-2 years": 3,
        "2-3 years": 4,
        "3-4 years": 5,
        "4-5 years": 6,
        "5-10 years": 7,
        "10-15 years": 8,
        "20+ years": 9
    },
    "Q24": {
        "I have never written code and I do not want to learn": 0,
        "I have never written code but I want to learn": 1,
        "< 1 year": 2,
        "1-2 years": 3,
        "3-5 years": 4,
        "5-10 years": 5,
        "10-20 years": 6,
        "20-30 years": 7,
        "30-40 years": 8,
        "40+ years": 9
    },
    "Q23": {
        "0% of my time": 0,
        "1% to 25% of my time": 1,
        "25% to 49% of my time": 2,
        "50% to 74% of my time": 3,
        "75% to 99% of my time": 4,
        "100% of my time": 5
    }
}

# Perform the actual encoding
for keys in ords.keys():
    name = feature_names[keys]
    kd[name] = kd_init[keys].apply(lambda x: ords[keys][x.replace("/", " ")])

# Binary encoding and feature renaming for questions: Q11, Q12, Q34, and Q35
# Q11 - Work Activity
for key in range(1, 7):
    keyn = 'Q11_Part_' + str(key)
    name = feature_names[keyn.split("_")[0]] + "_" + str(kd_init[keyn][kd_init[keyn].notnull()].iloc[0])
    kd[name] = np.where(kd_init[keyn].fillna(0) == 0, 0, 1)

# Q12 - Data Analysis Tools
q_feat = {1: "Spreadsheets", 2: "Advances Stats", 3: "BI Tools", 4: "Development Env",
          5: "Cloud SaaS", 6: "Other"}
for key in range(1, 6):
    keyn = 'Q12_Part_' + str(key) + "_TEXT"
    name = feature_names[keyn.split("_")[0]] + "_" + q_feat[key]
    kd[name] = np.where(kd_init[keyn].fillna(0) == 0, 0, 1)
kd[feature_names["Q12"] + "_Other"] = np.where(kd_init["Q12_OTHER_TEXT"].fillna(0) == 0, 0, 1)

# Q34 - Data Science Project Time
q_feat = {1: "Gathering data", 2: "Cleaning data", 3: "Visualizing data", 4: "Model building and selection",
          5: "Putting the model into production", 6: "Data insights and comms",
          7: "Other"}
for key in range(1, 6):
    keyn = 'Q34_Part_' + str(key)
    name = feature_names[keyn.split("_")[0]] + "_" + q_feat[key]
    kd[name] = np.where(kd_init[keyn].fillna(0) == 0, 0, 1)
kd[feature_names["Q34"] + "_Other"] = np.where(kd_init["Q34_OTHER_TEXT"].fillna(0) == 0, 0, 1)

# Q35 - ML Training (Work, Kaggle, or Uni)?
q_feat = {1: "Self-taught", 2: "Online courses", 3: "Work", 4: "University", 5: "Kaggle", 6: "Other"}
for key in range(1, 7):
    keyn = 'Q35_Part_' + str(key)
    name = feature_names[keyn.split("_")[0]] + "_" + q_feat[key]
    kd[name] = np.where(kd_init[keyn].fillna(0) == 0, 0, 1)

# Add questions for marginal distributions after clustering
kd["Q3"] = kd_init["Q3"]
kd["Q5"] = kd_init["Q5"]
kd["Q6"] = kd_init["Q6"]
kd["Q10"] = kd_init["Q10"]
kd["Q17"] = kd_init["Q17"]
kd["Q20"] = kd_init["Q20"]
kd["Q22"] = kd_init["Q22"]

print(kd.shape)
# Define features used for marginal exploration only
marginal_names = ["Q3", "Q5", "Q6", "Q10", "Q17", "Q20", "Q22"]
marginal_questions = ["Country of Residence", "Undergrad Major", "Current Title", 
                      "Employer Incorporates ML", "Most Used Programming Language",
                      "Most Used ML Framework", "Most Used Viz Tool"]

# Clustering features
cluster_features = [val for val in kd.columns if val not in marginal_names]
print("Cluster Features")
print(cluster_features)
# My responses
me = [5, 8, 7, 6, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1]
# Separate all self-identified students from identified non-students
kd_non_students = kd[(kd['Q6'] != "Student") & (kd_init['Q7'] != "I am a student")]
kd_students = kd[(kd['Q6'] == "Student") | (kd_init['Q7'] == "I am a student")]
print("Student feature space", kd_students.shape)
print("Non-student feature space", kd_non_students.shape)
# Z-score norm features for student and non-student clustering first to avoid any feature dynamic range dominance
stsc = StandardScaler()
kd_st = stsc.fit_transform(kd_students[cluster_features])
kd_nst = stsc.fit_transform(kd_non_students[cluster_features])

# Cluster for students personas - N.B. I added the names for the personas after initial interpretation of the clusters
personas = ['The Aspiring', 'The Coder', 'The Worker']
km = KMeans(n_clusters=3, random_state=222)
kmf = km.fit(kd_st)
st_centroids = kmf.cluster_centers_
st_results = pd.DataFrame(st_centroids, columns=cluster_features, index=personas).T
kd_students['cluster'] = kmf.predict(kd_st)

# Clustering for non-students - N.B. I added the names for the personas after initial interpretation of the clusters
personas_nst = ['The Freshman', 'The Data Engineer', 'The ML Researcher', 'The Practitioner']
km = KMeans(n_clusters=4, random_state=111)
kmn = km.fit(kd_nst)
nst_centroids = kmn.cluster_centers_
nst_results = pd.DataFrame(nst_centroids, columns=cluster_features, index=personas_nst).T
kd_non_students['cluster'] = kmn.predict(kd_nst)

# What is my cluster in Non-Student segment?
my_cluster = kmn.predict(np.asarray(me).reshape(1, -1))
print(my_cluster)

# Compute the silhouette_score is for each of our segments
student_ss_score = sl(kd_st, kd_students['cluster'])
nonstudent_ss_score = sl(kd_nst, kd_non_students['cluster'])
print("Student Clusters Silhouette Score", student_ss_score)
print("Non-Student Clusters Silhouette Score", nonstudent_ss_score)
# Heatmaps of student persona centroids
plt.figure(figsize=(18, 15))
ax = sbn.heatmap(st_results, cmap = 'GnBu', annot=True, cbar=True, vmin=-1, vmax=1)
plt.title('Self-Identified Student - Centroids vs Cluster Persona', fontsize=24)
plt.xlabel('Persona', fontsize=24)
plt.ylabel('Skill', fontsize=24)
plt.setp(ax.get_xticklabels(), rotation='30', fontsize=20)
plt.setp(ax.get_yticklabels(), rotation='horizontal', fontsize=12)
plt.show()
plt.savefig('Identified_Student_Centroids_vs_Cluster_Persona.png')
plt.close()
# Heatmap of non-student persona centroids
plt.figure(figsize=(18, 15))
ax = sbn.heatmap(nst_results, cmap='GnBu', annot=True, cbar=True, vmin=-0.9, vmax=0.9)
plt.title('Self-Identified Non-Student - Centroids vs Cluster Persona', fontsize=24)
plt.xlabel('Persona', fontsize=24)
plt.ylabel('Skill', fontsize=24)
plt.setp(ax.get_xticklabels(), rotation='30', fontsize=20)
plt.setp(ax.get_yticklabels(), rotation='horizontal', fontsize=12)
plt.show()
plt.savefig('Identified_Non_Student_Centroids_vs_Cluster_Persona.png')
plt.close()
# Plot Student marginal histograms for marginal_features questions
marginal_names_st = ["Q3", "Q5", "Q17", "Q20", "Q22"]
marginal_questions_st = ["Country of Residence", "Undergrad Major", "Most Used Programming Language",
                         "Most Used ML Framework", "Most Used Viz Tool"]
for j, key in enumerate(marginal_names_st):
    ct = 1
    plt.figure(figsize=(25, 25))
    for i, cluster in enumerate(personas):
        # print key, cluster
        # print kd_students["cluster"][kd_students['cluster'] == i].count(), kd_students[key][kd_students['cluster'] == i].count(), \
        #    kd_students[key][kd_students['cluster'] == i].count()/float(kd_students["cluster"][kd_students['cluster'] == i].count())

        total_resps = float(kd_students["cluster"][kd_students['cluster'] == i].count())
        count = kd_students[key][kd_students['cluster'] == i].value_counts()

        ax = plt.subplot(len(personas), 1, ct)
        plt.title(cluster + " " + marginal_questions_st[j], fontsize=20)
        plt.barh(count.index, count/float(total_resps))
        plt.setp(ax.get_yticklabels(), rotation='horizontal', fontsize=18)
        plt.setp(ax.get_xticklabels(), fontsize=18)
        plt.xlim(0, 0.6)
        plt.ylabel("Percent of Total Responses", fontsize=14)
        ct += 1
    plt.show()
# Plot Non-Student segment marginal histograms for some questions
for j, key in enumerate(marginal_names):
    ct = 1
    plt.figure(figsize=(25, 25))
    for i, cluster in enumerate(personas_nst):
        # print key, cluster
        # print kd_non_students["cluster"][kd_non_students['cluster'] == i].count(), kd_non_students[key][
        #    kd_non_students['cluster'] == i].count(), \
        #    kd_non_students[key][kd_non_students['cluster'] == i].count() / float(
        #        kd_non_students["cluster"][kd_non_students['cluster'] == i].count())

        total_resps = float(kd_non_students["cluster"][kd_non_students['cluster'] == i].count())
        count = kd_non_students[key][kd_non_students['cluster'] == i].value_counts()

        ax = plt.subplot(len(personas_nst), 1, ct)
        plt.title(cluster + " " + marginal_questions[j], fontsize=16)
        plt.barh(count.index, count / float(total_resps))
        plt.setp(ax.get_yticklabels(), rotation='horizontal', fontsize=18)
        plt.setp(ax.get_xticklabels(), fontsize=18)
        plt.xlim(0, 0.6)
        plt.ylabel("Percent of Total Responses", fontsize=14)
        ct += 1
    plt.show()