import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

data = pd.read_excel("../input/btech-results-iv1/26-5-IV B.Tech. I Semester (R15) Regular.xls");
data["Percentage"] = data["Total Marks"]/750;
marks = data[data["Total Marks"] > 100];
print("Top Five Scorers")
print(marks.sort_values("Total Marks", ascending=False)[["Hallticket No", "Total Marks", "Percentage"]].head(5))
avg = data[data["Total Marks"] < 100].groupby("Subject Name").mean();
plt.barh(avg.index, avg["External Marks"]);
plt.grid()
plt.title("Average External Marks")
plt.show()
total = len(data["Hallticket No"].unique())
print("Number of Failed Papers "+ str(sum(data["External Marks"] < 26)))
total_fail = data[data["External Marks"] < 26].groupby("Subject Name").count();
plt.barh(total_fail.index,total_fail["Credits"])
plt.title("Failures Per Subject");
plt.grid()
plt.show()
print("Percentage of failures Per Subject")
print(100*total_fail["Credits"]/total)
subjects = ["CLOUD COMPUTING", "COMPUTER GRAPHICS", "DATA WAREHOUSING AND DATA MINING",
            "DESIGN PATTERNS", "LINUX PROGRAMMING", "MACHINE LEARNING"]
total_cse1 = 61
print("ESTIMATED PASS PERCENTAGE FOR CSE-1 (only Normals and LEs)")
for subject in subjects:
    cse1_norm = data[(data["Subject Name"] == subject) &(data["Hallticket No"] < "15261A0561")]
    cse1_le = data[(data["Subject Name"] == subject) &(data["Hallticket No"] > "16265A0500") & (data["Hallticket No"] < "16265A0513")]

    pass_norm = sum(cse1_norm["External Marks"]>=26)
    pass_le = sum(cse1_le["External Marks"]>=26)

    total_pass = pass_norm+pass_le

    print(subject +":"+str(100*total_pass/total_cse1)+"%")
def getRank(rollno):
    sorted_marks = marks.sort_values("Total Marks", ascending = False).reset_index()
    print("Congrats! Your rank is: ")
    print(sorted_marks[sorted_marks["Hallticket No"] == rollno].index.values.astype(int)[0]+1)
    
getRank("15261A0551")
def distribution(spec_marks, title):
    fig, ax_l = plt.subplots(1,3, figsize = (20, 5))
    ax_l[0].scatter(range(spec_marks.shape[0]),spec_marks)
    ax_l[1].hist(spec_marks)
    ax_l[2].boxplot(spec_marks)
    ax_l[0].grid()
    ax_l[1].grid()
    fig.suptitle("Distribution of Marks in "+title, fontsize=16)
    plt.show()

distribution(marks["External Marks"], "Total External Marks")
lp = data[data["Subject Name"] == "LINUX PROGRAMMING"]["External Marks"]
dp = data[data["Subject Name"] == "DESIGN PATTERNS"]["External Marks"]
cc = data[data["Subject Name"] == "CLOUD COMPUTING"]["External Marks"]
dm = data[data["Subject Name"] == "DATA WAREHOUSING AND DATA MINING"]["External Marks"]
ml = data[data["Subject Name"] == "MACHINE LEARNING"]["External Marks"]
mc = data[data["Subject Name"] == "MOBILE COMPUTING"]["External Marks"]
cg = data[data["Subject Name"] == "COMPUTER GRAPHICS"]["External Marks"]
distribution(lp, "Linux Programming")
distribution(dp, "Design Patterns")
distribution(cc, "Cloud Computing")
distribution(dm, "Data Warehousing and Mining")
distribution(ml, "Machine Learning")
distribution(mc, "Mobile Computing")
distribution(cg, "Computer Graphics")
lp = data[data["Subject Name"] == "LINUX PROGRAMMING"]
dp = data[data["Subject Name"] == "DESIGN PATTERNS"]
cc = data[data["Subject Name"] == "CLOUD COMPUTING"]
dm = data[data["Subject Name"] == "DATA WAREHOUSING AND DATA MINING"]
ml = data[data["Subject Name"] == "MACHINE LEARNING"]
mc = data[data["Subject Name"] == "MOBILE COMPUTING"]
cg = data[data["Subject Name"] == "COMPUTER GRAPHICS"]

def findGod(subject_name):
    spec_data = data[data["Subject Name"] == subject_name]
    print("God of "+subject_name)
    print(spec_data[spec_data["External Marks"] == max(spec_data["External Marks"])][["Hallticket No","External Marks"]])
    
findGod("LINUX PROGRAMMING")
findGod("DESIGN PATTERNS")
findGod("CLOUD COMPUTING")
findGod("DATA WAREHOUSING AND DATA MINING")
findGod("MACHINE LEARNING")
findGod("MOBILE COMPUTING")
findGod("COMPUTER GRAPHICS")
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

X = data[["Internal Marks", "External Marks"]].values
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
kmeans.labels_
plt.figure(figsize = (7,7))
plt.xlabel("Internal Marks")
plt.ylabel("External Marks")
plt.scatter(data["Internal Marks"], data["External Marks"], c = kmeans.labels_)
plt.title("Clustering on Internal and External Marks")
plt.show()
X = data.pivot(index = "Hallticket No",columns = "Subject Name", values = "External Marks").fillna(0)
#kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
#kmeans.labels_

X.columns = ["TOTAL",'CLOUD COMPUTING','COMPUTER GRAPHICS', 'DATA WAREHOUSING AND DATA MINING',
        'DATA WAREHOUSING AND MINING LAB', 'DESIGN PATTERNS',
        'LINUX PROGRAMMING', 'LINUX PROGRAMMING LAB',
        'MACHINE LEARNING', 'MOBILE COMPUTING']
X = X.drop(columns = ["TOTAL"])
fig, ax = plt.subplots(figsize = (10,10))
ax.set_xticklabels(X.columns,rotation=90, fontsize=10)
ax.set_yticklabels(X.columns,rotation=0, fontsize=10)
ax.set_xticks(np.arange(len(X.columns)))
ax.set_yticks(np.arange(len(X.columns)))
ax.set_title("Correlation of External Marks between Subjects");
cax = ax.imshow(X.corr())

fig.colorbar(cax, ticks=[0.1,0.2,0.3,0.4,0.5,0.6,0.7, 0.8, 0.9, 1, 0.0, -0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7, -0.8, -0.9, -1])
plt.show()
#Decision factor
def sig(x):
    return 1/(1 + np.exp(-1*x))

def decision_factor(rollno):
    myscore = data[data["Hallticket No"] == rollno].groupby("Subject Name").sum();
    f = avg["External Marks"]-myscore["External Marks"]
    val = total_fail["External Marks"].multiply(sig(f), fill_value = 1).drop(["DATA WAREHOUSING AND MINING LAB","LINUX PROGRAMMING LAB"])
    print("Decision Factor for Revaluation: "+rollno)
    #print(total_fail.loc["CLOUD COMPUTING", "Credits"])
    if(val["MOBILE COMPUTING"] == 40.0):
        print(round(val.sort_values(ascending = False), 5).drop("MOBILE COMPUTING").head(3))
    else:
        print(round(val.sort_values(ascending = False), 5).drop("COMPUTER GRAPHICS").head(3))

hts = data["Hallticket No"].unique()
for ht in hts:
    decision_factor(ht)
#Might suffer from Effect of Small Number tendancy 