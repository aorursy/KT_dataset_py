%matplotlib inline
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
import xgboost
import operator
df1 = pd.read_csv('../input/survey_results_schema.csv')
df2 = pd.read_csv('../input/survey_results_public.csv')
plt.figure(1)
for_plot = df2.groupby(["Hobby"]).size()
for_plot = for_plot.reset_index()
sns.barplot(x = for_plot["Hobby"] , y = for_plot[0])
plt.ylabel("Count")
plt.show()
plt.figure(1)
for_plot = df2.groupby(["OpenSource"]).size()
for_plot = for_plot.reset_index()
sns.barplot(x = for_plot["OpenSource"] , y = for_plot[0])
plt.ylabel("Count")
plt.show()
df2["is_stack_learner"] = df2["SelfTaughtTypes"].str.contains("Stack Overflow")
for_plot = df2[-pd.isnull(df2["is_stack_learner"])]
for_plot = for_plot[for_plot["is_stack_learner"]]
for_plot = for_plot.groupby(["StackOverflowParticipate"]).size().reset_index()
for_plot[0] = for_plot[0].apply(lambda x : x/for_plot[0].sum())
plt.figure(1)
D = zip(for_plot["StackOverflowParticipate"], for_plot[0])
D = sorted( list(D) , key = lambda x : x[1])
plt.figure(1 , figsize = (30,30))
x_val = [x[0] for x in D]
y_val = [x[1] for x in D]
sns.barplot(x_val , y_val)
plt.xticks(rotation = 90)
plt.show()
#So about 55% of users who use stack overflow for learning end up participating very rarely. This subset must be 
#done with further investigation
for_plot = df2[-pd.isnull(df2["is_stack_learner"])]
for_plot = for_plot[for_plot["is_stack_learner"]]
for_plot = for_plot.groupby(["StackOverflowConsiderMember"]).size().reset_index()
for_plot[0] = for_plot[0].apply(lambda x : x/for_plot[0].sum())
plt.figure(1)
D = zip(for_plot["StackOverflowConsiderMember"], for_plot[0])
D = sorted( list(D) , key = lambda x : x[1])
plt.figure(1 , figsize = (30,30))
x_val = [x[0] for x in D]
y_val = [x[1] for x in D]
sns.barplot(x_val , y_val)
plt.show()
for_plot = df2[-pd.isnull(df2["is_stack_learner"])]
for_plot = for_plot[for_plot["is_stack_learner"]]
for_plot = for_plot.groupby(["StackOverflowVisit"]).size().reset_index()
for_plot[0] = for_plot[0].apply(lambda x : x/for_plot[0].sum())
plt.figure(1)
D = zip(for_plot["StackOverflowVisit"], for_plot[0])
D = sorted( list(D) , key = lambda x : x[1])
plt.figure(1 , figsize = (30,30))
x_val = [x[0] for x in D]
y_val = [x[1] for x in D]
sns.barplot(x_val , y_val)
plt.xticks(rotation = 90)
plt.show()
for_plot = df2[-pd.isnull(df2["is_stack_learner"])]
for_plot = for_plot[for_plot["is_stack_learner"]]
for_plot = for_plot[(for_plot["StackOverflowVisit"] == "Daily or almost daily") | (for_plot["StackOverflowVisit"] == "Multiple times per day")]
for_plot = for_plot.groupby(["StackOverflowParticipate"]).size().reset_index()
for_plot[0] = for_plot[0].apply(lambda x : x/for_plot[0].sum())
D = zip(for_plot["StackOverflowParticipate"], for_plot[0])
D = sorted( list(D) , key = lambda x : x[1])
plt.figure(1 , figsize = (5,5))
x_val = [x[0] for x in D]
y_val = [x[1] for x in D]
sns.barplot(x_val , y_val)
plt.xticks(rotation = 90)
plt.show()
tempo = df1[(df1["Column"].str.contains("StackOverflow")) | (df1["Column"].str.contains("Hypothetical"))].reset_index()
tempo
tempo.loc[9][2]  #Hypothetical Tools2
tempo.loc[0][2] #StackOverflowRecommend
tempo.loc[2][2] #StackOverflowHasAccount
tempo.loc[3][2] #StackOverflowParticipation
tempo.loc[7][2] #Positive 1.5
code  =  {'Extremely interested' :4 , 'Very interested' :3 , 'Somewhat interested' : 2 , 'A little bit interested' : 1 , 'Not at all interested' :0 }
code_reco = {'10 (Very Likely)' :4 , '7' :3 , '9' : 4 , '8' : 3,  '0' : 0 , '1' : 0 , '6' : 2 , '5' : 2 , '4' : 1.5, '3':1 , '2': 1 }
df2["Ease of Participation"] = -1.5 * df2["HypotheticalTools2"].map(code) #-1.5 weight
recommend = df2["StackOverflowRecommend"].map(code_reco)
df2["Ease of Participation"] += 1 * recommend # +1 weight
account = df2["StackOverflowHasAccount"].map({'Yes' : 1 , 'No' : 0 , 'I\'m not sure / I can\'t remember' : 0}) 
df2["Ease of Participation"] += account *4 * 0.25   #+0.25 weight ; Scaled to 0-4
participate_map = {'I have never participated in Q&A on Stack Overflow':0, 'Less than once per month or monthly' :1 , 'A few times per month or weekly' : 2 ,'A few times per week' : 3 ,'Daily or almost daily' : 4, 'Multiple times per day' : 4  }
participate_map
participates = df2["StackOverflowParticipate"].map(participate_map)
df2["Ease of Participation"] += (participates * 2) #+2 weight
memberso = df2["StackOverflowConsiderMember"].map({'Yes' : 1 , 'No' : 0 , 'I\'m not sure' : 0}) 
df2["Ease of Participation"] += memberso * 4 * 1.5 #+1.5 weight ; scaled to 0-4
plt.figure(1)
for_plot = df2[-pd.isnull(df2["is_stack_learner"])]
sns.boxplot(x = for_plot["is_stack_learner"] , y = for_plot["Ease of Participation"])# , order = meds)
plt.xticks(rotation = 90)
plt.show()

for_plot = df2.groupby(["Country"])["Ease of Participation"].mean().reset_index().dropna()
plt.figure(7 , figsize= (30,30))
#sns.barplot(x = for_plot["Country"] , height = for_plot["Ease of Participation"])
#plt.xticks(rotation = 90)
D = zip(for_plot["Country"], for_plot["Ease of Participation"])
D = sorted( list(D) , key = lambda x : x[1])
plt.figure(1 , figsize = (30,30))
x_val = [x[0] for x in D]
y_val = [x[1] for x in D]
sns.barplot(x_val , y_val)
plt.xticks(rotation = 90)
plt.show()
df2["is_windows"] = (df2["OperatingSystem"] == "Windows")
plt.figure(1)
plt.figure(1)
sns.boxplot(x = df2["is_windows"] , y = df2["Ease of Participation"])# , order = meds)
plt.xticks(rotation = 90)
plt.show()

plt.figure(1)
sns.boxplot(x = df2["Student"] , y = df2["Ease of Participation"])
plt.show()
plt.figure(1)
sns.boxplot(x = df2["YearsCoding"] , y = df2["Ease of Participation"])
plt.xticks(rotation = 90)
#pol.set_xticklabels(rotation=90)
plt.show()
plt.figure(1)
sns.boxplot(x = df2["Hobby"] , y = df2["Ease of Participation"])# , order = meds)
plt.xticks(rotation = 90)
#pol.set_xticklabels(rotation=90)
plt.show()
#Number of Monitors??
plt.figure(1)
sns.boxplot(x = df2["NumberMonitors"] , y = df2["Ease of Participation"])# , order = meds)
plt.xticks(rotation = 90)
#pol.set_xticklabels(rotation=90)
plt.show()
plt.figure(1)
sns.boxplot(x = df2["HoursComputer"] , y = df2["Ease of Participation"])# , order = meds)
plt.xticks(rotation = 90)
#pol.set_xticklabels(rotation=90)
plt.show()

MULTIPLE_CHOICE = [
    'CommunicationTools','EducationTypes','SelfTaughtTypes','HackathonReasons', 
    'DatabaseWorkedWith','DatabaseDesireNextYear','PlatformWorkedWith',
    'PlatformDesireNextYear','Methodology','VersionControl',
    'AdBlockerReasons','AdsActions','ErgonomicDevices','Gender',
    'SexualOrientation','RaceEthnicity', 'LanguageWorkedWith'
]
for c in MULTIPLE_CHOICE:
    # Check if there are multiple entries in this column
    temp = df2[c].str.split(';', expand=True)

    # Get all the possible values in this column
    new_columns = pd.unique(temp.values.ravel())
    for new_c in new_columns:
        if new_c and new_c is not np.nan:
            
            # Create new column for each unique column
            idx = df2[c].str.contains(new_c, regex=False).fillna(False)
            df2.loc[idx, f"{c}_{new_c}"] = 1

    # Info to the user
    print(f">> Multiple entries in {c}. Added {len(new_columns)} one-hot-encoding columns")

for_plot = df2
for_plot[[cols for cols in for_plot.columns.values if re.search('^Language\S+_' , cols)]] = for_plot[[cols for cols in for_plot.columns.values if re.search('^Language\S+_' , cols)]].fillna(0)
for_plot = for_plot[for_plot[[cols for cols in for_plot.columns.values if re.search('^Language\S+_' , cols)]].sum(axis=1)!=0]
for_plot = for_plot[-pd.isnull(for_plot["is_stack_learner"])]
for_plot = for_plot[for_plot["is_stack_learner"]] #Those who learn using stack overflow, probably the lanugage , therefore subsettin
for_plot = for_plot[[cols for cols in for_plot.columns.values if re.search('^Language\S+_' , cols)] + ['Ease of Participation']]
D = {}
for i in for_plot.columns.values[:-1]:
    D[i] = for_plot[for_plot[i] == 1]["Ease of Participation"].mean()
D = sorted(D.items(), key=operator.itemgetter(1))
plt.figure(1 , figsize = (30,30))
x_val = [x[0] for x in D]
y_val = [x[1] for x in D]
sns.barplot(x_val , y_val)
plt.xticks(rotation = 90)
plt.show()
for_plot = df2
for_plot[[cols for cols in for_plot.columns.values if re.search('^EducationTypes_' , cols)]] = for_plot[[cols for cols in for_plot.columns.values if re.search('^EducationTypes_' , cols)]].fillna(0)    
for_plot = for_plot[for_plot[[cols for cols in for_plot.columns.values if re.search('^EducationTypes_' , cols)]].sum(axis=1)!=0]
#Those who learn using stack overflow, probably the lanugage , therefore subsettin
for_plot = for_plot[[cols for cols in for_plot.columns.values if re.search('^EducationTypes_' , cols)] + ['Ease of Participation']]
D = {}
for i in for_plot.columns.values[:-1]:
    D[i] = for_plot[for_plot[i] == 1]["Ease of Participation"].mean()
D = sorted(D.items(), key=operator.itemgetter(1))
plt.figure(1 , figsize = (5,5))
x_val = [x[0] for x in D]
y_val = [x[1] for x in D]
sns.barplot(x_val , y_val)
plt.xticks(rotation = 90)
plt.show()
for_plot = df2
for_plot[[cols for cols in for_plot.columns.values if re.search('^VersionControl_' , cols)]] = for_plot[[cols for cols in for_plot.columns.values if re.search('^VersionControl_' , cols)]].fillna(0)
for_plot = for_plot[for_plot[[cols for cols in for_plot.columns.values if re.search('^VersionControl_' , cols)]].sum(axis=1)!=0]
for_plot = for_plot[[cols for cols in for_plot.columns.values if re.search('^VersionControl_' , cols)] + ['Ease of Participation']]
D = {}
for i in for_plot.columns.values[:-1]:
    D[i] = for_plot[for_plot[i] == 1]["Ease of Participation"].mean()
D = sorted(D.items(), key=operator.itemgetter(1))
plt.figure(1 , figsize = (5,5))
x_val = [x[0] for x in D]
y_val = [x[1] for x in D]
sns.barplot(x_val , y_val)
plt.xticks(rotation = 90)
plt.show()
for_plot = df2
for_plot[[cols for cols in for_plot.columns.values if re.search('^Methodology_' , cols)]] = for_plot[[cols for cols in for_plot.columns.values if re.search('^Methodology_' , cols)]].fillna(0)
for_plot = for_plot[for_plot[[cols for cols in df2.columns.values if re.search('^Methodology_' , cols)]].sum(axis=1)!=0]
#Those who learn using stack overflow, probably the lanugage , therefore subsettin
for_plot = for_plot[[cols for cols in for_plot.columns.values if re.search('^Methodology_' , cols)] + ['Ease of Participation']]
D = {}
for i in for_plot.columns.values[:-1]:
    D[i] = for_plot[for_plot[i] == 1]["Ease of Participation"].mean()
D = sorted(D.items(), key=operator.itemgetter(1))
plt.figure(1 , figsize = (5,5))
x_val = [x[0] for x in D]
y_val = [x[1] for x in D]
sns.barplot(x_val , y_val)
plt.xticks(rotation = 90)
plt.show()
for_plot = df2
for_plot[[cols for cols in for_plot.columns.values if re.search('CommunicationTools_' , cols)]] = for_plot[[cols for cols in for_plot.columns.values if re.search('CommunicationTools_' , cols)]].fillna(0)
for_plot = for_plot[for_plot[[cols for cols in for_plot.columns.values if re.search('CommunicationTools_' , cols)]].sum(axis=1)!=0]
#Those who learn using stack overflow, probably the lanugage , therefore subsettin
for_plot = for_plot[[cols for cols in for_plot.columns.values if re.search('CommunicationTools_' , cols)] + ['Ease of Participation']]
D = {}
for i in for_plot.columns.values[:-1]:
    D[i] = for_plot[for_plot[i] == 1]["Ease of Participation"].mean()
D = sorted(D.items(), key=operator.itemgetter(1))
plt.figure(1 , figsize = (5,5))
x_val = [x[0] for x in D]
y_val = [x[1] for x in D]
sns.barplot(x_val , y_val)
plt.xticks(rotation = 90)
plt.show()
uniqueness = []
for i in df2.columns.values[1:]:
    if len(pd.unique(df2[i])) > 15:
        continue
        #print(i)
    elif "StackOverflow" in i:
        continue
        #print("TARGET --->", i)
    elif "Hypo" in i:
        continue
        #print("TARGET --->", i) 
    else:
        uniqueness.append(i)
for_tree = df2
for_tree = for_tree[[cols for cols in uniqueness]]
for_tree = for_tree.drop("SexualOrientation", axis = 1)
listo = list(for_tree.dtypes == object)
for_tree2 = for_tree
for i,col in enumerate(for_tree.columns.values):
    if listo[i] == True:
        new_columns = pd.get_dummies(for_tree[col])
        for new_c in new_columns:
            #print(f"{col}_{new_c}")
            if new_c and new_c is not np.nan:
                for_tree2 = pd.concat([for_tree2, new_columns[new_c]] , axis = 1)
                for_tree2.columns.values[-1] = f"{col}_{new_c}"
        for_tree2 = for_tree2.drop(col,axis = 1)
for_tree2.head()
final_tree = pd.concat([for_tree2 , df2["Ease of Participation"]] , axis = 1)
final_tree = final_tree[-pd.isnull(final_tree["Ease of Participation"])]
final_tree.head(8)
import xgboost as xgb
# read in data
dtrain = xgb.DMatrix(final_tree.iloc[:,:-1] , label= final_tree["Ease of Participation"])
dtrain.get_label()
# specify parameters via map
param = {'max_depth':1, 'eta':0.01, 'silent' : 0 , 'objective':'reg:linear' , 'subset' : 0.5 }
num_round = 500
bst = xgb.train(param, dtrain, num_round)
dtest = xgb.DMatrix(final_tree.iloc[:,:-1])
preds = bst.predict(dtest)
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(bst, max_num_features=50, height=0.8, ax=ax)
plt.show()