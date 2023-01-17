# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt
import numpy as np  # linear algebra
import time
from collections import Counter 
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import f1_score
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
df.head()
def hist_vis(df, title):
    ctrdict = dict(Counter(df))
    labels, values = ctrdict.keys(), ctrdict.values()
    fig, ax = plot.subplots(figsize=(11, 9))
    plot.title(title, weight='bold')
    plot.barh(range(len(labels)),list(values), color='#cc593c')
    plot.yticks(range(len(labels)), labels)
    plot.plot()

def visualize_desc(desc):
    retdf = df[df["Descript"] == desc]
    if len(retdf) < 1:
        print ("No incidents found for " + desc)
        return 0
    else:
        print(len(retdf))
        print(desc)
        
        hist_vis(retdf["Resolution"], "Resolutions")
        hist_vis(retdf["DayOfWeek"], "Day of the Week")

def hplot_vis(category, value, title):
    fig, ax = plot.subplots(figsize=(11, 9))
    plot.title(title, weight='bold')
    ax.barh(category,value, color='#34b180')

    plot.show()
    print(len(category))
    

df.Category.unique()
hist_vis(df.Category, "Categories of Crimes")
len(df[df.Category=="OTHER OFFENSES"])

hist_vis(dict(Counter(df[df.Category=="OTHER OFFENSES"].Descript).most_common(30)), "Top 30 'OTHER' crimes")
hist_vis(dict(Counter(df.Descript).most_common(30)), "Top 30 of All Crimes")
df["PdDistrict"].unique()
hist_vis(dict(Counter(df["PdDistrict"]).most_common()), "Crimes by District")
hist_vis(dict(Counter(df[df["PdDistrict"]=="SOUTHERN"].Descript).most_common(25)), "Top 25 Crimes in the Southern District")
hist_vis(dict(Counter(df[df["PdDistrict"]=="RICHMOND"].Descript).most_common(25)), "Top 25 Crimes in the Richmond District")
hist_vis(dict(Counter(df[df["PdDistrict"]=="TENDERLOIN"].Descript).most_common(25)), "Top 25 Crimes in the Tenderloin")
hist_vis(dict(Counter(df[df["PdDistrict"]=="NORTHERN"].Descript).most_common(25)), "Top 25 Crimes in the Northern District")
visualize_desc("GRAND THEFT FROM LOCKED AUTO")
visualize_desc("STOLEN AUTOMOBILE")
visualize_desc("PLACING WIFE IN HOUSE OF PROSTITUTION")
visualize_desc("PLACING HUSBAND IN HOUSE OF PROSTITUTION")
visualize_desc("DANGER OF LEADING IMMORAL LIFE")
visualize_desc("MAYHEM WITH A DEADLY WEAPON")
visualize_desc("FORTUNE TELLING")
visualize_desc("ASSAULT, AGGRAVATED, W/ MACHINE GUN")
visualize_desc("DESTRUCTION OF PROPERTY WITH EXPLOSIVES")
df['Resolution'].unique()
hist_vis(dict(Counter(df['Address']).most_common(25)), "Top 25 Most Common Addresses")
len(Counter(df["Address"]))
hist_vis(dict(Counter(df[df["Address"] == '800 Block of BRYANT ST'].X).most_common(5)), "braynta")
hist_vis(dict(Counter(df[df["Address"] == '800 Block of BRYANT ST'].Y).most_common(5)), "braynta")
hist_vis(dict(Counter(df[df["Address"] == '800 Block of MARKET ST'].X).most_common(25)), "braynta")
hist_vis(dict(Counter(df[df["Address"] == '800 Block of MARKET ST'].Y).most_common(25)), "braynta")
#Clean the data

#800 Bryant street is the police station... What's going on here? 
#Let's remove everythign with this address
df_list = df.index[df["Address"] == "800 Block of BRYANT ST"].tolist()
cleaned_df = df.drop(df.index[df_list])
cleaned_df[cleaned_df["Address"] == "800 Block of BRYANT ST"]


cleaned_df.drop(["Dates", "Address", "Descript"], axis = 1, inplace=True)

df.head()
print(len(cleaned_df.Resolution))
hist_vis(dict(Counter(cleaned_df.Resolution)), "y")
%%time
classfs = []
accuracy = []
fscore = []
time_ls = []


cats = pd.Series(cleaned_df["Resolution"]).astype('category')

label_ints = cats.cat.codes
labels = cats
cleaned_df = cleaned_df.drop(["Resolution"], axis=1)
cleaned_df = pd.get_dummies(cleaned_df, prefix='sf_')
#Scale Data
scaler = StandardScaler()
scaler.fit(cleaned_df)
scaled_X_vals = scaler.transform(cleaned_df)
print(len(scaled_X_vals))
print(len(label_ints))
X_train, X_test, y_train, y_test = train_test_split(scaled_X_vals, label_ints, test_size=0.5)
start = time.time()
clf = DecisionTreeClassifier()
fitted = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


end = time.time()
time_ls.append(end - start)
classfs.append("DT")
decision_accuracy = accuracy_score(y_test, y_pred)
accuracy.append(decision_accuracy)

f1 = f1_score(y_test, y_pred,  average='weighted')
print("f1 score: {}".format(f1))
print("accuracy score: {}".format(decision_accuracy))
print("Runtime: " + str(time_ls))
# hplot_vis(labels.unique(), f1, "f1 scores")

def random_forest( trees, max_d, min_sample, name):
    start = time.time()
    
    clf = RandomForestClassifier(n_estimators=trees, max_depth=max_d, min_samples_split=min_sample)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    accuracy.append(accuracy_score(y_test, preds))
    fscore.append(f1_score(y_test, preds, average='weighted'))
    classfs.append(name)
    end = time.time()
    time_ls.append(end - start)
    print("accuracy score: {}".format(accuracy[-1]))
    print("Weighted f1: {}".format(fscore[-1]))
    print("Runtime: {} seconds".format(time_ls[-1]))

random_forest(30, 4, 50, "RF1")
random_forest(30, 4, 5, "RF2")
random_forest(30, 4, 100, "RF3")
random_forest(30, 1, 50, "RF4")
random_forest(60, 8, 50, "RF5")
random_forest(60, 16, 50, "RF6")
random_forest(120, 16, 50, "RF7")
timecolor = []
#Generate Colors on red/green axis based on execution time
for time in time_ls:
    percent_red = time/350
    percent_green = 1 - percent_red
    red_10 = int(percent_red * 255)
    green_10 = int(percent_green * 255)
    red_16 = str(hex(red_10))[-2:].replace("x", "0")
    green_16 = str(hex(green_10))[-2:].replace("x", "0")
    timecolor.append("#"+str(red_16)+str(green_16)+"88")

fig, ax = plot.subplots(figsize=(11, 9))
rects = ax.bar(classfs, accuracy, color=timecolor)
random_chance = 1/len(label_ints.unique())
plot.axhline(y=random_chance, color='r', linestyle='-')

# Indicate Times.
labels = ["%ds" % t for t in time_ls]


plot.plot()
for rect, label in zip(rects, labels):
    ax.text(rect.get_x() + rect.get_width() / 2, .1, label,ha='center', va='bottom')
