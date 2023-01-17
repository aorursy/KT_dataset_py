# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

# Set default fontsize for graphs
#plt.rcParams.update({'font.size': 12})
SMALL_SIZE, MEDIUM_SIZE, BIG_SIZE = 10, 12, 20
plt.rc('font', size=MEDIUM_SIZE)       
plt.rc('axes', titlesize=BIG_SIZE)     
plt.rc('axes', labelsize=MEDIUM_SIZE)  
plt.rc('xtick', labelsize=MEDIUM_SIZE) 
plt.rc('ytick', labelsize=MEDIUM_SIZE) 
plt.rc('legend', fontsize=SMALL_SIZE)  
plt.rc('figure', titlesize=BIG_SIZE)  
# Read the input training and test data
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
train_data.sample(5)
# Total number of records
print("Total number of records in training dataset:", train_data.shape)
print("Total number of records in test dataset:", test_data.shape)
# What are the features available and what are their data type?
train_data.dtypes
# Descriptive statistics of training data
train_data.describe().transpose()
# Is there any empty data in training dataset?
train_data.isnull().sum()/train_data.shape[0]
# Is there any empty data in test dataset?
test_data.isnull().sum()/test_data.shape[0]
print("Min PassengerId =", train_data.PassengerId.min(), "and Max PassengerId =", train_data.PassengerId.max())
print("Total number of unique PassengerId values =", len(train_data.PassengerId.unique()))
conditions = [train_data["Survived"] == 0, train_data["Survived"] == 1]
choices = ["Did not Survive", "Survived"]
train_data["Survived_str"] = np.select(conditions, choices)

# Plot a graph for Survived
fig, axes = plt.subplots(figsize=(8,5))
data = train_data["Survived_str"].value_counts(normalize=True)
axes.bar(data.index, data, color=['red', 'green'])
axes.set_title('Survival %', fontsize=15)
plt.show()
conditions = [train_data["Pclass"] == 1, train_data["Pclass"] == 2, train_data["Pclass"] == 3]
choices = ["Upper", "Middle", "Lower"]
train_data["Pclass_str"] = np.select(conditions, choices)

# Plot a graph for Pclass
fig, axes = plt.subplots(figsize=(10,5))
data = train_data["Pclass_str"].value_counts(normalize=True).sort_index()
axes.bar(data.index, data, color='orange')
axes.set_title('% of Passenger Class', fontsize=15)
plt.show()
train_data["Salutation"] = train_data.Name.str.split(',').str[1].str.split('.').str[0]

# Plot a graph for Salutation
fig, axes = plt.subplots(figsize=(15,8))
data = train_data["Salutation"].value_counts(normalize=True)
axes.bar(data.index, data, color='0.5')
axes.set_xticklabels(data.index, rotation='vertical')
axes.set_xlabel('Salutation')
axes.set_title('Distribution of Salutation', fontsize=15)
plt.show()
# Plot a graph for Sex or Gender
fig, axes = plt.subplots(figsize=(10,5))
data = train_data["Sex"].value_counts(normalize=True)
axes.bar(data.index, data, color='green')
axes.set_title('% of Passengers by Gender', fontsize=15)
plt.show()
# Plot a graph for Age
fig, axes = plt.subplots(figsize=(15,8))
sns.distplot(train_data.loc[train_data["Age"].notnull(), "Age"], color='orange', ax=axes)
axes.set_title('% of Passengers by Age', fontsize=15)
axes.set_xlim(0,100)
axes.set_xticks(np.arange(0, 100, 5))
axes.grid(True)
plt.show()
# Plot a graph for Sibsp and Parch
fig, axes = plt.subplots(1, 2, figsize=(15,8))

sns.barplot(
    x=train_data['SibSp'], 
    y=train_data['SibSp'], 
    estimator=lambda x: len(x) / len(train_data), 
    ax=axes[0])
axes[0].set(xlabel='# of Siblings/Spouses', ylabel='')
axes[0].set_title('% of Siblings/Spouses', fontsize=15)

sns.barplot(
    x=train_data['Parch'], 
    y=train_data['Parch'], 
    estimator=lambda x: len(x) / len(train_data), 
    ax=axes[1])
axes[1].set(xlabel='# of Parents/Children', ylabel='')
axes[1].set_title('% of Parents/Children', fontsize=15)

plt.show()
# <b>Let us combine SibSp and Parch and consider that as a single feature - "FamilySize". We'll perform EDA on FamilySize.</b>
train_data['FamilySize'] = train_data.SibSp + train_data.Parch
fig, axes = plt.subplots(figsize=(15,8))
sns.barplot(
    x=train_data['FamilySize'], 
    y=train_data['FamilySize'], 
    estimator=lambda x: len(x) / len(train_data), 
    ax=axes)
axes.set_title('% of Passengers by Family Size', fontsize=15)
plt.show()
# Save first alphabet of Cabin as CabinDeck
train_data["CabinDeck"] = train_data.Cabin.str[0]

# Plot graph for Fare, Cabin and Embarked
fig, axes = plt.subplots(3, 1, figsize=(10,25))

# Cabin Deck
data = train_data["CabinDeck"].value_counts(normalize=True).sort_index()
axes[0].bar(data.index, data, color='orange')
axes[0].set_title('% of Passengers per Cabin Deck', fontsize=15)

# Fare
sns.distplot(train_data['Fare'], color='r', ax=axes[1])
axes[1].set(xlabel='Fare', ylabel='')
axes[1].set_title('Distribution of Fare', fontsize=15)
axes[1].grid(True)
axes[1].set_xticks(np.arange(0, 550, 20))
axes[1].set_xlim(0,550)

# Embarked
data = train_data["Embarked"].value_counts(normalize=True).sort_index()
axes[2].bar(data.index, data, color='violet')
axes[2].set_title('% of Passengers per Port of Embarkment', fontsize=15)
axes[2].set_xticklabels(['Cherbourg', 'Queenstown', 'Southampton'])

plt.show()
# Fetch only the non-null age values and corresponding salutation
age_sal = train_data.loc[train_data['Age'].notnull(), ['Age', 'Salutation']]
age = age_sal.iloc[:,0]
sal = age_sal.iloc[:,1]

# Fetch Age grouped by Salutation and convert each set into a list
age_sal_grouped = age_sal.groupby("Salutation").Age.apply(list)

# Plot a boxplot
fig, axes = plt.subplots(figsize=(15,10))
axes.boxplot(age_sal_grouped, patch_artist=True)
axes.set_xticklabels(age_sal_grouped.index, rotation='vertical')
axes.set_yticks(np.arange(0, 85, 5))
axes.grid(True)
axes.set_xlabel('Salutation', fontsize=15)
axes.set_ylabel('Age', fontsize=12)

axes.set_title('Salutation vs Age - Data Distribution', fontsize=20)
plt.show()
# Fill missing Age values on train and test data
age_sal_grouped_median = train_data.groupby("Salutation", as_index=False).Age.median()
joined = train_data.merge(age_sal_grouped_median, on="Salutation", how="inner")
train_data.Age = train_data.Age.fillna(joined.Age_y, axis=0)
test_data.Age = test_data.Age.fillna(joined.Age_y, axis=0)
print("Missing values in Age column now: in training data =", train_data.Age.isnull().sum(), " and in test data =", test_data.Age.isnull().sum())

# Plot a graph for Age
fig, axes = plt.subplots(figsize=(15,8))
sns.distplot(train_data["Age"], color='orange', ax=axes)
axes.set_title('% of Passengers by Age', fontsize=15)
axes.set_xlim(0,100)
axes.set_xticks(np.arange(0, 100, 5))
axes.grid(True)
plt.show()
train_data.Embarked = train_data.Embarked.fillna('S', axis=0)
print("Missing values in Embarked column now:", train_data.Embarked.isnull().sum())
test_data.Fare = train_data.Age.fillna(train_data.Fare.mean(), axis=0)
print("Missing values in Fare column now: in test data =", test_data.Fare.isnull().sum())
fig, [axes1, axes2] = plt.subplots(1,2,figsize=(18,9))

# Plot relation between Cabin Deck and Fare
data = train_data[['CabinDeck', 'Fare', 'Pclass']].sort_values('CabinDeck')
sns.swarmplot(data.iloc[:,0], data.iloc[:,1], ax=axes1)
axes1.set_xlabel('Cabin Deck')
axes1.set_ylabel('Fare')
axes1.set_title('What is the relation between Cabin Deck and Fare?', fontsize=15)

# Plot relation between Cabin Deck and Passenger Class
sns.stripplot(data.iloc[:,0], data.iloc[:,2], size=25, ax=axes2)
axes2.set_xlabel('Cabin Deck')
axes2.set_ylabel('Passenger Class')
axes2.set_yticklabels(['Upper', 'Middle', 'Lower'])
axes2.set_yticks([1,2,3])
axes2.set_title('What is the relation between Cabin Deck and Passenger Class?', fontsize=15)

plt.show()
# Add CabinAvail
train_data["CabinAvail"] = train_data["Cabin"].notnull().astype('int')
test_data["CabinAvail"] = test_data["Cabin"].notnull().astype('int')

# Add Gender
sex_mapping = {"male": 0, "female": 1}
train_data['Gender'] = train_data['Sex'].map(sex_mapping)
test_data['Gender'] = test_data['Sex'].map(sex_mapping)

embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train_data['EmbarkPort'] = train_data['Embarked'].map(embarked_mapping)
test_data['EmbarkPort'] = test_data['Embarked'].map(embarked_mapping)
# Pie chart of Survival percentage
fig, axes = plt.subplots(figsize=(15,10))
width = 0.5

# Percentage of Survived vs Did not survive 
data = train_data.Survived.value_counts().sort_index()
axes.pie(
    data,
    labels=['Did not Survive', 'Survived'],
    #shadow=True,
    autopct='%1.1f%%',
    pctdistance=0.8,
    startangle=90,
    textprops={'color':'black', 'fontweight':'bold'},
    wedgeprops = {'width':width, 'edgecolor':'w'},
    radius=1
)

# Percentage of Sex based on Survival
data = train_data.groupby(["Survived", "Sex"]).size().reset_index()
axes.pie(
    data.iloc[:,2], 
    labels=list(data.Sex),
    #shadow=True,
    autopct='%1.1f%%',
    startangle=90,
    textprops={'color':'white', 'fontweight':'bold'},
    wedgeprops = {'width':width, 'edgecolor':'w'},
    radius=1-width,
    rotatelabels=True
)

axes.set_title('What is the sex ratio of the passengers who survived?', fontsize=15)
axes.legend(loc='best', bbox_to_anchor=(1,1))
axes.axis('equal')
plt.show()
fig, axes1 = plt.subplots(figsize=(10,10))

# Find index of train_data based on value of Survived
idxYes = np.where(train_data.Survived == 1)
idxNo = np.where(train_data.Survived == 0)

# Plot relation between Age and Survival
data = train_data[['Age', 'Survived']].sort_values('Survived')
sns.stripplot(train_data.iloc[idxYes].Sex, train_data.iloc[idxYes].Age, jitter=True, color='green', size=15, label="Survived", ax=axes1)
sns.stripplot(train_data.iloc[idxNo].Sex, train_data.iloc[idxNo].Age, jitter=True, color='orange', size=15, alpha=0.5, label="Did not Survive", ax=axes1)
axes1.set_xlabel('Sex')
axes1.set_ylabel('Age')
axes1.set_title('Does age and sex determine survival?', fontsize=15)
handles, labels = axes1.get_legend_handles_labels()
axes1.legend((handles[0], handles[2]), (labels[0], labels[2]), loc='best')
plt.show()
# Divide the ages into bins
bins = [0, 5, 12, 21, 40, 60, np.inf]
labels = ['[0-5]', '[5-12]', '[12-21]', '[21-40]', '[40-60]', '[60+]']
train_data['AgeGroup'] = pd.cut(train_data["Age"], bins, labels=labels)

# Draw a bar plot of Age vs. survival
fig, [axes1, axes2] = plt.subplots(1,2,figsize=(15,8))
sns.barplot(x="AgeGroup", y="Survived", data=train_data, ci=None, ax=axes1)
axes1.set_xlabel('Age Group', fontsize=12)
axes1.set_ylabel('Survival Percentage', fontsize=12)
axes1.set_ylim(0,1)
axes1.set_title('What is the chance of survival based on Age?', fontsize=15)

sns.barplot(x="AgeGroup", y="Survived", hue="Sex", data=train_data, ci=None, ax=axes2)
axes2.set_xlabel('Age Group and Sex', fontsize=12)
axes2.set_ylabel('Survival Percentage', fontsize=12)
axes2.set_ylim(0,1)
axes2.set_title('What is the chance of survival based on Age and Sex?', fontsize=15)

plt.show()
fig, axes1 = plt.subplots(figsize=(15,8))

# Divide Fare into bins and plot Survival for each Fare bin
# Range of Fare is from 0 to 512
bins = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, np.inf]
labels = ['[0-50]', '[50-100]', '[100-150]','[150-200]', '[200-250]','[250-300]', '[300-350]', '[350-400]', '[400-450]', '[450+]']
train_data['FareGroup'] = pd.cut(train_data["Fare"], bins, labels=labels)

# Draw a bar plot of Age vs. survival
sns.barplot(x="FareGroup", y="Survived", data=train_data, ci=None, ax=axes1)
axes1.set_xlabel('Fare', fontsize=12)
axes1.set_ylabel('Survival Percentage', fontsize=12)
axes1.set_ylim(0,1)
axes1.set_title('What is the chance of survival based on Fare?', fontsize=15)

plt.show()
# Pie chart of Survival percentage
fig, (axes1, axes2) = plt.subplots(2,1,figsize=(15,25))
width = 0.5

# Percentage of Survived vs Did not survive 
data = train_data.Survived.value_counts().sort_index()
axes1.pie(
    data,
    labels=['Did not Survive', 'Survived'],
    #shadow=True,
    autopct='%1.1f%%',
    pctdistance=0.8,
    startangle=90,
    textprops={'color':'black', 'fontweight':'bold'},
    wedgeprops = {'width':width, 'edgecolor':'w'},
    radius=1
)

# Percentage of Passenger Class based on Survival
data = train_data.groupby(["Survived", "Pclass_str"]).size().reset_index()
axes1.pie(
    data.iloc[:,2], 
    labels=list(data.Pclass_str),
    #shadow=True,
    autopct='%1.1f%%',
    startangle=90,
    textprops={'color':'white', 'fontweight':'bold'},
    wedgeprops = {'width':width, 'edgecolor':'w'},
    radius=1-width,
    rotatelabels=True
)

axes1.set_title('How is Passenger Class related to Survival?', fontsize=15)
#axes1.legend(loc='best', bbox_to_anchor=(0,1))
axes1.axis('equal')

# Now we look at Survival vs Passenger Class from the opposite perspective
# Percentage of Passenger Class 
data = train_data.Pclass.value_counts().sort_index()
axes2.pie(
    data,
    labels=['Upper', 'Middle','Lower'],
    #shadow=True,
    autopct='%1.1f%%',
    pctdistance=0.8,
    startangle=90,
    textprops={'color':'black', 'fontweight':'bold'},
    wedgeprops = {'width':width, 'edgecolor':'w'},
    radius=1
)

# Percentage of Survival based on Passenger Class
data = train_data.groupby(["Pclass", "Survived"]).size().reset_index()
axes2.pie(
    data.iloc[:,2], 
    labels=['Did not Survive', 'Survived']*3,
    #shadow=True,
    autopct='%1.1f%%',
    startangle=90,
    textprops={'color':'white', 'fontweight':'bold'},
    wedgeprops = {'width':width, 'edgecolor':'w'},
    radius=1-width,
    rotatelabels=True
)

axes2.set_title('In each Passenger Class, what is the percentage of Survival?', fontsize=15)
#axes2.legend(loc='best', bbox_to_anchor=(1,1))
axes2.axis('equal')

plt.show()
fig, axes = plt.subplots(figsize=(10,5))
# Pclass of passengers who survived
sns.barplot(x="Pclass_str", y="Survived", data=train_data, ci=None, ax=axes)
axes.set_xlabel('Passenger Class', fontsize=12)
axes.set_ylabel('Chance of Survival', fontsize=12)
axes.set_ylim(0,1)
axes.set_title('What is the chance of survival per Passenger Class?', fontsize=15)
plt.show()
fig, (axes1, axes2) = plt.subplots(2,1,figsize=(15,20))
width = 0.7

# Relation between family size and survival
data = train_data.iloc[idxYes].groupby(["FamilySize"]).size().reset_index()
axes1.pie(
    data.iloc[:,1], 
    labels=list(data.FamilySize),
    labeldistance=1.0,
    autopct=lambda pct: "{:.1f}%".format(pct) if pct > 5.0 else "",
    startangle=90,
    textprops={'color':'white', 'fontweight':'bold'},
    wedgeprops = {'width':width, 'edgecolor':'w'},
    #radius=1-width,
    rotatelabels=True,
    counterclock=False
)

axes1.set_title('Family Size of those who Survived', fontsize=15)
#axes1.legend(["Each Pie = Family Size"], loc='best', bbox_to_anchor=(1,0))
axes1.legend(data.FamilySize, title="Family Size", loc='best', bbox_to_anchor=(0,1))
axes1.axis('equal')

# Draw a bar plot of Family Size vs Survival
sns.barplot(x="FamilySize", y="Survived", data=train_data, ci=None, ax=axes2)
axes2.set_xlabel('Family Size', fontsize=12)
axes2.set_ylabel('Survival Percentage', fontsize=12)
axes2.set_ylim(0,1)
axes2.set_title('What is the chance of survival for each Family Size?', fontsize=15)

plt.show()
# Draw a bar plot of Passenger Class vs Survival
fig, [axes1, axes2] = plt.subplots(1,2,figsize=(18,9))
sns.barplot(x="CabinAvail", y="Survived", hue="Embarked", data=train_data, ci=None, ax=axes1)
axes1.set_xlabel('Cabin Availibility and Embarked From', fontsize=12)
axes1.set_ylabel('Survival Percentage', fontsize=12)
axes1.set_ylim(0,1)
axes1.set_title('What is the chance of survival based on availibility of Cabins?', fontsize=15)

# Draw a bar plot of Family Size vs Survival
sns.barplot(x="Embarked", y="Survived", data=train_data, ci=None, ax=axes2)
axes2.set_xlabel('Port of Embarkation', fontsize=12)
axes2.set_ylabel('Survival Percentage', fontsize=12)
axes2.set_ylim(0,1)
axes2.set_title('What is the chance of survival for each Port of Embarkation?', fontsize=15)

plt.show()
# Code to be added (work-in-progress)
