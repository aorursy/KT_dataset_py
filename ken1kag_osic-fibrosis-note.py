import pandas as pd
train_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
train_df
import seaborn as sns
import matplotlib.pyplot as plt
sns.regplot(x=train_df["Age"], y=train_df["FVC"])
sns.scatterplot(x=train_df["Age"], y=train_df["FVC"], hue=train_df['SmokingStatus'])

ax = plt.axes()
ax.set_title('Distribution of Age and FVC for unique patients')
plt.show()
sns.lmplot(x="Age", y="FVC", hue="SmokingStatus", data=train_df)
sns.lmplot(x="Age", y="FVC", hue="Sex", data=train_df)
import matplotlib.pyplot as plt
sns.swarmplot(x=train_df['Sex'],
              y=train_df['FVC'],hue=train_df['SmokingStatus'])
plt.figure(figsize=(10,10))
import matplotlib.pyplot as plt
sns.swarmplot(x=train_df['SmokingStatus'],
              y=train_df['FVC'],hue=train_df['Sex'])
plt.figure(figsize=(10,10))
new_df = train_df.groupby([train_df.Patient,train_df.Age,train_df.Sex, train_df.SmokingStatus])['Patient'].count() 
new_df.index = new_df.index.set_names(['id','Age','Sex','SmokingStatus'])
new_df = new_df.reset_index()
new_df.rename(columns = {'Patient': 'freq'},inplace = True) 
new_df
import matplotlib.pyplot as plt
for Patient in new_df['id']:
    train2=train_df.loc[train_df.Patient == Patient]
    graph = plt.plot(train2["Weeks"], train2["FVC"] )
    plt.xlabel("Weeks")
    plt.ylabel("FVC")
    plt.title("{}".format(Patient))
    plt.show()