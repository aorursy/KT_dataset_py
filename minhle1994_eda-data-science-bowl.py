import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
print("Reading Training data...")

train_df = pd.read_csv("/kaggle/input/data-science-bowl-2019/train.csv", parse_dates=['timestamp'])#, converters = {"event_data": json.loads})

print('The training data has {} rows and {} columns.'.format(train_df.shape[0], train_df.shape[1]))



print("Reading Test data...")

test_df = pd.read_csv("/kaggle/input/data-science-bowl-2019/test.csv", parse_dates = ['timestamp'])

print("The test data has {} rows and {} columns.".format(test_df.shape[0], test_df.shape[1]))



print("Reading Training Label data...")

train_labels_df = pd.read_csv("/kaggle/input/data-science-bowl-2019/train_labels.csv")

print("The training labels data has {} rows and {} columns.".format(train_labels_df.shape[0], train_labels_df.shape[1]))



print("Reading Specs Data...")

specs_df = pd.read_csv("/kaggle/input/data-science-bowl-2019/specs.csv")

print("The specs data has {} rows and {} columns.".format(specs_df.shape[0], specs_df.shape[1]))



print("Reading sample Submission...")

sample_submission_df = pd.read_csv("/kaggle/input/data-science-bowl-2019/sample_submission.csv")

print("The sample submission has {} rows and {} columns".format(sample_submission_df.shape[0], sample_submission_df.shape[1]))
print("The training data has {} missing values.".format(sum(train_df.isnull().sum())))

print("The training label data has {} missing values.".format(sum(train_labels_df.isnull().sum())))

print("The test data has {} missing values.".format(sum(test_df.isnull().sum())))

print("The specs data has {} missing values.".format(sum(specs_df.isnull().sum())))
for (name, data) in train_df.drop("event_data", axis = 1).iteritems():

    unique = len(data.unique())

    print("The column {} of the training data has {} unique entries.".format(name,unique))
for (name, data) in test_df.iteritems():

    unique = len(data.unique())

    print("The column {} of the test data has {} unique entries.".format(name,unique))
#Visualization of Type of Activity 

media_train = train_df.loc[:,["type", "event_id"]].groupby("type").count()

media_test = test_df.loc[:,["type", "event_id"]].groupby("type").count()



plt.figure(dpi = 100)

plt.subplot(2,1,1)

plt.barh(media_train.index, width = media_train["event_id"]/len(train_df), color = (0.36,0.54,0.66, 0.6))

plt.title('Media Types of the Training Data')

plt.xlabel('Percentage')

plt.subplot(2,1,2)

plt.barh(media_test.index, width = media_test["event_id"]/len(test_df), color = (0.77,0.38,0.06, 1))

plt.title('Media Types of the Test Data')

plt.xlabel('Percentage')

plt.subplots_adjust(hspace = 1)

plt.show()
world_train = train_df.loc[:,["world", "game_session"]].groupby("world")["game_session"].nunique()

world_test = test_df.loc[:,["world", "game_session"]].groupby("world")["game_session"].nunique()





plt.figure(dpi = 100)

plt.subplot(1,2,1)

plt.bar(world_train.index, world_train, color = (0.36,0.54,0.66, 0.6))

plt.ylabel("Count")

plt.xticks(rotation = "vertical")

plt.title("Worlds Training Data")



plt.subplot(1,2,2)

plt.bar(world_test.index, world_test, color = (0.77,0.38,0.06, 1))

plt.ylabel("Count")

plt.xticks(rotation = "vertical")

plt.title("Worlds Test Data")

plt.subplots_adjust(wspace = 0.7)

plt.show()
#Group the data so we can get the absolute count

label_group = train_labels_df.groupby("accuracy_group").count()



#Create the labels for the positioning of the text

labels_count = label_group["game_session"]

bar_position = label_group.index



#plot the graph

plt.figure(dpi=100)

plt.bar(x = label_group.index, height = label_group["game_session"], color = (0.36,0.54,0.66, 0.6))

plt.xlabel("Accuracy Group")

plt.xticks([0,1,2,3])

plt.ylabel("Count")

plt.title("Distribution of the Labels")

for i in range(len(labels_count)):

    plt.text(x = bar_position[i]-0.2, y = labels_count[i] - 1000, s = labels_count[i], size = 12)

plt.show()
def plot_by_group(dataset, column_name, groupby, xlabel, ylabel):

    

    plt.figure(dpi = 300)

    levels = dataset[column_name].unique()

    

    for i in range(len(levels)):

        

        df = dataset[dataset[column_name] == levels[i]].groupby(groupby).count()

        

        plt.subplot(round(len(levels)/2), round(len(levels)/2)+1, i+1)

        plt.bar(x = df.index, height = df["game_session"], color = (0.36,0.54,0.66, 0.6))

        plt.xticks([0,1,2,3], fontsize = 6)

        plt.yticks(fontsize = 6)

        plt.xlabel(xlabel, fontsize = 6)

        plt.ylabel(ylabel, fontsize = 6)

        plt.title(levels[i], fontdict = {"fontsize": 6})



    plt.subplots_adjust(hspace = 0.5, wspace = 1)

    plt.show()

    

plot_by_group(train_labels_df, "title", "accuracy_group", "Accuracy Group", "Count")
worlds = train_df["world"].unique()

worlds



plt.figure(dpi = 150)



for i in range(len(worlds)):

    world_filter = train_df[train_df["world"] == worlds[i]].groupby("type")["game_session"].nunique()

    

    plt.subplot(2,2,i+1)

    plt.bar(x = world_filter.index, height = world_filter, color =  (0.36,0.54,0.66, 0.6))

    plt.xlabel("Type of Media", fontsize = 6)

    plt.ylabel("Count", fontsize = 6)

    plt.xticks(fontsize = 6)

    plt.yticks(fontsize = 6)

    plt.title("World: {}".format(worlds[i]), fontdict = {"fontsize": 6})

    

plt.subplots_adjust(hspace = 0.5, wspace = 0.5)  

plt.show()
world_train = train_df.loc[:,["world", "event_id"]].groupby("world").count()

world_test = test_df.loc[:,["world", "event_id"]].groupby("world").count()





plt.figure(dpi = 100)

plt.subplot(1,2,1)

plt.bar(world_train.index, world_train["event_id"], color = (0.36,0.54,0.66, 0.6))

plt.ylabel("Count")

plt.xticks(rotation = "vertical")

plt.title("Actions per Worlds Training Data")



plt.subplot(1,2,2)

plt.bar(world_test.index, world_test["event_id"], color = (0.77,0.38,0.06, 1))

plt.ylabel("Count")

plt.xticks(rotation = "vertical")

plt.title("Actions per Worlds Test Data")

plt.subplots_adjust(wspace = 0.7)

plt.show()
plt.figure(dpi = 300)



for i in range(len(worlds)):

    world_filter = train_df[train_df["world"] == worlds[i]].groupby("title").count()

    

    plt.subplot(2,2,i+1)

    plt.bar(x = world_filter.index, height = world_filter["game_session"], color =  (0.36,0.54,0.66, 0.6))

    plt.xlabel("Title of Activity", fontsize = 6)

    plt.ylabel("Count", fontsize = 6)

    plt.xticks(fontsize = 6, rotation = "vertical")

    plt.yticks(fontsize = 6)

    plt.title("Action per Title, World: {}".format(worlds[i]), fontdict = {"fontsize": 6})

    

plt.subplots_adjust(hspace = 2.5, wspace = 0.5)  

plt.show()
df_title = train_df["title"].value_counts().sort_values(ascending=True)



plt.figure(dpi=150)

df_title.plot.barh(color =  (0.77,0.38,0.06, 1))

plt.xlabel("Count")

plt.ylabel("Title")

plt.yticks(size = 4)

plt.xticks(size = 4)

plt.show()
#Print the Type of Activity per Title for each city

for entry in worlds:

    world_filter = train_df[train_df["world"] == entry].groupby(["type", "title"]).count()

    

    world_filter = world_filter.index

    

    activity = [act[0] for act in world_filter]

    title = [act[1] for act in world_filter]

    

    df = pd.DataFrame({"Type of Activity": activity, "Title": title})

    

    print("World: {}".format(entry))

    print(df)

    print("")