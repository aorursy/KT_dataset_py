import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import re

import matplotlib as mpl



from matplotlib import rcParams

from sklearn.preprocessing import StandardScaler

from keras.models import Sequential

from keras.layers import Dropout, Dense

from keras.callbacks import EarlyStopping, ModelCheckpoint



np.random.seed(0)



# Plots the figure in the kernel rather than opening a window or tab.

%matplotlib inline



# Set the universal size for figure

rcParams['figure.figsize'] = (10, 8)

plt.style.use("ggplot")

mpl.rc("savefig", dpi = 200)
train_df = pd.read_csv("../input/titanic/train.csv")

test_df  = pd.read_csv("../input/titanic/test.csv")
print("[+] Basic Information on Training Dataset: \n")

print(train_df.info())



print('')

print("[+] Basic Information on Testing Dataset: \n")

print(test_df.info())
print("[+] Basic Statistics on Training DataFrame: \n")

print(train_df.describe())



print('')

print("[+] Basic Statistics on Testing DataFrame: \n")

print(test_df.describe())
print("[+] First Five Rows of Training DataFrame:")

print("##########################################\n")



print(train_df.head(5))

print("")





print("[+] First Five Rows of Testing DataFrame:")

print("##########################################\n")



print(test_df.head(5))
print("[+] The Name Column of Training Dataset:")

print("#######################################\n")



print(train_df['Name'].head(5))
train_df['Title'] = train_df["Name"].apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))







_ = plt.figure(figsize = (12, 5))

_ = plt.xlabel("Title", fontsize = 16)

_ = plt.ylabel("Count", fontsize = 16)

_ = plt.title("Occurances of Titles", fontsize = 20)

_ = plt.xticks(rotation = 90)



sns.countplot(x = 'Title', data = train_df, palette = "Blues_d")



plt.show()



# Repeat the same procedure for testing dataset

test_df['Title'] = test_df["Name"].apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))
Title_Dictionary = {

        "Capt":       "Officer",

        "Col":        "Officer",

        "Major":      "Officer",

        "Dr":         "Officer",

        "Rev":        "Officer",

        "Jonkheer":   "Royalty",

        "Don":        "Royalty",

        "Sir" :       "Royalty",

        "the Countess":"Royalty",

        "Dona":       "Royalty",

        "Lady" :      "Royalty",

        "Mme":        "Mrs",

        "Ms":         "Mrs",

        "Mrs" :       "Mrs",

        "Mlle":       "Miss",

        "Miss" :      "Miss",

        "Mr" :        "Mr",

        "Master" :    "Master"

                   }



train_df['Title'] = train_df['Title'].map(Title_Dictionary)

test_df['Title'] = test_df['Title'].map(Title_Dictionary)
print("[+] First 5 Rows of the  'Pclass' column: ")

print("########################################\n")

print(train_df['Pclass'].head())

print("")



print("[+] Count of Categories in Tabular Format: ")

print("##########################################\n")

print(train_df['Pclass'].value_counts())



_ = plt.figure(figsize = (12, 5))

_ = plt.xlabel("Pclass", fontsize = 15)

_ = plt.ylabel("Count", fontsize = 15)

_ = plt.title("Occurances of Pclass", fontsize = 15)

_ = plt.xticks(rotation = 0)



sns.countplot(x = 'Pclass', data = train_df, palette = "Blues_d")



plt.show()
# Plotting and basic EDA on the column



print("[+] First 5 Rows of the  'Sex' column: ")

print("########################################\n")

print(train_df['Sex'].head())

print("")



print("[+] Count of Genders in Tabular Format: ")

print("##########################################\n")

print(train_df['Sex'].value_counts())



_ = plt.figure(figsize = (12, 5))

_ = plt.xlabel("Sex", fontsize = 15)

_ = plt.ylabel("Count", fontsize = 15)

_ = plt.title("Occurances of Sex", fontsize = 15)

_ = plt.xticks(rotation = 0)



sns.countplot(x = 'Sex', data = train_df, palette = "Blues_d")



plt.show()
# Basic EDA and Imputation of Null Cells



print("[+] The first five rows of the 'Age' column: ")

print("##########################################\n")

print(train_df['Age'].head())

print("")



print("[+] Total Number of Null Values : ", train_df["Age"].isnull().sum())



# Imputing the NaN values from the column

train_df.loc[train_df.Age.isnull(), 'Age'] = train_df.groupby(['Sex','Pclass','Title'])['Age'].transform('median')

test_df.loc[test_df.Age.isnull(), 'Age']   = test_df.groupby(['Sex','Pclass','Title'])['Age'].transform('median')



print("[+] Total Null Values after Imputation: ", train_df["Age"].isnull().sum())
# Plotting the data



_ = plt.figure(figsize = (15,5))

_ = plt.title("Distribuition and density by Age")

_ = plt.xlabel("Age")

_ = plt.xticks(rotation = 0)



sns.distplot(train_df["Age"], bins = 24,  color = 'black')



plt.show()
plt.figure(figsize=(15,5))



plot = sns.FacetGrid(train_df, col = "Survived", size = 6.2)

plot = plot.map(sns.distplot, "Age", color = 'black')



plt.show()
age_intervals = (0, 5, 12, 18, 25, 35, 60, 120)

categories    = ['Babies', 'Children', 'Teen', 'Student', 'Young', 'Adult', 'Senior']



train_df["Age_Category"] = pd.cut(train_df['Age'], age_intervals, labels = categories)

test_df["Age_Category"]  = pd.cut(test_df['Age'], age_intervals, labels = categories)
print(pd.crosstab(train_df['Age_Category'], train_df['Survived']))



_ = plt.figure(figsize = (15, 5))

_ = plt.ylabel("Fare Distribution", fontsize=18)

_ = plt.xlabel("Age Categorys", fontsize=18)

_ = plt.title("Fare Distribution by Age Categorys ", fontsize=20)



sns.swarmplot(x = 'Age_Category',y = "Fare", data = train_df, hue = "Survived", palette = "PuBuGn_d")



plt.subplots_adjust(hspace = 0.5, top = 0.9)



plt.show()
# Figure Setup

_ = plt.figure(figsize=(15, 5))

_ = plt.ylabel("Count", fontsize = 18)

_ = plt.xlabel("Age Categorys", fontsize = 18)

_ = plt.title("Age Distribution ", fontsize = 20)



sns.countplot("Age_Category",data = train_df, hue = "Survived", palette = "PuBuGn_d")



# Plot the figure

plt.show()

# Remove the irrelevent columns

train_dataset = train_df.drop(columns = ['Fare', 'Ticket', 'Age', 'Cabin', 'Name'])

test_dataset  = test_df.drop(columns = ['Fare', 'Ticket', 'Age', 'Cabin', 'Name'])
train_dataset.head()
_ = plt.figure(figsize = (15, 5))

_ = plt.title("Sex Distribution According to Survived or Not")

_ = plt.xlabel("Sex Distribution")

_ = plt.ylabel("Count")



sns.countplot(x = "Sex", data = train_dataset, hue = "Survived", palette = 'PuBuGn_d')



plt.show()
train_dataset["Embarked"] = train_dataset["Embarked"].fillna('S')

test_dataset["Embarked"]  = test_dataset["Embarked"].fillna('S')



_ = plt.figure(figsize = (15, 5))

_ = plt.title("Pclass Distribution According Survival")

_ = plt.xlabel("Embarked")

_ = plt.ylabel("Count")



sns.countplot(x = 'Embarked', data = train_dataset, hue = 'Survived', palette = 'PuBuGn_d')



plt.show()
plot = sns.factorplot(x = 'SibSp', y = 'Survived', data = train_dataset, kind = 'bar', height = 5, aspect = 1.6, palette = "PuBuGn_d")

_    = plot.set_ylabels("Probability of Survival")

_    = plot.set_xlabels("SibSp Number")



plt.show()
train_dataset = pd.get_dummies(train_dataset, columns = ["Sex", "Embarked", "Age_Category","Title"], prefix = ["Sex", "Emb", "Age", "Prefix"], drop_first = True)

test_dataset  = pd.get_dummies(test_dataset, columns = ["Sex", "Embarked", "Age_Category","Title"], prefix = ["Sex", "Emb", "Age", "Prefix"], drop_first = True)
# Plotting the correlation matrix



_ = plt.figure(figsize = (18, 15))

_ = plt.title("Correlation Matrix of Features in Training Dataset")

_ = sns.heatmap(train_dataset.astype(float).corr(), vmax = 1.0, annot = False, cmap = "Blues")



plt.show()
train_dataset.columns.tolist()
test_dataset.columns.tolist()
training_data = train_dataset.drop(['Survived', 'PassengerId'], axis = 1)

training_target = train_dataset["Survived"]



testing_data = test_dataset.drop(["PassengerId"], axis = 1)



X_train, y_train = training_data.values, training_target.values 

X_test = (testing_data.values).astype(np.float64, copy = False)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test  = scaler.fit_transform(X_test)
print("[+] Shapes of Training and Testing Datasets: ")

print("############################################\n")



print('> Training Dataset = ', X_train.shape)

print('> Testing Dataset  = ', X_test.shape)
model = Sequential()

model.add(Dense(50, input_shape = (17, ), activation = 'relu'))

model.add(Dropout(0.5))

model.add(Dense(1, activation = 'sigmoid'))

model.summary()



callbacks = [EarlyStopping(monitor='val_loss', patience = 2, mode = 'min'), 

             ModelCheckpoint(filepath = 'best_model.h5', monitor = 'val_loss', save_best_only = True)]
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

network = model.fit(X_train, y_train, batch_size = 50, epochs = 100, verbose = True, callbacks = callbacks, validation_split = 0.2)
model.load_weights("best_model.h5")

y_preds = model.predict(X_test)



submission = pd.read_csv('../input/titanic/gender_submission.csv', index_col = 'PassengerId')

submission['Survived'] = y_preds.astype(int)

submission.to_csv('/kaggle/working/TitanicPreds.csv')
print("[+] Available Parameters in Model's History: ")

print("#############################################\n")



for index, key in enumerate(network.history.keys()): print(str(index + 1) + ". ", key)



_ = plt.figure(figsize = (15, 8))

_ = plt.plot(network.history['val_accuracy'])

_ = plt.plot(network.history['accuracy'])

_ = plt.title('Training and Validation Accuracy')

_ = plt.ylabel('Accuracy')

_ = plt.xlabel('Epoch')

_ = plt.legend(['train', 'validation'], loc = 'upper left')



plt.show()