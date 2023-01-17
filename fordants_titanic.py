import numpy as np
import pandas as pd
import pylab as plt 

#set the global default size of matplotlib figures 

plt.rc('figure', figsize=(10,5)) 

# size of matplotlib figures that contain subplots 
figsize_with_subplots = (10,10)

#size of matplotlib histogram bins
bin_size = 10 

# read the data
#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)

df_train = pd.read_csv("../input/train.csv") 
print(df_train.tail())
print(df_train.dtypes)
df_train.info()
df_train.describe()
#set up a grid of plots 
fig = plt.figure(figsize=figsize_with_subplots)
fig_dims= (3,2)

#plot death and survival counts
plt.subplot2grid(fig_dims, (0,0)) 
df_train['Survived'].value_counts().plot(kind="bar", title="Death and Survival Counts")

#plot Pclass counts
plt.subplot2grid(fig_dims, (0,1))
df_train['Pclass'].value_counts().plot(kind='bar',title="Passenger class counts")

#plot sex counts
plt.subplot2grid(fig_dims, (1,0))
df_train['Sex'].value_counts().plot(kind='bar',title="Gender counts")

plt.xticks(rotation=0)

#plot embarked counts 
plt.subplot2grid(fig_dims,(1,1)) 
df_train['Embarked'].value_counts().plot(kind='bar', title = 'Ports of embarkation counts')

#plot the age histogram
plt.subplot2grid(fig_dims, (2,0))
df_train['Age'].hist() 
plt.title('Age histogram')
pclass_xt = pd.crosstab(df_train['Pclass'], df_train['Survived'])
#normalize the cross tab to sum to 1:
pclass_xt_pct = pclass_xt.div(pclass_xt.sum(1).astype(float), axis = 0)
pclass_xt_pct.plot(kind='bar', stacked = True, title = "Survival rate by Passenger Classes")
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
sexes = sorted(df_train['Sex'].unique())
genders_mapping = dict(zip(sexes, range(0,len(sexes) + 1)))
genders_mapping
#transform sex from string to number representation
df_train['Sex_Val'] = df_train['Sex'].map(genders_mapping).astype(int)
df_train.head()
#plot normalized cross tab for sex_val and survived: 
sex_val_xt = pd.crosstab(df_train['Sex_Val'],df_train['Survived']) 
sex_val_xt_pct =sex_val_xt.div(sex_val_xt.sum(1).astype('float'),axis = 0)
sex_val_xt_pct.plot(kind='bar',stacked=True, title="Survival rate by Gender")
# Get the unique values of Pclass:
passenger_classes= sorted(df_train['Pclass'].unique())
for p_class in passenger_classes: 
   print ('M: ', p_class, len(df_train[(df_train['Sex'] == 'male') & 
                             (df_train['Pclass'] == p_class)]))
   print ('F: ', p_class, len(df_train[(df_train['Sex'] == 'female') & 
                             (df_train['Pclass'] == p_class)]))
    
#plot survival rates by sex and pclass
males_df = df_train[df_train['Sex'] == 'male']
males_xt = pd.crosstab(males_df['Pclass'], df_train['Survived'])
males_xt_pct = males_xt.div(males_xt.sum(1).astype(float), axis = 0)
males_xt_pct.plot(kind='bar',stacked=True,title="Male Survival Rate by Passenger Class")
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')

#plot survival rates by sex and pclass
females_df = df_train[df_train['Sex'] == 'female']
females_xt = pd.crosstab(females_df['Pclass'], df_train['Survived'])
females_xt_pct = females_xt.div(females_xt.sum(1).astype(float), axis = 0)
females_xt_pct.plot(kind='bar',stacked=True,title="Female Survival Rate by Passenger Class")
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')                  

df_train[df_train['Embarked'].isnull()]

 
#get the unique values of embarked
embarked_locs = sorted(df_train['Embarked'].unique())
embarked_locs_mapping = dict(zip(embarked_locs, range(0, len(embarked_locs) + 1)))

#df_train['Embarked'].unique()             
df_train['Embarked_Val'] = df_train['Embarked'].map(embarked_locs_mapping).astype(int)
df_train['Embarked_Val'].hist(bins=len(embarked_locs), range=(0,3))
plt.title('Port of Embarkation Histogram')
plt.xlabel('Port of Embarkation')
plt.ylabel('Count')
plt.show()
if len(df_train[df_train['Embarked'].isnull()] > 0):
    df_train.replace({'Embarked_Val' : 
                   { embarked_locs_mapping['nan'] : embarked_locs_mapping['S'] 
                   }
               }, 
               inplace=True)
embarked_locs_mapping
df_train.head()

embarked_val_xt = pd.crosstab(df_train['Embarked_Val'], df_train['Survived'])
embarked_val_xt_pct = embarked_val_xt.div(embarked_val_xt.sum(1).astype(float), axis  = 0) 
embarked_val_xt_pct.plot(kind='bar',stacked=True)
plt.title('Survival rate by port of embarkation')
plt.xlabel('Port of Embarkation')
plt.ylabel('Survival rate')
plt.show()
