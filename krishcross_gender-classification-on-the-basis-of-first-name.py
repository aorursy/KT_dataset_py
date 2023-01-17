

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df=pd.read_csv('../input/usa-national-names/NationalNames.csv')

df.drop_duplicates("Name")

df.columns
df.head
df.columns
print(df['Gender'])
m=0
f=0
for x in range(len(df['Gender'])):
    if(df['Gender'][x]=='F'):
        f+=1
    else:
        m+=1
print("The number of females are "+(str)(f))
print("The number of males are "+(str)(m))
import matplotlib.pyplot as plt
import matplotlib as mp

fig, ax = plt.subplots(figsize=(25,12))
ax.bar(["Males","Females"],[m,f],color=['Brown','Pink'])
plt.title("% of MALES vs FEMALES in our dataset")
plt.xticks(rotation=90)
plt.show()



fig1, ax1 = plt.subplots(figsize=(20,10))
names=['MALES','FEMALES']
ax1.pie([f,m], labels=names, autopct='%1.1f%%',
        shadow=True, startangle=90,frame=True,colors=["blue","pink"])
ax1.set_title("% of MALES vs FEMALES in our dataset")

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
def checkVowelEnd(name):
    if name[-1] in "aeiou":
        return "Vowel End"
    return "Consonant End"
df["Vowel/Consonant End"] = df["Name"].apply(checkVowelEnd)
df.head()
vend=0
mend=0
for x in range(len(df['Gender'])):
                      
    if(df['Gender'][x]=='F' and df['Vowel/Consonant End'][x]=='Vowel End'):
        vend+=1
    if(df['Gender'][x]=='M' and df['Vowel/Consonant End'][x]=='Vowel End'):
        mend+=1
import matplotlib.pyplot as plt
import matplotlib as mp


fig, ax = plt.subplots(figsize=(25,12))
ax.bar(['FEMALES WITH NAMES ENDING WITH VOWEL','FEMALES WITH NAMES ENDING WITH CONSONANT'],[vend,f-vend],color=['Red','Pink'])
plt.title("Analysis of FEMALES in our dataset")
plt.xticks(rotation=90)
plt.show()


fig1, ax1 = plt.subplots(figsize=(20,10))
names=['FEMALES WITH NAMES ENDING WITH VOWEL','FEMALES WITH NAMES ENDING WITH CONSONANT']
ax1.pie([vend,f-vend], labels=names, autopct='%1.1f%%',
        shadow=True, startangle=90,frame=True,colors=["blue","pink"])
ax1.set_title("Analysis of FEMALES in our dataset")

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()



fig, ax = plt.subplots(figsize=(25,12))
ax.bar(['MALES WITH NAMES ENDING WITH VOWEL','MALES WITH NAMES ENDING WITH CONSONANT'],[mend,m-vend],color=['Orange','Yellow'])
plt.title("Analysis of MALES in our dataset")
plt.xticks(rotation=90)
plt.show()


fig1, ax1 = plt.subplots(figsize=(20,10))
names=['MALES WITH NAMES ENDING WITH VOWEL','MALES WITH NAMES ENDING WITH CONSONANT']
ax1.pie([mend,m-mend], labels=names, autopct='%1.1f%%',
        shadow=True, startangle=90,frame=True,colors=["blue","Red"])
ax1.set_title("Analysis of MALES in our dataset")

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
def vowelConsonantStart(name):
    
    if name[0] in "aeiou" or name[0] in "AEIOU":
        return "Vowel Start"
    return "Consonant Start"

df["Vowel/Consonant Start"] = df["Name"].apply(vowelConsonantStart)


df.head()
fcnst=0
fvst=0
mvst=0
mcnst=0
for x in range(len(df['Gender'])):
                      
    if(df['Gender'][x]=='F' and df['Vowel/Consonant Start'][x]=='Consonant Start'):
        fcnst+=1
    if(df['Gender'][x]=='F' and df['Vowel/Consonant Start'][x]=='Vowel Start'):
        fvst+=1
    if(df['Gender'][x]=='M' and df['Vowel/Consonant Start'][x]=='Consonant Start'):
        mcnst+=1
    if(df['Gender'][x]=='M' and df['Vowel/Consonant Start'][x]=='Vowel Start'):
        mvst+=1
print("Males with names starting with vowels"+(str)(mvst))
print("Males with names starting with consonant"+(str)(mcnst))
print("Females with names starting with vowels"+(str)(fvst))
print("Females with names starting with consonant"+(str)(fcnst))




    
        
    
print("% Males with names starting with vowels    "+(str)(mvst*100/m))
print("% Males with names starting with consonant   "+(str)(mcnst*100/m))
print("% Females with names starting with vowels   "+(str)(fvst*100/f))
print("% Females with names starting with consonant   "+(str)(fcnst*100/f))
fig, ax = plt.subplots(figsize=(25,12))
ax.bar(['FEMALES WITH NAMES STARTING WITH A VOWEL','FEMALES WITH NAMES STARTING WITH CONSONANT'],[fvst,f-fvst],color=['Red','Pink'])
plt.title("Analysis of FEMALES in our dataset")
plt.xticks(rotation=90)
plt.show()


fig1, ax1 = plt.subplots(figsize=(20,10))
names=['FEMALES WITH NAMES STARTING WITH A VOWEL','FEMALES WITH NAMES STARTING WITH CONSONANT']
ax1.pie([fvst,f-fvst], labels=names, autopct='%1.1f%%',
        shadow=True, startangle=90,frame=True,colors=["blue","pink"])
ax1.set_title("Analysis of FEMALES in our dataset")

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
fig, ax = plt.subplots(figsize=(25,12))
ax.bar(['MALES WITH NAMES STARTING WITH A VOWEL','MALES WITH NAMES STARTING WITH CONSONANT'],[mvst,m-mvst],color=['Orange','Yellow'])
plt.title("Analysis of MALES in our dataset")
plt.xticks(rotation=90)
plt.show()


fig1, ax1 = plt.subplots(figsize=(20,10))
names=['MALES WITH NAMES STARTING WITH A VOWEL','MALES WITH NAMES STARTING WITH CONSONANT']
ax1.pie([mvst,m-mvst], labels=names, autopct='%1.1f%%',
        shadow=True, startangle=90,frame=True,colors=['Orange','Yellow'])
ax1.set_title("Analysis of MALES in our dataset")

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
avg_length=0
for x in df['Name']:
    avg_length=avg_length+len(x)
avg_length=avg_length/len(df['Name'])
print(" The average length of names for our dataset is:-"+(str)(avg_length))
vsl=0
msl=0
def shortLongName(name):
    if len(name) < 7:
        return "Short"
    return "Long"

df["Short/Long Name"] = df["Name"].apply(shortLongName)






        
df
for x in range(len(df['Gender'])):
    if(df['Gender'][x]=='F' and df['Short/Long Name'][x]=="Short" ):
        vsl+=1
    if(df['Gender'][x]=='M' and df['Short/Long Name'][x]=="Short" ):
        msl+=1
print("Males with length of name above 7 = "+(str)(m-msl))
print("Males with length of name less than 7 = "+(str)(msl))
print("FeMales with length of name above 7 = "+(str)(f-vsl))
print("FeMales with length of name less than 7 = "+(str)(vsl))

fig1, ax1 = plt.subplots(figsize=(20,10))
names=['MALES WITH NAMES WITH LENGTH LESS THAN 7','MALES WITH NAMES WITH LENGTH MORE THAN 7']
ax1.pie([msl,m-msl], labels=names, autopct='%1.1f%%',
        shadow=True, startangle=90,frame=True,colors=['Orange','Yellow'])
ax1.set_title("Analysis of MALES in our dataset")

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()



fig1, ax1 = plt.subplots(figsize=(20,10))
names=['FEMALES WITH NAMES WITH LENGTH LESS THAN 7','FEMALES WITH NAMES WITH LENGTH MORE THAN 7']
ax1.pie([vsl,f-vsl], labels=names, autopct='%1.1f%%',
        shadow=True, startangle=90,frame=True,colors=['Pink','Red'])
ax1.set_title("Analysis of FeMALES in our dataset")

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()





from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
df['Gender']=pd.get_dummies(df['Gender'])

import pandas as pd
df['Vowel/Consonant End']=pd.get_dummies(df['Vowel/Consonant End'])
df['Short/Long Name']=pd.get_dummies(df['Short/Long Name'])
df['Vowel/Consonant Start']=pd.get_dummies(df['Vowel/Consonant Start'])
df
trains=df[['Gender','Vowel/Consonant End','Short/Long Name']]
trains.head

train, test = train_test_split(trains, test_size = 0.20)
clf = DecisionTreeClassifier()
clf.fit(train[['Vowel/Consonant End','Short/Long Name']],train[['Gender']])
res=clf.predict(test[['Vowel/Consonant End','Short/Long Name']])
print(res)
accuracy_score(test["Gender"], res)
from sklearn import tree
with open("/kaggle/working/decidenames.dot", "w") as dot_file:
    dot_file = tree.export_graphviz(clf,
                            feature_names=["Vowel/Consonant End", "Short/Long Name"], out_file=dot_file)
!dot -Tpng decidenames.dot -o tree_limited.png -Gdpi=600
from IPython.display import Image
Image(filename = 'tree_limited.png')