import pandas as pd

df = pd.read_csv('../input/decision-tree-on-titanic-data/train.csv')
df
df.head()
df.tail()
# To select a coloums there 2 ways
# to select one coloum 
df.get('Pclass')
df['Gender']

# To select more then 1 coloum
df[['Pclass','Gender','Survived']]
# To select more then 1 coloum
df[['Pclass','Gender','Survived']]
df[['Pclass','Gender','Survived']].head(50).plot(kind ='bar',x ='Pclass',y ='Survived',figsize =[ 20,5])
# when we have repetative values in a coloum, eg male & female , class , survived,we can count how many type a particular value appears.it is same as group by in sql
# step first select the coloum whcich has repetative values.
#in this example we want to calculate how many male and females passengers are there 
df['Gender'].value_counts()
df['Gender'].value_counts().plot(kind='bar')
df['Survived'].value_counts().plot(kind='pie')
(df['Survived'].value_counts()/len(df['Survived']))*100
# whenever we have to calculate the percentage in value count,divide the value counts by len
(df['Gender'].value_counts()/len(df['Gender']))*100
df['Gender'][df['Survived']==0]
df['Gender'][df['Survived']==0].value_counts().plot(kind='bar')
df['Gender'][df['Survived']==0].value_counts().plot(kind='pie')
df['Age'].min()
df['Age'].max()
df['Age'].plot(kind='bar',figsize=[20,10])
bins = [0,10,20,30,40,50,60,70,80]
df['Agebin']=pd.cut(df['Age'],bins)
df['Agebin']
df['Agebin'].value_counts()

df['Agebin'].value_counts().sort_index().plot(kind='bar')
df[df['Survived']==1]['Agebin'].value_counts().sort_index().plot(kind='bar')
df[df['Pclass']==3]['Survived'].value_counts().sort_index().plot(kind='bar')
df[df['Pclass']==2]['Survived'].value_counts().sort_index().plot(kind='bar')
df[df['Pclass']==1]['Survived'].value_counts().sort_index().plot(kind='bar')
