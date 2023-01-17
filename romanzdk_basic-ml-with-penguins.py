import numpy as np
import pandas as pd
df = pd.read_excel("/kaggle/input/penguin data.xlsx")
df.head()
df.shape
df.info()
df.describe(include = 'all')
#Drop unnecessary columns
df.drop(['studyName', 'Sample Number', 'Region', 'Stage', 'Individual ID', 'Date Egg' ,'Delta 15 N (o/oo)', 'Delta 13 C (o/oo)', 'Comments'], axis = 1, inplace = True)
#Remove latin name of the penguins
df = df.replace('\(.*', '', regex=True)
#Check the records with the null values
df[df['Culmen Length (mm)'].isna()]
#Let's drop these records
df = df[df['Culmen Length (mm)'].notnull()]
#There are 3 types of sex
df['Sex'].value_counts()
#Select sex value "."
df[df['Sex'] == '.']
#Let's see counts by Island and Sex
df.groupby(['Species', 'Island', 'Sex']).count()
df['Sex'] = df['Sex'].replace('.', 'MALE')
#Some sex is missing
df[df['Sex'].isna()]
#Check the counts by Sex
df.groupby(['Sex']).count()
df['Sex'] = df['Sex'].fillna('MALE')
df.count()
df.head()
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(handle_unknown='ignore')
encoded = pd.DataFrame(ohe.fit_transform(df[['Island', 'Clutch Completion', 'Sex']]).toarray())
#join original dataframe with the encoded one
df = df.join(encoded)
#rename columns after one hot encoding
real = ohe.get_feature_names(['Island', 'Clutch Completion', 'Sex'])
keys = np.array(df.columns[8:])
names = dict(zip(keys,real))
df.rename(columns = names, inplace = True)
df.head()
#transform target variable into codes
df['Species'] = df.Species.astype('category')
df['Species'] = df.Species.cat.codes
#drop text attributes
df.drop(columns=['Island', 'Clutch Completion', 'Sex'], inplace = True)
#drop nulls
df.dropna(inplace = True)
df
#build a decision tree classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
model = DecisionTreeClassifier()
#fit the model
mytree = model.fit(X = df[df.columns[1:]], y = df['Species'])
#plot the tree
tree.plot_tree(mytree)
#export to pdf
import graphviz 
dot_data = tree.export_graphviz(mytree, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("penguins") 