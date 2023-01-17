#IMPORTING PACKAGES
import seaborn as sns                   #Visualization
import matplotlib.pyplot as plt         #Visualization
import pandas as pd                     #Statistics
import random                           #Generate random numbers


#CREATING A RANDOM DATAFRAME
matrix = []
for i in range(5):
  linha = []
  for j in range(5):
    x = random.randint(1,5)
    y = random.randint(1,5)
    z = random.randint(1,5)
    w = random.randint(1,5)
    linha.append([x, y, z, w])
  matrix.append(linha)
  df = pd.DataFrame(linha,columns=['Test 1','Test 2','Test 3','Test 4'],index=['Aluno 1','Aluno 2','Aluno 3','Aluno 4','Aluno 5'])
df
fig = plt.figure(figsize=(10,5))
sns.barplot(x=df.index, y='Test 1',data=df)
plt.title('Nota dos alunos no teste 1',fontsize=19)
plt.ylim(0,5.5);
fig = plt.figure(figsize=(10,5))
sns.countplot(x='Test 1',order=df['Test 1'].value_counts().index,data=df)
plt.title('Nota dos alunos',fontsize=19)
plt.ylim(0,5)
plt.xlabel('Scores',fontsize=13)
plt.ylabel('Quantity', fontsize=13);
fig = plt.figure(figsize=(15,5))
plt.subplot(2,2,1)
sns.countplot(x='Test 1',data=df)
plt.title('Test 1',fontsize=13)
plt.ylim(0,5)
plt.xlabel('Scores',fontsize=11)
plt.ylabel('Quantity', fontsize=11)

fig = plt.figure(figsize=(15,5))
plt.subplot(2,2,2)
sns.countplot(x='Test 2',data=df)
plt.title('Test 2',fontsize=13)
plt.ylim(0,5)
plt.xlabel('Scores',fontsize=11)
plt.ylabel('Quantity', fontsize=11);

fig = plt.figure(figsize=(15,5))
plt.subplot(2,2,3)
sns.countplot(x='Test 3',data=df)
plt.title('Test 3',fontsize=13)
plt.ylim(0,5)
plt.xlabel('Scores',fontsize=11)
plt.ylabel('Quantity', fontsize=11)

fig = plt.figure(figsize=(15,5))
plt.subplot(2,2,4)
sns.countplot(x='Test 4',data=df)
plt.title('Test 4',fontsize=13)
plt.ylim(0,5)
plt.xlabel('Scores',fontsize=11)
plt.ylabel('Quantity', fontsize=11);
from subprocess import check_output
print(check_output(["ls", "../input/videogamesales"]).decode("utf8"))
df = pd.read_csv('../input/videogamesales/vgsales.csv')
df.head()
sns.countplot(x='Platform',data=df)
fig = plt.figure(figsize=(15,5))
sns.countplot(x='Platform',data=df);
df['Platform'].value_counts().index
fig = plt.figure(figsize=(15,5))
sns.countplot(x='Platform',order=df['Platform'].value_counts().index,data=df);
fig = plt.figure(figsize=(15,5))
sns.countplot(x='Platform',order=df['Platform'].value_counts().index,data=df)
plt.xlim(0,10);
fig = plt.figure(figsize=(15,5))
sns.countplot(x='Platform',order=df['Platform'].value_counts().index,data=df)
plt.xlim(-0.5,9.5);
fig = plt.figure(figsize=(15,5))
sns.countplot(x='Platform',order=df['Platform'].value_counts().index,data=df)
plt.xlim(-0.5,9.5)
plt.xlabel('')
plt.ylabel('Quantity sold',fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title('The 10 platform with more sold games',fontsize=21);
fig = plt.figure(figsize=(15,5))
sns.countplot(x='Platform',order=df['Platform'].value_counts().index,color= 'green',data=df)
plt.xlim(-0.5,9.5)
plt.xlabel('')
plt.ylabel('Quantity sold',fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title('The 10 platform with more sold games',fontsize=21);
palette_platform = sns.light_palette("green",10,reverse=True)
fig = plt.figure(figsize=(15,5))
sns.countplot(x='Platform',order=df['Platform'].value_counts().index,palette=palette_platform,data=df)
plt.xlim(-0.5,9.5)
plt.xlabel('')
plt.ylabel('Quantity sold',fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title('The 10 platform with more sold games',fontsize=21);
palette_platform = sns.light_palette("green",15,reverse=True)
fig = plt.figure(figsize=(15,5))
sns.countplot(x='Platform',order=df['Platform'].value_counts().index,palette=palette_platform,data=df)
plt.xlim(-0.5,9.5)
plt.xlabel('')
plt.ylabel('Quantity sold',fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title('The 10 platform with more sold games',fontsize=21);
#Final Graphic

palette_platform_cubehelix = sns.cubehelix_palette(15, start=2, rot=0, reverse=True)
fig = plt.figure(figsize=(15,5))
sns.countplot(x='Platform',order=df['Platform'].value_counts().index,palette=palette_platform_cubehelix,data=df)
plt.xlim(-0.5,9.5)
plt.xlabel('')
plt.ylabel('Quantity sold',fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.title('The 10 platform with more sold games',fontsize=21);
palette_genre_cubehelix = sns.cubehelix_palette(15,reverse=True)
fig = plt.figure(figsize=(15,5))
sns.countplot(x='Genre',order=df['Genre'].value_counts().index,palette=palette_genre_cubehelix,data=df)
plt.xlim(-0.5,9.5)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('')
plt.ylabel('Quantity sold',fontsize=15)
plt.title('The 10 genre more sold',fontsize=21);
palette_pub_cubehelix = sns.cubehelix_palette(15, start=3, rot=0, reverse=True)
fig = plt.figure(figsize=(19,5))
sns.countplot(x='Publisher',order=df['Publisher'].value_counts().index,palette=palette_pub_cubehelix,data=df)
plt.xlim(-0.5,9.5)
plt.xticks(fontsize=12,rotation=13)
plt.xlabel('')
plt.ylabel('Quantity sold', fontsize=15)
plt.yticks(fontsize=13)
plt.title('Avoid it!',fontsize=21);
fig = plt.figure(figsize=(15,5))
publi = df.Publisher.value_counts().sort_values(ascending=False).iloc[0:10]
sns.barplot(publi.values, publi.index,palette=palette_pub_cubehelix,data=df)
plt.yticks(fontsize=13)
plt.xticks(fontsize=13)
plt.title('The 10 publishers with more games sold',fontsize=21);