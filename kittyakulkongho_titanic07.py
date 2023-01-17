df=pd.read_csv('../input/TitanicDataset/titanic_data.csv')
print(df[:])
df=pd.read_csv('../input/TitanicDataset/titanic_data.csv')
df[df.Name.str.match('Sandstrom, Miss. Marguerite Rut')]
df=pd.read_csv('../input/TitanicDataset/titanic_data.csv')
df[df.Name.str.match('Samaan, Mr. Yousse')]
df=pd.read_csv('../input/TitanicDataset/titanic_data.csv')
df[['Fare','Age']].plot.bar()
df=pd.read_csv('../input/TitanicDataset/titanic_data.csv')
df[99:109]
df=pd.read_csv('../input/TitanicDataset/titanic_data.csv')
df[:]
sns.scatterplot(x='Age',y='Survived',data=df)  