import pandas as  pd



d={'Name':pd.Series(['tashi','Akash','shamu']),

  'Age':pd.Series([23,43,12]),

  'Rating':pd.Series([23,54.65,34.56])}
df = pd.DataFrame(d)

print(df)
#add the dataset

print(df.sum())
#mean of dataset

print(df.mean())
print(df['Age'].mean())

#finding the mean of age column
#standard deviation

print(df.std())
#to find minimum

print(df.min())
#Maximum

print(df.max())
#to print absolute values

print(df['Age'].abs())
#to print product

print(df.prod())
#print commulative sum

print(df.cumsum())
#print commulativ product

print(df['Age'].cumprod())
#to print mode of the age

print(df['Age'].mode())