import pandas as pd
# Data
df = pd.DataFrame([
  [1, '3 inch screw', 0.5, 0.75],
  [2, '2 inch nail', 0.10, 0.25],
  [3, 'hammer', 3.00, 5.50],
  [4, 'screwdriver', 2.50, 3.00]
],
  columns=['Product ID', 'Description', 'Cost to Manufacture', 'Price']
)
print(df)
df['Sold in Bulk?'] = ['Yes', 'Yes', 'No', 'No']
print(df)
df['Is taxed?'] = 'Yes'
print(df)
df['Revenue'] = df['Price'] - df['Cost to Manufacture']
print(df)
df = pd.DataFrame([
  ['JOHN SMITH', 'john.smith@gmail.com'],
  ['Jane Doe', 'jdoe@yahoo.com'],
  ['joe schmo', 'joeschmo@hotmail.com']
],
columns=['Name', 'Email'])
print(df)
df['Email Provider'] = df.Email.apply(lambda x : x.split('@')[-1])
print(df)
df['Message'] = df.apply(
    lambda row: row.Name + ' uses gmail'
            if row['Email Provider'] == 'gmail.com'
            else row.Name + ' uses ' + row['Email Provider'],
    axis = 1
)
print(df)
df.columns = ['Full Name', 'Email Address', 'Email Provider', 'Message']
print(df.info())
df.rename(columns={
    'Full Name': 'Full_Name',
    'Email Address': 'Email_Address',
    'Email Provider': 'Email_Provider'
}, inplace=True)
print(df.info())