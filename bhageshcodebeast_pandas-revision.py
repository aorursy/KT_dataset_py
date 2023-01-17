try:

    import pandas as pd

except:

    print('Import error')
asus_dataset = pd.read_csv('../input/asus-laptops-2020-jun/AsusLaptops.csv')

asus_dataset.head(5)
asus_dataset.columns
nRow, nCol = asus_dataset.shape



print(f"there are {nRow} rows & {nCol} columns")
new_asus_data = asus_dataset.drop('Unnamed: 0',axis = 1)

new_asus_data.head(5)
new_asus_data.columns = new_asus_data.columns.str.lower()

new_asus_data.head(5)
df = new_asus_data.rename(columns= {

    'mrp':'price','modelname':'model','colour&chassis':'build'

})



df.head(5)
sorted_byprice = df.sort_values(by='price')

sorted_byprice.head(5)
sorted_byprice.describe()
df.loc[df.price <= 80000]
condition_1 = df.loc[

    df.description.str.contains('^R5.*512GB*') & 

    (df.price<=90000)

]

condition_1
condition_2 = df.loc[df.description.str.contains('.*GTX 1650*')]

condition_2.head(5)
condition_3 = df.loc[

    df.description.str.contains('^R.*SSD*') & 

    (df.description.str.contains('.*HDD*') == False) &

    (df.price <= 90000)

]



condition_3.iloc[:,[1,2,-1]]
condition_3.description
data2save = condition_3.iloc[:,[1,2,-1]]

data2save.to_csv('filename.csv')
data2save