import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
data_f = pd.read_csv(r'../input/BlackFriday.csv')
data_f.head()
data_f.info()
data_f.describe()
plt.figure(figsize=(9, 6))

colors = ['dodgerblue', 'pink']
labels = ['Male', 'Female']

patches, l_text, p_text = plt.pie(
    data_f.Gender.value_counts(),
    labels=labels,
    colors=colors,
    explode=(0, 0.08),
    autopct='%4.2f%%',
    startangle=90,
    shadow=True)

for t in l_text + p_text:
    t.set_size(20)
for t in p_text:
    t.set_color('white')
plt.legend(fontsize=15, loc='best', title='Gender', frameon=False)
plt.axis('equal')
plt.figure(figsize=(9, 6))

Gender_M = data_f[data_f.Gender == 'M'].Purchase.sum()
Gender_F = data_f[data_f.Gender == 'F'].Purchase.sum()
colors = ['blueviolet', 'lightcoral']
labels = ['Male', 'Female']

patches, l_text, p_text = plt.pie([Gender_M, Gender_F],
                                  labels=labels,
                                  colors=colors,
                                  explode=(0, 0.08),
                                  autopct='%4.2f%%',
                                  startangle=90,
                                  shadow=True)

for t in l_text + p_text:
    t.set_size(20)
for t in p_text:
    t.set_color('white')
plt.legend(fontsize=15, loc='best', title='Total-Purchase', frameon=False)
plt.axis('equal')
plt.figure(figsize=(10, 6))

sizes = data_f.City_Category.value_counts()
labels = data_f.City_Category.value_counts().index
colors = ['sandybrown', 'deepskyblue', 'limegreen']

patches, l_text, p_text = plt.pie(
    sizes,
    labels=labels,
    colors=colors,
    explode=(0.08, 0, 0),
    autopct='%3.1f%%',
    startangle=90,
    shadow=True)

for t in l_text + p_text:
    t.set_size(20)
for t in p_text:
    t.set_color('white')
plt.legend(fontsize=15, loc='best', title='City-Purchases', frameon=False)
plt.axis('equal')
plt.figure(figsize=(10, 6))

sizes = data_f.groupby('City_Category')['Purchase'].sum().sort_values(
    ascending=False)
labels = data_f.groupby('City_Category')['Purchase'].sum().sort_values(
    ascending=False).index
colors = ['tomato', 'deepskyblue', 'limegreen']

patches, l_text, p_text = plt.pie(
    sizes,
    labels=labels,
    colors=colors,
    explode=(0.08, 0, 0),
    autopct='%3.1f%%',
    startangle=90,
    shadow=True)

for t in l_text + p_text:
    t.set_size(20)
for t in p_text:
    t.set_color('white')
plt.legend(fontsize=15, loc='best', title='City-Purchase', frameon=False)
plt.axis('equal')
group_City_Gender = data_f.groupby(['City_Category', 'Gender'])
group_City_Gender.count()
fig = plt.figure(figsize=(10, 8))
# ax1 = fig.add_subplot(121)
P_M = data_f.City_Category[data_f['Gender'] == 'M'].value_counts()
P_F = data_f.City_Category[data_f['Gender'] == 'F'].value_counts()
df = pd.DataFrame({'Female': P_F, 'Male': P_M})
df.plot.bar()
plt.title('city-gender', fontsize=20)
plt.xlabel('City_Category', fontsize=20)
plt.ylabel('Number_Of_Times', fontsize=20)
plt.grid(axis='y', linestyle=':')
ax = plt.gca()
ax.set_xticklabels(df.index, rotation=0, fontsize=15)

fig = plt.figure(figsize=(10, 8))
# ax2 = fig.add_subplot(122)
df_p = data_f.groupby(['City_Category',
                       'Gender'])['Purchase'].sum().unstack().sort_values(
                           by='M', ascending=False)
df_p.plot.bar()
plt.title('city-gender-Purchase', fontsize=20)
plt.ylabel('Purchase-sum', fontsize=20)
plt.xlabel('City_Category', fontsize=20)
plt.legend(['Female', 'Male'])
plt.grid(axis='y', linestyle=':')
ax = plt.gca()
ax.set_xticklabels(df_p.index, rotation=0, fontsize=15)

plt.tight_layout()
data_f.groupby('City_Category')[[
    'Product_Category_1', 'Product_Category_2', 'Product_Category_3'
]].count()
plt.figure(figsize=(10, 8))
df = data_f.groupby('City_Category')[[
    'Product_Category_1', 'Product_Category_2', 'Product_Category_3'
]].count().sort_values(
    by='Product_Category_1', ascending=False)
df.plot.bar()
plt.grid(axis='y', linestyle=':')
plt.xlabel('City_Category', fontsize=20)
plt.ylabel('Number_Of_Times', fontsize=20)
ax = plt.gca()
ax.set_xticklabels(df.index, rotation=0, fontsize=15)
df_A = data_f.groupby('Age')['Purchase'].sum()
df_A.plot.bar(figsize=(10, 8))
plt.title('Age-Purchase', fontsize=20)
plt.xlabel('Age', fontsize=20)
plt.ylabel('Purchase', fontsize=20)
plt.grid(axis='y', linestyle=':')
ax = plt.gca()
ax.set_xticklabels(
    df_A.index, fontsize=12, rotation=30, horizontalalignment='center')
plt.legend()
df_A = data_f.groupby(['Age', 'Gender'])['Purchase'].count().unstack()
df_A.plot.bar(figsize=(10, 8))
plt.title('Age-Gender', fontsize=20)
plt.xlabel('Age', fontsize=20)
plt.ylabel('Number_Of_Times', fontsize=20)
plt.grid(axis='y', linestyle=':')
ax = plt.gca()
ax.set_xticklabels(
    df_A.index, fontsize=12, rotation=30, horizontalalignment='center')
plt.legend(['Female', 'Male'],
           fontsize=15,
           loc='best',
           title='Gender',
           frameon=False)
df_A = data_f.groupby(['Age', 'Gender'])['Purchase'].sum().unstack()
df_A.plot.bar(figsize=(10, 8))
plt.title('Age-Gender-Purchase', fontsize=20)
plt.xlabel('Age', fontsize=20)
plt.ylabel('Purchase', fontsize=20)
plt.grid(axis='y', linestyle=':')
ax = plt.gca()
ax.set_xticklabels(
    df_A.index, fontsize=12, rotation=30, horizontalalignment='center')
plt.legend(['Female', 'Male'],
           fontsize=15,
           loc='best',
           title='Gender',
           frameon=False)
df_C_A = data_f.groupby(['City_Category', 'Age'])['Purchase'].sum().unstack()
df_C_A.plot.bar(figsize=(10, 8))
plt.title('City_Category-Age-Purchase', fontsize=20)
plt.xlabel('City_Category', fontsize=20)
plt.ylabel('Purchase', fontsize=20)
plt.grid(axis='y', linestyle=':')
ax = plt.gca()
ax.set_xticklabels(
    df_C_A.index, fontsize=12, rotation=0, horizontalalignment='center')
plt.legend(df_C_A.columns, fontsize=15, loc='best', title='Age', frameon=False)
df_A_P = data_f.groupby('Age')[[
    'Product_Category_1', 'Product_Category_2', 'Product_Category_3'
]].count()
df_A_P.plot.bar(figsize=(10, 8))
plt.title('Age-Product_Category', fontsize=20)
plt.xlabel('Age', fontsize=20)
plt.ylabel('Number_Of_Times', fontsize=20)
plt.grid(axis='y', linestyle=':')
ax = plt.gca()
ax.set_xticklabels(
    df_A_P.index, fontsize=12, rotation=0, horizontalalignment='center')
plt.legend(
    df_A_P.columns,
    fontsize=15,
    loc='best',
    title='Product_Category',
    frameon=False)
plt.figure(figsize=(10, 6))

sizes = data_f.groupby('Marital_Status')['Purchase'].sum().sort_values(
    ascending=False)
labels = data_f.groupby('Marital_Status')['Purchase'].sum().sort_values(
    ascending=False).rename(index={
        0: 'NO',
        1: 'YES'
    }).index
colors = ['deepskyblue', 'tomato']

patches, l_text, p_text = plt.pie(
    sizes,
    labels=labels,
    colors=colors,
    explode=(0.08, 0),
    autopct='%3.1f%%',
    startangle=90,
    shadow=True)

for t in l_text + p_text:
    t.set_size(20)
for t in p_text:
    t.set_color('white')
plt.legend(
    fontsize=15, loc='best', title='Marital_Status-Purchase', frameon=False)
plt.axis('equal')
df_M_A = data_f.groupby(['Marital_Status', 'Age'])['Purchase'].sum().unstack()
df_M_A.plot.bar(figsize=(10, 8))
plt.title('Age-Marital_Status-Purchase', fontsize=20)
plt.xlabel('Marital_Status', fontsize=20)
plt.ylabel('Purchase', fontsize=20)
plt.grid(axis='y', linestyle=':')
ax = plt.gca()
ax.set_xticklabels(
    df_M_A.index, fontsize=12, rotation=0, horizontalalignment='center')
plt.legend(df_M_A.columns, fontsize=15, loc='best', title='Age', frameon=False)
df_M_P = data_f.groupby('Marital_Status')[[
    'Product_Category_1', 'Product_Category_2', 'Product_Category_3'
]].count()
df_M_P.plot.bar(figsize=(10, 8))
plt.title('Marital_Status-Product_Category', fontsize=20)
plt.xlabel('Marital_Status', fontsize=20)
plt.ylabel('Number_Of_Times', fontsize=20)
plt.grid(axis='y', linestyle=':')
ax = plt.gca()
ax.set_xticklabels(
    df_M_P.index, fontsize=12, rotation=0, horizontalalignment='center')
plt.legend(
    df_M_P.columns,
    fontsize=15,
    loc='best',
    title='Product_Category',
    frameon=False)
df_M_P = data_f.groupby('Marital_Status')[[
    'Product_Category_1', 'Product_Category_2', 'Product_Category_3'
]].sum()
df_M_P.plot.bar(figsize=(10, 8))
plt.title('Marital_Status-Product_Category', fontsize=20)
plt.xlabel('Marital_Status', fontsize=20)
plt.ylabel('Number_Of_Product', fontsize=20)
plt.grid(axis='y', linestyle=':')
ax = plt.gca()
ax.set_xticklabels(
    df_M_P.index, fontsize=12, rotation=0, horizontalalignment='center')
plt.legend(
    df_M_P.columns,
    fontsize=15,
    loc='best',
    title='Product_Category',
    frameon=False)
plt.figure(figsize=(15, 8))

sizes = data_f.groupby('Stay_In_Current_City_Years')['Purchase'].sum(
).sort_values(ascending=False)
labels = data_f.groupby('Stay_In_Current_City_Years')['Purchase'].sum(
).sort_values(ascending=False).rename(
    index={
        '0': 'Geust',
        '1': 'First Year',
        '2': 'Second Year',
        '3': 'Third Year',
        '4+': 'More Than Four Years'
    }).index
colors = ['tomato', 'deepskyblue', 'aqua', 'lime', 'violet']

patches, l_text, p_text = plt.pie(
    sizes,
    labels=labels,
    colors=colors,
    explode=(0.08, 0.08, 0, 0, 0),
    autopct='%3.1f%%',
    startangle=90,
    shadow=True)

for t in l_text + p_text:
    t.set_size(20)
for t in p_text:
    t.set_color('white')
plt.legend(
    fontsize=12,
    loc='best',
    title='Stay_In_Current_City_Years-Purchase',
    frameon=False)
plt.axis('equal')
df_Y_P = data_f.groupby(
    'Stay_In_Current_City_Years')['Purchase'].sum().sort_values(ascending=True)
df_Y_P.plot.bar(figsize=(10, 8))
plt.title('Stay_In_Current_City_Years-Purchase', fontsize=20)
plt.xlabel('Stay_In_Current_City_Years', fontsize=20)
plt.ylabel('Purchase', fontsize=20)
plt.grid(axis='y', linestyle=':')
ax = plt.gca()
ax.set_xticklabels(
    df_Y_P.index, fontsize=12, rotation=0, horizontalalignment='center')
df_Y_P = data_f.groupby('Stay_In_Current_City_Years')[[
    'Product_Category_1', 'Product_Category_2', 'Product_Category_3'
]].sum()
df_Y_P.plot.bar(figsize=(10, 8))
plt.title('Stay_In_Current_City_Years-Product', fontsize=20)
plt.xlabel('Stay_In_Current_City_Years', fontsize=20)
plt.ylabel('Number_Of_Product', fontsize=20)
plt.grid(axis='y', linestyle=':')
ax = plt.gca()
ax.set_xticklabels(
    df_Y_P.index, fontsize=12, rotation=0, horizontalalignment='center')
plt.legend(
    df_Y_P.columns,
    fontsize=15,
    loc='best',
    title='Product_Category',
    frameon=False)
df_O_P = data_f.groupby('Occupation')[[
    'Product_Category_1', 'Product_Category_2', 'Product_Category_3'
]].count().sort_values(
    by='Product_Category_2', ascending=True)
df_O_P.plot.barh(figsize=(15, 15))
plt.title('Occupation-Product-Num', fontsize=20)
plt.xlabel('number of times', fontsize=20)
plt.ylabel('Occupation', fontsize=20)
plt.grid(axis='x', linestyle=':')
ax = plt.gca()
ax.set_yticklabels(
    df_O_P.index, fontsize=12, rotation=0, horizontalalignment='center')
plt.legend(
    df_O_P.columns,
    fontsize=15,
    loc='best',
    title='Product_Category',
    frameon=False)
df_O_P = data_f.groupby('Occupation')[[
    'Product_Category_1', 'Product_Category_2', 'Product_Category_3'
]].sum().sort_values(
    by='Product_Category_2', ascending=True)
df_O_P.plot.barh(figsize=(15, 15))
plt.title('Occupation-Product-Sum', fontsize=20)
plt.xlabel('Number of pieces', fontsize=20)
plt.ylabel('Occupation', fontsize=20)
plt.grid(axis='x', linestyle=':')
ax = plt.gca()
ax.set_yticklabels(
    df_O_P.index, fontsize=12, rotation=0, horizontalalignment='center')
plt.legend(
    df_O_P.columns,
    fontsize=15,
    loc='best',
    title='Product_Category',
    frameon=False)
df_O_P = data_f.groupby('Occupation')['Purchase'].sum().sort_values(
    ascending=True)
df_O_P.plot.barh(figsize=(15, 15))
plt.title('Occupation-Purchase', fontsize=20)
plt.xlabel('Purchase', fontsize=20)
plt.ylabel('Occupation', fontsize=20)
plt.grid(axis='x', linestyle=':')
ax = plt.gca()
ax.set_yticklabels(
    df_O_P.index, fontsize=12, rotation=0, horizontalalignment='center')
plt.legend()
df_O_P = data_f.groupby('Product_ID')['Purchase'].count().nlargest(
    10).sort_values(ascending=True)
df_O_P.plot.barh(figsize=(10, 10))
plt.title('Product_ID-Count-TOP10', fontsize=20)
plt.xlabel('count', fontsize=20)
plt.ylabel('Product_ID', fontsize=20)
plt.grid(axis='x', linestyle=':')
ax = plt.gca()
ax.set_yticklabels(
    df_O_P.index, fontsize=12, rotation=0, horizontalalignment='right')
plt.legend()
df_O_P = data_f.groupby('Product_ID')['Purchase'].sum().nlargest(
    10).sort_values(ascending=True)
df_O_P.plot.barh(figsize=(10, 10))
plt.title('Product_ID-Purchase-TOP10', fontsize=20)
plt.xlabel('Purchase', fontsize=20)
plt.ylabel('Product_ID', fontsize=20)
plt.grid(axis='x', linestyle=':')
ax = plt.gca()
ax.set_yticklabels(
    df_O_P.index, fontsize=12, rotation=0, horizontalalignment='right')
plt.legend()
df_M_PID = data_f[data_f['Gender'] == 'M'].groupby(
    'Product_ID')['Purchase'].sum().nlargest(10).sort_values(ascending=True)
df_M_PID.plot.barh(figsize=(10, 10))
plt.title('Product_ID-Purchase-TOP10-M', fontsize=20)
plt.xlabel('Purchase', fontsize=20)
plt.ylabel('Product_ID', fontsize=20)
plt.grid(axis='x', linestyle=':')
ax = plt.gca()
ax.set_yticklabels(
    df_M_PID.index, fontsize=12, rotation=0, horizontalalignment='right')
plt.legend()
df_F_PID = data_f[data_f['Gender'] == 'F'].groupby(
    'Product_ID')['Purchase'].sum().nlargest(10).sort_values(ascending=True)
df_F_PID.plot.barh(figsize=(10, 10))
plt.title('Product_ID-Purchase-TOP10-F', fontsize=20)
plt.xlabel('Purchase', fontsize=20)
plt.ylabel('Product_ID', fontsize=20)
plt.grid(axis='x', linestyle=':')
ax = plt.gca()
ax.set_yticklabels(
    df_M_PID.index, fontsize=12, rotation=0, horizontalalignment='right')
plt.legend()

