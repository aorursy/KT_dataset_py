import pandas as pd 



file = open('../input/cholera-dataset/data.csv', 'r')

df = pd.read_csv(file)
df['Number of reported cases of cholera'].iloc[1059] = '35' # Iraq in 2016 had 35 cases, in dataset it is written as '3 5'

df['Number of reported cases of cholera'] = df['Number of reported cases of cholera'].fillna('0')

df['Number of reported cases of cholera'] = df['Number of reported cases of cholera'].apply(int)
df = df[df['Year'] < 2010] # filter by year later than 2010

check = df['Number of reported cases of cholera'] != 0 # filter by finding all nonzero reported cases

count = 0

ind_true = []

for i in check:

    if i is True:

        ind_true.append(count)

    count += 1



        

df_orig_ind = df.index.values.tolist()

ind = [x for x in df_orig_ind if x not in ind_true]

ans = df[df.index.isin(ind)]

ans.drop_duplicates(subset ="Country", keep = False, inplace = True) 

print(ans['Country'])