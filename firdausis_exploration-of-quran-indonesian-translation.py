import pandas as pd

df = pd.read_csv('../input/Indonesian.csv', sep='*', header=None, names=['row'])
df.iloc[2101].row = df.iloc[2101].row.replace('tentu|ah', 'tentulah')
df = pd.DataFrame(df.row.str.split('|').tolist(), 
                  columns = ['surah','ayah','text'])
df.surah = df.surah.astype(int)
df.ayah = df.ayah.astype(int)
print('Setup completed')
df[df.surah == 1]