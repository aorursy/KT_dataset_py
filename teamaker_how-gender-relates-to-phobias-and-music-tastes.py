import pandas as pd

df = pd.read_csv('../input/responses.csv', header=0)



df.rename(columns = {'Metal or Hardrock':'Metal',

                     'Number of siblings':'NumSiblings',

                    'Left - right handed':'Hand',

                    'House - block of flats':'Block'}, inplace=True)



music = df.columns[:19].tolist()

phobias = df.columns[63:73].tolist()

demographics = df.columns[-10:].tolist()

df = df[music+phobias+demographics]



print (df.shape)
print (demographics)

print (music)

print (phobias)

df_metal_male = df[df['Gender']=='male']['Metal']

df_metal_female = df[df['Gender']=='female']['Metal']

print (df_metal_male.mean(), df_metal_male.std())

print (df_metal_female.mean(), df_metal_female.std())

from scipy import stats

stats.ttest_ind(df_metal_male.dropna(), df_metal_female.dropna())

#df_metal_male.value_counts()

#df_metal_male[isnan(df_metal_male)]

df_metal_male.value_counts().plot(kind='bar')
df_metal_female.value_counts().plot(kind='bar')
import math

def effect_size(series1, series2):    

    diff = series1.mean() - series2.mean()

    var1 = series1.var()

    var2 = series2.var()

    n1, n2 = len(series1), len(series2)

    pooled_var = (n1*var1 + n2*var2)/(n1+n2)

    return diff/ math.sqrt(pooled_var)



print (effect_size(df_metal_male, df_metal_female))
df_height_male =  df[df['Gender']=='male']['Height']

df_height_female =  df[df['Gender']=='female']['Height']

print (effect_size(df_height_male, df_height_female))
df_height_male =  df[df['Gender']=='male']['Weight']

df_height_female =  df[df['Gender']=='female']['Weight']

print (effect_size(df_height_male, df_height_female))
effect_sizes = []

for col in music+phobias:

    df_male = df[df['Gender']=='male'][col]

    df_female = df[df['Gender']=='female'][col]

    effect_sizes.append((col, effect_size(df_male, df_female)))

df_final = pd.DataFrame(sorted(effect_sizes, key=lambda a:abs(a[1]), reverse=True))

df_final.head(10)