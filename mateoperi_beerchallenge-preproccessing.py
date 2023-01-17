import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



from sklearn.preprocessing    import LabelEncoder, OneHotEncoder
df = pd.read_csv("data/beer_train.csv")

df.head()
cat = df.select_dtypes(include=[object]).columns

print("\nCategorical features:\n", cat.values)
style_encoder = LabelEncoder()

style_encoder.fit(df['Style'])



#df[cat] = df[cat].apply(LabelEncoder().fit_transform)

#df['Style'] = df[cat].apply(style_encoder.fit_transform)



#df['Style'] = df['Style'].apply(style_encoder.fit_transform)

df['Style'] = style_encoder.fit_transform(df['Style'])

df.head()
def encode_and_bind(original_dataframe, feature_to_encode):

    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])#, prefix=feature_to_encode)

    res = pd.concat([original_dataframe, dummies], axis=1)

    res = res.drop([feature_to_encode], axis=1)

    return(res)
for c in cat.values:

  if c != 'Style':

    df = encode_and_bind(df, c)

df.head()
from sklearn.impute import SimpleImputer



# Imputar

imputed_df = pd.DataFrame(SimpleImputer().fit_transform(df))

# Restaurar nombres de columnas

imputed_df.columns = df.columns

df = imputed_df

df.head()
# Features

X = df[['Size(L)', 'OG', 'FG', 'ABV', 'IBU', 'Color', 'BoilSize', 'BoilTime',

        'BoilGravity', 'Efficiency', 'MashThickness', 'PitchRate',

        'PrimaryTemp', 'SugarScale_Plato', 'SugarScale_Specific Gravity',

        'BrewMethod_All Grain', 'BrewMethod_BIAB', 'BrewMethod_Partial Mash', 'BrewMethod_extract'

       ]]



# Label

y = df["Style"].values

X[0:5]
final_df = pd.read_csv("data/5.csv")

final_df.head()
# Generate solution Dataframe

final_df = final_df[['predict']]

# Name Index to 'Id'

final_df.index.name = 'Id'

final_df.columns = ['Style']

final_df.head()
# Decode values

final_df['Style'] = style_encoder.inverse_transform(final_df['Style'].astype('int')) # como pasamos el encoder?

final_df.head()
# Save to file

final_df.to_csv("5_final.csv")