import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.preprocessing import OneHotEncoder



from keras.models import Model

from keras.layers import Input, Dense, BatchNormalization
df1 = pd.read_pickle("/kaggle/input/death-and-population-in-france-19902019/preprocessed_data/INSEE_deces_1990_1999.pkl")

df2 = pd.read_pickle("/kaggle/input/death-and-population-in-france-19902019/preprocessed_data/INSEE_deces_2000_2009.pkl")

df3 = pd.read_pickle("/kaggle/input/death-and-population-in-france-19902019/preprocessed_data/INSEE_deces_2010_2019.pkl")



df = pd.concat([df1, df2, df3], ignore_index=True)



df = df.rename(columns={'departement_deces': 'departement'})
df = df[df['date_deces']>='1990-01-01']
df = df.groupby(["year","weeknumber", "age_bin", "departement"], as_index=False)["nb_deces"].sum()
df.head()
df['nb_deces'] = df['nb_deces'].astype('float') # matplotlib struggles with pandas int
adf = df.copy()

adf['nb_deces_jan_feb'] = df['nb_deces'] * (df['weeknumber'] < 10)

adf['nb_deces_jul_aug'] = df['nb_deces'] * ((df['weeknumber'] > 25) & (df['weeknumber'] < 35))

adf['nb_deces_all_year'] = df['nb_deces']
tmp = adf.groupby(by=['year', 'departement', 'age_bin']).sum()[['nb_deces_jan_feb', 'nb_deces_jul_aug', 'nb_deces_all_year']]

gdf = df.join(tmp, how='left', on=['year', 'departement', 'age_bin'])

gdf.head()
# Read population dataframe



population = pd.read_csv('/kaggle/input/death-and-population-in-france-19902019/data/population_insee_dept_year_sex_age.csv', delimiter=';')



# Map age_bins to classe_age



population['age_bin'] = population['classe_age'].map({

    '00 à 19 ans' : 1,

    '20 à 39 ans' : 20,

    '40 à 59 ans' : 40,

    '60 à 74 ans' : 60,

    '75 ans et plus' : 75

})



population = population[['annee', 'departement_code', 'age_bin', 'population']]

population = population.rename(columns={

    'departement_code' : 'departement',

    'annee' : 'year'})



# Sum male and female in population

population = population.groupby(by=['year', 'departement', 'age_bin'], as_index=False).sum()



# Process Corsica (sum departements '2A' and '2B', and store the result in departement '20')

corse = population[(population['departement'] == '2A') | (population['departement'] == '2B')].groupby(by=['year', 'age_bin'], as_index=False).sum()

corse['departement'] = '20'



# Concat Corsica with other departements

population = pd.concat([population, corse], axis=0, sort=False, ignore_index=True)



# Set index

population = population.set_index(['year', 'departement', 'age_bin'])



population.head()
# Join population to death dataframe

# Note that we use an inner join, therefore departements for which we don't have the population will not be kept



jdf = gdf.join(population, how='inner', on=['year', 'departement', 'age_bin'])

jdf.head()
# Remove weeknumber



kdf = jdf.groupby(by=['year', 'age_bin', 'departement']).max()[['nb_deces_jan_feb', 'nb_deces_jul_aug', 'nb_deces_all_year', 'population']]

kdf = kdf.reset_index()

kdf.head()
# Define a year for train / test split (before this year: train, this year and after: test)

train_test_split_year = 2013
# Features engineering

kdf['ratio_jan_feb_pop'] = kdf['nb_deces_jan_feb'] / kdf['population']

kdf['ratio_jul_aug_pop'] = kdf['nb_deces_jul_aug'] / kdf['population']

kdf['ratio_all_year_pop'] = kdf['nb_deces_all_year'] / kdf['population']



# Get only train data for average features

kdf_train = kdf[kdf['year'] < train_test_split_year]



mean_df = kdf_train.groupby(by=['age_bin', 'departement']).mean()[['ratio_jan_feb_pop', 'ratio_jul_aug_pop', 'ratio_all_year_pop']]

mean_df = mean_df.rename(columns={

                                'ratio_jan_feb_pop' : 'avg_jan_feb_pop',

                                'ratio_jul_aug_pop' : 'avg_jul_aug_pop',

                                'ratio_all_year_pop': 'avg_all_year_pop'})



ldf = kdf.join(mean_df, how='left', on=['age_bin', 'departement'])



ldf = ldf[ldf['avg_jan_feb_pop'].notnull()]



ldf.tail()
# Create X



def preprocess_X(df):

    # Transform numerical values to numpy array

    numerical = df[['population', 'ratio_jan_feb_pop', 'avg_jan_feb_pop', 'avg_jul_aug_pop', 'avg_all_year_pop']]

    numerical = numerical.to_numpy()



    # One-hot encoding of categorcal values, and transform into numpy array

    categorical = df['age_bin'].to_numpy()[..., np.newaxis]

    onehot_encoder = OneHotEncoder(sparse=False)

    onehot_encoder = onehot_encoder.fit(categorical)

    onehot_encoded = onehot_encoder.transform(categorical)



    # Concatenate numpy arrays (numerical and categorical)

    X = np.append(numerical, onehot_encoded, axis=1)

    

    return X



#X = preprocess_X(ldf)

#X.shape
# Create y



# df: dataframe with weeknumbers

# ldf: dataframe without weeknumbers, same dataframe than the one given to preprocess_X

def preprocess_y(df, ldf):

    # Put weeks in columns

    pivot = pd.pivot_table(df, index=['year', 'age_bin', 'departement'], columns='weeknumber', values='nb_deces')

    # Keep only 52 weeks

    pivot = pivot.iloc[:, :52]



    y = ldf.join(pivot, how='left', on=['year', 'age_bin', 'departement']).iloc[:, -52:]

    y = y.to_numpy()

    

    return y

    

#y = preprocess_y(df, ldf)

#y.shape
# Create train / test split



train_df = ldf[ldf['year'] < train_test_split_year]

test_df = ldf[ldf['year'] >= train_test_split_year]



X_train = preprocess_X(train_df)

y_train = preprocess_y(df, train_df)

X_test = preprocess_X(test_df)

y_test = preprocess_y(df, test_df)



# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



X_train.shape[1]
# Create model



def get_model(input_shape=9, output_shape=52):

    x_in = Input(shape=(input_shape,))

    x = BatchNormalization()(x_in)

    x = Dense(500, activation='relu')(x)

    x = Dense(500, activation='relu')(x)

    x = Dense(1000, activation='relu')(x)



    output = Dense(output_shape)(x)

    model = Model(inputs=x_in, outputs=output)

    

    model.compile(loss='mse', optimizer='adam')



    return model



model = get_model(input_shape=X_train.shape[1])



print(model.summary())
# Train model



model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=256, epochs=100)
# Predict on validation dataset

y_pred = model.predict(X_test)
# Plot one prediction





index = 15052 - X_train.shape[0]



ax = plt.subplot(1,1,1)

p1, = ax.plot(np.arange(52), y_pred[index].astype('float'), '-', label="pred");

p2, = ax.plot(np.arange(52), y_test[index].astype('float'), '-', label="ground truth");



ax.legend()

plt.title("Death records in France according to INSEE")

plt.xlabel("Week number")

plt.ylabel("Number of deaths")

plt.show()
# Plot several predictions



# Number of plots (lines, columns)

i,j = 12,3



# Choose random samples in the test dataset

max_index = X_test.shape[0]

np.random.seed(40)

indexes = [np.random.randint(max_index) for i in range(i*j)]



plt.figure(figsize=(20,60))



for k in range(i*j):

    index = indexes[k]

    ax = plt.subplot(i,j,k + 1)

    p1, = ax.plot(np.arange(52), y_pred[index].astype('float'), '-', label="pred");

    p2, = ax.plot(np.arange(52), y_test[index].astype('float'), '-', label="ground truth");



    sample = test_df.iloc[index]

    title = "year: " + str(sample['year']) + " age_bin:" + str(sample['age_bin']) + " dpt:" + sample['departement']

    

    plt.title(title)

    

    ax.legend()

    plt.xlabel("Week number")

    plt.ylabel("Number of deaths")

plt.show()