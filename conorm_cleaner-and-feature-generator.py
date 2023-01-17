import pandas as pd
import numpy as np

def gen_features(train, test):

    train.columns = train.columns.str.lower()
    test.columns = test.columns.str.lower()
    train_len = len(train)
    train.rename(columns={'animalid' : 'id'}, inplace=True)
    full = train.append(test)
    
    
    full['named'] = np.where(full.name.isnull(), 0, 1)
    df = full['ageuponoutcome'].str.split(' ', expand = True)
    full['age1'], full['age2'] = df.ix[:, 0], df.ix[:, 1]
    full['age2'] = full['age2'].str.replace('s', '')


    multiplier = {'year' : 365,
                'week' : 7,
                'month' : 30.5,
                'day' : 1,
                None : 0}

    full['age_mult'] = full.age2.map(multiplier)
    full['ages'] = np.where((full['age1'] == 'nan'), np.NaN, full.age1)
    full['ages'] = full['ages'].fillna(value=4).astype(int) # mean ages is 3.6
    full['age_days'] = full.ages * full.age_mult
    full['pupkit'] = (full.age_days <= 365.0).astype(int)

    df_sex = full.sexuponoutcome.str.split(' ', expand=True)
    full['intact'], full['gend'] = df_sex.iloc[:, 0], df_sex.iloc[:, 1]
    full['intactness'] = full.sexuponoutcome.str.contains('Intact')
    full['intactness'] = np.where(full['intactness'] == True, 1, 0)
    full['gend'] = full['gend'].fillna('Female')
    full['gend'] = np.where(full['gend'] == 'Male', 1, 0)

    full['breeds'] = full.breed.str.split('/').str.get(0).str.replace('Mix', '').str.strip()
    full['breed_id'], breed_fac = pd.factorize(full.breeds)

    full['colors_n'] = full.color.str.replace('/', ' ').str.split(' ').str.len()
    full['colour'] = full.color.str.replace('/', ' ').str.split(' ').str.get(0)
    full['colour'], col_fac = pd.factorize(full.colour)

    full['dtime'] = pd.to_datetime(full['datetime'])
    full['hour'] = full['dtime'].dt.hour
    full['wday'] = full['dtime'].dt.weekday
    full['mday'] = full['dtime'].dt.day
    full['month'] = full['dtime'].dt.month
    full['year'] = full['dtime'].dt.year

    full['animaltype'] = (full.animaltype == 'Dog').astype(int) # 1 = dog
    full['mix_breed'] = (full.breed.str.contains('Mix').astype(int))

    full.drop(['id', 'name', 'datetime', 'dtime' 'outcomesubtype', 'sexuponoutcome', 'ageuponoutcome', 'breed',
               'color', 'breeds', 'age1', 'age2', 'age_mult', 'intact'], 1, inplace=True)

    train = full.iloc[0:train_len, :]
    test = full.iloc[train_len:len(full), :]

    return train, test