import pandas as pd

import numpy as np

from sklearn.linear_model import LogisticRegression

import csv



def prep_data(path, train_means, train_merken):

    df = pd.read_csv(path, parse_dates=['Datum eerste toelating', 'Datum tenaamstelling'])

    df.fillna(train_means, inplace=True)

    df['Datum eerste toelating'].fillna(df['Datum tenaamstelling'], inplace=True)

    features = ['Catalogusprijs', 'Massa ledig voertuig', 'Wielbasis',

                'Aantal cilinders', 'Cilinderinhoud']

    for merk in train_merken:

        df['merk_%s' % merk] = (df['Merk'] == merk).astype('int')

        features.append('merk_%s' % merk)

    for name, short in [

        ('Datum eerste toelating', 'dat_toel'),

        ('Datum tenaamstelling', 'dat_naam')]:

        df['%s_year' % short] = df[name].dt.year

        df['%s_weekday' % short] = df[name].dt.weekday

        df['%s_yearday' % short] = df[name].dt.dayofyear

        df['%s_d70' % short] = (df[name] - pd.Timestamp('1970-01-01')).dt.days

        features.extend(['%s_year' % short, '%s_weekday' % short,

                         '%s_yearday' % short, '%s_d70' % short])

    df['days_between'] = (df['Datum tenaamstelling'] - df['Datum eerste toelating']).dt.days

    features.append('days_between')

    current_date = max(df['Datum tenaamstelling'])

    current_days = [(current_date - dat_naam).days for dat_naam in df['Datum tenaamstelling']]

    total_days = [(current_date - dat_toel).days for dat_toel in df['Datum eerste toelating']]

    df['percent_current'] = [100.0*cur/tot if tot>0 else 100. for cur, tot in zip(current_days, total_days)]

    features.append('percent_current')

    df['log_cata_prijs'] = np.log10(df['Catalogusprijs'])

    features.append('log_cata_prijs')

    

    return features, df[features], df['WAM verzekerd'] if 'WAM verzekerd' in df else df['Kenteken']



train_df = pd.read_csv('../input/train.csv')

train_means = train_df[['Wielbasis', 'Massa ledig voertuig']].dropna().mean()

train_merken = train_df['Merk'].value_counts()[:50].index



names, xtrain, ytrain = prep_data("../input/train.csv", train_means, train_merken)

names, xtest, idtest = prep_data("../input/test.csv", train_means, train_merken)



model = LogisticRegression()

model.fit(xtrain, ytrain)

for name, coef in zip(names, model.coef_[0]):

    print('%32s %f' % (name, coef))



pred = model.predict_proba(xtest)[:,1]

with open('submission.csv', 'w') as f:

    w = csv.writer(f)

    w.writerow(['Kenteken', 'Prediction'])

    for row in zip(idtest, pred):

        w.writerow(row)