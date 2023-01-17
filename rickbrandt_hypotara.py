import datetime as dt 



import pandas as pd
def bereken_lasten(rente_perc, maandlasten, n_maanden):

    restschuld = 128772

#     n_maanden = 14*12 + 9 

    datum = 2019 * 12 + 9

#     maandlasten = 706

#     rente_perc = 4.3



    D = []



    for _m in range(n_maanden):

        _rente = restschuld * rente_perc / 100 / 12

        _aflossing = maandlasten - _rente

        restschuld -= _aflossing

        D.append([datum, _rente, _aflossing, restschuld])

        datum += 1



    df = pd.DataFrame(D, columns = ['datum', 'rente','aflossing','restschuld'])

    df.datum = df.datum.map(lambda i: dt.date(int(i / 12), i % 12 + 1, 1))

    df.set_index('datum', inplace=True)

    

    return df

# n_m = 14 * 12 + 9 

# n_m = 24  

n_m = 180



df0 = bereken_lasten(4.3, 706, n_m)

df1 = bereken_lasten(3.1, 620, n_m)
%pylab inline 



f, ax = plt.subplots()

df0[['rente', 'aflossing']].plot(ax=ax)

df1[['rente', 'aflossing']].plot(ax=ax)
df0.restschuld.plot()

df1.restschuld.plot()
pd.concat([df0.sum(), df1.sum()], axis=1).applymap(int)
pd.concat([df0.sum() - df1.sum()], axis=1)