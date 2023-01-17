import pandas as pd

import numpy as np

import requests



save_path = "../output/"



import matplotlib.pyplot as plt



from scipy.optimize import curve_fit

from scipy.stats import norm
giturl = "https://api.github.com/repos/beoutbreakprepared/nCoV2019/contents/fatality_rates?ref=master"



res=requests.get(giturl)

js = res.json()



dfs = {}

for ent in js:

    fname = ent['name']

    fext = fname[fname.find('.')+1:]

    

    if not(fname in dfs) and fext=='csv':

        try:

            df = pd.read_csv(ent['download_url'], encoding = "ISO-8859-1")

            dfs[ent['name']] = df

        except:

            print(ent['name'] + " ERROR", flush=True)

    else:

        print("skipping " + fname + " - " + fext)
url = "https://github.com/beoutbreakprepared/nCoV2019/raw/master/fatality_rates/cfr_age_sources.xlsx"

sources = pd.read_excel(url)
for fn in sources.data_filename:

    if not fn in dfs:

        print("missing: " + fn)
all_cols = {}

all_dfs = []

for fn in dfs:

    fn_elems = fn.split(sep='.')[0].split(sep='_')

    df = dfs[fn].reset_index()

    

    # convert column names to lowercase

    cols = df.columns.array

    for c in cols:

        if c != c.lower() and not(c.lower() in cols):

            df.rename(columns={c: c.lower()}, inplace=True)

    

    # skip if data not recognized

    if not('age_start' in df.columns) or not('cases' in df.columns) and not('deaths' in df.columns):

        print("ignoring " + fn)

        continue

        

    # extract info from file_name   

    fn_sex = 'both'

    fn_date = ''

    fn_kind = 'cases'

    xnum = 1

    for xc in ['x_1', 'x_2', 'x_3', 'x_4']:

        if xc in df.columns:

            df.drop([xc], axis=1)

    for i in range(1, len(fn_elems)):

        if fn_elems[i] in ['both', 'male', 'female']:

            fn_sex = fn_elems[i]

        elif fn_elems[i].isdigit() and len(fn_elems[i]) == 8:

            fn_date = fn_elems[i]

        elif fn_elems[i] in ['cases', 'mortality', 'deaths']:

            fn_kind = fn_elems[i]

        elif fn_elems[i] in ['siensano', 'linelist', 'bolletino', 'COVID19', 'SOURCE']:

            pass

        else:

            df['x_' + str(xnum)] = fn_elems[i]

            xnum = xnum+1

            

    df['file_name'] = fn

    df['country_code'] = fn_elems[0]

    df['file_sex'] = fn_sex

    df['file_date'] = fn_date

    df['file_kind'] = fn_kind

    

    all_dfs = all_dfs + [df]



    for cn in df.columns:

        all_cols[cn] = True

        

all_cols = sorted(all_cols)



for df in all_dfs:

    for cn in all_cols:

        if not cn in df.columns:

            df[cn] = ''



all_data = pd.concat(all_dfs)

all_data = all_data[all_cols]





# keep only the latest records

x = all_data

u = pd.DataFrame(x.groupby(by=['country_code', 'file_kind', 'file_sex']).file_date.max()).reset_index()

x.set_index(['country_code', 'file_kind', 'file_sex', 'file_date'], inplace=True)

u.set_index(['country_code', 'file_kind', 'file_sex', 'file_date'], inplace=True)

x = x.join(u, how='inner').reset_index().sort_values(by=['country_code', 'age_start', 'age_end', 'file_sex', 'file_kind'])



x.to_csv("covid_agegroup.csv")

x.shape



x.deaths = x.deaths.map(lambda v: int(v) if str(v).isdigit() else 0)

x.cases = x.cases.map(lambda v: int(v) if str(v).isdigit() else 1)

x.age_end = x.age_end.map(lambda v: int(v) if str(v).isdigit() else 110)
x[(x.country_code=='BEL') & (x.age_end>=20) & (x.age_start<30) & (x.file_sex=='both')]
x[x.file_sex=='both'].country_code.unique().shape
df = pd.read_csv("covid_agegroup.csv")

df.age_end = df.apply(lambda r: r.age_end if str(r.age_end).isnumeric() else r.age_start+9, axis=1).astype('int')
ag = pd.DataFrame({'age_start': np.linspace(0,100, 11)})

ag['age_end'] = ag['age_start'] + 9
agok = ag.set_index(['age_start', 'age_end']).join(df.set_index(['age_start', 'age_end']), how='left', lsuffix='_')

agok.country_code.unique().shape

agok
df = pd.read_csv("covid_agegroup.csv")

df = df[(df.cases>=0) & (df.deaths>=0) & (df.file_sex=='both')].sort_values(by=['country_code', 'age_start'])
def left_gauss_scale(xv, mean, std, a, b):

    return [ a + b*norm.pdf(x if x<mean else mean, mean, std) for x in xv ]



features = ['country_code', 'age_start', 'age_end', 'cases', 'deaths']

xx = np.linspace(-1, 100)

fig, ax = plt.subplots(figsize=(20, 10))

ax.set_yscale('log')

ax.grid()





for cc in df[df.country_code.isin(['FRA', 'ESP', 'DNK', 'DEU'])].country_code.unique():

    try:

        dfc = df[df.country_code == cc].sort_values(by='age_start')[features]

        dfc.age_start = dfc.age_start.astype('int')

        dfc['age_end'] = dfc.apply(lambda r: int(r.age_end) if r.age_end.isdigit() else r.age_start+10, axis=1)

        x = (dfc.age_end + dfc.age_start) / 2

        y = dfc.deaths / dfc.cases



        p0 = [90, 70, 0.02, 0.5]

        bounds1_min = [40, 5, -1, 0]

        bounds1_max = [200, 70, 5, 50]

        popt1, pcov1 = curve_fit(left_gauss_scale, x, y, maxfev = 10000, p0=p0, sigma=y**0.9+1e-4, bounds=(bounds1_min, bounds1_max))

        #popt1, pcov1 = curve_fit(left_gauss_scale, x, y, maxfev = 10000, p0=p0, sigma=y**0.9, bounds=(bounds1_min, bounds1_max))



        ax.plot(x,y, label=f"cfr {cc}")

        ax.plot(xx, left_gauss_scale(xx, *popt1), label=f"fit cfr {cc}")

        

        print(f"PARAMS [{cc}]: {popt1}")

    except:

        print("ERROR: " + cc)

        

ax.legend(loc="upper left")

ax.set_yticks([0.0008*2**x for x in range(10)])